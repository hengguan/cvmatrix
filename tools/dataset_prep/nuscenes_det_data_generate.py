# -*- encoding: utf-8 -*-
'''
@File    :   nuscenes_det_data_generate.py
@Time    :   2022-11-02 14:13:30
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from cvmatrix.datasets.transforms.nuscenes_cvt import get_pose

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

VERSIONS = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
CLASSES_NAME = list(set([v for k, v in NameMapping.items()]))
print(CLASSES_NAME)


class NuScenesDetDatasetGenerate():

    def __init__(self, dataset_dir, save_dir, version='v1.0-trainval'):
        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=True)

        assert version in VERSIONS
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError('unknown')

        # filter existing scenes.
        all_scenes = [s for s in self.nusc.scene]
        all_scenes_name = [s['name'] for s in all_scenes]
        train_scenes_rec = [
            all_scenes[all_scenes_name.index(s)] for s in train_scenes]
        val_scenes_rec = [
            all_scenes[all_scenes_name.index(s)] for s in val_scenes]

        print(f'train scenes nums: {len(train_scenes_rec)}, val scenes num: {len(val_scenes_rec)}')
        print('-'*30 + ' parse train data ' + '-'*30)
        self.save_dir = save_dir
        self.parse_scene(train_scenes_rec, split='train')

        print('-'*30 + ' parse val data ' + '-'*30)
        self.parse_scene(val_scenes_rec, split='val')
    
    def parse_scene(self, scenes_rec, split):
        data = []
        len_scenes = len(scenes_rec)
        for idx in tqdm(range(len_scenes), position=0, leave=False):
            scene_rec = scenes_rec[idx]
            sample_token = scene_rec['first_sample_token']
            scene_name = scene_rec['name']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                sample_data = dict(scene=scene_name)
                self.parse_sample(sample, sample_data)
                data.append(sample_data)
                sample_token = sample['next']
                
        # Write all info for loading to json
        save_path = Path(self.save_dir) / f'{split}.json'
        with open(str(save_path), 'w') as f:
            json.dump(data, f)
        # save_path.write_text(json.dumps(data))
        # return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample(self, sample, sample_data):
        lidar_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path, boxes, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor',
                             lidar_record['calibrated_sensor_token'])
        annotations = [
            self.nusc.get('sample_annotation', token)
            for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                            for b in boxes]).reshape(-1, 1)
        valid_flag = np.array(
            [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                for anno in annotations],
            dtype=bool).reshape(-1)

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        # names = np.array(names)
        # we need to convert box size to
        # the format of our lidar coordinate system
        # which is x_size, y_size, z_size (corresponding to l, w, h)
        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'
        
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        lidar2egolidar = self.parse_pose(lidar_cs_record, inv=False)

        world_from_egolidarflat = self.parse_pose(egolidar, flat=False)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=False, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []
        egocam2cams = []
        world2egocams = []

        for cam_channel in CAMERAS:
            cam_token = sample['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            # extrinsics.append(E.tolist())
            egocam2cams.append(cam_from_egocam.tolist())
            world2egocams.append(egocam_from_world.tolist())
            images.append(image_path)

        # gt_boxes_ = [box.tolist() for box in gt_boxes]
        sample_data.update(dict(
            lidar_path=lidar_path,
            sample_token=sample['token'],
            gt_boxes=gt_boxes.tolist(),
            gt_names=names,
            valid_flag=valid_flag.tolist(),
            # pose=world_from_egolidarflat.tolist(),
            # pose_inverse=egolidarflat_from_world.tolist(),
            # 'cam_ids': list(camera_rig),
            cam_channels=cam_channels,
            intrinsics=intrinsics,
            lidar2egolidar=lidar2egolidar.tolist(),
            egolidar2world=world_from_egolidarflat.tolist(),
            world2egocam=world2egocams,
            egocam2cam=egocam2cams,
            # extrinsics=extrinsics,
            images=images,
        ))


def verify_generated_dataset(dataset_dir):
    data_file = Path(dataset_dir) / 'train.json'
    with open(str(data_file), 'r') as f:
        data = json.load(f)
    # data = data_file.read_text().strip().split('\n')
    print(f'data length: {len(data)}')

    import cv2
    for sample in data:
        gt_boxes = np.array(sample['gt_boxes'])
        n, d = gt_boxes.shape
        gt_names = sample['gt_names']
        intr = np.array(sample['intrinsics'])
        # extr = np.array(sample['extrinsics'])
        egolidar2world = np.array(sample['egolidar2world'])
        lidar2egolidar = np.array(sample['lidar2egolidar'])
        world2egocam = np.array(sample['world2egocam'])
        egocam2cam = np.array(sample['egocam2cam'])
        
        gt_boxes_loc = np.hstack([gt_boxes[:, :3], np.ones((n, 1))]).T
        
        intr_ = np.zeros((6, 3, 4))
        intr_[:, :, :3] = intr

        image = cv2.imread(str(Path(dataset_dir)/sample['images'][0]))
        h, w, c = image.shape

        pixels_ = intr_[0] @ egocam2cam[0] @ world2egocam[0] @ egolidar2world @ lidar2egolidar @ gt_boxes_loc
        pixels = pixels_[:2, :] / pixels_[2, :]
        pixels = pixels.T.astype(np.int32)

        for p_, vf, name in zip(pixels_.T, sample['valid_flag'], gt_names):
            p = p_[:2] / p_[2]
            if p[0]<0 or p[1] < 0 or p[0] >= w or p[1] >= h or not vf or p_[2] < 0:
                continue
            print(p, p_[2], vf, name)
            cv2.circle(image, tuple(p.astype(np.int32)), 5, (0, 0, 255), thickness=-1)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        for box, name in zip(gt_boxes, gt_names):
            if name not in CLASSES_NAME:
                continue
            loc = box[:3]
            print(loc)
            if loc[0] >50 or loc[0]<-50 or loc[1]>50 or loc[1]<-50:
                continue
            x = (loc[0] - (-50)) / 0.5
            y = (loc[1] - (-50)) / 0.5
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), thickness=-1)
        cv2.circle(img, (100, 100), 2, (0, 0, 255), thickness=-1)
        img = cv2.flip(img, 0)
        cv2.imshow('vis', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    dataset_dir = "D:\\data\\nuScenes\\nuscenes"
    labels_dir = "D:\\data\\nuScenes\\cvt_labels"
    split = 'val'
    version = 'v1.0-trainval'
    # datasets = get_data(dataset_dir, labels_dir, split, version)
    # dataset = torchdata.ConcatDataset(datasets)
    # dataset = NuScenesDetDatasetGenerate(dataset_dir, dataset_dir, version)
    verify_generated_dataset(dataset_dir)