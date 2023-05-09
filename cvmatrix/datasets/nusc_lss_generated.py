# -*- encoding: utf-8 -*-
'''
@File    :   nusc_lss_generated.py
@Time    :   2022-11-02 13:56:38
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import os
import os.path as osp
import json
import numpy as np
from pyquaternion import Quaternion

import torch
import torch.utils.data as torchData
from PIL import Image
import cv2
from nuscenes.utils.data_classes import Box

from cvmatrix.model.utils.geometry import (
    get_lidar_data, 
    img_transform, 
    normalize_img, 
    gen_dx_bx
)
from .build import DATASETS_REGISTRY


@DATASETS_REGISTRY.register()
class NuscLSSGenerated(torchData.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    def __init__(self, data_root, label_file, is_train, data_aug_conf, grid_conf, split='train', **kwargs):
        super(NuscLSSGenerated, self).__init__(**kwargs)

        fn = 'train_lss.json' if is_train else 'val_lss.json'
        with open(label_file, 'r') as f:
            self.samples = json.load(f)
        
        self.data_root = data_root
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _get_image(self, sample, cams_index):
        imgs, post_rots, post_trans = [], [], []
        intrins, rots, trans = [], [], []
        for i in cams_index:
            img = Image.open(os.path.join(self.data_root, sample['img_path'][i]))

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(sample['intr'][i])
            rot = torch.Tensor(sample['rots'][i])
            tran = torch.Tensor(sample['trans'][i])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (
            torch.stack(imgs, dim=0), torch.stack(rots), 
            torch.stack(trans), torch.stack(intrins), 
            torch.stack(post_rots, dim=0), torch.stack(post_trans, dim=0))

    def _get_target(self, sample):
        trans = -np.array(sample['lidar2ego_trans'])
        rot = Quaternion(sample['lidar2ego_rots']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for b in sample['boxes']:
            box = Box(b['translation'], b['size'], Quaternion(b['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def _choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(
                self.data_aug_conf['cams'], 
                self.data_aug_conf['Ncams'],
                replace=False
            ).tolist()

        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        cams = self._choose_cams()
        cams_index = [self.CAMERAS.index(c) for c in cams]
        imgs, rots, trans, intrins, post_rots, post_trans = self._get_image(sample, cams_index)
        target = self._get_target(sample)
        
        # return imgs, rots, trans, intrins, post_rots, post_trans, target
        return {
            'image': imgs, 
            'rots': rots, 
            'trans': trans, 
            'intrins': intrins, 
            'post_rots': post_rots, 
            'post_trans': post_trans, 
            'target': target
        }


# def worker_rnd_init(x):
#     np.random.seed(13 + x)


# def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
#                  nworkers, parser_name):
#     traindata = NuscGenerated(dataroot, True, data_aug_conf, grid_conf)
#     valdata = NuscGenerated(dataroot, False, data_aug_conf, grid_conf)
#     trainloader = torchData.DataLoader(traindata, batch_size=bsz,
#                                               shuffle=True,
#                                               num_workers=nworkers,
#                                               drop_last=True,
#                                               worker_init_fn=worker_rnd_init)
#     valloader = torchData.DataLoader(valdata, batch_size=bsz,
#                                             shuffle=False,
#                                             num_workers=nworkers)

#     return trainloader, vallo