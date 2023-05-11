# -*- encoding: utf-8 -*-
'''
@File    :   changan_dataset.py
@Time    :   2022-11-02 13:58:28
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import torch
import torch.utils.data as torchdata
import torchvision

import os
import os.path as osp
from PIL import Image
import numpy as np
import cv2

from .changan_sensor_calib import get_all_camIE

from .build import DATASETS_REGISTRY


@DATASETS_REGISTRY.register()
class CADataset(torchdata.Dataset):
    CAMERAS = [
        'cam6_front-left',
        'cam1_front', 
        'cam2_front-right',
        'cam5_rear-left',
        'cam4_rear',
        'cam3_rear-right',
        ]

    def __init__(self, img_root, cameras_param=None):
        
        img_dir = osp.join(img_root, self.CAMERAS[0], 'images')
        imgs = []
        for item in os.listdir(img_dir):
            if item.endswith('jpg'):
                imgs.append(item)
        imgs = sorted(imgs)

        self.samples = []
        for img in imgs:
            flag = True
            batch_imgs = []
            for cam in self.CAMERAS:
                img_path = osp.join(img_root, cam, 'images', img)
                if not osp.exists(img_path):
                    flag = False
                    break
                batch_imgs.append(img_path)
            if flag:
                self.samples.append(batch_imgs)
        
        cam_params = get_all_camIE(flat=False)
        
        intrs = np.zeros((6, 3, 3))
        extrs = np.zeros((6, 4, 4))
        for i, (k, v) in enumerate(cam_params.items()):
            intrs[i, ...] = np.array(v["camera_intrinsic"])
            # r = v["rotation"]
            # t = np.array(v["translation"])
            # rm = np.array(v["rot_mat"])
            # extr = np.eye(4)
            # extr[:3, :3] = rm
            # extr[:3, 3] = t
            extrs[i, ...] = v['extrinsic']

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.intrs = torch.from_numpy(intrs)
        self.extrs = torch.from_numpy(extrs.astype(np.float32))
        print(self.intrs.shape, self.extrs.shape)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.to_rgb = False
        self.norm_resnet = False
    
    def _normalize_img(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # cv2 inplace normalization does not accept uint8
        img = img.copy().astype(np.float32)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        if not self.norm_resnet:
            img /= 255.0
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs_path = self.samples[idx]
        
        intrinsics = []
        images = []
        for image_path, I_original in zip(imgs_path, self.intrs):
            if 'cam1_front' in image_path:
                h, w, top_crop = 224, 480, 46
            else:
                h, w, top_crop = 224, 480, 136

            h_resize = h + top_crop
            w_resize = w

            image = Image.open(image_path)

            
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop

            images.append(self.to_tensor(self._normalize_img(image_new)))
            intrinsics.append(torch.from_numpy(I))

        return {
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': self.extrs,
        }


if __name__ == '__main__':
    from tqdm import tqdm
    import cv2
    img_root = "D:\\data\\changan\\imgs"
    dataset = CADataset(img_root)
    dataloader = torchdata.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in tqdm(dataloader):
        images = batch['image']
        imgs = images[0].numpy().transpose(0, 2, 3, 1)
        viz = np.zeros((448, 1440, 3), dtype=np.uint8)
        i, j = 1, 0
        for img in imgs:
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            viz[j*224:(j+1)*224, i*480:(i+1)*480, :] = img
            i = (i + 1) % 3
            if i == 0:
                j = (j+1) % 2
        cv2.imshow('ver', viz)
        cv2.waitKey(0)

        print(batch['intrinsics'])
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4
        print(I_inv)

