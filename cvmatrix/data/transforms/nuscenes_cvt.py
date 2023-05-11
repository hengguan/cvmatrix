# -*- encoding: utf-8 -*-
'''
@File    :   nuscenes_cvt.py
@Time    :   2022-11-02 14:00:16
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   Cross-View-Transformer data process code.
'''

import pathlib

import torch
import torchvision
import numpy as np

from PIL import Image
import cv2

"""
https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/transforms.py
"""
import imgaug.augmenters as iaa
from pyquaternion import Quaternion

__all__ = ['get_view_matrix', 'get_transformation_matrix', 'get_pose', 'encode', 'decode',
            'StrongAug', 'GeometricAug', 'Sample', 'SaveDataTransform', 'LoadDataTransform']

INTERPOLATION = cv2.LINE_8


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    print(rotation)
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)


def encode(x):
    """
    (h, w, c) np.uint8 {0, 255}
    """
    n = x.shape[2]

    # assert n < 16
    assert x.ndim == 3
    assert x.dtype == np.uint8
    assert all(x in [0, 255] for x in np.unique(x))

    shift = np.arange(n, dtype=np.int32)[None, None]

    binary = (x > 0)
    binary = (binary << shift).sum(-1)
    binary = binary.astype(np.int32)

    return binary


def decode(img, n):
    """
    returns (h, w, n) np.int32 {0, 1}
    """
    shift = np.arange(n, dtype=np.int32)[None, None]

    x = np.array(img)[..., None]
    x = (x >> shift) & 1

    return x


class AugBase(torchvision.transforms.ToTensor):
    def __init__(self):
        super().__init__()

        self.augment = self.get_augment().augment_image

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return self.augment(x)


class StrongAug(AugBase):
    def get_augment(self):
        return iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.AddToBrightness((-60, 40)),
            # iaa.AddToHue((-20, 20)),
        ])


class GeometricAug(AugBase):
    def get_augment(self):
        return iaa.Affine(rotate=(-2., 2.),
                          translate_percent=(-0.01, 0.01),
                          scale=(0.975, 1.025),
                          mode='symmetric')


class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        intrinsics,
        extrinsics,
        images,
        view,
        bev,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene

        self.view = view
        self.bev = bev

        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class Normalize(torchvision.transforms.ToTensor):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, 
                norm_resnet=False,
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.norm_resnet = norm_resnet

    def __call__(self, img):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
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

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics
        }

    def get_bev(self, batch: Sample):
        result = {
            'view': batch.view,
        }

        scene_dir = self.labels_dir / batch.scene

        bev_path = f'bev_{batch.token}.png'
        Image.fromarray(encode(batch.bev)).save(scene_dir / bev_path)

        result['bev'] = bev_path

        # Auxilliary labels
        if batch.get('aux') is not None:
            aux_path = f'aux_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux=batch.aux)

            result['aux'] = aux_path

        # Visibility mask
        if batch.get('visibility') is not None:
            visibility_path = f'visibility_{batch.token}.png'
            Image.fromarray(batch.visibility).save(scene_dir / visibility_path)

            result['visibility'] = visibility_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none', normalize=None):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes

        # mean=[103.530, 116.280, 123.675]
        # std=[57.375, 57.120, 58.395]
        norm = dict(normalize)
        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [Normalize(**norm)] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        # self.to_tensor = super().__call__
        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()

        for image_path, I_original in zip(sample.images, sample.intrinsics):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
        }

    def get_bev(self, sample: Sample):
        scene_dir = self.labels_dir / sample.scene
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            bev = decode(bev, self.num_classes)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)

        result = {
            'bev': bev,
            'view': torch.tensor(sample.view),
        }

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])

        if 'pose' in sample:
            result['pose'] = np.float32(sample['pose'])

        return result

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)

        result = dict()
        result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))

        return result
