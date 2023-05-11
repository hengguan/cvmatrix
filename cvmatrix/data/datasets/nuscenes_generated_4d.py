# -*- encoding: utf-8 -*-
'''
@File    :   nuscenes_generated_4d.py
@Time    :   2022-11-02 13:57:42
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import json
import torch
import numpy as np

from pathlib import Path

from ..build import DATASETS_REGISTRY
from ..transforms.nuscenes_cvt import LoadDataTransform, Sample


@DATASETS_REGISTRY.register()
def get_data(
    num_classes=12,
    version='',
    normalize=None,
    split='train',
    data_dir: str = None,
    labels_root: str = None,
    labels_dir_name: str = None,
    image: dict = None,
    cameras=None,
    augment='none',
    num_time=None,
    **kwargs
):

    dataset_dir = Path(data_dir)
    labels_dir = Path(labels_root) / labels_dir_name

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(
        dataset_dir, labels_dir, image, num_classes, augment, normalize)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_path = Path(labels_root) / f'{split}.txt'
    split_scenes = split_path.read_text().strip().split('\n')

    return [NuScenesGeneratedDataset(s, labels_dir, transform=transform, num_time=num_time) for s in split_scenes]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None, num_time=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform
        self.num_time = num_time
        self.scene_name = int(scene_name[-4:])
        assert num_time>=1, "number of time stamp must be more than 1"

    def __len__(self):
        return len(self.samples) - (self.num_time-1)

    def __getitem__(self, idx):
        datas = []
        for i in range(self.num_time):
            data = Sample(**self.samples[idx+i])

            if self.transform is not None:
                data = self.transform(data)
            
            datas.append(data)
        
        dd = dict()
        for k, v in datas[0].items():
            dd[k] = [v]
        
        for d in datas[1:]:
            for k, v in d.items():
                dd[k] += [v]
        for k, v in dd.items():
            if isinstance(v[0], np.ndarray):
                dd[k] = np.stack(v, axis=0)
            else:
                dd[k] = torch.stack(v, axis=0)
        dd['scene_name'] = self.scene_name
        return dd
