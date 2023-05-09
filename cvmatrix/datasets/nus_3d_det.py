# -*- encoding: utf-8 -*-
'''
@File    :   nus_3d_det.py
@Time    :   2022-11-02 13:52:15
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import json
from pathlib import Path

from torch.utils.data import Dataset

from .build import DATASETS_REGISTRY
from .transforms.nuscenes_cvt import LoadDataTransform, Sample


@DATASETS_REGISTRY.register()
class NuscenesDet3D(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.
    """

    CLASSES = None
    PALETTE = None

    def __init__(self, 
                *,
                num_classes=12,
                version='',
                normalize=None,
                split='train',
                data_dir: str = None,
                labels_root: str = None,
                labels_dir_name: str = None,
                image: dict = None,
                cameras=None,
                augment='none',):

        data_dir = Path(data_dir)
        labels_dir = Path(labels_root) / labels_dir_name
        # Override augment if not training
        augment = augment if split=='train' else 'none'
        self.transform = LoadDataTransform(
            data_dir, labels_dir, image, num_classes, augment, normalize)

        # Format the split name
        # split = 'val' if test_mode else 'train'
        split = f'mini_{split}' if version == 'v1.0-mini' else split

        split_path = Path(labels_root) / f'{split}.txt'
        split_scenes = split_path.read_text().strip().split('\n')

        self.samples = []
        for s in split_scenes:
            self.samples += json.loads((Path(labels_dir) / f'{s}.json').read_text())

    def __len__(self):
        """Total number of samples of data."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        data = Sample(**self.samples[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data

    