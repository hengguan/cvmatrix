# -*- encoding: utf-8 -*-
'''
@File    :   nuscenes_generated.py
@Time    :   2022-11-02 13:57:55
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import json
from pathlib import Path

from torch.utils.data import Dataset

# from cvmatrix.utils.file_io import PathManager
from .build import DATASETS_REGISTRY
from .transforms.nuscenes_cvt import LoadDataTransform, Sample


@DATASETS_REGISTRY.register()
class NuscenesGenerated(Dataset):
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

        split_path = Path(labels_root) / 'nuscenes' / f'{split}.txt'
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

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
