# -*- encoding: utf-8 -*-
'''
@File    :   nusc_lss_eval.py
@Time    :   2022-11-02 14:06:49
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import torch
from torchmetrics import Metric

from .build import EVALUATION_REGISTRY


__all__ = ['CVTIoUMetrix', 'LSSIoUMetrix']

@EVALUATION_REGISTRY.register()
class CVTIoUMetrix(Metric):

    full_state_update: bool = False

    def __init__(self, thresholds=[0.2, 0.3, 0.4, 0.5]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        thresholds = torch.FloatTensor(thresholds)

        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, batch):
        label = batch['target']
        pred = pred.detach().sigmoid().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}


@EVALUATION_REGISTRY.register()
class LSSIoUMetrix(Metric):

    full_state_update: bool = False

    def __init__(self, thresholds=[0.4, 0.5]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        thresholds = torch.FloatTensor(thresholds)

        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('intersect', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('union', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, batch):
        label = batch['target']
        pred = pred.detach().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.intersect += (pred & label).sum(0).float()
        self.union += (pred | label).sum(0).float()

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.intersect / (self.union + 1e-7)

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}
