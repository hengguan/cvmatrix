# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2022-11-02 14:14:23
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

# from hydra import core, initialize, compose
# from omegaconf import OmegaConf
import time
import imageio
# import ipywidgets as widgets
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import cv2
from torchmetrics import MetricCollection

from cvmatrix.model import build_model
from cvmatrix.utils.config import LazyConfig
from cvmatrix.data.build import DATASETS_REGISTRY
from cvmatrix.utils.checkpoint import DetectionCheckpointer
# from vis.nuscenes_viz import NuScenesViz
from cvmatrix.evaluation.build import build_evaluator


"""
example: 
python scripts/demo.py \
    --ckpt path/to/xxx.ckpt \
    --dataset ./data/nuscenes \
    --labels path/to/labels
"""


def build_dataloader(cfg, split='val', subsample=5, is_val4d=False):
    datasets = []
    data_cfg = cfg.datasets.train.copy()
    assert 'type' in data_cfg.keys(), f'"type" not be assigned.'
    dataset_name = data_cfg.pop("type")
    data_cfg.split = 'val'
    if is_val4d:
        data_cfg.update(dict(num_time=1))

    datasets = DATASETS_REGISTRY.get(dataset_name)(**data_cfg)
    if not isinstance(datasets, list):
        datasets = [datasets]

    dataset = torch.utils.data.ConcatDataset(datasets)
    if isinstance(subsample, int) and subsample > 0:
        dataset = torch.utils.data.Subset(
            dataset, range(0, len(dataset), subsample))
    print(f'length of dataset: {len(dataset)}')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2)
    return dataloader


def load_checkpoint(model, ckpt_path, prefix=''):
    # model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(ckpt_path)
    print(f'loading checkpoint from "{ckpt_path}"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='', help="config path")
    parser.add_argument("--ckpt", type=str, default='', help="checkpoint path")
    parser.add_argument("--split", type=str, default='val', help="split dataset must be val/train/test")
    parser.add_argument("--subsample", type=int, default=0, help="the number of subsample dataset")
    parser.add_argument("--val4d", action='store_true', help="valuation for 4d temporal")
    # parser.add_argument("--output", type=str, default='./predictions.gif', help="output result path")
    args = parser.parse_args()

    # resolve config references
    # OmegaConf.resolve(cfg)
    cfg = LazyConfig.load(args.config)
    print(list(cfg.keys()))

    # Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
    SPLIT = args.split
    SUBSAMPLE = args.subsample

    # model, data, metrics
    network = build_model(cfg)
    load_checkpoint(network, args.ckpt)

    loader = build_dataloader(cfg, SPLIT, SUBSAMPLE, args.val4d)

    metrics = {k: build_evaluator(v) for k, v in cfg.evaluator.items()}
    metrics = MetricCollection(metrics)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network.to(device)
    network.eval()
    metrics.to(device)

    images = list()

    with torch.no_grad():
        for batch in tqdm(loader):
            if args.val4d:
                gt = dict(
                    bev=batch['bev'].flatten(0, 1),
                    center=batch['center'].flatten(0, 1),
                    visibility=batch['visibility'].flatten(0, 1)
                )
                batch.update(gt)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch, test_mode=True)
           
            metrics.update(pred, batch)

    results = metrics.compute()

    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, val in value.items():
                print(f'/metrics/{key}{subkey}', val)
        else:
            print(f'/metrics/{key}', value)

    metrics.reset()
