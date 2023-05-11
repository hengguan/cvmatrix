# -*- encoding: utf-8 -*-
'''
@File    :   stich_viz.py
@Time    :   2022-11-02 14:14:14
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
import ipywidgets as widgets
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import cv2

from cvmatrix.model import build_model
from cvmatrix.utils.config import LazyConfig
from cvmatrix.data.build import DATASETS_REGISTRY
from cvmatrix.utils.checkpoint import DetectionCheckpointer
from vis.nuscenes_stich_viz import NuScenesStitchViz


"""
example: 
python scripts/demo.py \
    --ckpt path/to/xxx.ckpt \
    --dataset ./data/nuscenes \
    --labels path/to/labels
"""


def build_dataloader(cfg, split='val', subsample=5):
    datasets = []
    data_cfg = cfg.datasets.copy()
    assert 'type' in data_cfg.keys(), f'"type" not be assigned.'
    dataset_name = data_cfg.pop("type")
    data_cfg.update(dict(split=split))
    datasets.append(
            DATASETS_REGISTRY.get(dataset_name)(**data_cfg))

    dataset = torch.utils.data.ConcatDataset(datasets)
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
    parser.add_argument("--main-config", type=str, default='', help="config path")
    parser.add_argument("--aux-config", type=str, default='', help="config path")
    parser.add_argument("--main-ckpt", type=str, default='', help="checkpoint path")
    parser.add_argument("--aux-ckpt", type=str, default='', help="checkpoint path")
    parser.add_argument("--output", type=str, default='./predictions.gif', help="output result path")
    args = parser.parse_args()

    # resolve config references
    # OmegaConf.resolve(cfg)
    main_cfg = LazyConfig.load(args.main_config)
    aux_cfg = LazyConfig.load(args.aux_config)
    print(list(main_cfg.keys()))

    # Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
    SPLIT = 'val'
    SUBSAMPLE = 100

    # model, data, viz = setup_experiment(cfg)
    road_network = build_model(main_cfg)
    load_checkpoint(road_network, args.main_ckpt)
    vehicle_network = build_model(aux_cfg)
    load_checkpoint(vehicle_network, args.aux_ckpt)

    loader = build_dataloader(main_cfg, SPLIT, SUBSAMPLE)

    # label_indices = [[4, 5, 6, 7, 8, 10, 11]]
    # viz = NuScenesViz(label_indices=label_indices)
    # Show more confident predictions, note that if show_images is True, GIF quality with be degraded.
    viz = NuScenesStitchViz(vehicle_threshold=0.4, road_threshold=0.4, show_images=False)

    GIF_PATH = args.output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vehicle_network.to(device)
    vehicle_network.eval()

    road_network.to(device)
    road_network.eval()

    images = list()

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            vehicle_pred = vehicle_network(batch, test_mode=True)['bev']
            road_pred = road_network(batch, test_mode=True)['bev']

            # visualization = np.vstack(viz(batch=batch, pred=pred))
            vis_res = viz(batch, road_pred, vehicle_pred)
            visualization = np.vstack(vis_res)

            images.append(visualization)

    # Save a gif
    duration = [1 for _ in images[:-1]] + [5 for _ in images[-1:]]
    imageio.mimsave(GIF_PATH, images, duration=duration)

    # html = f'''
    # <div align="center">
    # <img src="{GIF_PATH}?modified={time.time()}" width="80%">
    # </div>
    # '''

    # display(widgets.HTML(html))
