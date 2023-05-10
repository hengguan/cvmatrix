# -*- encoding: utf-8 -*-
'''
@File    :   build.py
@Time    :   2022-11-02 13:58:44
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   build train dataloader and test dataloader, stolen from FAIR's
            detectron2.
'''

import itertools
import logging
import numpy as np
import operator
import pickle
from typing import Any, Callable, Dict, List, Optional, Union
from functools import partial

import torch
import torch.utils.data as torchdata
# from tabulate import tabulate
# from termcolor import colored
import omegaconf

from cvmatrix.utils.config import configurable
# from cvmatrix.structures import BoxMode
from cvmatrix.utils.comm import get_world_size
from cvmatrix.utils.env import seed_all_rng
from cvmatrix.utils.file_io import PathManager
from cvmatrix.utils.logger import _log_api_usage, log_first_n
from cvmatrix.utils.registry import Registry

# from mmcv.parallel import collate

# from .catalog import DatasetCatalog, MetadataCatalog
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
# from .dataset_mapper import DatasetMapper
# from .detection_utils import check_metadata_consistency


DATASETS_REGISTRY = Registry("DATASETS")
DATASETS_REGISTRY.__doc__ = """
build registry for datasets.
"""

PIPELINES_REGISTRY = Registry("PIPELINES")
PIPELINES_REGISTRY.__doc__ = """
build registry for data processing pipeline.
"""


def worker_rnd_init(x):
    np.random.seed(13 + x)


def build_pipeline(cfg, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    _cfg = cfg.copy()
    _name = _cfg.pop("type")
    if isinstance(_cfg, omegaconf.dictconfig.DictConfig):
        _cfg = omegaconf.OmegaConf.to_object(_cfg)
    return PIPELINES_REGISTRY.get(_name)(**_cfg, **kwargs)


def build_pipelines(cfg_list: List, **kwargs):
    pipelines = []
    for cfg in cfg_list:
        pipelines.append(build_pipeline(cfg))
    
    return pipelines


def build_dataset(cfg, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    _cfg = cfg.copy()
    _name = _cfg.pop("type")
    if isinstance(_cfg, omegaconf.dictconfig.DictConfig):
        _cfg = omegaconf.OmegaConf.to_object(_cfg)
    return DATASETS_REGISTRY.get(_name)(**_cfg, **kwargs)


def build_dataloader(cfg, loader_cfg, **kwargs):
    datasets = []
    datasets = build_dataset(cfg)
    if not isinstance(datasets, list):
        datasets = [datasets]

    dataset = torchdata.ConcatDataset(datasets)

    for k, v in loader_cfg.items():
        if isinstance(v, str):
            loader_cfg[k] = eval(v)

    if loader_cfg['num_workers'] == 0:
        loader_cfg['prefetch_factor'] = 2

    # TODO: shuffle should be False in non-training step 
    shuffle = True
    return torch.utils.data.DataLoader(
        dataset, shuffle=shuffle,
        **loader_cfg)

def build_test_dataloader(cfg, split='val', subsample=5, is_val4d=False):
    datasets = []
    data_cfg = cfg.copy()
    cfg.split = 'val'
    assert 'type' in data_cfg.keys(), f'"type" not be assigned.'
    dataset_name = data_cfg.pop("type")
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

# def build_train_dataloader(dataset, test_mode, loader_cfg):

#     datasets = torchdata.ConcatDataset([dataset])
#     # logger = logging.getLogger(__name__)
#     # logger.info(" Length of Loading dataset: {}".format(len(dataset)))
#     print("==========> Length of Loading dataset: {}".format(len(dataset)))

#     if loader_cfg['num_workers'] == 0:
#         loader_cfg['prefetch_factor'] = 2

#     shuffle = False if test_mode else True
#     return torch.utils.data.DataLoader(datasets, shuffle=shuffle, **loader_cfg)

    
