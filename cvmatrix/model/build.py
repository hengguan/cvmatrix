import torch
from typing import Dict
import omegaconf

from cvmatrix.utils.logger import _log_api_usage
from cvmatrix.utils.registry import Registry


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images.
"""

NECK_REGISTRY = Registry('NECK')
NECK_REGISTRY.__doc__ = """
build register for neck.
"""

HEAD_REGISTRY = Registry('HEAD')
HEAD_REGISTRY.__doc__ = """
Registry for model head.
"""

TRANSFORMER_REGISTRY = Registry("TRANSFORMER")
TRANSFORMER_REGISTRY.__doc__ = """
Registry for transformer.
"""

DETECTOR_REGISTRY = Registry("DETECTOR")
DETECTOR_REGISTRY.__doc__ = """
build registry for detector.
"""

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
build registry for loss.
"""

BEV_REGISTRY = Registry('BEV')
BEV_REGISTRY.__doc__ = """
build registry for BEV.
"""

BOX_CODER_REGISTRY = Registry('BOX_CODER')
BOX_CODER_REGISTRY.__doc__ = """
build registry for bbox encoder.
"""


MODEL_TYPES = ["bev", "det", "seg", "multitask"]


def build_bbox_coder(cfg, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    _cfg = cfg.copy()
    _name = _cfg.pop("type")
    if isinstance(_cfg, omegaconf.dictconfig.DictConfig):
        _cfg = omegaconf.OmegaConf.to_object(_cfg)
    return BOX_CODER_REGISTRY.get(_name)(**_cfg, **kwargs)


def build_backbone(cfg):
    """
    Build a backbone from `cfg.model.backbone`.

    Returns:
        an instance of :class:`nn.Module`
    """
    assert "type" in cfg.keys(), f'"type" not be implemented in backbone of config'
    backbone_cfg = cfg.copy()
    model_name = backbone_cfg.pop('type')
    backbone = BACKBONE_REGISTRY.get(model_name)(**backbone_cfg)
    return backbone


def build_neck(cfg, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    neck_cfg = cfg.copy()
    _name = neck_cfg.pop("type")
    if isinstance(neck_cfg, omegaconf.dictconfig.DictConfig):
        neck_cfg = omegaconf.OmegaConf.to_object(neck_cfg)
    return NECK_REGISTRY.get(_name)(**neck_cfg, **kwargs) 


def build_head(cfg, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    head_cfg = cfg.copy()
    head_name = head_cfg.pop("type")
    if isinstance(head_cfg, omegaconf.dictconfig.DictConfig):
        head_cfg = omegaconf.OmegaConf.to_object(head_cfg)
    return HEAD_REGISTRY.get(head_name)(**head_cfg, **kwargs)


def build_transformer(cfg, **kwargs):
    """
    Build a transformer from `cfg.model.transformer`.

    Returns:
        an instance of :class:`transformer`
    """
    assert "type" in cfg.keys(), f'"type" not in config'
    former_cfg = cfg.copy()
    transformer_name = former_cfg.pop("type")
    return TRANSFORMER_REGISTRY.get(transformer_name)(**former_cfg, **kwargs)


def build_loss(cfg: Dict, **kwargs):
    assert "type" in cfg.keys(), f'"type" not in config'
    loss_cfg = cfg.copy()
    loss_model_name = loss_cfg.pop('type')
    return LOSS_REGISTRY.get(loss_model_name)(**loss_cfg)


def build_detector(cfg):
    det_cfg = dict(cfg)
    assert 'type' in det_cfg.keys(), f'key words "type" not in config {det_cfg}'
    model_name = det_cfg.pop('type')
    return DETECTOR_REGISTRY.get(model_name)(**det_cfg)


def build_bev_model(cfg):
    bev_cfg = cfg.copy()
    if 'model_type' in bev_cfg.keys():
        bev_cfg.pop('model_type')
    assert 'type' in bev_cfg.keys(), f'"type" not key words item of config {bev_cfg}'
    model_name = bev_cfg.pop('type')
    return BEV_REGISTRY.get(model_name)(**bev_cfg)


def build_model(cfg):
    assert 'model_type' in cfg.model.keys(), f'key "model_type" not be certain.'
    model_type = cfg.model.model_type
    assert model_type in MODEL_TYPES, f'MODEL TYPE: {model_type} \
        not in the MODEL TYPES ZOO: {MODEL_TYPES}'
    
    if model_type == 'bev':
        model = build_bev_model(cfg.model)
    
    elif model_type == 'det':
        model = build_detector(cfg.model)

    elif model_type == 'seg':
        model = build_segmentation(cfg.model)

    elif model_type == 'multitask':
        model = build_segmentation(cfg.model)
    else:
        raise ValueError(f'model type "{model_type}" is not support.')

    model.to(torch.device(cfg.train.device))
    _log_api_usage("modeling type: " + model_type)
    # model.to(torch.device(cfg.model.device))
    return model