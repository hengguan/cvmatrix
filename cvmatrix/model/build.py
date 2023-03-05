from cvmatrix.utils.registry import Registry
from hydra.utils import instantiate


MODEL_REGISTRY = Registry("model")
MODEL_REGISTRY.__doc__ = """
build registry for model.
"""


def build_model(cfg):
    return instantiate(cfg.model)
# def build_model(cfg, ):
#     model_name = cfg._target_
#     model_params = dict(cfg)
#     return MODEL_REGISTRY.get(cfg._target_)(**model_params)