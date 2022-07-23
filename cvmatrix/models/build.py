from camatrix.utils.registry import Registry

MODEL_REGISTRY = Registry("model")
MODEL_REGISTRY.__doc__ = """
build registry for model.
"""

def build_model(cfg, ):
    model_name = cfg._target_
    model_params = dict(cfg)
    return MODEL_REGISTRY.get(cfg._target_)(**model_params)