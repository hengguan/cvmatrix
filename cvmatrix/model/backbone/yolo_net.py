from .yolo import YoloModel

from ..build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class Yolov7Net(YoloModel):

    def __init__(self, net_cfg):
        super().__init__(net_cfg)