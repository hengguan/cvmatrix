# -*- encoding: utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2022-11-02 14:10:42
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import torch
import torch.nn as nn
from torchvision.models import resnet101, resnet50, resnet34, resnet18

from ..build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class ResnetExtractor(nn.Module):
    MODELS = ['resnet50', 'resnet101', 'resnet34', 'resnet18']
    MODULES = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    def __init__(self, 
                depth=50, 
                output_layers=['c3', 'c4'], 
                pretrained=True, 
                checkpoint='//torchvision'):
        super().__init__()

        model_name = f'resnet{depth}'
        assert model_name in self.MODELS, f'model name "{model_name}" not implemented...'
        resnet = eval(model_name)(pretrained=True, checkpoint=checkpoint)

        for m in self.MODULES:
            self.add_module(m, getattr(resnet, m))
        
        self.output_layers = output_layers

    def forward(self, x):
        out = []
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        if 'c1' in self.output_layers:
            out.append(c1)
        c2 = self.layer2(c1)
        if 'c2' in self.output_layers:
            out.append(c2)
        c3 = self.layer3(c2)
        if 'c3' in self.output_layers:
            out.append(c3)
        c4 = self.layer4(c3)
        if 'c4' in self.output_layers:
            out.append(c4)
        
        return out



# if __name__ == "__main__":
#     model = ResnetExtractor(model_name='resnet50')

#     dump = torch.randn((2, 3, 640, 1280))
#     outs = model(dump)
#     for out in outs:
#         print(out.shape)