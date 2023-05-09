# -*- encoding: utf-8 -*-
'''
@File    :   lss_head.py
@Time    :   2022-11-02 14:12:38
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34

from ..build import HEAD_REGISTRY


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


@HEAD_REGISTRY.register()
class LSSBevEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LSSBevEncoder, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = UpConcat(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channel, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)  # 2x
        x = self.bn1(x)
        x = self.relu(x)  

        x1 = self.layer1(x)  # 4x
        x = self.layer2(x1)  # 8x
        x = self.layer3(x)   # 16x

        x = self.up1(x, x1)  # 8x
        x = self.up2(x)      # 4x

        return x