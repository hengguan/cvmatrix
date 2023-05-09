# -*- encoding: utf-8 -*-
'''
@File    :   efficientnet.py
@Time    :   2022-11-02 14:10:30
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

from ..build import BACKBONE_REGISTRY


__all__ = ['EfficientNetEPExtractor', 'EfficientNetExtractor']

# Precomputed aliases
MODELSEP = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b3', 'efficientnet-b4']
ENDPOINTS= ['reduction_1', 'reduction_2', 'reduction_3', 'reduction_4', 'reduction_5', 'reduction_6']


@BACKBONE_REGISTRY.register()
class EfficientNetEPExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, *, model_name=None, output_layers=['reduction_6',], input_size=None):
        super().__init__()

        assert model_name is not None and model_name in MODELSEP, f'architecture: {model_name} not supported \
            by this model, support arch: {MODELSEP}'
        assert all(k in ENDPOINTS for k in output_layers), \
        'return layers is wrong! please check endpoints'

        self.layer_names = output_layers
        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)

        self.stem = nn.Sequential(net._conv_stem, net._bn0, net._swish)

        blocks = []
        # Blocks
        for idx, block in enumerate(net._blocks):
            drop_connect_rate = net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(net._blocks) # scale drop connect_rate

            blocks.append((block, [drop_connect_rate]))
        self.blocks = SequentialWithArgs(*blocks)
        # self.drop = net._global_params.drop_connect_rate

    def forward(self, x):
        endpoints = dict()

        # Stem
        x = self.stem(x)
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self.blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x
        
        return [endpoints[ln] for ln in self.layer_names]


# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 11)),
        ('reduction_4', (11, 23)),
    ]
}


@BACKBONE_REGISTRY.register()
class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, output_layers, model_name='efficientnet-b4'):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in output_layers)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in output_layers:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max):
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = output_layers
        self.idx_pick = [layer_to_idx[l] for l in output_layers]

        # Pass a dummy tensor to precompute intermediate shapes
        # dummy = torch.rand(1, 3, image_height, image_width)
        # output_shapes = [x.shape for x in self(dummy)]
        # print(f'feats shapes: {output_shapes}')
        # self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

            result.append(x)

        return [result[i] for i in self.idx_pick]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x
