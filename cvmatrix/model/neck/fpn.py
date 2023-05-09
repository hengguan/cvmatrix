# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from cvmatrix.model.backbone.layers import get_norm, Conv2d

from ..build import NECK_REGISTRY

__all__ = ["FPN"]


@NECK_REGISTRY.register()
class FPN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        in_channels,
        out_channels,
        norm="",
        act=False,
        last_level=5,
        fuse_type="sum",
    ):
        """
        Args:
            in_channels (int): number of channels in the input feature maps.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(FPN, self).__init__()
        assert in_channels, in_channels

        lateral_convs = []
        output_convs = []

        activation = nn.ReLU(inplace=True) if act else None
        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            out_channel = out_channels[idx]
            lateral_norm = get_norm(norm, out_channel)
            output_norm = get_norm(norm, out_channel)

            lateral_conv = Conv2d(
                in_channel, 
                out_channel, 
                kernel_size=1, 
                bias=use_bias, 
                norm=lateral_norm,
                activation=activation
            )
            output_conv = Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=activation
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            # stage = int(math.log2(strides[idx]))
            # self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            # self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

        self.top_block = None
        if last_level == 6:
            self.top_block = LastLevelMaxPool()
        elif last_level == 7:
            self.top_block = LastLevelP6P7(in_channels[-1], out_channels[-1])

        self.out_channels = out_channels
        self.last_level = last_level

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](x[-1])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(x[-idx-1])
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.last_level == 6:
            results.extend(self.top_block(results[-1]))
        elif self.last_level == 7:
            results.extend(self.top_block(x[-1]))

        assert len(self.out_channels) == len(results)
        return results


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


if __name__ == "__main__":
    cfg = dict(
        in_channels=[256, 512],
        out_channels=[128, 128],
        norm="",
        last_level=5,
        fuse_type="sum",
    )
    fpn = FPN(**cfg)

    s1 = torch.randn((4, 256, 28, 60))
    s2 = torch.randn((4, 512, 14, 30))

    out = fpn([s1, s2])

    for o in out:
        print(o.shape)