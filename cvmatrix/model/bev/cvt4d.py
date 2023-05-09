# -*- encoding: utf-8 -*-
'''
@File    :   cvt4d.py
@Time    :   2022-11-02 14:11:36
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from einops import rearrange, repeat

from ..build import (
    BEV_REGISTRY, 
    build_backbone,
    build_transformer,
    build_head,
    build_loss
)


__all__ = ['CrossViewTransformerExp4D']


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class BEVTemporal(nn.Module):
    def __init__(
        self,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        blocks: list,
    ):
        """
        Only real arguments are:
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(blocks))
        w = bev_width // (2 ** len(blocks))

        # map from bev coordinates to ego frame
        V = get_view_matrix(h, w, h_meters, w_meters, offset)  # 3 3
        ego2bev = torch.FloatTensor(V)  # .unsqueeze(0)  # 1 3 3
        bev2ego = ego2bev.inverse()
        bev_size = torch.FloatTensor([w, h])
        self.register_buffer('ego2bev', ego2bev, persistent=False)                    # 3 h w
        self.register_buffer('bev2ego', bev2ego, persistent=False)                    # 3 h w
        self.register_buffer('bev_size', bev_size, persistent=False)                    # 3 h w


@BEV_REGISTRY.register()
class CrossViewTransformerExp4D(nn.Module):
    def __init__(self, 
                *,
                backbone,
                transformer,
                head,
                dim_last=128,
                outputs=[],
                input_size=[],
                losses=None,
            ):
        super().__init__()
        
        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            self.backbone = build_backbone(backbone)

        feats_shapes = None
        if input_size is not None:
            dummy = torch.rand(1, 3, *input_size)
            feats_shapes = [x.shape for x in self.backbone(dummy)]
            print(f'feats shapes: {feats_shapes}')

        # last bev and scene name
        self.last_bev = None
        self.last_scene = None
        # map from bev coordinates to ego frame
        # bh = transformer.bev_embedding['bev_height']
        # bw = transformer.bev_embedding['bev_width']
        self.bev_param = BEVTemporal(**transformer.bev_embedding)
        bw, bh = self.bev_param.bev_size
        self.calib = torch.FloatTensor([
            [1., bh/bw, 2/bw],
            [bh/bw, 1., 2/bh]
        ]).cuda()

        if isinstance(transformer, nn.Module):
            self.transformer = transformer
        else:
            transformer.feats_shapes = feats_shapes
            self.transformer = build_transformer(transformer)

        if isinstance(head, nn.Module):
            self.head = head
        else:
            self.head = build_head(head)

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        dim_last = dim_last
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.head.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))

        self.loss_funcs = dict()
        for k, loss in losses.items():
            # loss_cfg = loss.copy()
            self.loss_funcs[k] = build_loss(loss)

        self.count = 0

    def losses(self, pred, batch):
        # losses = []
        losses_dict = dict()
        for k, loss in self.loss_funcs.items():
            losses_dict[k] = loss(pred, batch)
            # losses.append(losses_dict[k])
        # losses_dict['total_loss'] = sum(losses)
        return losses_dict

    def forward(self, batch, test_mode=False):

        b, t, n, _, _, _ = batch['image'].shape
        
        image = batch['image'].flatten(0, 2)            # (b t n) c h w
        poses = batch['pose'][:, :, [0, 1, 3], :][..., [0, 1, 3]]    # b t 3 3
        poses_inv = poses.inverse()

        # training step not use temporal info between batch and 
        # val step not use temporal info different scene
        use_temporal_info = False
        if test_mode:
            use_temporal_info = batch['scene_name'][0] == self.last_scene
            self.last_scene = batch['scene_name'][0]

        if not use_temporal_info:
            self.last_bev = None

        gt = dict(
            bev=batch['bev'].flatten(0, 1),
            center=batch['center'].flatten(0, 1),
            visibility=batch['visibility'].flatten(0, 1)
        )

        feats = self.backbone(image)
        feats = [rearrange(
            feat, '(b t n) ... -> b t n ...', b=b, t=t, n=n) for feat in feats]
        bevs = list()
        for i in range(t):
            # align time t-1 bev feature to time t bev
            if i>0 or use_temporal_info: 
                pose_t = self.bev_param.ego2bev @ poses_inv[:, i] @ poses[:, i-1] @ self.bev_param.bev2ego  # b 3 3
                # convert affine transform matric to affine_grid "theta"
                pose_t = pose_t.inverse()
                theta = pose_t[:, :2, :] * self.calib[None]
                theta[:, 0, 2] += pose_t[:, 0, 0] + pose_t[:, 0, 1] - 1
                theta[:, 1, 2] += pose_t[:, 1, 0] + pose_t[:, 1, 1] - 1

                grid = F.affine_grid(
                    theta, self.last_bev.size(), align_corners=False)
                last_bev = F.grid_sample(self.last_bev, grid, align_corners=False)  # b d H W
            else:
                last_bev = self.last_bev
            camera = dict(
                intrinsics=batch['intrinsics'][:, i],
                extrinsics=batch['extrinsics'][:, i]
            )
            
            features = [feat[:, i] for feat in feats]
            x = self.transformer(features, camera, last_bev=last_bev)
            # add to list
            bevs.append(x)   # list(b 1 d H W)
            self.last_bev = x.clone()
        xs = torch.stack(bevs, axis=1) # b t d H W
        xs = xs.flatten(0, 1)
        y = self.head(xs)
        z = self.to_logits(y)
        
        pred = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        if test_mode:
            return pred

        return self.losses(pred, gt)
