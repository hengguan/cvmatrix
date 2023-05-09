# -*- encoding: utf-8 -*-
'''
@File    :   lss_cva.py
@Time    :   2022-11-02 14:12:16
@Author  :   Guan Heng 
@Version :   v0.1
@Contact :   202208034@Any3.com
@License :   Copyright (c) ChangAn Auto, Inc., SDA-S group.
@Desc    :   None
'''

from einops import rearrange, repeat
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from cvmatrix.model.utils.geometry import gen_dx_bx, cumsum_trick, QuickCumsum

import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


from ..build import (
    BEV_REGISTRY, 
    build_backbone,
    build_transformer,
    build_head,
    build_neck,
    build_loss
)

__all__ = ['LSS']


@BEV_REGISTRY.register()
class LSSCVA(nn.Module):

    def __init__(self, 
            grid_conf, 
            data_aug_conf, 
            num_classes, 
            input_size=[],
            backbone=None,
            neck=None,
            transformer=None,
            head=None,
            losses=None, 
            dim_last=128,
            **kwargs):
        super(LSSCVA, self).__init__()

        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            self.backbone = build_backbone(backbone)

        feats_shapes = None
        if input_size is not None:
            dummy = torch.rand(1, 3, *input_size)
            feats_shapes = [x.shape for x in self.backbone(dummy)]
            print(f'feats shapes: {feats_shapes}')

        if isinstance(neck, nn.Module):
            self.neck = neck
        else:
            neck.in_channels = [fs[1] for fs in feats_shapes]
            self.neck = build_neck(neck)
        # print(neck, self.neck)
        if isinstance(transformer, nn.Module):
            self.transformer = transformer
        else:
            dummys = [torch.randn(fs) for fs in feats_shapes]
            mlv_feats_shape = [f.shape for f in self.neck(dummys)]
            transformer.feats_shapes = mlv_feats_shape[1:]
            self.transformer = build_transformer(transformer)

        self.feature_linear = nn.Conv2d(neck.out_channels[0], dim_last, 1, bias=False)

        if isinstance(head, nn.Module):
            self.head = head
        else:
            self.head = build_head(head)

        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 8
        self.camC = 64
        self.frustum = self._create_frustum()
        self.D, _, _, _ = self.frustum.shape
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        self.loss_funcs = dict()
        for k, loss in losses.items():
            # loss_cfg = loss.copy()
            self.loss_funcs[k] = build_loss(loss)

    def _create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def _get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def _voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def losses(self, pred, batch):
        # losses = []
        losses_dict = dict()
        for k, loss in self.loss_funcs.items():
            losses_dict[k] = loss(pred, batch)
            # losses.append(losses_dict[k])
        # losses_dict['total_loss'] = sum(losses)
        return losses_dict

    def forward(self, batch, test_mode=False):
        image = batch.pop('image')
        target = batch.pop('target')

        b, n, _, _, _ = image.shape

        # image = image.flatten(0, 1)            # (b n) c h w
        x = self.backbone(image.flatten(0, 1))
        mlv_feats = self.neck(x)                        
        depth_embed = self.transformer(mlv_feats[1:], batch['intrins'])    # b n d m

        # generate voxel feature    
        _, _, h, w = mlv_feats[0].shape
        out_feat = self.feature_linear(mlv_feats[0])
        feat = rearrange(out_feat, '(b n) d h w -> b n (h w) d', b=b, n=n) # b n (h w) d
        depth = feat @ depth_embed
        feat = rearrange(depth, 'b n (h w) m -> (b n) m h w', h=h, w=w) # (b n) m h w
        depth_feat = feat.sigmoid()[:, None] * out_feat.unsqueeze(2)
        depth_feat = rearrange(depth_feat, '(b n) c m h w -> b n m h w c', b=b, n=n) # b n m h w c

        geom = self._get_geometry(**batch)
        y = self._voxel_pooling(geom, depth_feat)
        z = self.head(y)

        if test_mode:
            batch['target'] = target
            return z
        
        return self.losses(z, target)
