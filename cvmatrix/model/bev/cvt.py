import logging

import torch
import torch.nn as nn

# from ..backbone import build_backbone
# from ..transformer import build_transformer
# from ..head import build_head
# from ..loss import build_loss
from ..build import (
    BEV_REGISTRY, 
    build_backbone,
    build_transformer,
    build_head,
    build_loss
)


__all__ = ['CrossViewTransformerExp']


@BEV_REGISTRY.register()
class CrossViewTransformerExp(nn.Module):
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

        # for x in self.backbone.parameters():
        #     print(x.requires_grad)

        feats_shapes = None
        if input_size is not None:
            dummy = torch.rand(1, 3, *input_size)
            feats_shapes = [x.shape for x in self.backbone(dummy)]
            print(f'feats shapes: {feats_shapes}')

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

    def losses(self, pred, batch):
        # losses = []
        losses_dict = dict()
        for k, loss in self.loss_funcs.items():
            losses_dict[k] = loss(pred, batch)
            # losses.append(losses_dict[k])
        # losses_dict['total_loss'] = sum(losses)
        return losses_dict

    def forward(self, batch, test_mode=False):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        camera = dict(
            intrinsics=batch['intrinsics'],
            extrinsics=batch['extrinsics']
        )

        feats = self.backbone(image)
        x = self.transformer(feats, camera)
        y = self.head(x)
        z = self.to_logits(y)

        pred = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        if test_mode:
            return pred

        return self.losses(pred, batch)
