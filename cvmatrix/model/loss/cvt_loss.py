import torch
import logging

from fvcore.nn import sigmoid_focal_loss

from ..build import LOSS_REGISTRY

logger = logging.getLogger(__name__)


__all__ = ['BinarySegmentationLoss', 'CenterLoss']


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)


@LOSS_REGISTRY.register()
class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        *,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0,
        weight=1.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.weight = weight

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean() * self.weight


@LOSS_REGISTRY.register()
class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        *,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0,
        weight=1.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility
        self.weight = weight

    def forward(self, pred, batch):
        pred = pred['center']
        label = batch['center']
        
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean() * self.weight
