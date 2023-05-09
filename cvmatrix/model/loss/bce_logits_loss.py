import torch
import torch.nn as nn

from ..build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class BCELogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(BCELogitsLoss, self).__init__()
        pw = torch.Tensor([pos_weight]).cuda()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss