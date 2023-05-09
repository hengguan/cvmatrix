# -*- encoding: utf-8 -*-
'''
@File    :   bev_depth.py
@Time    :   2023/03/09 15:42:14
@Author  :   Guan Heng 
@Version :   0.1
@Contact :   202208034
@Desc    : 
'''

from torch import nn

from .lss_bevdpth import BaseLSSFPN
from cvmatrix.model.build import (
    BEV_REGISTRY,
    build_backbone,
    build_neck,
    build_head
)

__all__ = ['BaseBEVDepth']


@BEV_REGISTRY.register()
class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, 
                img_backbone, 
                img_neck, 
                lss_conf, 
                bev_backbone, 
                bev_neck, 
                head):
        super(BaseBEVDepth, self).__init__()
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.bev_backbone = build_backbone(bev_backbone)
        self.bev_neck = build_neck(bev_neck)

        self.lss_model = BaseLSSFPN(**lss_conf)
        self.head = build_head(head)
        # self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        x = self.img_neck(self.img_backbone(x))
        if self.is_train_depth and self.training:
            x, depth_pred = self.lss_model(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
        else:
            x = self.lss_model(x, mats_dict, timestamps)
            depth_pred = None
        
        y = self.bev_backbone(x)
        y = self.bev_neck(y)
        z = self.head(y)
        if depth_pred is not None:
            return z, depth_pred
        return z

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
