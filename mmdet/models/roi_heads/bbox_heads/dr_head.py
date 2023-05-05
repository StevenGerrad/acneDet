import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, medical_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmcv.cnn import ConvModule
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import math

@HEADS.register_module()
class DRHead(BaseModule):
    """
    DR-loss
    the subnet structure is same as rank head of ltr model.
    """

    def __init__(self,
                 num_rank_convs=0,
                 num_rank_fcs=0,
                 with_avg_pool=False,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=10,
                 rank_class_agnostic=False,     # rank_class_agnostic 必须为false
                 bbox_coder=None,
                 score_fusion_bf_softmax=True,
                 rank_predictor_cfg=dict(type='Linear'),
                 loss_rank=dict(
                     type='SigmoidDRLoss',
                     margin=0.5,
                     pos_lambda=1,
                     neg_lambda=0.1 / math.log(3.5),
                     L=6.,
                     tau=4.,
                     loss_weight=1.0
                 ),
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super(DRHead, self).__init__(init_cfg)
        self.num_rank_convs = num_rank_convs
        self.num_rank_fcs = num_rank_fcs
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.rank_predictor_cfg = rank_predictor_cfg
        self.rank_class_agnostic = rank_class_agnostic
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        self.score_fusion_bf_softmax = score_fusion_bf_softmax
        self.loss_rank = build_loss(loss_rank)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.rank_convs, self.rank_fcs, self.rank_last_dim = \
            self._add_conv_fc_branch(
                self.num_rank_convs,
                self.num_rank_fcs,
                self.in_channels,
                True)

        # in_channels = self.in_channels
        # if self.with_avg_pool:
        #     self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        # else:
        #     in_channels *= self.roi_feat_area

        if self.num_rank_fcs == 0:
            self.rank_last_dim *= self.roi_feat_area

        out_dim_rank = 1 if rank_class_agnostic else num_classes
        self.fc_rank = build_linear_layer(
            self.rank_predictor_cfg,
            in_features=self.rank_last_dim,
            out_features=out_dim_rank)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            # TODO: 考虑参考DW模型的参数初始化
            self.init_cfg += [
                dict(
                    type='Normal', std=0.01, override=dict(name='fc_rank'))
            ]
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='rank_fcs'),
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    @auto_fp16()
    def forward(self, x):
        x_rank = x
        for conv in self.rank_convs:
            x_rank = conv(x_rank)
        if x_rank.dim() > 2:
            if self.with_avg_pool:
                x_rank = self.avg_pool(x_rank)
            x_rank = x_rank.flatten(1)
        for fc in self.rank_fcs:
            x_rank = self.relu(fc(x_rank))

        rank_score = self.fc_rank(x_rank)
        return rank_score

    @force_fp32(apply_to=('rank_score'))
    def loss(self,
             rank_score,
             targets):
        losses = dict()
        if rank_score is not None:
            if rank_score.numel() > 0:
                loss_rank_ = self.loss_rank(rank_score, targets)

                if isinstance(loss_rank_, dict):
                    losses.update(loss_rank_)
                else:
                    losses['loss_rank'] = loss_rank_

        return losses


    # def _get_target_single(self, rank_scores, proposal, gt_bboxes, gt_labels):
    #     num_bboxes = rank_scores.size(0)
    #
    #     target = rank_scores.new_full((num_bboxes,), 0, dtype=torch.long)
    #     if gt_labels.size(0) == 0:
    #         return target, None
    #
    #     ious = bbox_overlaps(proposal[:, :4], gt_bboxes)
    #     max_iou, max_gt_inds = ious.max(-1)
    #
    #     # TODO: 阈值如何选取
    #     pos_inds = max_iou >= 0.5
    #     if pos_inds.any():
    #         target[pos_inds] = gt_labels[max_gt_inds[pos_inds]] + 1
    #
    #     return target, None

    def _get_target_single(self, rank_scores, proposal, gt_bboxes, gt_labels):
        """
        2212121942
        :param rank_scores:
        :param proposal:
        :param gt_bboxes:
        :param gt_labels:
        :return:
        """
        num_bboxes = rank_scores.size(0)

        target = rank_scores.new_full((num_bboxes,), -1, dtype=torch.long)
        if gt_labels.size(0) == 0:
            return target, None

        ious = bbox_overlaps(proposal[:, :4], gt_bboxes)
        max_iou, max_gt_inds = ious.max(-1)

        # TODO: 阈值如何选取
        pos_inds = max_iou >= 0.5
        neg_inds = max_iou < 0.4
        if pos_inds.any():
            target[pos_inds] = gt_labels[max_gt_inds[pos_inds]] + 1
        if neg_inds.any():
            target[neg_inds] = 0

        return target, None

    def get_targets(self,
                    rank_scores,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    gt_masks=None,
                    concat=True):
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rank_scores_list = rank_scores.split(num_proposals_per_img, 0)

        # 似乎这个multi_apply不能只传一个，否则会报错
        targets, _ = multi_apply(
            self._get_target_single,
            rank_scores_list,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )

        if concat:
            targets = torch.cat(targets, 0)
        return targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   rank_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   check_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:  # True
        #     scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        if rank_score.size(1) == 1:
            # rank_class_agnostic
            sr = rank_score.expand(rank_score.size(0), cls_score.size(1))
        else:
            sr = torch.cat((rank_score, rank_score.new_ones(rank_score.size(0), 1)), dim=1)

        # TODO: 虽然好像应该是先softmax再加rank_score但效果并不好
        # if self.score_fusion_bf_softmax:
        #     # TODO: 这样的话，sr对应的bg类别有问题，简单得用扩充不符合逻辑
        #     scores = self.beta * sr + (1 - self.beta) * cls_score
        #     scores = F.softmax(scores, dim=-1)
        # else:
        #     scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None
        #     scores = self.beta * sr + (1 - self.beta) * scores

        # 220603:
        scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None
        scores = (scores * sr.sigmoid()).sqrt()

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # 这块是对ROI输出的一次矫正, 这块是最终模型输出的关键 # mmdet/core/post_processing/bbox_nms.py
            # bboxes:N*40 scores:N*11 score_thr:0.05 nms: IOU0.5 max_per_img:100
            if check_nms:
                det_bboxes, det_labels, bbox_score_label_bfnms = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms,
                                                                                cfg.max_per_img, check_nms=check_nms)
                # det_bboxes, det_labels, bbox_score_label_bfnms = medical_nms(bboxes, scores, cfg.score_thr, cfg.nms,
                #                                                                 cfg.max_per_img, check_nms=check_nms)
                return det_bboxes, det_labels, bbox_score_label_bfnms + [cls_score, bbox_pred, rank_score]
            else:
                det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
                # det_bboxes, det_labels = medical_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
                return det_bboxes, det_labels