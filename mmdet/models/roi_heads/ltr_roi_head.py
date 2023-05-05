# import numpy as np
import torch
# import torch.nn as nn
# from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms, roi2bbox)
from ..builder import HEADS, build_head, build_roi_extractor, build_shared_head
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin
from .standard_roi_head import StandardRoIHead
from mmdet.core import multi_apply
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import numpy as np

def Curve_Fitting(x, y, deg):
    parameter = np.polyfit(x, y, deg)    #拟合deg次多项式
    p = np.poly1d(parameter)             #拟合deg次多项式

    R2 = np.corrcoef(y, p(x))[0, 1]**2
    return R2

@HEADS.register_module()
class LTRRoIHead(StandardRoIHead):
    """LTR roi head.

    https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html
    """
    def __init__(self,
                 gen_quantize_fg=False,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 ltr_add_gts=True,
                 rank_rois_attention=None,
                 rank_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 cls_roi_extractor=None,
                 reg_roi_scale_factor=1.0,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.rank_head = rank_head
        super(LTRRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if rank_head is not None:
            self.init_rank_head(rank_head)
        self.ltr_add_gts = ltr_add_gts
        self.rank_rois_attention = None
        if rank_rois_attention is not None:
            self.rois_attention = build_shared_head(rank_rois_attention)

        self.gen_quantize_fg = gen_quantize_fg
        if self.gen_quantize_fg:
            self.gen_random_form(interval_num=10)

        self.reg_roi_scale_factor = reg_roi_scale_factor
        self.cls_roi_extractor = None
        if cls_roi_extractor is not None:
            self.cls_roi_extractor = build_roi_extractor(cls_roi_extractor)

    # init_bbox_head 继承 StandardRoIHead
    # init_mask_head 继承 StandardRoIHead
    # init_assigner_sampler 继承 StandardRoIHead
    # forward_dummy 继承 StandardRoIHead
    # _bbox_forward 继承 StandardRoIHead
    # _mask_forward 继承 StandardRoIHead
    # _mask_forward_train 继承 StandardRoIHead
    # aug_test 不考虑（继承 StandardRoIHead）

    def init_rank_head(self, rank_head):
        self.rank_head = build_head(rank_head)

    def bbox_head_get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_is_gt, gt_bboxes, gt_labels, rank_score, cfg):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.bbox_head.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

        # 220607 >>>>>>
        if gt_bboxes.size(0) > 0:
            all_bboxes = torch.cat((pos_bboxes, neg_bboxes), dim=0)
            ious = bbox_overlaps(all_bboxes, gt_bboxes)
            max_iou, max_gt_inds = ious.max(-1)

            # TODO: 分gt查看rank score的R^2 —— 是否要考虑一下每个gt的rois可能不同
            pre_gt_weights = pos_bboxes.new_ones(gt_labels.size(0))
            for g in range(gt_labels.size(0)):
                gt_mask = (max_gt_inds == g) & (max_iou > 0.05)
                if gt_mask.sum() <= 1:
                    continue
                det_iou_g = ious[gt_mask, g].cpu()
                if self.rank_head.rank_class_agnostic:
                    sr_g = rank_score[gt_mask].squeeze().cpu()
                else:
                    sr_g = rank_score[gt_mask, gt_labels[g]].cpu()

                R2_item = Curve_Fitting(det_iou_g, sr_g, 1)
                pre_gt_weights[g] = max(R2_item, 1e-3)

            pre_gt_weights = (pre_gt_weights - 1).exp()
        # 220607 <<<<<< 分别应用到pos和neg的label_weight中

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if gt_bboxes.size(0) > 0:
                label_weights[:num_pos] = pre_gt_weights[max_gt_inds[:num_pos]]

            if not self.bbox_head.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_head.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
            if gt_bboxes.size(0) > 0:
                label_weights[-num_neg:][max_iou[-num_neg:] > 0] = pre_gt_weights[max_gt_inds[-num_neg:][max_iou[-num_neg:] > 0]]

        return labels, label_weights, bbox_targets, bbox_weights

    def bbox_head_get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rank_score_list,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        # ADDED
        pos_is_gt_list = [res.pos_is_gt for res in sampling_results]

        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self.bbox_head_get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_is_gt_list,
            gt_bboxes,
            gt_labels,
            rank_score_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        if self.cls_roi_extractor is not None:
            bbox_cls_feats = self.cls_roi_extractor(
                x[:self.cls_roi_extractor.num_inputs], rois)
            bbox_reg_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor)
            if self.with_shared_head:
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats=bbox_cls_feats)
        else:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _rank_forward_train(self, x, proposal_list, gt_bboxes,
                            gt_labels, pos_is_gts):
        rois = bbox2roi([p[:, :4] for p in proposal_list])
        rank_results = self._rank_forward(x, rois)
        rank_targets = self.rank_head.get_targets(
            rank_results['rank_score'], proposal_list, gt_bboxes, gt_labels)

        if self.rank_head.__class__.__name__ == 'LTRRankHeadADV':
            loss_rank = self.rank_head.loss(rank_results['rank_score'], *rank_targets, proposal_list, gt_bboxes,
                            gt_labels, pos_is_gts)
        elif self.rank_head.__class__.__name__ in ['APLossHead', 'DRHead']:
            loss_rank = self.rank_head.loss(rank_results['rank_score'], rank_targets)
        else:
            # loss_rank = self.rank_head.loss(rank_results['rank_score'], *rank_targets, pos_is_gts)
            loss_rank = self.rank_head.loss(rank_results['rank_score'], *rank_targets)
        # TODO: 220614 DR loss 不加 *
        # loss_rank = self.rank_head.loss(rank_results['rank_score'], rank_targets)

        rank_results.update(loss_rank=loss_rank)
        return rank_results

    def _rank_forward(self, x, rois, return_var=False):
        rank_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.rank_rois_attention is not None:
            # print('c={:.6f}'.format(self.cls_rois_attention.gamma.item()), end=' ')
            # bbox_cls_feats = self.cls_rois_attention.unit_enhance(bbox_cls_feats, rois)
            if return_var:
                # cls_attention = self.cls_rois_attention.get_attn_map(bbox_cls_feats, rois)
                # bbox_cls_feats, cls_attention = self.cls_rois_attention.unit_enhance(bbox_cls_feats, rois, return_var=return_var, gt_bboxes=gt_bboxes)
                rank_feats, rank_attention = self.rank_rois_attention.unit_enhance(rank_feats, rois,
                                                                                     return_var=return_var)
            else:
                # bbox_cls_feats = self.cls_rois_attention.unit_enhance(bbox_cls_feats, rois, gt_bboxes=gt_bboxes)
                rank_feats = self.rank_rois_attention.unit_enhance(rank_feats, rois)

        # do not support caffe_c4 model anymore
        rank_score = self.rank_head(rank_feats)

        # rank_results = dict(rank_score=rank_score, rank_feats=rank_feats)
        rank_results = dict(rank_score=rank_score)
        return rank_results


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # BBoxHead 阶段
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # # 原参数 MaxIoUAssigner
                # Return: assign_result: {gt_inds, max_overlaps, num_gts(int), num_preds(int), labels(None), info({})}
                #           gt_inds: [num_anchors, ](int)
                #           max_overlaps: [num_anchors, ](float)
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_train_cfg = self.train_cfg
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    bbox_train_cfg)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        # TODO: 220601 这一步参照cascade会把gt去掉，但cascade本来就会重新加上
        # 实际上这些gt在经过'修正'后iou已经不是1了，但pos_is_gts这个变量可以沿用
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # pos_is_gts = [res.pos_is_gt.new_zeros(res.pos_is_gt.size(0)) for res in sampling_results]

        # bbox_targets is a tuple
        # 参考Cascade R-CNN处理阶段间的proposal
        roi_labels = bbox_results['bbox_targets'][0]
        with torch.no_grad():
            cls_score = bbox_results['cls_score']
            if self.bbox_head.custom_activation:
                cls_score = self.bbox_head.loss_cls.get_activation(
                    cls_score)
            # TODO: 注意此处，若label为bg，则取cls_score判断的类，否则仍取label
            roi_labels = torch.where(
                roi_labels == self.bbox_head.num_classes,
                cls_score[:, :-1].argmax(1), roi_labels)
            proposal_list_refine = self.bbox_head.refine_bboxes(
                bbox_results['rois'], roi_labels,
                bbox_results['bbox_pred'], pos_is_gts, img_metas)

        if self.gen_quantize_fg:
            proposal_list_refine = self.gen_extra_fg(proposal_list_refine, gt_bboxes, gt_labels)

        # ADD: 220714 将gt重新加进去
        # 220716 发现不能直接用之前的pos_is_gts，有问题，pos_is_gts是针对pos_ind来的
        if self.ltr_add_gts:
            pos_is_gts = []
            rois_num_cnt = 0
            for i in range(len(img_metas)):
                proposal_list_refine[i] = torch.cat((gt_bboxes[i], proposal_list_refine[i]), dim=0)
                # pos_is_gts[i] = torch.cumsum(pos_is_gts[i], dim=0) + rois_num_cnt
                pos_is_gts_i = torch.arange(gt_bboxes[i].size(0)).to(gt_bboxes[i].device) + rois_num_cnt
                rois_num_cnt += proposal_list_refine[i].size(0)
                if not self.rank_head.rank_class_agnostic:
                    # pos_is_gts[i] *= self.rank_head.num_classes
                    # pos_is_gts[i] = pos_is_gts[i] + gt_labels[i]
                    pos_is_gts_i *= self.rank_head.num_classes
                    pos_is_gts_i = pos_is_gts_i + gt_labels[i]
                pos_is_gts.append(pos_is_gts_i)
            pos_is_gts = torch.cat(pos_is_gts, dim=0)

            gt_labels_ = torch.cat(gt_labels, dim=0)
            pos_is_gts_per_cat = []
            for c in range(self.rank_head.num_classes):
                if (gt_labels_ == c).sum().float() > 1:
                    pos_is_gts_per_cat.append(pos_is_gts[gt_labels_ == c])
                else:
                    pos_is_gts_per_cat.append(None)
        else:
            pos_is_gts_per_cat = []

        # rank head阶段
        rank_results = self._rank_forward_train(x, proposal_list_refine,
                                                gt_bboxes, gt_labels, pos_is_gts_per_cat)
        losses.update(rank_results['loss_rank'])

        return losses

    def gen_extra_fg(self, proposal_list, gt_bboxes_list, gt_labels_list):
        num_imgs = len(proposal_list)
        new_proposal_list = []
        for i in range(num_imgs):
            gt_bboxes = gt_bboxes_list[i]
            gt_labels = gt_labels_list[i]
            bboxes = proposal_list[i]

            num_gt = len(gt_bboxes)
            # 统计现有iou所在区间
            if num_gt == 0:
                new_proposal_list.append(bboxes)
                continue
            ious = bbox_overlaps(bboxes[:, :4], gt_bboxes)
            max_iou, max_gt_inds = ious.max(-1)
            quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)

            gen_boxes = []
            # gen_gt_inds = []
            # gen_ious = []
            # gen_labels = []
            for g in range(num_gt):
                g_w = gt_bboxes[g, 2] - gt_bboxes[g, 0]
                g_h = gt_bboxes[g, 3] - gt_bboxes[g, 1]
                for k in range(1, 11):
                    num_k = (quantize_iou[max_gt_inds == g] == k).sum().float()
                    # TODO: 每个 gt 在每个区间至少有一个
                    if num_k > 0:
                        continue
                    choice_ind = torch.randint(0, self.gen_form[k - 1].size(0), (1,)).item()
                    choice_offset = self.gen_form[k - 1][choice_ind, :]

                    choice_box = torch.tensor([gt_bboxes[g, 0] + g_w * choice_offset[0],
                                               gt_bboxes[g, 1] + g_h * choice_offset[1],
                                               gt_bboxes[g, 2] + g_w * choice_offset[2],
                                               gt_bboxes[g, 3] + g_h * choice_offset[3]])
                    gen_boxes.append(choice_box)
                    # gen_gt_inds.append(g + 1)
                    # gen_labels.append(gt_labels[g])
                    # gen_ious.append(choice_offset[4])

            if len(gen_boxes) == 0:
                new_proposal_list.append(bboxes)
                continue

            gen_boxes = torch.stack(gen_boxes).to(bboxes.device)

            

            # 补充assign_result
            bboxes = torch.cat([gen_boxes, bboxes], dim=0)

            new_proposal_list.append(bboxes)

        return new_proposal_list

    def gen_random_form(self, interval_num=10):
        """
        对于一个anchor的四条边在gt上进行的 t、b、l、r 偏移，偏移后：
        iou =   (w - min(-t, 0) - min(b, 0)) * (h - min(l, 0) - min(-r, 0)) /
                w * h + (w + r - l) * (h + t - b) - <分子>
        """
        gen_num = 100000
        # offset = torch.rand(gen_num, 4) - 0.5
        offset = (torch.randn(gen_num, 4) * 0.2).clamp(min=-0.5, max=0.5)
        l = offset[:, 0]
        t = offset[:, 1]
        r = offset[:, 2]
        b = offset[:, 3]
        zeors_vec = torch.zeros(gen_num)

        iou_numerator = (1.0 - torch.maximum(-1 * t, zeors_vec) - torch.maximum(b, zeors_vec)) * \
                        (1.0 - torch.maximum(l, zeors_vec) - torch.maximum(-1 * r, zeors_vec))
        iou_denominator = 1.0 + (1.0 + r - l) * (1.0 + t - b) - iou_numerator
        ious = iou_numerator / iou_denominator
        # quantize_iou = torch.ceil(torch.maximum(ious - 0.5, ious.new_zeros(ious.size())) / 0.05)
        quantize_iou = torch.ceil(torch.maximum(ious - 0.5, ious.new_zeros(ious.size())) / (0.5 / interval_num))

        offset = torch.cat([offset, ious.reshape(-1, 1)], dim=1)
        self.gen_form = []
        print("LTRRoIHead--rank head generate offset form ", interval_num, " :")
        # TODO: 有可能某一区间为空
        for k in range(0, interval_num):
            k_mask = quantize_iou == (k + 1)
            self.gen_form.append(offset[k_mask, :])
            print(k_mask.sum().int().item(), end=' ')
        print("\nform finished")

    def simple_test(self, x, proposal_list, img_metas, rescale=False, check_nms=False):
        """
        Test without augmentation. 模仿Cascade R-CNN的simple_test函数写的
        将本函数注释掉即可忽略多出的rank net，只用原有的mask rcnn结构
        """
        rois = bbox2roi(proposal_list)
        rcnn_test_cfg = self.test_cfg
        num_imgs = len(proposal_list)

        if rois.shape[0] == 0:
            batch_size = len(proposal_list)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor # 拆分到各个图像中
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposal_list)

        # rank_head阶段
        if self.bbox_head.custom_activation:
            cls_score = [
                self.bbox_head.loss_cls.get_activation(s)
                for s in cls_score
            ]
        bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
        rois_refine = torch.cat([
            self.bbox_head.regress_by_class(rois[j], bbox_label[j], bbox_pred[j], img_metas[j])
            for j in range(num_imgs)
        ])

        # rois_refine = bbox2roi(proposal_list_refine)
        proposal_list_refine = roi2bbox(rois_refine)
        rank_results = self._rank_forward(x, rois_refine)
        num_proposals_per_img_refine = tuple(len(p) for p in proposal_list_refine)
        rank_score = rank_results['rank_score'].split(num_proposals_per_img_refine, 0)

        

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        bbox_score_label_bfnms_list = []
        for i in range(len(proposal_list)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                # 进入 mmdet/models/roi_heads/bbox_heads/bbox_head.py
                if check_nms:
                    det_bbox, det_label, bbox_score_label_bfnms = self.rank_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        # cls_score_refine[i],
                        bbox_pred[i],
                        rank_score[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        check_nms=check_nms)
                    # 220527: 不需要nms，直接只保存 BBoxHead 的 cls_score 和 bbox_pred
                    # bbox_score_label_bfnms.append(rank_score[i])
                    bbox_score_label_bfnms_list.append(bbox_score_label_bfnms[-3:])
                else:
                    det_bbox, det_label = self.rank_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        # cls_score_refine[i],
                        bbox_pred[i],
                        rank_score[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        check_nms=check_nms)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            if check_nms:
                return bbox_results, bbox_score_label_bfnms_list
            else:
                return bbox_results
        else:
            segm_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            # bbox_results, segm_results = self.medical_post_process(bbox_results, segm_results)
            if check_nms:
                return list(zip(bbox_results, segm_results)), bbox_score_label_bfnms_list
            else:
                return list(zip(bbox_results, segm_results))

