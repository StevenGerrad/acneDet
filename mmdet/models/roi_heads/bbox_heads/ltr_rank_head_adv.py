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
import numpy as np

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None



def inds_cartesian_product(pos_inds, neg_inds):
    """
    笛卡尔积
    """
    num_pos = pos_inds.size(0)
    num_neg = neg_inds.size(0)
    pos_inds = pos_inds[:, None].expand(num_pos, num_neg).reshape(-1)
    neg_inds = neg_inds[None, :].expand(num_pos, num_neg).reshape(-1)

    return pos_inds, neg_inds

def Curve_Fitting(x, y, deg):
    parameter = np.polyfit(x, y, deg)    # 拟合deg次多项式
    p = np.poly1d(parameter)             # 拟合deg次多项式
    R2 = np.corrcoef(y, p(x))[0, 1]**2
    return R2

def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def nms_topk(scores, bboxes, topk, iouThr=0.5):
    num_bbox = bboxes.size(0)

    rank_inds = torch.sort(scores, descending=True)[1]
    bboxes = bboxes[rank_inds, :]
    ious = bbox_overlaps(bboxes, bboxes)

    keep_mask = torch.zeros(num_bbox, device=bboxes.device)
    t_ind = torch.arange(num_bbox, device=bboxes.device)
    p = 0
    keep_mask[p] = 1
    keep_mask[(ious[p, :] >= iouThr) & (t_ind > p)] = -1
    p += 1
    for i in range(1, topk):
        while p < num_bbox and keep_mask[p] == -1:
            p += 1
        if p >= num_bbox:
            break
        keep_mask[p] = 1
        keep_mask[(ious[p, :] >= iouThr) & (t_ind > p)] = -1
        p += 1

    return rank_inds[keep_mask == 1]



@HEADS.register_module()
class LTRRankHeadADV(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 num_rank_convs=0,
                 num_rank_fcs=0,
                 with_avg_pool=False,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=10,
                 rank_class_agnostic=True,
                 beta=0.15,
                 alpha=0.5,
                 toph=7,
                 bbox_coder=None,
                 score_fusion_bf_softmax=True,
                 rank_loss_weight=1.0,
                 L=100,
                 add_bg_loss=0,
                 bg_loss_weight=0.1,
                 add_cls_feat=False,
                 rank_predictor_cfg=dict(type='Linear'),
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super(LTRRankHeadADV, self).__init__(init_cfg)
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
        self.beta = beta
        self.alpha = alpha
        self.toph = toph
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        self.score_fusion_bf_softmax = score_fusion_bf_softmax
        self.rank_loss_weight = rank_loss_weight
        self.L = L
        self.add_bg_loss = add_bg_loss
        self.bg_loss_weight = bg_loss_weight
        self.add_cls_feat = add_cls_feat
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.rank_convs, self.rank_fcs, self.rank_last_dim = \
            self._add_conv_fc_branch(
                self.num_rank_convs,
                self.num_rank_fcs,
                self.in_channels,
                True)

        self.extra_fc = None
        self.roi_fcs_attn = None
        if self.add_cls_feat:
            self.extra_fc = nn.Linear(self.fc_out_channels * 2, self.fc_out_channels)

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
             pos_inds,
             neg_inds,
             soft_alphas=None,
             soft_weight=None,
             # pair_mask=None,
             proposal_list=None,
             gt_bboxes=None,
             gt_labels=None,
             pos_is_gts=None,
             ):
        losses = dict()
        if rank_score is not None:
            if rank_score.numel() > 0:
                rank_score = rank_score.view(-1)
                if pos_inds is not None:
                    num_pair = pos_inds.numel()

                    if soft_alphas is not None:

                        # ADD 221009
                        loss_rank_ = torch.maximum(
                            rank_score[neg_inds] - rank_score[pos_inds] + soft_alphas,
                            pos_inds.new_zeros(num_pair))
                        
                    else:
                        loss_rank_ = torch.maximum(
                            rank_score[neg_inds] - rank_score[pos_inds] + self.alpha,
                            pos_inds.new_zeros(num_pair))

                    

                    if soft_weight is None:
                        # 220605
                        loss_rank_ = self.rank_loss_weight * loss_rank_.sum() / num_pair
                    else:
                        # ADD 221009
                        loss_rank_ = (loss_rank_ * soft_weight).sum() / num_pair

                    # loss_rank_ = self.get_bg_loss(rank_score, proposal_list, gt_bboxes, gt_labels)

                    if self.add_bg_loss > 0:

                        loss_rank_bg = self.get_bg_loss(rank_score, proposal_list, gt_bboxes, gt_labels)
                        if loss_rank_bg is not None:
                            loss_rank_ += loss_rank_bg

                    

                    if isinstance(loss_rank_, dict):
                        losses.update(loss_rank_)
                    else:
                        losses['loss_rank'] = loss_rank_
                else:
                    # TODO：或者我在这里正好可以抑制一下极值？
                    temp_ind = (rank_score > 0) & (rank_score < 0)
                    # 参考bbox_head还是要有个值
                    losses['loss_rank'] = rank_score[temp_ind].sum()

        return losses


    def _get_bg_loss_single(self, rank_scores, proposal, gt_bboxes, gt_labels):
        if gt_labels.size(0) == 0:
            return None, None
        rank_scores = rank_scores.squeeze()

        # gt_labels = torch.cat(gt_labels, dim=0)
        ious = bbox_overlaps(proposal[:, :4], gt_bboxes)
        max_iou, max_gt_inds = ious.max(-1)
        quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)
        # ADD 221001 >>>>>>
        # max_iou = torch.floor(max_iou / 0.05) * 0.05
        # ADD 221001 <<<<<<

        # TODO: 对于每个iou为0的rois，选取一定量（数量不能太多）的iou>0.5的pos与其配对
        # avg_match_num = int(sum(neg_pair_cnt) / len(neg_pair_cnt))
        # sample_neg_num = self.toph
        # sample_neg_num = 5
        sample_neg_num = self.add_bg_loss
        bg_mask = (max_iou == 0)
        # loss_rank_bg_ = torch.tensor(0.0).to(rank_scores.device)
        loss_rank_bg_ = []
        soft_weight_res = []
        num_valid_section = 0
        if bg_mask.sum().float() > 0:
            # TODO: 是否要区分类别，但这样要引入cls的预测label信息
            # pos_inds_ = torch.nonzero(fg_mask).squeeze(1)
            neg_inds_ = torch.nonzero(bg_mask).squeeze(1)

            if self.rank_class_agnostic:
                neg_rank_scores = rank_scores[neg_inds_]
                # pos_rank_scores = rank_scores[pos_inds_]
            else:
                neg_rank_scores = rank_scores[neg_inds_, gt_labels[neg_inds_]]
                # pos_rank_scores = rank_scores[pos_inds_, gt_labels[pos_inds_]]

            # neg_inds_ = neg_inds_[torch.sort(neg_rank_scores, descending=True)[1][:sample_neg_num]]
            # pos_inds_ = pos_inds_[torch.sort(pos_rank_scores, descending=False)[1][:avg_match_num]]

            neg_rank_scores = torch.sort(neg_rank_scores, descending=True)[0][:sample_neg_num]
            num_negs = neg_rank_scores.size(0)
            for i in range(1, 11):
                fg_mask = quantize_iou == i
                if fg_mask.sum().int() > 0:
                    num_valid_section += 1
                    if self.rank_class_agnostic:
                        pos_rank_scores = rank_scores[fg_mask]
                    else:
                        pos_rank_scores = rank_scores[fg_mask, max_gt_inds[fg_mask]]
                    pos_rank_score = pos_rank_scores.mean()
                    # loss_rank_bg_ += torch.maximum(
                    #     neg_rank_scores - pos_rank_score,
                    #     neg_rank_scores.new_zeros(num_negs)).sum() / num_negs
                    loss_rank_bg_.append(
                        torch.maximum(neg_rank_scores - pos_rank_score, neg_rank_scores.new_zeros(num_negs)))

                    soft_weight_ = neg_rank_scores.new_ones(num_negs) * self.rank_loss_weight
                    soft_weight_res.append(soft_weight_)

        # loss_rank_bg_ *= self.rank_loss_weight

        # if num_valid_section > 0:
        #     loss_rank_bg_ /= num_valid_section

        if len(loss_rank_bg_) == 0:
            return None, None

        loss_rank_bg = torch.cat(loss_rank_bg_)
        soft_weight_res = torch.cat(soft_weight_res)
        return loss_rank_bg, soft_weight_res


    def get_bg_loss(self, rank_scores, proposal_list, gt_bboxes, gt_labels):
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rank_scores_list = rank_scores.split(num_proposals_per_img, 0)

        # ADD 221119
        if len(rank_scores_list) % 2 == 0:
            batch_size = len(rank_scores_list)
            loss_bg_list, _ = multi_apply(
                self.get_bg_batch_loss,
                [[rank_scores_list[i], rank_scores_list[i + 1]] for i in range(0, batch_size, 2)],
                [[proposal_list[i], proposal_list[i+1]] for i in range(0, batch_size, 2)],
                [[gt_bboxes[i], gt_bboxes[i+1]] for i in range(0, batch_size, 2)],
                [[gt_labels[i], gt_labels[i+1]] for i in range(0, batch_size, 2)]
            )
            # loss_bg_list, _ = multi_apply(
            #     self.get_bg_batch_loss,
            #     [[rank_scores_list[i]] for i in range(0, batch_size)],
            #     [[proposal_list[i]] for i in range(0, batch_size)],
            #     [[gt_bboxes[i]] for i in range(0, batch_size)],
            #     [[gt_labels[i]] for i in range(0, batch_size)]
            # )
            loss_bg_ = torch.tensor(loss_bg_list)
            return loss_bg_.mean()

        else:
            loss_bg_list, soft_weight_list = multi_apply(
                self._get_bg_loss_single,
                rank_scores_list,
                proposal_list,
                gt_bboxes,
                gt_labels
            )

            loss_bg_list = [i for i in loss_bg_list if i is not None]
            soft_weight_list = [i for i in soft_weight_list if i is not None]
            if len(loss_bg_list) == 0:
                return torch.tensor(0.0).to(rank_scores.device)

            loss_bg_ = torch.cat(loss_bg_list)
            soft_weight = torch.cat(soft_weight_list)
            loss_bg_ = loss_bg_ * soft_weight
            return loss_bg_.sum() / loss_bg_.numel()

    def get_bg_batch_loss(self,
                    rank_scores,
                    proposal_list,
                    gt_bboxes,
                    gt_labels):
        """
        fg(iou > 0.5) : bg(iou = 0.5) = quantize_mean : toph
        """
        num_img = len(proposal_list)
        # cum_rois = np.cumsum([0, ] + [i.size(0) for i in proposal_list])
        cum_gts = np.cumsum([0, ] + [i.size(0) for i in gt_labels])

        # TODO: 考虑有些过大/过小的proposal，是否要设立最小的iou阈值？
        if cum_gts[-1] == 0:
            return None, None
        if isinstance(rank_scores, list):
            rank_scores = torch.cat(rank_scores).view(-1, 1 if self.rank_class_agnostic else self.num_classes)
        rank_scores = rank_scores.squeeze()

        # ious = rank_scores.new_zeros(cum_rois[-1], cum_gts[-1])
        # for i in range(num_img):
        #     if gt_labels[i].size(0) == 0:
        #         continue
        #     iou_i = bbox_overlaps(proposal_list[i][:, :4], gt_bboxes[i])
        #     ious[cum_rois[i]:cum_rois[i+1], cum_gts[i]:cum_gts[i+1]] = iou_i
        #
        # gt_labels = torch.cat(gt_labels, dim=0)
        # max_iou, max_gt_inds = ious.max(-1)
        # quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)

        max_iou_batch = []
        max_gt_inds_batch = []
        for i in range(num_img):
            if gt_labels[i].size(0) == 0:
                ious_i = rank_scores.new_zeros(proposal_list[i].size(0), 1)
            else:
                ious_i = bbox_overlaps(proposal_list[i][:, :4], gt_bboxes[i])
            max_iou_i, max_gt_inds_i = ious_i.max(-1)
            max_iou_batch.append(max_iou_i)
            max_gt_inds_batch.append(max_gt_inds_i + cum_gts[i])

        gt_labels = torch.cat(gt_labels, dim=0)
        max_iou = torch.cat(max_iou_batch)
        max_gt_inds = torch.cat(max_gt_inds_batch)
        quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)

        # ADD 221001 >>>>>>
        # max_iou = torch.floor(max_iou / 0.05) * 0.05
        # ADD 221001 <<<<<<

        # TODO: 对于每个iou为0的rois，选取一定量（数量不能太多）的iou>0.5的pos与其配对
        # avg_match_num = int(sum(neg_pair_cnt) / len(neg_pair_cnt))
        # sample_neg_num = self.toph
        # sample_neg_num = 5
        sample_neg_num = self.add_bg_loss
        bg_mask = (max_iou == 0)
        loss_rank_bg_ = torch.tensor(0.0).to(rank_scores.device)
        num_valid_section = 0
        if bg_mask.sum().float() > 0:
            # TODO: 是否要区分类别，但这样要引入cls的预测label信息
            # pos_inds_ = torch.nonzero(fg_mask).squeeze(1)
            neg_inds_ = torch.nonzero(bg_mask).squeeze(1)

            if self.rank_class_agnostic:
                neg_rank_scores = rank_scores[neg_inds_]
                # pos_rank_scores = rank_scores[pos_inds_]
            else:
                # TODO: 有问题，取不了toph，不知道按什么类
                neg_rank_scores = rank_scores[neg_inds_, gt_labels[neg_inds_]]
                # pos_rank_scores = rank_scores[pos_inds_, gt_labels[pos_inds_]]

            # neg_inds_ = neg_inds_[torch.sort(neg_rank_scores, descending=True)[1][:sample_neg_num]]
            # pos_inds_ = pos_inds_[torch.sort(pos_rank_scores, descending=False)[1][:avg_match_num]]

            neg_rank_scores = torch.sort(neg_rank_scores, descending=True)[0][:sample_neg_num]

            num_negs = neg_rank_scores.size(0)
            for i in range(1, 11):
                fg_mask = quantize_iou == i
                if fg_mask.sum().int() > 0:
                    num_valid_section += 1
                    if self.rank_class_agnostic:
                        pos_rank_scores = rank_scores[fg_mask]
                    else:
                        pos_rank_scores = rank_scores[fg_mask, max_gt_inds[fg_mask]]
                    pos_rank_score = pos_rank_scores.mean()
                    loss_rank_bg_ += torch.maximum(
                            neg_rank_scores - pos_rank_score,
                            neg_rank_scores.new_zeros(num_negs)).sum() / num_negs

        # loss_rank_bg_ *= self.rank_loss_weight
        loss_rank_bg_ *= self.bg_loss_weight

        if num_valid_section > 0:
            loss_rank_bg_ /= num_valid_section

        return loss_rank_bg_, None

    def _get_target_single(self, rank_scores, proposal, gt_bboxes, gt_labels):
        # TODO: 考虑有些过大/过小的proposal，是否要设立最小的iou阈值？
        if gt_labels.size(0) == 0:
            # return None, None
            return None, None, None, None
        rank_scores = rank_scores.squeeze()



        ious = bbox_overlaps(proposal[:, :4], gt_bboxes)
        # ious = cal_centerness(proposal[:, :4], gt_bboxes)
        max_iou, max_gt_inds = ious.max(-1)
        quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)

        

        pos_ind_res = []
        neg_ind_res = []
        soft_alpha_res = []
        soft_weight_res = []

        for g in range(gt_labels.size(0)):
            

            gt_mask = (max_gt_inds == g) & (max_iou >= 0.05)
            

            if not gt_mask.any():
                continue
            for k in range(1, 11):
                pos_candidate = (quantize_iou == k) & gt_mask
                neg_candidate = (quantize_iou < k) & gt_mask
                
                if pos_candidate.any() and neg_candidate.any():
                    pos_inds_ = torch.nonzero(pos_candidate).squeeze(1)
                    neg_inds_ = torch.nonzero(neg_candidate).squeeze(1)
                    # bg_inds_ = torch.nonzero(bg_candidate).squeeze(1)
                    if neg_inds_.size(0) > self.toph:
                    # if neg_inds_.size(0) > 0:
                        # TODO：之前写的是 > self.toph感觉有点问题
                        if self.rank_class_agnostic:
                            neg_rank_scores = rank_scores[neg_inds_]
                            # bg_rank_scores = rank_scores[bg_inds_]
                        else:
                            neg_rank_scores = rank_scores[neg_inds_, gt_labels[g]]
                            # bg_rank_scores = rank_scores[bg_inds_, gt_labels[g]]
                        # neg_inds_ = neg_inds_[torch.sort(neg_rank_scores)[1][:self.toph]]
                        # TODO: 220531: 似乎应该是降序，之前写成了升序
                        neg_inds_ = neg_inds_[torch.sort(neg_rank_scores, descending=True)[1][:self.toph]]

                        
                        num_pos = pos_inds_.size(0)
                        num_neg = neg_inds_.size(0)
                        pos_inds_ = pos_inds_[:, None].expand(num_pos, num_neg).reshape(-1)
                        neg_inds_ = neg_inds_[None, :].expand(num_pos, num_neg).reshape(-1)
                        if not self.rank_class_agnostic:
                            pos_inds_ = pos_inds_ * self.num_classes
                            pos_inds_ = pos_inds_ + gt_labels[g]
                            neg_inds_ = neg_inds_ * self.num_classes
                            neg_inds_ = neg_inds_ + gt_labels[g]

                        pos_ind_res.append(pos_inds_)
                        neg_ind_res.append(neg_inds_)

                        # soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha * (1 + k * 0.15)
                        soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha
                        # ADD 220913 >>>>>>
                        # soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha * pre_gt_weights[g]
                        # ADD 220913 <<<<<<
                        soft_alpha_res.append(soft_alpha_)

                        
                        soft_weight_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.rank_loss_weight
                        soft_weight_res.append(soft_weight_)

                        
        if len(pos_ind_res) == 0:
            return None, None, None, None
        pos_ind_res = torch.cat(pos_ind_res)
        neg_ind_res = torch.cat(neg_ind_res)

        soft_alpha_res = torch.cat(soft_alpha_res)

        
        soft_weight_res = torch.cat(soft_weight_res)

        

        return pos_ind_res, neg_ind_res, soft_alpha_res, soft_weight_res

    

    def _get_target_single_im(self, rank_scores, proposal, gt_bboxes, gt_labels):
        # TODO: 考虑有些过大/过小的proposal，是否要设立最小的iou阈值？
        if gt_labels.size(0) == 0:
            # return None, None
            return None, None, None, None
        rank_scores = rank_scores.squeeze()

        ious = bbox_overlaps(proposal[:, :4], gt_bboxes)
        # ious = cal_centerness(proposal[:, :4], gt_bboxes)
        max_iou, max_gt_inds = ious.max(-1)
        quantize_iou = torch.ceil(torch.maximum(max_iou - 0.5, max_iou.new_zeros(max_iou.size())) / 0.05)

        pos_ind_res = []
        neg_ind_res = []
        soft_alpha_res = []
        soft_weight_res = []
        
        gt_mask = (max_iou >= 0.5)
        if not gt_mask.any():
            return None, None, None, None
        for k in range(1, 11):
            pos_candidate = (quantize_iou == k) & gt_mask
            neg_candidate = (quantize_iou < k) & gt_mask

            # TODO: 220601: 这个区间不贴着取 —— 淦，结果这样炸得更快
            # neg_candidate = (quantize_iou < max(k - 2, 1)) & gt_mask
            if pos_candidate.any() and neg_candidate.any():
                pos_inds_ = torch.nonzero(pos_candidate).squeeze(1)
                neg_inds_ = torch.nonzero(neg_candidate).squeeze(1)
                # bg_inds_ = torch.nonzero(bg_candidate).squeeze(1)
                if neg_inds_.size(0) > self.toph:
                # if neg_inds_.size(0) > 0:
                    # TODO：之前写的是 > self.toph感觉有点问题
                    if self.rank_class_agnostic:
                        neg_rank_scores = rank_scores[neg_inds_]
                        # bg_rank_scores = rank_scores[bg_inds_]
                    else:
                        neg_rank_scores = rank_scores[neg_inds_, gt_labels[max_gt_inds[neg_inds_]]]
                        # bg_rank_scores = rank_scores[bg_inds_, gt_labels[g]]
                    # neg_inds_ = neg_inds_[torch.sort(neg_rank_scores)[1][:self.toph]]
                    # TODO: 220531: 似乎应该是降序，之前写成了升序
                    neg_inds_ = neg_inds_[torch.sort(neg_rank_scores, descending=True)[1][:self.toph]]

                    

                    num_pos = pos_inds_.size(0)
                    num_neg = neg_inds_.size(0)
                    pos_inds_ = pos_inds_[:, None].expand(num_pos, num_neg).reshape(-1)
                    neg_inds_ = neg_inds_[None, :].expand(num_pos, num_neg).reshape(-1)
                    if not self.rank_class_agnostic:
                        pos_inds_l = pos_inds_ + gt_labels[max_gt_inds[pos_inds_]]
                        pos_inds_ = pos_inds_ * self.num_classes + pos_inds_l
                        neg_inds_l = neg_inds_ + gt_labels[max_gt_inds[neg_inds_]]
                        neg_inds_ = neg_inds_ * self.num_classes + neg_inds_l

                    pos_ind_res.append(pos_inds_)
                    neg_ind_res.append(neg_inds_)

                    # soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha * (1 + k * 0.15)
                    soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha
                    # ADD 220913 >>>>>>
                    # soft_alpha_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.alpha * pre_gt_weights[g]
                    # ADD 220913 <<<<<<
                    soft_alpha_res.append(soft_alpha_)

                    
                    soft_weight_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.rank_loss_weight
                    # soft_weight_ = pos_inds_.new_ones(pos_inds_.size(0)) * self.rank_loss_weight * pre_gt_weights[g]
                    soft_weight_res.append(soft_weight_)



        if len(pos_ind_res) == 0:
            return None, None, None, None
        pos_ind_res = torch.cat(pos_ind_res)
        neg_ind_res = torch.cat(neg_ind_res)

        soft_alpha_res = torch.cat(soft_alpha_res)
        soft_weight_res = torch.cat(soft_weight_res)

        return pos_ind_res, neg_ind_res, soft_alpha_res, soft_weight_res

    def get_targets(self,
                    rank_scores,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    concat=True):
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rank_scores_list = rank_scores.split(num_proposals_per_img, 0)

        pos_inds, neg_inds, soft_alphas, soft_weight = multi_apply(
            # self._get_target_single,
            self._get_target_single_im,
            rank_scores_list,
            proposal_list,
            gt_bboxes,
            gt_labels
        )

        

        if concat:
            reind_pos_inds = []
            reind_neg_inds = []
            soft_alphas_list = []
            soft_weight_list = []
            # pair_mask_list = []
            proposal_cnt = 0
            for p in range(0, len(proposal_list)):
                if pos_inds[p] is not None:
                    reind_pos_inds.append(pos_inds[p] + proposal_cnt)
                    reind_neg_inds.append(neg_inds[p] + proposal_cnt)
                    soft_alphas_list.append(soft_alphas[p])
                    soft_weight_list.append(soft_weight[p])
                    # pair_mask_list.append(pair_mask[p])

                proposal_cnt += rank_scores_list[p].numel()
            if len(reind_pos_inds) == 0:
                return None, None, None, None
            pos_inds = torch.cat(reind_pos_inds, 0)
            neg_inds = torch.cat(reind_neg_inds, 0)
            soft_alphas = torch.cat(soft_alphas_list, 0)
            soft_weight = torch.cat(soft_weight_list, 0)
            # pair_mask = torch.cat(pair_mask_list, 0)
        return pos_inds, neg_inds, soft_alphas, soft_weight


    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'rank_score'))
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

        if self.roi_fcs_attn is not None:
            print('{:.7f}'.format(self.roi_fcs_attn.gamma.item()), end=' ')

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