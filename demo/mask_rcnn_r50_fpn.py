# model settings
model = dict(
    type='MaskRCNN',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50, # 网络层数
        num_stages=4,  # resnet的stage数量
        out_indices=(0, 1, 2, 3),  # 输出的stage的序号
        frozen_stages=-1,  # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；
        # 如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # resnet50，输入的各个stage的通道数
        # in_channels=[64, 128, 256, 512], # resnet18/34，输入的各个stage的通道数
        out_channels=256,  # 输出的特征层的通道数
        num_outs=4,
        # asff=False,  # 0919: wjy这里暂时还没有实现asff
    ),  # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',
        # type='RPNHeadNwd', is_rpn_nwd=True, loss_bboxScore_type='nwd',
        # type='RPNHeadIoU', loss_bboxScore_type='nwd',
        in_channels=256,  # RPN网络的输入通道数
        feat_channels=256,  # 特征层的通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[8, 16], # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
            scales=[8],
            ratios=[0.5, 1.0, 2.0],  # anchor的宽高比
            # strides=[4, 8, 16, 32, 64] # 在每个特征层上的anchor的步长（对应于原图）
            strides=[4, 8, 16, 32] # 在每个特征层上的anchor的步长（对应于原图）
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],  # 均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 方差
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # RoIExtractor类型
            # XXX: ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
            # sampling_ratio(int): number of inputs samples to take for each
            # output sample. 0 to take samples densely for current models.
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), 
            # roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=128), 
            out_channels=256,  # 输出通道数
            # 特征图的步长
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',  # 全连接层类型
            # type='ConvFCBBoxHead', num_shared_convs=2, num_shared_fcs=0,
            in_channels=256,  # 输入通道数
            fc_out_channels=1024,  # 输出通道数
            roi_feat_size=7,  # ROI特征层尺寸
            # num_classes=80,
            # num_classes=6,
            num_classes=10,
            # num_classes=7, # 分类器的类别数量+1，+1是因为多了一个背景的类别
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，
            # 续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            # num_classes=80,
            # num_classes=6,
            num_classes=10,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',  # RCNN网络正负样本划分
                # type='MaxNWDAssigner',  # RCNN网络正负样本划分
                pos_iou_thr=0.5,   # default: 0.7
                # pos_iou_thr=0.5,   # zjw TUNED
                neg_iou_thr=0.3,
                # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，
                # 则忽略所有的anchors，否则保留最大IOU的anchor
                min_pos_iou=0.3,
                match_low_quality=True,
                # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
                ignore_iof_thr=-1),
            # assigner=dict(type='ATSSAssigner', topk=9),
            sampler=dict(
                type='RandomSampler',  # 正负样本提取器类型
                num=256,  # 需提取的正负样本数量
                pos_fraction=0.5,  # 正样本比例
                neg_pos_ub=-1,  # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
                add_gt_as_proposals=False),
            # sampler=dict(
            #     type='OHEMSampler',
            #     num=256,
            #     pos_fraction=0.5,
            #     neg_pos_ub=-1,
            #     add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,  # 正样本权重，-1表示不改变原始的权重
            debug=False, # debug模式
            iou_label=False, # ADD：用于nwd相关计算
        ),  
        rpn_proposal=dict(
#             nms_across_levels=False,
#             nms_pre=2000,
#             nms_post=1000,
#             max_num=1000,
#             nms_thr=0.7,
#             # nms_thr=0.3,
#             min_bbox_size=0
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # type='MaxNWDAssigner',  # RCNN网络正负样本划分
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            # assigner=dict(type='ATSSAssigner', topk=9),
            # assigner=dict(type='ATSSAssigner', topk=12),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            # sampler=dict(
            #     type='OHEMSampler',
            #     num=512,
            #     pos_fraction=0.25,
            #     neg_pos_ub=-1,
            #     add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
#             nms_across_levels=False,  # 在所有的fpn层内做nms
#             nms_pre=1000,  # 在nms之前保留的的得分最高的proposal数量
#             nms_post=1000,  # 在nms之后保留的的得分最高的proposal数量
#             max_num=1000,  # 在后处理完成之后保留的proposal数量
#             nms_thr=0.7,  # nms阈值
#             min_bbox_size=0),  # 最小bbox尺寸
            # nms_pre=1000, # default: 1000
            nms_pre=2000,
            # nms_pre=6000,
            max_per_img=1000,
            # max_per_img=6000,
            nms=dict(type='nms', iou_threshold=0.7),
            # min_bbox_size=0,  # default: 0
            min_bbox_size=10,  # 最小bbox尺寸
        ),
        rcnn=dict(
            score_thr=0.05,
            # score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.5),
            # max_per_img=100, # default: max_per_img表示最终输出的det bbox数量
            max_per_img=200,
            # max_per_img=600,
            mask_thr_binary=0.5,
            # min_bbox_size=10 # 最小bbox尺寸，原本rcnn没有
        )))
