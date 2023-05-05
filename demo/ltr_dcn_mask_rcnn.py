_base_ = [
    'mask_rcnn_r50_fpn.py',
    # 'faster_rcnn_r50_fpn.py',
    'coco_instance.py',
    'schedule_1x.py',
    'default_runtime.py'
]


################################################################################################ 220614：

# import math

# model = dict(
#     roi_head=dict(
#         type='LTRRoIHead',
#         bbox_head=dict(
#             reg_class_agnostic=True,
#         ),
#         rank_head=dict(
#             type='DRHead',
#             num_rank_convs=0,
#             num_rank_fcs=2,
#             with_avg_pool=False,
#             roi_feat_size=7,
#             in_channels=256,
#             num_classes=10,
#             rank_class_agnostic=False,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             score_fusion_bf_softmax=True,
#             loss_rank=dict(
#                 type='SigmoidDRLoss',
#                 margin=0.5,
#                 pos_lambda=1,
#                 # neg_lambda=0.1/math.log(3.5),
#                 neg_lambda=0.079824,
#                 L=8.,
#                 tau=4.,
#                 loss_weight=0.1
#             )
#         )
#     )
# )
#
# log_config = dict(interval=20)
# evaluation = dict(interval=1, start=8, metric=['bbox', 'segm', 'proposal_fast'], classwise=True)

################################################################################################ 220616

# model = dict(
#     roi_head=dict(
#         type='LTRRoIHead',
#         bbox_head=dict(
#             reg_class_agnostic=True,
#         ),
#         rank_head=dict(
#             type='APLossHead',
#             num_rank_convs=0,
#             num_rank_fcs=2,
#             with_avg_pool=False,
#             roi_feat_size=7,
#             in_channels=256,
#             num_classes=10,
#             rank_class_agnostic=False,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             score_fusion_bf_softmax=True,
#             loss_rank=dict(
#                 # type='APLoss', delta=1.0, loss_weight=1.0
#                 type="AutoGradAPLoss"
#             )
#         )
#     )
# )
#
# log_config = dict(interval=20)
# evaluation = dict(interval=1, start=8, metric=['bbox', 'segm', 'proposal_fast'], classwise=True)



################################################################################################ 220711:
model = dict(
    roi_head=dict(
        type='LTRRoIHead',
        # gen_quantize_fg=True,
        bbox_head=dict(
            # reg_class_agnostic=False,
            reg_class_agnostic=True,
        ),
        rank_head=dict(
            # type='LTRRankHead',
            type='LTRRankHeadADV',
            num_rank_convs=0,
            num_rank_fcs=2,
            with_avg_pool=False,
            roi_feat_size=7,
            in_channels=256,
            num_classes=10,
            rank_class_agnostic=True,
            # rank_class_agnostic=False,
            beta=0.15,
            alpha=0.1,
            toph=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            score_fusion_bf_softmax=True,
            rank_loss_weight=0.1,
            L=100,
        )
    )
)

log_config = dict(interval=20)
evaluation = dict(interval=1, start=8, metric=['bbox', 'segm', 'proposal_fast'], classwise=True)

lr_config = dict(
    # step=[8, 11],
    step=[8, 12],
)

# runner = dict(type='EpochBasedRunner', max_epochs=16)
runner = dict(type='EpochBasedRunner', max_epochs=18)


# data_root = '/home/xxxx/data/facedata_croped_1209/'
data = dict(
    # 要调试train的hook流程，缩短运行时间，可直接把train换成val
    train=dict(
        ann_file=data_root + 'annotations/val_crop_1024.json',
        img_prefix=data_root + 'images/val_crop_1024/',
    )
)


################################################################################################ 221010

# model = dict(
#     roi_head=dict(
#         type='LTRRoIHead',
#         # gen_quantize_fg=True,
#         bbox_head=dict(
#             # type='ConvFCBBoxHead',
#             # num_shared_fcs=0,
#             # num_cls_fcs=1,
#             # num_reg_fcs=1,
#             # reg_class_agnostic=False,
#             reg_class_agnostic=True,
#         ),
#         rank_head=dict(
#             type='LTRRankHead',
#             # type='LTRRankHeadADV', add_bg_loss=5, bg_loss_weight=0.1,
#             num_rank_convs=0,
#             num_rank_fcs=2,
#             with_avg_pool=False,
#             roi_feat_size=7,
#             in_channels=256,
#             num_classes=10,
#             rank_class_agnostic=True,
#             # rank_class_agnostic=False,
#             # beta=0.15,
#             alpha=0.1,
#             toph=7,
#             # toph=10,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             score_fusion_bf_softmax=True,
#             rank_loss_weight=0.1,
#             L=10,
#         )
#     ),
#     train_cfg=dict(
#         rcnn=dict(
#             # sampler=dict(
#             #     type='RandomSamplerGEN'
#             # ),
#             # sampler=dict(
#             #     type='RandomSampler',
#             #     num=512,
#             #     # pos_fraction=0.25,
#             #     pos_fraction=0.5,
#             #     neg_pos_ub=-1,
#             #     add_gt_as_proposals=True),
#         )
#     ),
# )
#
# # data = dict(
# #     samples_per_gpu=8,
# #     workers_per_gpu=4,
# #     # 要调试train的hook流程，缩短运行时间，可直接把train换成val
# #     # train=dict(
# #     #     ann_file=data_root + 'annotations/val_crop_1024.json',
# #     #     img_prefix=data_root + 'images/val_crop_1024/',
# #     # )
# # )
#
# log_config = dict(interval=20)
# # evaluation = dict(interval=1, start=8, metric=['bbox', 'segm', 'proposal_fast'], classwise=True)
# evaluation = dict(interval=1, start=8, metric=['bbox'], classwise=True)





############################################################################################################### 221215

# model = dict(
#     roi_head=dict(
#         type='LTRRoIHead',
#         bbox_head=dict(
#             reg_class_agnostic=True,
#         ),
#         rank_head=dict(
#             type='RSRankHead',
#             num_rank_convs=0,
#             num_rank_fcs=2,
#             with_avg_pool=False,
#             roi_feat_size=7,
#             in_channels=256,
#             num_classes=10,
#             # num_classes=20,
#             rank_class_agnostic=False,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             score_fusion_bf_softmax=True,
#             loss_rank=dict(
#                 # type='APLoss', delta=1.0, loss_weight=1.0,
#                 # type='RankSort',
#                 type='APELoss', lamb=8, topk=100000, loss_weight=1,
#             ),
#             delta=0.5,
#         )
#     )
# )

# log_config = dict(interval=20)
# # evaluation = dict(interval=1, start=8, metric=['mAP', 'recall'],
# #                   iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])     # voc

# evaluation = dict(interval=1, start=8, metric=['bbox', 'segm', 'proposal_fast'], classwise=True)
