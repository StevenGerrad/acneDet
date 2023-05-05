dataset_type = 'CocoDataset'

data_root = '/home/xxxx/data/facedata_croped_1209/'



img_norm_cfg = dict(
    mean=[144.578, 107.304, 90.519], std=[78.271, 63.2, 56.992], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[150.88884, 111.05136, 95.698715], std=[75.45097, 61.343353, 56.674812], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

    dict(type='RandomFlip', flip_ratio=0.5),  # default

    dict(type='Normalize', **img_norm_cfg),

    dict(type='Pad', size_divisor=32), # size_divisor：保证图像大小为32的倍数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#         img_scale=[(2300, 1724)],
        scale_factor = 1.0, # 没有img_scale就需要有scale_factor
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute'),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),

            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),

        ])
]
data = dict(
    samples_per_gpu=2,
#     samples_per_gpu=5,
    workers_per_gpu=4,
#     workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_crop_1024.json',
        img_prefix=data_root + 'images/train_crop_1024/',
        pipeline=train_pipeline),
    val=dict(
        # samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + 'annotations/val_crop_1024.json',
        img_prefix=data_root + 'images/val_crop_1024/',
        pipeline=test_pipeline),
    test=dict(
        # samples_per_gpu=1,
        type=dataset_type,
        ann_file=data_root + 'annotations/val_crop_1024.json',
        img_prefix=data_root + 'images/val_crop_1024/',
        pipeline=test_pipeline))
evaluation = dict(interval=1,
                start=8,
                metric=['bbox', 'segm'],
                classwise=True)
