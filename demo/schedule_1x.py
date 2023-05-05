# optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)   # default
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',  # 优化策略
#     warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
#     warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
#     warmup_ratio=0.001,  # 起始的学习率
#     # step=[8, 11],
#     step=[7, 10, 13],
#     # step=[16, 22], # 在第16和22个epoch时降低学习率
#     # step=[10, 20, 40, 80],
#     # step=[8, 16, 30]
# )  
# total_epochs = 100  # 最大epoch数
# runner = dict(type='EpochBasedRunner', max_epochs=16)
# runner = dict(type='EpochBasedRunner', max_epochs=50)

base_epoch_num = 1
lr_config = dict(
    policy='step',  # 优化策略
    warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=50,  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.001,  # 起始的学习率
    step=[8, 11],
)

runner = dict(type='EpochBasedRunner', max_epochs=16)
