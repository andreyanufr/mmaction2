_base_ = ['../../_base_/default_runtime.py']

model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=3,
        spatial_type='avg',
        dropout_ratio=0.5,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


#load_from="work_dirs/x3d_s_2_frame/epoch_35.pth"
# dataset settings
dataset_type = 'RawframeDataset'
experiment = 'experiment_1'
dirname = f'data/ucf101/{experiment}'
data_root_train = f'{dirname}/rawframes_train'
data_root_val   = f'{dirname}/rawframes_val'
data_root_test  = f'{dirname}/rawframes_test'

ann_file_train = f'{dirname}/train_list_rawframes.txt'
ann_file_val = f'{dirname}/val_list_rawframes.txt'
ann_file_test = f'{dirname}/test_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),

    dict(type='TorchvisionTrans', tr_type='ColorJitter'),
    dict(type='TorchvisionTrans', tr_type='RandomAffine', degrees=5),
    dict(type='TorchvisionTrans', tr_type='RandomGrayscale', p=0.1),
    dict(type='TorchvisionTrans', tr_type='RandomPerspective'),
    dict(type='TorchvisionTrans', tr_type='RandomRotation', degrees=5),

    dict(type='TorchvisionTrans', tr_type='GaussianBlur', kernel_size=(5, 5)),
    dict(type='TorchvisionTrans', tr_type='RandomInvert'),
    dict(type='TorchvisionTrans', tr_type='RandomPosterize', bits=5),
    dict(type='TorchvisionTrans', tr_type='RandomAdjustSharpness', sharpness_factor=3),
    #dict(type='TorchvisionTrans', tr_type='AugMix'),


    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root_train,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_test,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

#optimizer
# optimizer = dict(
#     type='SGD', lr=0.0001, momentum=0.9,
#     weight_decay=0.01)  # this lr is used for 8 gpus

optimizer = dict(
    type='Adam',
    lr=0.001
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=10)
total_epochs = 50

# runtime settings
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
checkpoint_config = dict(interval=1)


dist_params = dict(backend='nccl')

# fp16 settings
#fp16 = dict()


work_dir = './work_dirs/x3d_s_13_frame_fp16'
find_unused_parameters = False
gpu_ids=range(0,1)

dist_params = dict(backend='nccl')
