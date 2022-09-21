_base_ = ['../../_base_/default_runtime.py']
num_classes = 3
num_samples = 12
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MoViNetBase',
        name="MoViNetA0",
        num_classes=num_classes),
    cls_head=dict(
        type='MoViNetHead',
        in_channels=480,
        hidden_dim = 2048,
        num_classes=num_classes,
        spatial_type='avg',
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/jester/rawframes'
data_root_val = 'data/jester/rawframes'
# ann_file_train = 'data/jester/jester_train_list_rawframes.txt'
# ann_file_val = 'data/jester/jester_val_list_rawframes.txt'
# ann_file_test = 'data/jester/jester_val_list_rawframes.txt'
seed=3
ann_file_train = f'data/jester/SC_jester_{num_classes}cls_{num_samples}_samples_seed_{seed}/train_list_rawframes.txt'
ann_file_val = f'data/jester/SC_jester_{num_classes}cls_{num_samples}_samples_seed_{seed}/val_list_rawframes.txt'
ann_file_test = f'data/jester/SC_jester_{num_classes}cls_{num_samples}_samples_seed_{seed}/test_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    to_bgr=False
)

clip_len=8
frame_interval=4

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
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

    #dict(type='PytorchVideoTrans', tr_type='AugMix'),
    #dict(type='PytorchVideoTrans', tr_type='RandAugment'),


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
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=1,
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
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=1,
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
    videos_per_gpu=10,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

#optimizer
# optimizer = dict(
#     type='SGD', lr=0.0001, momentum=0.9,
#     weight_decay=0.01)  # this lr is used for 8 gpus

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
)

optimizer_config = dict(grad_clip=dict(max_norm=40.0, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=5)
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=1)


# fp16 settings
#fp16 = dict()


find_unused_parameters = False
gpu_ids=range(0,1)

dist_params = dict(backend='nccl')
load_from = "../CLIP_checkpoints/modelA0_statedict_v3"