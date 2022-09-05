_base_ = ['../../_base_/default_runtime.py']
# from mmaction/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb.py
# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='EVLTransformer',
        num_frames=32,
        decoder_qkv_dim=1024,
        decoder_num_heads=16,
        backbone_name="ViT-L/14-lnpre",
        backbone_path='/home/jeom/workspace/mmaction/ViT-L-14.pt'
        ),
    cls_head=dict(type='EVLHead', num_classes=101, in_channels=1024),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ucf101/rawframes/'
data_root_val = 'data/ucf101/rawframes/'
split = 3  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
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
        clip_len=32,
        frame_interval=4,
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
        clip_len=32,
        frame_interval=4,
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
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], save_best='top1_acc')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15
fp16=dict(loss_scale='dynamic')
# runtime settings
work_dir = './work_dirs/EVL_UCF101_ViT-L-14'
load_from = '../EVL_ViT-L-14.pth'
