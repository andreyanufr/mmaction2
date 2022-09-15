_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='EVLTransformer',
        num_frames=8,
        decoder_qkv_dim=768,
        decoder_num_heads=12,
        backbone_name="ViT-B/16-lnpre",
        backbone_path='/local_ssd3/jeom/CLIP_checkpoints/ViT-B-16.pt'
        ),
    cls_head=dict(type='EVLHead', num_classes=400, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/videos_train_sampled'
data_root_val = 'data/kinetics400/videos_val_sampled'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos_sampled.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos_sampled.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos_sampled.txt'

img_norm_cfg = dict(
    mean=[122.77, 116.74, 104.09], std=[68.50, 66.63, 70.32], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=16, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='PytorchVideoTrans', tr_type='RandAugment', magnitude=7, num_layers=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
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
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    weight_decay=0.05,
)

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 60

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/EVL_ViT-B16-8f_kinetics400'
fp16=dict(loss_scale='dynamic')
