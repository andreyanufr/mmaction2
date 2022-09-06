# global parameters
# num_videos_per_gpu = 12
# num_workers_per_gpu = 3
# train_sources = ('custom_dataset', )
# test_sources = ('custom_dataset', )

root_dir = 'data'
work_dir = None
load_from = None
resume_from = None
reset_layer_prefixes = ['cls_head']
reset_layer_suffixes = None

# model settings
input_img_size = 172
input_clip_length = 11
input_frame_interval = 6

# training settings
enable_clip_mixing = False
num_train_clips = 2 if enable_clip_mixing else 1

# model definition
model = dict(
    test_cfg=dict(average_clips='prob'),
)

# model training and testing settings
train_cfg = dict(
    self_challenging=dict(enable=False, drop_p=0.33),
    clip_mixing=dict(enable=enable_clip_mixing, mode='logits', num_clips=num_train_clips,
                     scale=10.0, weight=0.2),
    loss_norm=dict(enable=False, gamma=0.9),
    sample_filtering=dict(enable=True, warmup_epochs=1),
)
test_cfg = dict(
    average_clips=None
)

# dataset settings
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_bgr=False
# )

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    to_bgr=False
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics600/rawframes_train'
data_root_val = 'data/kinetics600/rawframes_val'
ann_file_train = 'data/kinetics600/kinetics600_train_list_rawframes.txt'
ann_file_val = 'data/kinetics600/kinetics600_val_list_rawframes.txt'
ann_file_test = 'data/kinetics600/kinetics600_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

clip_len = 10
frame_interval = 4
shape = 172

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    to_bgr=False
)

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MoViNetBase',
        name="MoViNetA0",
        num_classes=600),
    cls_head=dict(
        type='MoViNetHead',
        in_channels=480,
        hidden_dim = 2048,
        num_classes=600,
        spatial_type='avg',
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=50,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, shape)),
    #dict(type='ThreeCrop', crop_size=shape),
    dict(type='CenterCrop', crop_size=shape),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=50,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

dist_params = dict(backend='nccl')
#gpu_ids=range(2, 3)
gpu_ids=range(0, 1)