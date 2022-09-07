# bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb.py 8 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split1_V100_8gpu --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/movinet_SC_eperiment/movinets_A0_base_UCF101_split1.py 8 --validate --work-dir work_dirs/MoViNet_A0_base_UCF101_V100_8gpu --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb.py 8 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split1_movinet_setting_V100_8gpu --seed 0 --deterministic

# bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_2.py 8 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split2_V100_8gpu --seed 0 --deterministic
# bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_2.py 8 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split2_V100_8gpu --seed 0 --deterministic

# bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_3.py 8 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split3_V100_8gpu --seed 0 --deterministic
# bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_3.py 8 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split3_V100_8gpu --seed 0 --deterministic

# bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_slowonly_cfg.py 8 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split1_slowonly_cfg_V100_8gpu --seed 0 --deterministic
# bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_slowonly_cfg.py 8 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split1_slowonly_cfg_V100_8gpu --seed 0 --deterministic


# # python3 tools/train.py configs/recognition/evl/evl_hmdb51_rgb.py --gpus 0 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split1_V100_8gpu --seed 0 --deterministic


# bash tools/dist_train.sh configs/recognition/movinet_SC_eperiment/movinets_A0_base_UCF101_split1.py 8 --validate --work-dir work_dirs/MoViNet_A0_base_UCF101_V100_8gpu --seed 0 --deterministic

# python3 tools/train.py configs/recognition/movinet_SC_eperiment/movinets_A0_base_UCF101.py --gpus 1 --validate --work-dir work_dirs/MoViNet_A0_base_UCF101_V100_1gpu --seed 0 --deterministic
