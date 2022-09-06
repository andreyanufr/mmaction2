bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb.py 4 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split1 --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb.py 4 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split1 --seed 0 --deterministic

bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_2.py 4 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split2 --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_2.py 4 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split2 --seed 0 --deterministic

bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_3.py 4 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split3 --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_3.py 4 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split3 --seed 0 --deterministic

bash tools/dist_train.sh configs/recognition/evl/evl_hmdb51_rgb_slowonly_cfg.py 4 --validate --work-dir work_dirs/EVL_HMDB51_ViT-L-14_split1_slowonly_cfg --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/evl/evl_ucf101_rgb_slowonly_cfg.py 4 --validate --work-dir work_dirs/EVL_UCF101_ViT-L-14_split1_slowonly_cfg --seed 0 --deterministic
