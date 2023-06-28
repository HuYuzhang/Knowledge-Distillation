# CUDA_VISIBLE_DEVICES=2 python train_video_rain.py -checkpoint_dir ./checkpoints/P202_video_rain_pretrain_formal_code/ -data_dir Video_rain -list_filename ./lists/video_rain_removal_train.txt -crop_size 64
CUDA_VISIBLE_DEVICES=$2 python -u $1 -checkpoint_dir $3 -data_dir Video_rain -list_filename ./lists/video_rain_removal_train_wGT_full.txt -crop_size 128 | tee $4
