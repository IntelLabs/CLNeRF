#!/bin/bash

# export ROOT_DIR=dataset/WOT

# task_number=5
# task_curr=4
# scene_name=breville


# downsample=1.0
# rep=$1
# python train_ngpgv2_lb.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_lb \
#     --exp_name ${scene_name}_${rep}_autoScale8.0_8.0  \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips \
#     --task_number $task_number --task_curr $task_curr --dim_a 48 --dim_g 16 --scale 8.0 --downsample ${downsample} --rep_size $rep --vocab_size=5


python render_video_EWC.py --train_split train --scene $1 --rep_size $2 --vocab_size $3 --max_steps 50000 --task_number $3 --data_root $4 --frame_start $5 --frame_end $6
