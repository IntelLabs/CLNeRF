#!/bin/bash

export ROOT_DIR=dataset/WOT

task_number=5
task_curr=4
scene_name=street

downsample=1.0
scale=32.0
rep=$1
python train_ngpgv2_lb.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_lb \
    --exp_name ${scene_name}_${rep}_autoscale8.0_${scale}  \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --dim_a 48 --scale ${scale} --downsample ${downsample} --rep_size $rep --vocab_size=${task_number}
