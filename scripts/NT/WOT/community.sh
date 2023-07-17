#!/bin/bash

export ROOT_DIR=dataset/WOT

task_number=10
task_curr=9
scene_name=community

downsample=1.0
rep=$1
python train_ngpgv2_lb.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_lb \
    --exp_name ${scene_name}_${rep}_autoscale8.0_32.0  \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --dim_a 48 --scale 32.0 --downsample ${downsample} --rep_size $rep --vocab_size ${task_number}
