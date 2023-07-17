#!/bin/bash

export ROOT_DIR=dataset/WOT

task_number=10
scene_name=community
downsample=1.0

rep=$1
for ((i=0; i<$task_number; i++))
do
    python train_ngpgv2_MEIL.py \
        --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_MEIL \
        --exp_name ${scene_name}_${rep}_autoScale8.0_32.0 \
        --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number --dim_a 48 --scale 32.0 --downsample ${downsample} --vocab_size ${task_number}
done
