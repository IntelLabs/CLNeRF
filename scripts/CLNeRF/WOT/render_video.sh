#!/bin/bash
export ROOT_DIR=dataset/WOT

task_number=$1
task_curr=$2
scene_name=$3
rep=0
model_path=$4
downsample=1.0


python render_NGP_WOT.py \
	--root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
	--exp_name ${scene_name}_${rep} \
	--num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
	--task_curr ${task_curr} --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number} \
	--weight_path ${model_path} --val_only

