#!/bin/bash
export ROOT_DIR=dataset/WAT

task_number=$1
task_curr=$2
scene_name=$3
model_path=$4
rep=$5
scale=$6
render_fname=$7
downsample=1.0

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH


python render_NGP_WAT.py \
	--root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf_render \
	--exp_name ${scene_name}_${rep} \
	--num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
	--task_curr ${task_curr} --task_number $task_number --dim_a 48 --scale ${scale} --downsample ${downsample} --vocab_size ${task_number} \
	--weight_path ${model_path} --render_fname ${render_fname} --val_only

