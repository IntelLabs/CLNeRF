#!/bin/bash

export ROOT_DIR=dataset/Synthetic_NeRF


rep=10
task_number=10
task_curr=9

python train_lb.py \
    --root_dir $ROOT_DIR/Chair --dataset_name nsvf_lb \
    --exp_name Chair_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Drums --dataset_name nsvf_lb \
    --exp_name Drums_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Ficus --dataset_name nsvf_lb \
    --exp_name Ficus_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Hotdog --dataset_name nsvf_lb \
    --exp_name Hotdog_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Lego --dataset_name nsvf_lb \
    --exp_name Lego_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Materials --dataset_name nsvf_lb \
    --exp_name Materials_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Mic --dataset_name nsvf_lb \
    --exp_name Mic_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep

python train_lb.py \
    --root_dir $ROOT_DIR/Ship --dataset_name nsvf_lb \
    --exp_name Ship_10task --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --task_number $task_number --task_curr $task_curr --rep_size $rep