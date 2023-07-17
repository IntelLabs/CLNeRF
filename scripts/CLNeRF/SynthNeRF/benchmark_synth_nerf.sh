#!/bin/bash

export ROOT_DIR=dataset/Synthetic_NeRF

rep=$1
task_number=10

# Lego
data=Lego
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done

# Chair
data=Chair
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done

# Drums
data=Drums
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done


# Ficus
data=Ficus
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done

# Hotdog
data=Hotdog
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done


# Materials
data=Materials
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done


# Mic
data=Mic
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done


# Ship
data=Ship
for ((i=0; i<$task_number; i++))
do
    python train_CLNerf.py \
        --root_dir $ROOT_DIR/$data --dataset_name nsvf_CLNerf \
        --exp_name $data'_'$task_number'_'$rep \
        --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number
done