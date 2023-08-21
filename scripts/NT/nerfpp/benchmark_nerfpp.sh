#!/bin/bash

export ROOT_DIR=dataset/tanks_and_temples

task_number=10
task_curr=9

data=tat_intermediate_M60
python train_lb.py \
    --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_lb \
    --exp_name $data'_'$task_number'task' --no_save_test \
    --num_epochs 20 --scale 4.0 \
    --task_number $task_number --task_curr $task_curr

data=tat_intermediate_Playground
python train_lb.py \
    --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_lb \
    --exp_name $data'_'$task_number'task' --no_save_test \
    --num_epochs 20 --scale 4.0 \
    --task_number $task_number --task_curr $task_curr


data=tat_intermediate_Train
python train_lb.py \
    --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_lb \
    --exp_name $data'_'$task_number'task' --no_save_test \
    --num_epochs 20 --scale 16.0 --batch_size 4096 \
    --task_number $task_number --task_curr $task_curr


data=tat_training_Truck
python train_lb.py \
    --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_lb \
    --exp_name $data'_'$task_number'task' --no_save_test \
    --num_epochs 20 --scale 16.0 --batch_size 4096 \
    --task_number $task_number --task_curr $task_curr
