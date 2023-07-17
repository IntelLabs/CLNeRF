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

# python train.py \
#     --root_dir $ROOT_DIR/tat_intermediate_Playground --dataset_name nerfpp_lb \
#     --exp_name tat_intermediate_Playground --no_save_test \
#     --num_epochs 20 --scale 4.0

# python train.py \
#     --root_dir $ROOT_DIR/tat_intermediate_Train --dataset_name nerfpp_lb \
#     --exp_name tat_intermediate_Train --no_save_test \
#     --num_epochs 20 --scale 16.0 --batch_size 4096

# python train.py \
#     --root_dir $ROOT_DIR/tat_training_Truck --dataset_name nerfpp_lb \
#     --exp_name tat_training_Truck --no_save_test \
#     --num_epochs 20 --scale 16.0 --batch_size 4096

# export ROOT_DIR=/home/ubuntu/data/nerf_data/lf_data

# python train.py \
#     --root_dir $ROOT_DIR/africa --dataset_name nerfpp \
#     --exp_name africa --no_save_test \
#     --num_epochs 20 --scale 16.0 --eval_lpips

# # basket fails for some unknown reason (black stripes appear in test image)
# # python train.py \
# #     --root_dir $ROOT_DIR/basket --dataset_name nerfpp \
# #     --exp_name basket --no_save_test \
# #     --num_epochs 20 --scale 16.0 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/ship --dataset_name nerfpp \
#     --exp_name ship --no_save_test \
#     --num_epochs 20 --scale 8.0 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/statue --dataset_name nerfpp \
#     --exp_name statue --no_save_test \
#     --num_epochs 20 --scale 16.0 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/torch --dataset_name nerfpp \
#     --exp_name torch --no_save_test \
#     --num_epochs 20 --scale 32.0 --eval_lpips