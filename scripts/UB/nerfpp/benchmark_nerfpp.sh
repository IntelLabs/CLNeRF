#!/bin/bash

export ROOT_DIR=dataset/tanks_and_temples

python train.py \
    --root_dir $ROOT_DIR/tat_intermediate_M60 --dataset_name nerfpp \
    --exp_name tat_intermediate_M60 \
    --num_epochs 20 --scale 4.0

python train.py \
    --root_dir $ROOT_DIR/tat_intermediate_Playground --dataset_name nerfpp \
    --exp_name tat_intermediate_Playground \
    --num_epochs 20 --scale 4.0

python train.py \
    --root_dir $ROOT_DIR/tat_intermediate_Train --dataset_name nerfpp \
    --exp_name tat_intermediate_Train \
    --num_epochs 20 --scale 16.0 --batch_size 4096

python train.py \
    --root_dir $ROOT_DIR/tat_training_Truck --dataset_name nerfpp \
    --exp_name tat_training_Truck \
    --num_epochs 20 --scale 16.0 --batch_size 4096
