#!/bin/bash

export ROOT_DIR=/export/share/datasets/Carla_Dynamic/NeRF/nerf_synthetic_NSVF/Synthetic_NeRF

python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name Drums \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name Ficus \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name Hotdog \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name Materials \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name Mic \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name Ship \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips