#!/bin/bash

export ROOT_DIR=dataset/WOT
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

# scene_name=breville
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=car
python train_ngpgv2.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=community
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 32.0 --eval_lpips --vocab_size=10 

scene_name=grill
python train_ngpgv2.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=kitchen
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

# scene_name=living_room
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=mac
python train_ngpgv2.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=6 

scene_name=ninja
python train_ngpgv2.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

# scene_name=spa
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=street
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 32.0 --eval_lpips --vocab_size=5 
