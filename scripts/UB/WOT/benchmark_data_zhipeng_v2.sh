#!/bin/bash

export ROOT_DIR=dataset/WOT

# scene_name=breville
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 4.0 --eval_lpips --vocab_size=5 

# scene_name=breville
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoScale4.0_4.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 4.0 --eval_lpips --vocab_size=5 

# scene_name=breville
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoScale8.0_8.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 



# scene_name=kitchen
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoscale8.0_8.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=spa
python train_ngpgv2.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name}_autoscale8.0_16.0 --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 


# scene_name=home
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoscale8.0_16.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=spa
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=community
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 4.0 --eval_lpips --vocab_size=8 

# scene_name=community2
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_scale16 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=8 

# scene_name=street
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_scale16 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=4 


# scene_name=street2
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_scale16 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 

# scene_name=home_1x
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoscale8.0_8.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 

# scene_name=street_sunny
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoscale8.0_64.0_minD1e-4 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 32.0 --eval_lpips --vocab_size=5


# scene_name=spa_sunny
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_autoscale8.0_16.0 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5


# scene_name=community_sunny
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 64.0 --eval_lpips --vocab_size=7

# scene_name=community3
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 4.0 --eval_lpips --vocab_size=10 

# scene_name=community4
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_scale16 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=6 

# scene_name=spa2
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name}_scale16 --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=6 


# scene_name=community2
# python train_ngpgv2.py \
#     --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#     --exp_name ${scene_name} --downsample 1.0 \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --dim_g 16 --scale 4.0 --eval_lpips --vocab_size=5 
