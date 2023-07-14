#!/bin/bash

export ROOT_DIR=/mnt/beegfs/mixed-tier/work/zcai/WorkSpace/NeRF/nerf_zoo/data_continual_nerf/geometry_change

task_number=5
scene_name=kitchen
downsample=1.0

# rep=0
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpgv2_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_autoScale8.0_8.0 \
#         --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
# done

rep=5
for ((i=0; i<$task_number; i++))
do
    python train_ngpgv2_CLNerf.py \
        --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
        --exp_name ${scene_name}_${rep}_autoScale8.0_8.0 \
        --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
done

# rep=10
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpgv2_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_autoScale8.0_8.0 \
#         --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
# done

# rep=20
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpgv2_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_autoScale8.0_8.0 \
#         --num_epochs 20 --batch_size 8192 --lr 1e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 8.0 --downsample ${downsample} --vocab_size ${task_number}
# done


# rep=5
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpa_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_ds${downsample} \
#         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 4.0 --downsample ${downsample}
# done


# rep=10
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpa_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_ds${downsample} \
#         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 4.0 --downsample ${downsample}
# done

# rep=20
# for ((i=0; i<$task_number; i++))
# do
#     python train_ngpa_CLNerf.py \
#         --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa_CLNerf \
#         --exp_name ${scene_name}_${rep}_ds${downsample} \
#         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
#         --task_curr $i --task_number $task_number --dim_a 48 --scale 4.0 --downsample ${downsample}
# done

# # # Chair
# # data=Chair
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done

# # # Drums
# # data=Drums
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done


# # # Ficus
# # data=Ficus
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done

# # # Hotdog
# # data=Hotdog
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done


# # # Materials
# # data=Materials
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done


# # # Mic
# # data=Mic
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number
# # done


# # # Ship
# # data=Ship
# # for ((i=0; i<$task_number; i++))
# # do
# #     python train_ngpa_CLNerf.py \
# #         --root_dir $ROOT_DIR/$data --dataset_name nsvf_ngpa_CLNerf \
# #         --exp_name $data'_'$task_number'_'$rep --no_save_test \
# #         --num_epochs 20 --batch_size 16384 --lr 2e-2 --rep_size $rep --eval_lpips \
# #         --task_curr $i --task_number $task_number --dim_a 16
# # done