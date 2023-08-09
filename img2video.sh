#!/bin/bash

# inputs:
# 1. scene name
# 2. task_number

export ROOT_DIR=dataset/WOT

scene_name=$1
python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/'${scene_name}'_0/video' \
	${ROOT_DIR}/${scene_name} \
	$2


python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/'${scene_name}'_10/video' \
	${ROOT_DIR}/${scene_name} \
	$2

python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/EWC/'${scene_name}'_0/video' \
	${ROOT_DIR}/${scene_name} \
	$2

