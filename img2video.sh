#!/bin/bash

# inputs:
# 1. scene name
# 2. task_number

export ROOT_DIR=dataset/WAT

scene_name=$1
python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WAT/NT_ER/'${scene_name}'_0/video' \
	${ROOT_DIR}/${scene_name} \
	$2


python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WAT/NT_ER/'${scene_name}'_10/video' \
	${ROOT_DIR}/${scene_name} \
	$2

python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WAT/EWC/'${scene_name}'_0/video' \
	${ROOT_DIR}/${scene_name} \
	$2

