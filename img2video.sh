#!/bin/bash

# inputs:
# 1. path to baseline video
# 2. path to CLNeRF video
# 3. path to UB video
# 4. path to output video 
# 5. baseline name
# 6. root_dir of the data folder (to extract task ids)
# 7. task_number

export ROOT_DIR=dataset/WOT

# breville
# write code to specify the task number
scene_name=breville
# python img2video.py \
# 	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/breville_0/video' \
# 	${ROOT_DIR}/${scene_name} \
# 	5

python img2video.py \
	'/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/breville_10/video' \
	${ROOT_DIR}/${scene_name} \
	5