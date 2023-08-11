#!/bin/bash

# inputs:
# 1. scene name
# 2. task_number

export ROOT_DIR=dataset/WOT


scene_name=$1
python merge_video.py \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_0/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/EWC/${scene_name}_0/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_10/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/${scene_name}_0_MEIL/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/CLNeRF/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/UB/colmap_ngpa_CLNerf_render/${scene_name}_0_UB/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/comparisons_UB.mp4" \
	'NT' \
	'EWC' \
	'ER' \
	'MEIL-NeRF' \
	${ROOT_DIR}/${scene_name} \
	$2 \
	1
