#!/bin/bash
# inputs:
# 1. path to baseline video
# 2. path to CLNeRF video
# 3. path to UB video
# 4. path to output video 
# 5. baseline name
python merge_video.py \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/ninja_10_MEIL/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/ninja_10_CLNeRF/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/test_videos/output_ref_bb.mp4" \
	'MEIL'
