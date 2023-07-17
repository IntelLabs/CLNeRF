#!/bin/sh

python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate $1 $2
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/community
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/kitchen
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/living_room
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/spa
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/street
