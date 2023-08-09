#!/bin/bash

python render_video_EWC.py --train_split train --scene $1 --rep_size $2 --vocab_size $3 --max_steps 50000 --task_number $3 --data_root $4 --frame_start $5 --frame_end $6
