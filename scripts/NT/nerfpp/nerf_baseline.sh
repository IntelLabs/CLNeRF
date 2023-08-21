#!/bin/bash

python train_nerfpp_NT.py --train_split train --scene $1 --rep_size $2 --max_steps 50000 --task_number $3 --data_root $4
