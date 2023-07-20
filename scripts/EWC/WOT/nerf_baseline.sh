#!/bin/bash

python train_WOT_EWC.py --train_split train --scene $1 --rep_size $2 --vocab_size $3 --max_steps 50000 --task_number $3 --data_root $4 --EWC_weight 1e5
