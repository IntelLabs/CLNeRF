#!/bin/bash

# LB on WOT dataset
rep=0
bash scripts/NT/WOT/nerf_baseline.sh breville ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh community ${rep} 10 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh kitchen ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh living_room ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh spa ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh street ${rep} 5 dataset/WOT

# # LB on Synth-NeRF dataset
# bash scripts/LB/SynthNeRF/benchmark_synth_nerf.sh

# # # CLNeRF on NeRF++ dataset
# bash scripts/LB/nerfpp/benchmark_nerfpp.sh

