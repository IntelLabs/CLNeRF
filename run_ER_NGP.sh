#!/bin/bash

# LB on WAT dataset
rep=0
bash scripts/ER_NGP/WAT/breville.sh
bash scripts/NT/WAT/nerf_baseline.sh community ${rep} 10 dataset/WAT
bash scripts/NT/WAT/nerf_baseline.sh kitchen ${rep} 5 dataset/WAT
bash scripts/NT/WAT/nerf_baseline.sh living_room ${rep} 5 dataset/WAT
bash scripts/NT/WAT/nerf_baseline.sh spa ${rep} 5 dataset/WAT
bash scripts/NT/WAT/nerf_baseline.sh street ${rep} 5 dataset/WAT

# # # # Synth-NeRF dataset
bash scripts/NT/SynthNeRF/benchmark_synth_nerf.sh


# # # #  NeRF++ dataset
bash scripts/NT/nerfpp/benchmark_nerfpp.sh
