#!/bin/bash

# LB on WOT dataset
rep=0
bash scripts/NT/WOT/nerf_baseline.sh breville ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh community ${rep} 10 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh kitchen ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh living_room ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh spa ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh street ${rep} 5 dataset/WOT
bash scripts/NT/WOT/nerf_baseline.sh car_resized ${rep} 5 dataset/WOT

# # # # Synth-NeRF dataset
bash scripts/NT/SynthNeRF/nerf_baseline.sh chair ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh drums ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh ficus ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh hotdog ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh lego ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh materials ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh mic ${rep} 10 dataset/nerf_synthetic
bash scripts/NT/SynthNeRF/nerf_baseline.sh ship ${rep} 10 dataset/nerf_synthetic

# # # #  NeRF++ dataset
bash scripts/NT/nerfpp/nerf_baseline.sh tat_intermediate_M60 ${rep} 10 dataset/tanks_and_temples
bash scripts/NT/nerfpp/nerf_baseline.sh tat_intermediate_Playground ${rep} 10 dataset/tanks_and_temples
bash scripts/NT/nerfpp/nerf_baseline.sh tat_intermediate_Train ${rep} 10 dataset/tanks_and_temples
bash scripts/NT/nerfpp/nerf_baseline.sh tat_training_Truck ${rep} 10 dataset/tanks_and_temples
