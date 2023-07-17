#!/bin/bash

# LB on WOT dataset
rep=0
bash scripts/LB/WOT/breville.sh ${rep}
bash scripts/LB/WOT/community.sh ${rep}
bash scripts/LB/WOT/kitchen.sh ${rep}
bash scripts/LB/WOT/living_room.sh ${rep}
bash scripts/LB/WOT/spa.sh ${rep}
bash scripts/LB/WOT/street.sh ${rep}

# LB on Synth-NeRF dataset
bash scripts/LB/SynthNeRF/benchmark_synth_nerf.sh

# # CLNeRF on NeRF++ dataset
bash scripts/LB/nerfpp/benchmark_nerfpp.sh

