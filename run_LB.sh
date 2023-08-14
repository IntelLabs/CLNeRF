#!/bin/bash

# LB on WAT dataset
rep=0
bash scripts/LB/WAT/breville.sh ${rep}
bash scripts/LB/WAT/community.sh ${rep}
bash scripts/LB/WAT/kitchen.sh ${rep}
bash scripts/LB/WAT/living_room.sh ${rep}
bash scripts/LB/WAT/spa.sh ${rep}
bash scripts/LB/WAT/street.sh ${rep}

# LB on Synth-NeRF dataset
bash scripts/LB/SynthNeRF/benchmark_synth_nerf.sh

# # CLNeRF on NeRF++ dataset
bash scripts/LB/nerfpp/benchmark_nerfpp.sh

