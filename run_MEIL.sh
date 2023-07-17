#!/bin/bash
# Note: with the un-optimized code, MEIL-NeRF sometimes needs 48gb memory, so please prepare a high-end GPU for this experiment
# MEIL-NeRF on WOT dataset
rep=10
bash scripts/MEIL/WOT/breville.sh ${rep}
bash scripts/MEIL/WOT/community.sh ${rep}
bash scripts/MEIL/WOT/kitchen.sh ${rep}
bash scripts/MEIL/WOT/living_room.sh ${rep}
bash scripts/MEIL/WOT/spa.sh ${rep}
bash scripts/MEIL/WOT/street.sh ${rep}

# # MEIL-NeRF on Synth-NeRF dataset
rep=10
bash scripts/MEIL/SynthNeRF/benchmark_synth_nerf.sh ${rep}

# MEIL-NeRF on NeRF++ dataset
rep=10
bash scripts/MEIL/nerfpp/benchmark_nerfpp.sh ${rep}

