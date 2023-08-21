#!/bin/bash
# Note: with the un-optimized code, MEIL-NeRF sometimes needs 48gb memory, so please prepare a high-end GPU for this experiment
# MEIL-NeRF on WAT dataset
rep=10
bash scripts/MEIL/WAT/breville.sh ${rep}
bash scripts/MEIL/WAT/community.sh ${rep}
bash scripts/MEIL/WAT/kitchen.sh ${rep}
bash scripts/MEIL/WAT/living_room.sh ${rep}
bash scripts/MEIL/WAT/spa.sh ${rep}
bash scripts/MEIL/WAT/street.sh ${rep}

# # MEIL-NeRF on Synth-NeRF dataset
rep=10
bash scripts/MEIL/SynthNeRF/benchmark_synth_nerf.sh ${rep}

# MEIL-NeRF on NeRF++ dataset
rep=10
bash scripts/MEIL/nerfpp/benchmark_nerfpp.sh ${rep}

