#!/bin/bash

# CLNeRF on WOT dataset
rep=10
bash scripts/CLNeRF/WOT/breville.sh ${rep}
bash scripts/CLNeRF/WOT/community.sh ${rep}
bash scripts/CLNeRF/WOT/kitchen.sh ${rep}
bash scripts/CLNeRF/WOT/living_room.sh ${rep}
bash scripts/CLNeRF/WOT/spa.sh ${rep}
bash scripts/CLNeRF/WOT/street.sh ${rep}
bash scripts/CLNeRF/WOT/car.sh ${rep}
bash scripts/CLNeRF/WOT/grill.sh ${rep}
bash scripts/CLNeRF/WOT/mac.sh ${rep}
bash scripts/CLNeRF/WOT/ninja.sh ${rep}


# CLNeRF on Synth-NeRF dataset
rep=10
bash scripts/CLNeRF/SynthNeRF/benchmark_synth_nerf.sh ${rep}

# CLNeRF on NeRF++ dataset
rep=10
bash scripts/CLNeRF/nerfpp/benchmark_nerfpp.sh ${rep}

