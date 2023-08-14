#!/bin/bash

# CLNeRF on WAT dataset
rep=10
bash scripts/CLNeRF/WAT/breville.sh ${rep}
bash scripts/CLNeRF/WAT/community.sh ${rep}
bash scripts/CLNeRF/WAT/kitchen.sh ${rep}
bash scripts/CLNeRF/WAT/living_room.sh ${rep}
bash scripts/CLNeRF/WAT/spa.sh ${rep}
bash scripts/CLNeRF/WAT/street.sh ${rep}
bash scripts/CLNeRF/WAT/car.sh ${rep}
bash scripts/CLNeRF/WAT/grill.sh ${rep}
bash scripts/CLNeRF/WAT/mac.sh ${rep}
bash scripts/CLNeRF/WAT/ninja.sh ${rep}


# CLNeRF on Synth-NeRF dataset
rep=10
bash scripts/CLNeRF/SynthNeRF/benchmark_synth_nerf.sh ${rep}

# CLNeRF on NeRF++ dataset
rep=10
bash scripts/CLNeRF/nerfpp/benchmark_nerfpp.sh ${rep}

