# CLNeRF
Official implementation of 'CLNeRF: Continual Learning Meets NeRF'

some demo video/image here

# Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 12GB (Tested with RTX3090 Ti and RTX6000), CUDA 11.3 (might work with older version)

## Environment setup
* Clone this repo: `https://github.com/ZhipengCai/CLNeRF.git`
* simply run the code in `setup_env.sh` line by line (to avoid failure in specific line so that your own environment is damaged)

## Dataset prepare (Naming follows Fig.4 of the main paper, currently support WOT, SynthNeRF and NeRF++)

```bash
bash prepare_datasets.sh
```

# Run experiments

```bash
# run experiments on CLNeRF (WOT, SynthNeRF and NeRF++ datasets are currently supported)
bash run_CLNeRF.sh
# run experiments on MEIL-NeRF (WOT, SynthNeRF and NeRF++ datasets are currently supported)
bash run_MEIL.sh
```

# Contact
Please contact Zhipeng Cai (homepage: https://zhipengcai.github.io/, email: czptc2h@gmail.com) if you have questions, comments or want to collaborate on this repository to make 
