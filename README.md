# CLNeRF
Official implementation of 'CLNeRF: Continual Learning Meets NeRF'

We study the problem of continual learning in the context of NeRFs. We propose a new dataset World Over Time (WOT) for this purpose, where during continual learning, the scene appearance and geometry can change over time (at different time step/task of continual learning). We propose a simple yet effective method CLNeRF which combines generative replay with advanced NeRF architectures so that a single NeRF model can efficiently adapt to gradually revealed new data, i.e., render scenes at different time with potential appearance and geometry changes, without the need to store historical images.

To facilitate future research on continual NeRF, we provide the code to run different continual learning methods on different NeRF datasets (including WOT).

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
# run experiments on MEIL-NeRF
bash run_MEIL.sh
# run experiments on ER (experience replay)
bash run_ER.sh
# run experiments on EWC 
bash run_EWC.sh
# run experiments on NT (naive training/finetuning on the sequential data)
bash run_NT.sh

```

# Contact
Please contact Zhipeng Cai (homepage: https://zhipengcai.github.io/, email: czptc2h@gmail.com) if you have questions, comments or want to collaborate on this repository to make 
