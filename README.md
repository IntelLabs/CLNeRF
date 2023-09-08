# CLNeRF
Official implementation of 'CLNeRF: Continual Learning Meets NeRF' (accepted to ICCV'23)

[[Paper](https://arxiv.org/abs/2308.14816)] [[Video](https://youtu.be/nLRt6OoDGq0)] [[Dataset](https://huggingface.co/datasets/IntelLabs/WAT-WorldAcrossTime)]  [Web Demo (coming soon)]

![Example Image](https://github.com/ZhipengCai/CLNeRF/blob/main/demo/teaser.png)

We study the problem of continual learning in the context of NeRFs. We propose a new dataset World Across Time (WAT) for this purpose, where during continual learning, the scene appearance and geometry can change over time (at different time step/task of continual learning). We propose a simple yet effective method CLNeRF which combines generative replay with advanced NeRF architectures so that a single NeRF model can efficiently adapt to gradually revealed new data, i.e., render scenes at different time with potential appearance and geometry changes, without the need to store historical images.

To facilitate future research on continual NeRF, we provide the code to run different continual learning methods on different NeRF datasets (including WAT).

Please give us a star or cite our paper if you find it useful.

# Contact
Please contact Zhipeng Cai (homepage: https://zhipengcai.github.io/, email: czptc2h@gmail.com) if you have questions, comments or want to collaborate on this repository to make it better.

We are actively looking for good research interns, contact Zhipeng if you are interested (multiple bases are possible, e.g., US, Munich, China).

```bash
@inproceedings{iccv23clnerf,
title={CLNeRF: Continual Learning Meets NeRF},
author={Zhipeng Cai, Matthias Müller},
year={2023},
booktitle={ICCV},
}
```

# Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 12GB (Tested with RTX3090 Ti and RTX6000), CUDA 11.3 (might work with older version)

## Environment setup
* Clone this repo and submodules (pycolmap): `git clone --recurse-submodules https://github.com/IntelLabs/CLNeRF.git`
* simply run the code in `setup_env.sh` line by line (to avoid failure in specific line so that your own environment is damaged)

## Dataset prepare (Naming follows Fig.4 of the main paper, currently support WAT, SynthNeRF and NeRF++)

```bash
bash prepare_datasets.sh
```

# Run experiments

```bash
# run experiments on CLNeRF (WAT, SynthNeRF and NeRF++ datasets are currently supported)
bash run_CLNeRF.sh
# run experiments on MEIL-NeRF
bash run_MEIL.sh
# run experiments on ER (experience replay)
bash run_ER.sh
# run experiments on EWC 
bash run_EWC.sh
# run experiments on NT (naive training/finetuning on the sequential data)
bash run_NT.sh
# render video using CLNeRF model
scene=breville
task_number=5
task_curr=4
rep=10
scale=8.0 # change to the right scale according to the corresponding training script (scripts/NT/WAT/breville.sh)
ckpt_path=/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/${scene}_10/epoch=19-v4.ckpt # change to your ckpt path
bash scripts/CLNeRF/WAT/render_video.sh $task_number $task_curr $scene $ckpt_path $rep $scale $render_fname
```
# License

This repository is under the Apache 2.0 License, it is free for non-commercial use. Please contact Zhipeng for other use cases.
