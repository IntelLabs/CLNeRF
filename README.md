# CLNeRF
Official implementation of CLNeRF (Coming soon)

some demo video/image here

# Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 12GB (Tested with RTX3090 Ti and A6000), CUDA 11.3 (might work with older version)

## Software

* Clone this repo: `https://github.com/ZhipengCai/CLNeRF.git`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n CLNeRF python=3.8` to create a conda environment and activate it by `conda activate CLNeRF`)
* libraries
    * Install pytorch by `pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `bash install_cuda_module.sh` (please run this each time you `pull` the code)

## Dataset prepare

* Synth-NeRF

* NeRF ++

* WOT

# Code running

	Synth-NeRF
	NeRF++
	WOT (introduce the dataset here)


