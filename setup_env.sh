#!/bin/sh
# initialize environment
conda create -n CLNeRF python=3.8 
conda activate CLNeRF
# link to your cuda 11 folder
export CUDA_HOME=/usr/local/cuda-11.6 # change to your own cuda 11 repository (11.3 will be the best)

# install pytorch
pip install torch==1.11.0 torchvision==0.12.0 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.9.5 # must use old version to be compatible with the code
# torch scatter (see https://github.com/rusty1s/pytorch_scatter#installation for more details)
pip install --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# tinycudann (see https://github.com/NVlabs/tiny-cuda-nn#requirements)
pip install --force-reinstall git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# apex (see https://github.com/NVIDIA/apex#linux)
# note that you can comment out the cuda version check at line 32 of setup.py in apex folder so that if you don't have exactly cuda 11.3 it will still work
git clone https://github.com/NVIDIA/apex
cd apex
pip install --force-reinstall packaging
pip install flit_core
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..
rm -rf apex
# cuda extension (modified NGP architecture with appearance and geometric embeddings, please run this line each time you pull the code)
bash install_cuda_module.sh

# nerfacc (for vanilla NeRF baselines)
pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.11.0_cu113.html

# install some necessary libraries
pip install imageio==2.19.3
pip install opencv-python
pip install einops
pip install tqdm
pip install kornia
pip install pandas
pip install torchmetrics
pip install torchmetrics[image]
pip install -U 'tensorboardX'

# the followings are for creating WAT dataset for your custom scenes, can ignore if you dont want them
pip install scikit-image==0.19.3
conda install -c conda-forge colmap
