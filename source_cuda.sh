#!/bin/bash
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
# export PATH=${PATH}:/usr/local/cuda/bin
# export CUDA_HOME=/usr/local/cuda-11 # change to your own cuda 11 repository (not necessarily 11.3, I think as long as it is 11 it would work)
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
# export PATH=${PATH}:/usr/local/cuda/bin

# export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
# export CPATH=/usr/local/cuda-11/include${CPATH:+:${CPATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
