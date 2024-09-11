# Define base image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Install required apt packages and clear cache afterwards
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    dos2unix \
    git \
    ninja-build \
    screen \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Copy required files and folders
COPY setup_env_docker.sh .
COPY install_cuda_module.sh .
COPY models models/
RUN dos2unix setup_env_docker.sh && \
    dos2unix install_cuda_module.sh

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh && \
    bash /opt/miniconda-installer.sh -b -u -p /opt/miniconda3

# Set up conda environment and activate by default
ENV PATH=/opt/miniconda3/bin:$PATH
RUN conda init bash 

# Install required packages
RUN bash setup_env_docker.sh && \
    echo "conda activate CLNeRF" >> ~/.bashrc

# Clean up
RUN rm setup_env_docker.sh && \
    rm install_cuda_module.sh && \
    rm -rf models

# Switch to workspace folder (this is where the code will be mounted)
WORKDIR /workspace/CLNeRF

# Bash as default entrypoint
CMD ["/bin/bash"]
