#!/usr/bin/env bash
#
# Copyright 2020 Axel Huebl
#
# License: BSD-3-Clause-LBNL

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

# Parse NVCC version from the command line
nvcc_version_dotted=${1:?Usage: $0 <nvcc_version>}
nvcc_version_dashed=${nvcc_version_dotted/./-}  # replace first occurence of "." with "-"

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    g++-11              \
    gfortran-11         \
    gnupg               \
    libopenmpi-dev      \
    openmpi-bin         \
    pkg-config          \
    wget

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update
sudo apt-get install -y \
    cuda-command-line-tools-${nvcc_version_dashed} \
    cuda-compiler-${nvcc_version_dashed}           \
    cuda-cupti-dev-${nvcc_version_dashed}          \
    cuda-minimal-build-${nvcc_version_dashed}      \
    cuda-nvml-dev-${nvcc_version_dashed}           \
    cuda-nvtx-${nvcc_version_dashed}               \
    libcurand-dev-${nvcc_version_dashed}           \
    libcusparse-dev-${nvcc_version_dashed}
sudo ln -s cuda-${nvcc_version_dotted} /usr/local/cuda
