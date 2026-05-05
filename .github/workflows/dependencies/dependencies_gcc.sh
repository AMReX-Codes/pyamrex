#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

# Parse GCC version from the command line
gcc_version=${1:?Usage: $0 <gcc_version>}

sudo apt-get update

sudo apt-get install -y --no-install-recommends\
    build-essential \
    g++-${gcc_version} \
    libopenmpi-dev \
    openmpi-bin
