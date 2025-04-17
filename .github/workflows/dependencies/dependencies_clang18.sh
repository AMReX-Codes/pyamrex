#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y  \
    build-essential      \
    clang-18             \
    lld                  \
    libc++-18-dev        \
    libopenmpi-dev       \
    openmpi-bin
