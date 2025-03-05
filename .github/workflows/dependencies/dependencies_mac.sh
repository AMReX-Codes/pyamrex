#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

brew update
brew install gfortran || true
brew install libomp || true
brew install open-mpi || true
brew install ccache || true

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip setuptools[core] wheel pytest
python3 -m pip install -U cmake
