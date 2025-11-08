#!/usr/bin/env bash
#
# Copyright 2021-2023 The AMReX Community
#
# This script updates the .pyi stub files for documentation and interactive use.
# To run this script, pyAMReX needs to be installed (all dimensions) and importable.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
set -eu -o pipefail

# we are in the source directory, .github/
this_dir=$(cd $(dirname $0) && pwd)

pybind11-stubgen --exit-code --enum-class-locations="GrowthStrategy:amrex.space1d" -o ${this_dir}/../src/ amrex.space1d
pybind11-stubgen --exit-code --enum-class-locations="GrowthStrategy:amrex.space2d" -o ${this_dir}/../src/ amrex.space2d
pybind11-stubgen --exit-code --enum-class-locations="GrowthStrategy:amrex.space3d" -o ${this_dir}/../src/ amrex.space3d
