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

# we are in the source directory, .github/
this_dir=$(cd $(dirname $0) && pwd)

pybind11-stubgen --ignore-all-errors -o ${this_dir}/../src/amrex/ amrex.space1d
pybind11-stubgen --ignore-all-errors -o ${this_dir}/../src/amrex/ amrex.space2d
pybind11-stubgen --ignore-all-errors -o ${this_dir}/../src/amrex/ amrex.space3d
