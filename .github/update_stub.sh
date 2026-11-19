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
pybind11-stubgen --exit-code --enum-class-locations="GrowthStrategy:amrex.space2d" --enum-class-locations="EBSupport:amrex.space2d" -o ${this_dir}/../src/ amrex.space2d
pybind11-stubgen --exit-code --enum-class-locations="GrowthStrategy:amrex.space3d" --enum-class-locations="EBSupport:amrex.space3d" -o ${this_dir}/../src/ amrex.space3d

# Fix circular default argumetn for
#   strategy: GrowthStrategy = amrex.space3d.GrowthStrategy.Poisson
sed -i 's/amrex.space1d.GrowthStrategy/GrowthStrategy/g' src/amrex/space1d/amrex_1d_pybind/__init__.pyi
sed -i 's/amrex.space2d.GrowthStrategy/GrowthStrategy/g' src/amrex/space2d/amrex_2d_pybind/__init__.pyi
sed -i 's/amrex.space3d.GrowthStrategy/GrowthStrategy/g' src/amrex/space3d/amrex_3d_pybind/__init__.pyi

sed -i 's/= GrowthStrategy.Poisson/= "GrowthStrategy.Poisson"/g' src/amrex/space1d/amrex_1d_pybind/__init__.pyi
sed -i 's/= GrowthStrategy.Poisson/= "GrowthStrategy.Poisson"/g' src/amrex/space2d/amrex_2d_pybind/__init__.pyi
sed -i 's/= GrowthStrategy.Poisson/= "GrowthStrategy.Poisson"/g' src/amrex/space3d/amrex_3d_pybind/__init__.pyi

# Fix circular default argument for
#   support: EBSupport = amrex.space3d.EBSupport.full
sed -i 's/amrex.space2d.EBSupport/EBSupport/g' src/amrex/space2d/amrex_2d_pybind/__init__.pyi
sed -i 's/amrex.space3d.EBSupport/EBSupport/g' src/amrex/space3d/amrex_3d_pybind/__init__.pyi

sed -i 's/= EBSupport.full/= "EBSupport.full"/g' src/amrex/space2d/amrex_2d_pybind/__init__.pyi
sed -i 's/= EBSupport.full/= "EBSupport.full"/g' src/amrex/space3d/amrex_3d_pybind/__init__.pyi
