#!/usr/bin/env bash
#
# Copyright 2021-2026 The AMReX Community
# License: BSD-3-Clause-LBNL
#
# Regenerate the committed native-extension stubs with nanobind. Configure the
# build with -DpyAMReX_BUILD_STUBS=ON first. The public package stubs are kept
# separately because they describe the pure-Python convenience layer.
set -eu -o pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
build_dir=${1:-"${repo_dir}/build"}
python=${PYTHON:-python3}
pycache_dir=${PYTHONPYCACHEPREFIX:-"${TMPDIR:-/tmp}/pyamrex-pycache"}

cmake --build "${build_dir}" --target pyAMReX_stubs

# A generated stub is executable Python syntax. Catch malformed docstrings or
# annotations before the stale-stub workflow compares files.
PYTHONPYCACHEPREFIX="${pycache_dir}" "${python}" -m py_compile \
    "${repo_dir}/src/amrex/space1d/amrex_1d_pybind/__init__.pyi" \
    "${repo_dir}/src/amrex/space2d/amrex_2d_pybind/__init__.pyi" \
    "${repo_dir}/src/amrex/space3d/amrex_3d_pybind/__init__.pyi"
