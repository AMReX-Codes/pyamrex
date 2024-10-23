# pyAMReX

[![Python3][api-py3]](https://www.python.org/)
![Python3 API: Beta][dev-beta]
[![Documentation Status](https://readthedocs.org/projects/pyamrex/badge/?version=latest)](https://pyamrex.readthedocs.io)
[![Discussions](https://img.shields.io/badge/chat-discussions-turquoise.svg)](https://github.com/AMReX-Codes/pyamrex/discussions)  
![Linux](https://github.com/AMReX-Codes/pyamrex/actions/workflows/ubuntu.yml/badge.svg?branch=development)
![macOS](https://github.com/AMReX-Codes/pyamrex/actions/workflows/macos.yml/badge.svg?branch=development)
![Windows](https://github.com/AMReX-Codes/pyamrex/actions/workflows/windows.yml/badge.svg?branch=development)  
[![License pyAMReX](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-blue.svg)](https://spdx.org/licenses/BSD-3-Clause-LBNL.html)
[![DOI (source)](https://img.shields.io/badge/DOI%20(source)-10.5281/zenodo.8408733-blue.svg)](https://doi.org/10.5281/zenodo.8408733)

[api-py3]: https://img.shields.io/badge/language-Python3-yellowgreen "Python3 API"
[dev-beta]: https://img.shields.io/badge/phase-beta-yellowgreen "Status: Beta"

The Python binding pyAMReX bridges the compute in AMReX block-structured codes and data science:
it provides zero-copy application GPU data access for AI/ML, in situ analysis, application coupling and enables rapid, massively parallel prototyping.
pyAMReX enhances the [Block-Structured AMR Software Framework AMReX](https://amrex-codes.github.io) and its applications.

## Users

pyAMReX [can be installed](https://pyamrex.readthedocs.io/en/latest/install/users.html) with package managers or from source.


### Usage

Please see the [manual](https://pyamrex.readthedocs.io/en/latest/usage/how_to_run.html) and our [test cases](https://github.com/AMReX-Codes/pyamrex/tree/development/tests) for detailed examples.

Use AMReX objects and APIs from Python:
```python
import amrex.space3d as amr

small_end = amr.IntVect()
big_end = amr.IntVect(2, 3, 4)

b = amr.Box(small_end, big_end)
print(b)

# ...
```

## Developers

If you are new to CMake, [this short tutorial](https://hsf-training.github.io/hsf-training-cmake-webpage/) from the HEP Software foundation is the perfect place to get started with it.

If you just want to use CMake to build the project, jump into sections *1. Introduction*, *2. Building with CMake* and *9. Finding Packages*.

### Dependencies

pyAMReX depends on the following popular third party software.

- a mature [C++17](https://en.wikipedia.org/wiki/C%2B%2B17) compiler, e.g., GCC 8, Clang 7, NVCC 11.0, MSVC 19.15 or newer
- [CMake 3.24.0+](https://cmake.org)
- [AMReX *development*](https://amrex-codes.github.io): we automatically download and compile a copy of AMReX
- [pybind11](https://github.com/pybind/pybind11/) 2.13.0+: we automatically download and compile a copy of pybind11 ([new BSD](https://github.com/pybind/pybind11/blob/master/LICENSE))
  - [Python](https://python.org) 3.9+
  - [Numpy](https://numpy.org) 1.15+

Optional dependencies include:
- [mpi4py](https://mpi4py.readthedocs.io) 2.1+: for multi-node and/or multi-GPU execution
- [CCache](https://ccache.dev): to speed up rebuilds (for CUDA support, needs 3.7.9+ and 4.2+ is recommended)
- further [optional dependencies of AMReX](https://github.com/AMReX-Codes/amrex/)
- [pandas](https://pandas.pydata.org/) 2+: for DataFrame support
- [pytest](https://docs.pytest.org/en/stable/) 6.2+: for running unit tests

Optional CUDA-capable dependencies for tests include:
- [cupy](https://github.com/cupy/cupy#installation) 11.2+
- [numba](https://numba.readthedocs.io/en/stable/user/installing.html) 0.56+
- [torch](https://pytorch.org/get-started/locally/) 1.12+

### Install Dependencies

macOS/Linux:
```bash
spack env activate -d .
# optional:
# spack add cuda
spack install
```
(in new terminals, re-activate the environment with `spack env activate -d .` again)

or macOS/Linux:
```bash
brew update
brew install ccache cmake libomp mpi4py numpy open-mpi python
```

Now, `cmake --version` should be at version 3.24.0 or newer.

Or go:
```bash
python3 -m pip install -U pip
python3 -m pip install -U build packaging setuptools wheel
python3 -m pip install -U cmake
```

If you wish to run unit tests, then please install `pytest`

```bash
python3 -m pip install -U pytest
```

Some of our tests depend on optional third-party modules (e.g., `pandas`, `cupy`, `numba`, and/or `pytorch`).
If these are not installed then their tests will be skipped.


### Configure your compiler

For example, using the Clang compiler:
```bash
export CC=$(which clang)
export CXX=$(which clang++)
```

If you also want to select a CUDA compiler:
```bash
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which clang++)
```


### Build

From the base of the pyAMReX source directory, execute:
```bash
# optional controls (example):
#export AMREX_SPACEDIM=3
#export AMREX_MPI=ON
#export AMREX_OMP=ON
#export AMREX_GPU_BACKEND=CUDA
#export AMREX_SRC=$PWD/../amrex
#export CMAKE_BUILD_PARALLEL_LEVEL=8

python3 -m pip install -U -r requirements.txt
python3 -m pip install -v --force-reinstall --no-deps .
```

If you are iterating on builds, it will faster to rely on ``ccache`` and to let CMake call the ``pip`` install logic:
```bash
cmake -S . -B build -DAMReX_SPACEDIM="1;2;3"
cmake --build build --target pip_install -j 8
```

### Test

After successful installation, you can run the unit tests (assuming `pytest` is
installed). If `AMREX_MPI=ON`, then please prepend the following commands with `mpiexec -np <NUM_PROCS>`

```bash
# Run all tests
python3 -m pytest tests/

# Run tests from a single file
python3 -m pytest tests/test_intvect.py

# Run a single test (useful during debugging)
python3 -m pytest tests/test_intvect.py::test_iv_conversions

# Run all tests, do not capture "print" output and be verbose
python3 -m pytest -s -vvvv tests/
```

### Build Options

If you are using the pip-driven install, selected [AMReX CMake options](https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#building-with-cmake) can be controlled with environment variables:

| Environment Variable         | Default & Values                           | Description                                                  |
|------------------------------|--------------------------------------------|--------------------------------------------------------------|
| `AMREX_OMP`                  | ON/**OFF**                                 | Enable OpenMP                                                |
| `AMREX_GPU_BACKEND`          | **NONE**/SYCL/CUDA/HIP                     | On-node, accelerated GPU backend                             |
| `AMREX_MPI`                  | ON/**OFF**                                 | Enable MPI                                                   |
| `AMREX_PRECISION`            | SINGLE/**DOUBLE**                          | Precision of AMReX Real type                                 |
| `AMREX_SPACEDIM`             | "1;2;3"                                    | Dimension(s) of AMReX as a ``;``-separated list              |
| `AMREX_BUILD_SHARED_LIBS`    | ON/**OFF**                                 | Build the core AMReX library as shared library               |
| `AMREX_SRC`                  | *None*                                     | Absolute path to AMReX source directory (preferred if set)   |
| `AMREX_REPO`                 | `https://github.com/AMReX-Codes/amrex.git` | Repository URI to pull and build AMReX from                  |
| `AMREX_BRANCH`               | `development`                              | Repository branch for `AMREX_REPO`                           |
| `AMREX_INTERNAL`             | **ON**/OFF                                 | Needs a pre-installed AMReX library if set to `OFF`          |
| `PYBIND11_INTERNAL`          | **ON**/OFF                                 | Needs a pre-installed pybind11 library if set to `OFF`       |
| `CMAKE_BUILD_PARALLEL_LEVEL` | 2                                          | Number of parallel build threads                             |
| `PYAMREX_LIBDIR`             | *None*                                     | If set, search for pre-built a pyAMReX library               |
| `PYAMREX_CCACHE`             | **ON**/OFF                                 | Search and use CCache to speed up rebuilds                   |
| `PYAMREX_IPO`                | **ON**/OFF                                 | Compile with interprocedural/link optimization (IPO/LTO)     |
| `PY_PIP_OPTIONS`             | `-v`                                       | Additional options for ``pip``, e.g., ``-vvv;-q``            |
| `PY_PIP_INSTALL_OPTIONS`     | *None*                                     | Additional options for ``pip install``, e.g., ``--user;-q``  |

Furthermore, pyAMReX adds a few selected CMake build options:

| CMake Option                 | Default & Values                           | Description                                                   |
|------------------------------|--------------------------------------------|---------------------------------------------------------------|
| `AMReX_SPACEDIM`             | **3**, use `"1;2;3"` for all               | Dimension(s) of AMReX as a ``;``-separated list               |
| `pyAMReX_CCACHE`             | **ON**/OFF                                 | Search and use CCache to speed up rebuilds                    |
| `pyAMReX_IPO`                | **ON**/OFF                                 | Compile with interprocedural/link optimization (IPO/LTO)      |
| `pyAMReX_INSTALL`            | **ON**/OFF                                 | Enable install targets for pyAMReX                            |
| `pyAMReX_amrex_src`          | *None*                                     | Absolute path to AMReX source directory (preferred if set)    |
| `pyAMReX_amrex_internal`     | **ON**/OFF                                 | Needs a pre-installed AMReX library if set to `OFF`           |
| `pyAMReX_amrex_repo`         | `https://github.com/AMReX-Codes/amrex.git` | Repository URI to pull and build AMReX from                   |
| `pyAMReX_amrex_branch`       | `development`                              | Repository branch for `pyAMReX_amrex_repo`                    |
| `pyAMReX_pybind11_src`       | *None*                                     | Absolute path to pybind11 source directory (preferred if set) |
| `pyAMReX_pybind11_internal`  | **ON**/OFF                                 | Needs a pre-installed pybind11 library if set to `OFF`        |
| `pyAMReX_pybind11_repo`      | `https://github.com/pybind/pybind11.git`   | Repository URI to pull and build pybind11 from                |
| `pyAMReX_pybind11_branch`    | `v2.13.6`                                  | Repository branch for `pyAMReX_pybind11_repo`                 |
| `Python_EXECUTABLE`          | (newest found)                             | Path to Python executable                                     |

As one example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in `$HOME/src/amrex`, then `export AMREX_SRC=$HOME/src/amrex`.

Or as a one-liner, assuming your AMReX source directory is located in `../amrex`:
```bash
AMREX_SRC=$PWD/../amrex python3 -m pip install -v --force-reinstall .
```
Note that you need to use absolute paths for external source trees, because pip builds in a temporary directory.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to `https://github.com/WeiqunZhang/amrex/` in a branch `new-feature` then

```bash
unset AMREX_SRC  # preferred if set
AMREX_REPO=https://github.com/WeiqunZhang/amrex.git AMREX_BRANCH=new-feature python3 -m pip install -v --force-reinstall .
```

You can speed up the install further if you pre-install AMReX, e.g. with a package manager.
Set `AMREX_INTERNAL=OFF` and add installation prefix of AMReX to the environment variable [CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html).
Please see the [short CMake tutorial that we linked above](#Developers) if this sounds new to you.


## Acknowledgements

This work was supported by the Laboratory Directed Research and Development Program of Lawrence Berkeley National Laboratory under U.S. Department of Energy Contract No. DE-AC02-05CH11231.


## Copyright Notice

pyAMReX Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory, National Renewable Energy
Laboratory Alliance for Sustainable Energy, LLC and Lawrence Livermore
National Security, LLC (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

Please see the full license agreement in [LICENSE](LICENSE).  
Please see the notices in [NOTICE](NOTICE).  
The SPDX license identifier is `BSD-3-Clause-LBNL`.
