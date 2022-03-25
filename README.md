# pyAMReX

[![Python3][api-py3]](https://www.python.org/) ![Python3 API: Pre-Alpha][dev-pre-alpha]
[![License AMReX](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-blue.svg)](https://spdx.org/licenses/BSD-3-Clause-LBNL.html)  
![linux](https://github.com/AMReX-Codes/pyamrex/workflows/linux/badge.svg?branch=development)
![macos](https://github.com/AMReX-Codes/pyamrex/workflows/macos/badge.svg?branch=development)
![windows](https://github.com/AMReX-Codes/pyamrex/workflows/windows/badge.svg?branch=development)

[api-py3]: https://img.shields.io/badge/language-Python3-yellowgreen "Python3 API"
[dev-pre-alpha]: https://img.shields.io/badge/phase-pre--alpha-yellowgreen "Status: Pre-Alpha"

pyAMReX is part of AMReX.

Due to its **highly experimental** nature, we develop it currently in a separate respository.

We will add further information here once first development versions are ready for testing.

## Users

*to do*

- pip/pypa
- conda-forge
- spack
- brew
- ...

### Usage

*to do*

```python
import amrex

small_end = amrex.Int_Vect()
big_end = amrex.Int_Vect(2, 3, 4)

b = amrex.Box(small_end, big_end)
print(b)

# ...
```

## Developers

If you are new to CMake, [this short tutorial](https://hsf-training.github.io/hsf-training-cmake-webpage/) from the HEP Software foundation is the perfect place to get started with it.

If you just want to use CMake to build the project, jump into sections *1. Introduction*, *2. Building with CMake* and *9. Finding Packages*.

### Dependencies

pyAMReX depends on the following popular third party software.

- a mature [C++17](https://en.wikipedia.org/wiki/C%2B%2B17) compiler, e.g., GCC 7, Clang 7, NVCCC 11.0, MSVC 19.15 or newer
- [CMake 3.18.0+](https://cmake.org)
- [AMReX *development*](https://amrex-codes.github.io): we automatically download and compile a copy of AMReX
- [pybind11](https://github.com/pybind/pybind11/) 2.9.1+: we automatically download and compile a copy of pybind11 ([new BSD](https://github.com/pybind/pybind11/blob/master/LICENSE))
  - [Python](https://python.org) 3.6+
  - [Numpy](https://numpy.org) 1.15+

Optional dependencies include:
- [mpi4py](https://www.openmp.org) 2.1+: for multi-node and/or multi-GPU execution
- [CCache](https://ccache.dev): to speed up rebuilds (needs 3.7.9+ for CUDA)
- further [optional dependencies of AMReX](https://github.com/AMReX-Codes/amrex/)
- [pytest](https://docs.pytest.org/en/stable/) 6.2+: for running unit tests

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

Now, `cmake --version` should be at version 3.18.0 or newer.

Or go:
```bash
# optional:                                    --user
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U cmake
```

If you wish to run unit tests, then please install `pytest`

```bash
python3 -m pip install -U pytest
```

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
cmake -S . -B build
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
```

### Build Options

If you are using the pip-driven install, selected [AMReX CMake options](https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#building-with-cmake) can be controlled with environment variables:

| Environment Variable         | Default & Values                           | Description                                                  |
|------------------------------|--------------------------------------------|--------------------------------------------------------------|
| `AMREX_OMP`                  | ON/**OFF**                                 | Enable OpenMP                                                |
| `AMREX_GPU_BACKEND`          | **NONE**/SYCL/CUDA/HIP                     | On-node, accelerated GPU backend                             |
| `AMREX_MPI`                  | ON/**OFF**                                 | Enable MPI                                                   |
| `AMREX_PRECISION`            | SINGLE/**DOUBLE**                          | Precision of AMReX Real type                                 |
| `AMREX_SPACEDIM`             | 1/2/**3**                                  | Dimension of AMReX                                           |
| `AMREX_BUILD_SHARED_LIBS`    | ON/**OFF**                                 | Build the core AMReX library as shared library               |
| `AMREX_SRC`                  | *None*                                     | Absolute path to AMReX source directory (preferred if set)   |
| `AMREX_REPO`                 | `https://github.com/AMReX-Codes/amrex.git` | Repository URI to pull and build AMReX from                  |
| `AMREX_BRANCH`               | `development`                              | Repository branch for `AMREX_REPO`                           |
| `AMREX_INTERNAL`             | **ON**/OFF                                 | Needs a pre-installed AMReX library if set to `OFF`          |
| `CMAKE_BUILD_PARALLEL_LEVEL` | 2                                          | Number of parallel build threads                             |
| `PYAMREX_LIBDIR`             | *None*                                     | If set, search for pre-built a pyAMReX library               |
| `PYINSTALLOPTIONS`           | *None*                                     | Additional options for ``pip install``, e.g., ``-v --user``  |

For example, one can also build against a local AMReX copy.
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


## License

pyAMReX Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Innovation & Partnerships Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.

License for pyamrex can be found at [LICENSE](LICENSE).
