# pyAMReX

![Supported Platforms][api-platforms]
[![License AMReX](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-blue.svg)](https://spdx.org/licenses/BSD-3-Clause-LBNL.html)

[api-platforms]: https://img.shields.io/badge/platforms-linux%20|%20osx%20|%20win-blue "Supported Platforms"


pyAMReX is part of AMReX.

Due to its **highly experimental** nature, we develop it currently in a separate respository.

We will add further information here once first development versions are ready for testing.

## Dependencies

pyAMReX depends on the following popular third party software.

- a mature [C++14](https://en.wikipedia.org/wiki/C%2B%2B14) compiler: e.g. g++ 5.0+, clang 5.0+, VS 2017+
- [CMake 3.18.0+](https://cmake.org)
- [AMReX *development*](https://amrex-codes.github.io): we automatically download and compile a copy of AMReX
- [pybind11](https://github.com/pybind/pybind11/) 2.6.2+: we automatically download and compile a copy of pybind11 ([new BSD](https://github.com/pybind/pybind11/blob/master/LICENSE))
  - [Python](https://python.org) 3.6+
  - [Numpy](https://numpy.org) 1.15+


Optional dependencies include:
- [mpi4py](https://www.openmp.org) 2.1+: for multi-node and/or multi-GPU execution
- [CCache](https://ccache.dev): to speed up rebuilds (needs 3.7.9+ for CUDA)
- further [optional dependencies of AMReX](https://github.com/AMReX-Codes/amrex/)

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
brew install ccache cmake fftw libomp mpi4py numpy open-mpi python
```

Now, `cmake --version` should be at version 3.18.0 or newer.

Or go:
```bash
# optional:                                    --user
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U cmake
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

## Build & Test

From the base of the pyAMReX source directory, execute:
```bash
# TODO: implement
#export AMREX_MPI=ON
#export AMREX_COMPUTE=...
#export AMREX_SRC=...

# optional:                --force-reinstall --user
python3 -m pip install -v .
```

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
