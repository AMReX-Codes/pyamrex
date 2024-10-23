.. _install-dependencies:

Dependencies
============

pyAMReX depends on the following popular third party software.
Please see installation instructions below.

- a mature `C++17 <https://en.wikipedia.org/wiki/C%2B%2B17>`__ compiler, e.g., GCC 8.4+, Clang 7, NVCC 11.0, MSVC 19.15 or newer
- `CMake 3.24.0+ <https://cmake.org>`__
- `Git 2.18+ <https://git-scm.com>`__
- `AMReX <https://amrex-codes.github.io>`__: we automatically download and compile a copy
- `pybind11 2.13.0+ <https://github.com/pybind/pybind11/>`__: we automatically download and compile a copy
- `Python 3.9+ <https://www.python.org>`__

  - `numpy 1.15+ <https://numpy.org>`__

Optional dependencies include:

- `MPI 3.0+ <https://www.mpi-forum.org/docs/>`__: for multi-node and/or multi-GPU execution
- for on-node accelerated compute *one of either*:

  - `OpenMP 3.1+ <https://www.openmp.org>`__: for threaded CPU execution or
  - `CUDA Toolkit 11.0+ (11.3+ recommended) <https://developer.nvidia.com/cuda-downloads>`__: for Nvidia GPU support (see `matching host-compilers <https://gist.github.com/ax3l/9489132>`_) or
  - `ROCm 5.2+ (5.5+ recommended) <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-rocm-installation-readme/>`__: for AMD GPU support
- `CCache <https://ccache.dev>`__: to speed up rebuilds (For CUDA support, needs version 3.7.9+ and 4.2+ is recommended)
- `Ninja <https://ninja-build.org>`__: for faster parallel compiles
- further `optional dependencies of AMReX <https://github.com/AMReX-Codes/amrex/>`__
- `Python dependencies <https://www.python.org>`__

  - `mpi4py 2.1+ <https://mpi4py.readthedocs.io>`__: for multi-node and/or multi-GPU execution
  - `cupy 11.2+ <https://github.com/cupy/cupy#installation>`__
  - `numba 0.56+ <https://numba.readthedocs.io/en/stable/user/installing.html>`__
  - `pandas 2+ <https://pandas.pydata.org>`__: for DataFrame support
  - `torch 1.12+ <https://pytorch.org/get-started/locally/>`__

For all other systems, we recommend to use a **package dependency manager**:
Pick *one* of the installation methods below to install all dependencies for pyAMReX development in a consistent manner.


Conda (Linux/macOS/Windows)
---------------------------

`Conda <https://conda.io>`__/`Mamba <https://mamba.readthedocs.io>`__ are cross-compatible, user-level package managers.

.. tip::

   We recommend to configure your conda to use the faster ``libmamba`` `dependency solver <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`__.

   .. code-block:: bash

      conda update -n base conda
      conda install -n base conda-libmamba-solver
      conda config --set solver libmamba

   We recommend to deactivate that conda self-activates its ``base`` environment.
   This `avoids interference with the system and other package managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__.

   .. code-block:: bash

      conda config --set auto_activate_base false

.. tab-set::

   .. tab-item:: With MPI (only Linux/macOS)

      .. code-block:: bash

         conda create -n pyamrex-cpu-mpich-dev -c conda-forge boost ccache cmake compilers git python numpy pandas scipy yt pkg-config make matplotlib mamba ninja mpich pip virtualenv
         conda activate pyamrex-cpu-mpich-dev

         # compile pyAMReX with -DAMReX_MPI=ON
         # for pip, use: export AMREX_MPI=ON

   .. tab-item:: Without MPI

      .. code-block:: bash

         conda create -n pyamrex-cpu-dev -c conda-forge boost ccache cmake compilers git python numpy pandas scipy yt pkg-config make matplotlib mamba ninja pip virtualenv
         conda activate pyamrex-cpu-dev

         # compile pyAMReX with -DAMReX_MPI=OFF
         # for pip, use: export AMREX_MPI=OFF

For OpenMP support, you will further need:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: bash

         conda install -c conda-forge libgomp

   .. tab-item:: macOS or Windows

      .. code-block:: bash

         conda install -c conda-forge llvm-openmp

For Nvidia CUDA GPU support, you will need to have `a recent CUDA driver installed <https://developer.nvidia.com/cuda-downloads>`__ or you can lower the CUDA version of `the Nvidia cuda package <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation>`__ and `conda-forge to match your drivers <https://docs.cupy.dev/en/stable/install.html#install-cupy-from-conda-forge>`__ and then add these packages:

.. code-block:: bash

   conda install -c nvidia -c conda-forge cuda cupy

More info for `CUDA-enabled ML packages <https://twitter.com/jeremyphoward/status/1697435241152127369>`__.


Spack (Linux/macOS)
-------------------

`Spack <https://spack.readthedocs.io>`__ is a user-level package manager.
It is primarily written for Linux, with slightly less support for macOS, and future support for Windows.

Please see `WarpX for now <https://warpx.readthedocs.io/en/latest/install/dependencies.html#spack-linux-macos>`__.


Brew (macOS/Linux)
------------------

`Homebrew (Brew) <https://brew.sh>`__ is a user-level package manager primarily for `Apple macOS <https://en.wikipedia.org/wiki/MacOS>`__, but also supports Linux.

.. code-block:: bash

   brew update
   brew tap openpmd/openpmd
   brew install ccache
   brew install cmake
   brew install git
   brew install libomp
   brew unlink gcc
   brew link --force libomp
   brew install open-mpi

If you also want to compile with PSATD in RZ, you need to manually install BLAS++ and LAPACK++:

.. code-block:: bash

   sudo mkdir -p /usr/local/bin/
   sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
   sudo chmod a+x /usr/local/bin/cmake-easyinstall

   cmake-easyinstall --prefix=/usr/local git+https://github.com/icl-utk-edu/blaspp.git \
       -Duse_openmp=OFF -Dbuild_tests=OFF -DCMAKE_VERBOSE_MAKEFILE=ON
   cmake-easyinstall --prefix=/usr/local git+https://github.com/icl-utk-edu/lapackpp.git \
       -Duse_cmake_find_lapack=ON -Dbuild_tests=OFF -DCMAKE_VERBOSE_MAKEFILE=ON

Compile pyAMReX with ``-DAMReX_MPI=ON``.
For ``pip``, use ``export AMREX_MPI=ON``.


APT (Debian/Ubuntu Linux)
-------------------------

The `Advanced Package Tool (APT) <https://en.wikipedia.org/wiki/APT_(software)>`__ is a system-level package manager on Debian-based Linux distributions, including Ubuntu.

.. tab-set::

   .. tab-item:: With MPI (only Linux/macOS)

      .. code-block:: bash

         sudo apt update
         sudo apt install build-essential ccache cmake g++ git libhdf5-openmpi-dev libopenmpi-dev pkg-config python3 python3-matplotlib python3-numpy python3-pandas python3-pip python3-scipy python3-venv

         # optional:
         # for CUDA, either install
         #   https://developer.nvidia.com/cuda-downloads (preferred)
         # or, if your Debian/Ubuntu is new enough, use the packages
         #   sudo apt install nvidia-cuda-dev libcub-dev

         # compile pyAMReX with -DAMReX_MPI=ON
         # for pip, use: export AMREX_MPI=ON

   .. tab-item:: Without MPI

      .. code-block:: bash

         sudo apt update
         sudo apt install build-essential ccache cmake g++ git libhdf5-dev pkg-config python3 python3-matplotlib python3-numpy python3-pandas python3-pip python3-scipy python3-venv

         # optional:
         # for CUDA, either install
         #   https://developer.nvidia.com/cuda-downloads (preferred)
         # or, if your Debian/Ubuntu is new enough, use the packages
         #   sudo apt install nvidia-cuda-dev libcub-dev

         # compile pyAMReX with -DAMReX_MPI=OFF
         # for pip, use: export AMREX_MPI=OFF
