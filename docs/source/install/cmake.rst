.. _install-developers:
.. _building-cmake:
.. _building-cmake-intro:

Developers
==========

`CMake <https://cmake.org>`_ is our primary build system.
If you are new to CMake, `this short tutorial <https://hsf-training.github.io/hsf-training-cmake-webpage/>`_ from the HEP Software foundation is the perfect place to get started.
If you just want to use CMake to build the project, jump into sections `1. Introduction <https://hsf-training.github.io/hsf-training-cmake-webpage/01-intro/index.html>`__, `2. Building with CMake <https://hsf-training.github.io/hsf-training-cmake-webpage/02-building/index.html>`__ and `9. Finding Packages <https://hsf-training.github.io/hsf-training-cmake-webpage/09-findingpackages/index.html>`__.

Dependencies
------------

Before you start, you will need a copy of the pyAMReX source code:

.. code-block:: bash

   git clone https://github.com/AMReX-Codes/pyamrex.git $HOME/src/pyamrex
   cd $HOME/src/pyamrex

pyAMReX depends on popular third party software.
On your development machine, :ref:`follow the instructions here <install-dependencies>`.

.. toctree::
   :hidden:

   dependencies

.. note::

   Preparation: make sure you work with up-to-date Python tooling.

   .. code-block:: bash

      python3 -m pip install -U pip
      python3 -m pip install -U build packaging setuptools[core] wheel pytest
      python3 -m pip install -U -r requirements.txt


Compile
-------

From the base of the pyAMReX source directory, execute:

.. code-block:: bash

   # find dependencies & configure
   cmake -S . -B build -DAMReX_SPACEDIM="1;2;3"

   # compile & install, here we use four threads
   cmake --build build -j 4 --target pip_install

That's all!

You can inspect and modify build options after running ``cmake -S . -B build`` with either

.. code-block:: bash

   ccmake build

or by adding arguments with ``-D<OPTION>=<VALUE>`` to the first CMake call, e.g.:

.. code-block:: bash

   cmake -S . -B build -DAMReX_GPU_BACKEND=CUDA -DAMReX_MPI=OFF -DAMReX_SPACEDIM="1;2;3"

**That's it!**

Developers could now change the pyAMReX source code and then call the install lines again to refresh the installation.

.. tip::

   If you do *not* develop with :ref:`a user-level package manager <install-dependencies>`, e.g., because you rely on a HPC system's environment modules, then consider to set up a virtual environment via `Python venv <https://docs.python.org/3/library/venv.html>`__.


Build Options
-------------

=============================== ============================================ ===========================================================
CMake Option                    Default & Values                             Description
=============================== ============================================ ===========================================================
``BUILD_TESTING``               **ON**/OFF                                   Build tests
``CMAKE_BUILD_TYPE``            RelWithDebInfo/**Release**/Debug             Type of build, symbols & optimizations
``CMAKE_INSTALL_PREFIX``        system-dependent path                        Install path prefix
``CMAKE_VERBOSE_MAKEFILE``      ON/**OFF**                                   Print all compiler commands to the terminal during build
``AMReX_OMP``                   ON/**OFF**                                   Enable OpenMP
``AMReX_GPU_BACKEND``           **NONE**/SYCL/CUDA/HIP                       On-node, accelerated GPU backend
``AMReX_MPI``                   **ON**/OFF                                   Enable MPI
``AMReX_PRECISION``             SINGLE/**DOUBLE**                            Precision of AMReX Real type
``AMReX_SPACEDIM``              ``3``                                        Dimension(s) of AMReX as a ``;``-separated list
``AMReX_BUILD_SHARED_LIBS``     ON/**OFF**                                   Build AMReX library as shared (required for app extensions)
``pyAMReX_INSTALL``             **ON**/OFF                                   Enable install targets for pyAMReX
``PY_PIP_OPTIONS``              ``-v``                                       Additional options for ``pip``, e.g., ``-vvv;-q``
``PY_PIP_INSTALL_OPTIONS``      *None*                                       Additional options for ``pip install``, e.g., ``--user;-q``
``Python_EXECUTABLE``           (newest found)                               Path to Python executable
=============================== ============================================ ===========================================================

pyAMReX can be configured in further detail with options from AMReX, which are documented in the AMReX manual:

* `general AMReX build options <https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options>`__
* `GPU-specific options <https://amrex-codes.github.io/amrex/docs_html/GPU.html#building-gpu-support>`__.

**Developers** might be interested in additional options that control dependencies of pyAMReX.
By default, the most important dependencies of pyAMReX are automatically downloaded for convenience:

============================= ============================================== ===========================================================
CMake Option                  Default & Values                               Description
============================= ============================================== ===========================================================
``BUILD_SHARED_LIBS``         ON/**OFF**                                     Build shared libraries for dependencies
``pyAMReX_CCACHE``            **ON**/OFF                                     Search and use CCache to speed up rebuilds.
``pyAMReX_IPO``               **ON**/OFF                                     Compile with interprocedural/link optimization (IPO/LTO)
``pyAMReX_amrex_src``         *None*                                         Path to AMReX source directory (preferred if set)
``pyAMReX_amrex_repo``        ``https://github.com/AMReX-Codes/amrex.git``   Repository URI to pull and build AMReX from
``pyAMReX_amrex_branch``      *we set and maintain a compatible commit*      Repository branch for ``pyAMReX_amrex_repo``
``pyAMReX_amrex_internal``    **ON**/OFF                                     Needs a pre-installed AMReX library if set to ``OFF``
``pyAMReX_pybind11_src``      *None*                                         Path to pybind11 source directory (preferred if set)
``pyAMReX_pybind11_repo``     ``https://github.com/pybind/pybind11.git``     Repository URI to pull and build pybind11 from
``pyAMReX_pybind11_branch``   *we set and maintain a compatible commit*      Repository branch for ``pyAMReX_pybind11_repo``
``pyAMReX_pybind11_internal`` **ON**/OFF                                     Needs a pre-installed pybind11 module if set to ``OFF``
============================= ============================================== ===========================================================

For example, one can also build against a local AMReX copy.
Assuming AMReX' source is located in ``$HOME/src/amrex``, add the ``cmake`` argument ``-DpyAMReX_amrex_src=$HOME/src/amrex``.
Relative paths are also supported, e.g. ``-DpyAMReX_amrex_src=../amrex``.

Or build against an AMReX feature branch of a colleague.
Assuming your colleague pushed AMReX to ``https://github.com/WeiqunZhang/amrex/`` in a branch ``new-feature`` then pass to ``cmake`` the arguments: ``-DpyAMReX_amrex_repo=https://github.com/WeiqunZhang/amrex.git -DpyAMReX_amrex_branch=new-feature``.

You can speed up the install further if you pre-install these dependencies, e.g. with a package manager.
Set ``-DpyAMReX_<dependency-name>_internal=OFF`` and add installation prefix of the dependency to the environment variable `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html>`__.
Please see the :ref:`introduction to CMake <building-cmake-intro>` if this sounds new to you.

If you re-compile often, consider installing the `Ninja <https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages>`__ build system.
Pass ``-G Ninja`` to the CMake configuration call to speed up parallel compiles.


Configure Your Compiler
-----------------------

If you don't want to use your default compiler, you can set the following environment variables.
For example, using a Clang/LLVM:

.. code-block:: bash

   export CC=$(which clang)
   export CXX=$(which clang++)

If you also want to select a CUDA compiler:

.. code-block:: bash

   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=$(which clang++)

.. note::

   Please clean your build directory with ``rm -rf build/`` after changing the compiler.
   Now call ``cmake -S . -B build`` (+ further options) again to re-initialize the build configuration.


Run
---

We provide the public imports ``amrex.space1d``, ``amrex.space2d`` and ``amrex.space3d``, mirroring the compile-time option ``AMReX_SPACEDIM``.

Due to limitations in AMReX, currently, only one of the imports can be used at a time in the same Python process.
For example:

.. code-block:: python

   import amrex.space3d as amr

A 1D or 2D AMReX run needs its own Python process.
Another dimensionality *cannot be imported into the same Python process* after choosing a specific dimensionality for import.
