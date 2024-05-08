AMReX-MPMD
==========

AMReX-MPMD utilizes the Multiple Program Multiple Data (MPMD) feature of MPI to provide cross-functionality for AMReX-based codes.

Test
====

The current test leverages the AMReX-MPMD capability to perform a data (MultiFab) transfer between *main.cpp* and *main.py* scripts.
The test is based on a two component MultiFab. The first component is populated in *main.cpp* before it is transferred to *main.py*.
Then *main.py* script fills the second component based on the obtained first component value.
Finally, the second component is transferred back from *main.py* to *main.cpp*.
**This test requires MPI and mpi4py**.

pyAMReX compile
---------------

.. code-block:: bash

   # find dependencies & configure
   # Include -DAMReX_GPU_BACKEND=CUDA for gpu version
   cmake -S . -B build -DAMReX_SPACEDIM="1;2;3" -DAMReX_MPI=ON -DpyAMReX_amrex_src=/path/to/amrex

   # compile & install, here we use four threads
   cmake --build build -j 4 --target pip_install

main.cpp compile
----------------

.. code-block:: bash

   # Include USE_CUDA=TRUE for gpu version
   make AMREX_HOME=/path/to/amrex -j 4

Run
---

Please note that MPI ranks attributed to each application/code need to be continuous, i.e., MPI ranks 0-7 are for *main.cpp* and 8-11 are for *main.py*.
This may be default behaviour on several systems.

.. code-block:: bash

   mpirun -np 8 ./main3d.gnu.DEBUG.MPI.ex : -np 4 python main.py
