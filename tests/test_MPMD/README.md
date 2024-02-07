# AMReX::MPMD in pyAMReX
**This directory is currently a work in progress.**

## Changes required during pyAMReX compilation
[Developer installation instructions of pyAMReX](https://pyamrex.readthedocs.io/en/latest/install/cmake.html#developers) were utilized for compilation.

**Important**

After running `cmake -S . -B build -DAMReX_SPACEDIM="1;2;3"` copy the `AMReX_MPMD.*` files in the current directory to `build/_deps/fetchedamrex-src/Src/Base/`.
This should be followed by `cmake --build build -j 4 --target pip_install` as mentioned in the instructions.

## Execution
The tests provided below can be executed as follows:

`mpirun -np 8 main.ccp_executable : -np 4 python main.py`

## Tests

### mpi4py_Comm_C_Comm
This test checks if [mpi4py.MPI.Comm](https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm) 
can be passed to `amr.initialize_when_MPMD` to run an MPI simulation in pyAMReX.

### MultiFab_cpp_py
This test checks if a `MultiFab` populated in the **cpp** code can be *sent* to an empty `MultiFab` in **python** script.

### MultiFab_py_cpp
This test checks if a `MultiFab` populated in the **python** script can be *sent* to an empty `MultiFab` in **cpp** code.

### lammps_tests/bench
This test checks if the pythonic version of [LAMMPS](https://docs.lammps.org/Python_launch.html#running-lammps-and-python-in-parallel-with-mpi)
can run **ONLY** on the resources provided to the python script.
