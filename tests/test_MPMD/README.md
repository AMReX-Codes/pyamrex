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

### MultiFab_two_way
This test deals with a two component MultiFab. The first component is populated on **cpp** side and
 the second component is populated on the **python** side.
The **cpp** code sends the first component to the **python** script.
This is followed by a receive of the second component by the **cpp** code from the **python** script.

### lammps_tests/bench
This test checks if the pythonic version of [LAMMPS](https://docs.lammps.org/Python_launch.html#running-lammps-and-python-in-parallel-with-mpi)
can run **ONLY** on the resources provided to the python script.

### pytorch_tests/ddp_cpu
This test builds on top of **MultiFab_two_way** case.
In addition to *MultiFab* data transfer it creates a 
[torch.distributed process group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
on the resources allocated to the python script. The test then creates a PyTorch Tensor which is only populated on
rank 0 and broadcasts it to all involved python processes.

Please note that the order of applications matters for this case. Why? See **rank** Parameter description
[here](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)

Execution:

`mpirun -np 4 python main.py : -np 8 main.ccp_executable`

**This test was only performed on a single node and it involves communication of CPU based PyTorch Tensors.**
