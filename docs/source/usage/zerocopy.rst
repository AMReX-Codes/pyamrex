.. _usage-zerocopy:

Zero-Copy
=========

The Python binding pyAMReX bridges the compute in AMReX block-structured codes and data science.
As such, it includes zero-copy GPU data access for AI/ML, in situ analysis, application coupling by implementing :ref:`standardized data interfaces <developers-implementation>`.


CPU: NumPy
----------

zero-copy read and write access.
CPU as well as managed memory CPU/GPU.

Call ``.to_numpy()`` on data objects of pyAMReX.
See the optional arguments of this API.

Writing to the created NumPy array will also modify the underlying AMReX memory.


GPU: CuPy
---------

GPU zero-copy read and write access.

Call ``.to_cupy()`` on data objects of pyAMReX.
See the optional arguments of this API.

Writing to the created CuPy array will also modify the underlying AMReX memory.


CPU/GPU Agnostic Code: NumPy/CuPy
---------------------------------

The previous examples can be written in CPU/GPU agnostics manner.
Either using NumPy (``np``) or CuPy (``cp``), we provide a `common short-hand abbreviation <https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code>`__ named ``xp`` .

Call ``.to_xp()`` on data objects of pyAMReX.
See the optional arguments of this API.

Writing to the created NumPy/CuPy array will also modify the underlying AMReX memory.


GPU: numba
----------

GPU zero-copy read and write access.

After ``from numba import cuda``, create a zero-copy tensor on a GPU array via ``marr_numba = cuda.as_cuda_array(marr)``.

Writing to the created numba array will also modify the underlying AMReX memory.


AI/ML: pyTorch
--------------

CPU and GPU zero-copy read and write access.

Create a zero-copy tensor on a GPU array via ``torch.as_tensor(amrex_array_here, device="cuda")``.

Writing to the created PyTorch tensor will also modify the underlying AMReX memory.
