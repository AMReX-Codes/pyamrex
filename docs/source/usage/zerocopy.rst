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

GPU zero-copy read and write access on NVIDIA (CUDA) and AMD (ROCm) GPUs.

Call ``.to_cupy()`` on data objects of pyAMReX.
See the optional arguments of this API.

Writing to the created CuPy array will also modify the underlying AMReX memory.


GPU: dpnp
---------

GPU zero-copy read and write access on Intel (SYCL) GPUs, exchanged via `DLPack <https://dmlc.github.io/dlpack/latest/>`__.

Call ``.to_dpnp()`` on data objects of pyAMReX.
See the optional arguments of this API.

Writing to the created dpnp array will also modify the underlying AMReX memory.


CPU/GPU Agnostic Code: NumPy/CuPy/dpnp
--------------------------------------

The previous examples can be written in CPU/GPU agnostics manner.
Either using NumPy (``np``), CuPy (``cp``) or dpnp (``dp``), we provide a `common short-hand abbreviation <https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code>`__ named ``xp`` .

Call ``.to_xp()`` on data objects of pyAMReX.
It returns NumPy arrays on CPU builds, CuPy arrays on CUDA/ROCm builds and dpnp arrays on SYCL builds.
See the optional arguments of this API.

Writing to the created NumPy/CuPy/dpnp array will also modify the underlying AMReX memory.


GPU: numba
----------

GPU zero-copy read and write access.

After ``from numba import cuda``, create a zero-copy tensor on a GPU array via ``marr_numba = cuda.as_cuda_array(marr)``.

Writing to the created numba array will also modify the underlying AMReX memory.


AI/ML: pyTorch
--------------

CPU and GPU zero-copy read and write access.

Create a zero-copy tensor on a GPU array via ``torch.as_tensor(amrex_array_here, device="cuda")`` or ``torch.from_dlpack(amrex_array_here)``.

Writing to the created PyTorch tensor will also modify the underlying AMReX memory.


Everything Else: DLPack
-----------------------

Frameworks not covered above can exchange data with pyAMReX through the standardized `DLPack <https://dmlc.github.io/dlpack/latest/python_spec.html>`__ protocol, on CPU as well as on CUDA, ROCm and SYCL GPUs.

Data objects of pyAMReX implement ``__dlpack__`` and ``__dlpack_device__`` and can be consumed by, e.g., ``numpy.from_dlpack``, ``cupy.from_dlpack``, ``dpnp.from_dlpack``, ``torch.from_dlpack`` or ``jax.dlpack``.
Both DLPack 1.x ("versioned") and legacy consumers are supported, including device-to-host transfers via ``from_dlpack(..., device="cpu")``.

An exception is the particle ``ArrayOfStructs``: DLPack cannot describe its record (struct) element type.
Use its structured ``__array_interface__``/``__cuda_array_interface__`` or the per-component particle struct-of-arrays data instead.
