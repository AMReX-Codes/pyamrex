.. _usage-compute:

Compute
=======

With zero-copy read and write access to data structures, this section presents how to compute in pyAMReX.

Since the pyAMReX data containers are wrappers to the C++ AMReX objects, it is worth reading:

* `AMReX Basics <https://amrex-codes.github.io/amrex/docs_html/Basics_Chapter.html>`__ and
* `AMReX parallelization strategy for MPI+X (e.g, GPUs, CPUs) <https://amrex-codes.github.io/amrex/docs_html/GPU.html>`__.

As a very short, simplified overview, this narrows down to:

* AMReX decomposes **index space** into **rectangular, block-structured, regular grids**,
* block are often intentionally slightly **over-decomposed**, there is >= one block per compute unit (CPU core or GPU),
* **particles** are chunked/tiled and usually decomposed like the field blocks,
* **refinement levels** are represented as (potentially sparse) levels of blocks.

Computations are thus performed (mostly) on whole blocks, which enables to use compute advanced acceleration techniques on CPUs or GPUs.


.. _usage-compute-fields:

Fields
------

The most common data structure to interact with is a `MultiFab <https://amrex-codes.github.io/amrex/docs_html/Basics.html#fabarray-multifab-and-imultifab>`__, which is a collection of boxes with associated field data.
The field data can have more than one component (in the slowest varying index), but all components have the same `staggering/centering <https://amrex-codes.github.io/amrex/docs_html/Basics.html#box>`__.

This is how to iterate and potentially compute for all blocks assigned to a local process in pyAMReX:

.. tab-set::

   .. tab-item:: Simple

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Simple START
         :end-before: # Manual: Compute Mfab Simple END

   .. tab-item:: Detailed

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Detailed START
         :end-before: # Manual: Compute Mfab Detailed END

   .. tab-item:: Global

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Global START
         :end-before: # Manual: Compute Mfab Global END


For a complete physics example that uses CPU/GPU agnostic Python code for computation on fields, see:

* `Heat Equation example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/HeatEquation/Source/main.py>`__

For many small CPU and GPU examples on how to compute on fields, see the following test cases:

* `MultiFab example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/MultiFab/main.py>`__

* .. dropdown:: Examples in ``test_multifab.py``

     .. literalinclude:: ../../../tests/test_multifab.py
        :language: python3
        :caption: This files is in ``tests/test_multifab.py``.


.. _usage-compute-portable:

Portable Kernels, OpenMP and Threading
--------------------------------------

A frequent question is how OpenMP parallelizes a pyAMReX ``MFIter`` loop.
It does not, and it cannot: in C++, OpenMP works by wrapping the loop in ``#pragma omp parallel``, so that ``MFIter`` hands each thread a subset of the tiles.
There is no way to express that from Python, and the GIL serializes the loop body anyway.
**Your Python** ``for mfi in mfab:`` **loop is always single-threaded, whether or not you built with** ``AMREX_OMP=ON``.

OpenMP is still doing work for you, just not there.
Every AMReX operation you call from Python runs its own OpenMP-parallel loop internally: :py:meth:`~amrex.space3d.MultiFab.saxpy`, ``lin_comb``, the norms and reductions, ``FillBoundary``, ``ParallelCopy``, ``average_down``, plotfile I/O, particle ``Redistribute``, and so on.
So the first lever is to prefer AMReX's own operations over a hand-written loop wherever one exists.

For everything else, this is how to write a custom kernel once and have it run CPU-serial, CPU-threaded and on GPU:

.. literalinclude:: ../../../tests/test_multifab.py
   :language: python3
   :dedent: 4
   :start-after: # Manual: Portable Kernel START
   :end-before: # Manual: Portable Kernel END

:py:func:`~amrex.space3d.for_each_tile` is the ``MFIter`` loop as a decorator, so the Python reads in the same order as the C++ it replaces.
The kernel receives the tilebox followed by one ``Array4`` per field.
Index those with the box: ``f(bx)`` is the tile, and ``px(bx, di=-1)`` is the analogue of C++ ``px(i-1,j,k)`` over that tile.
Unlike :py:meth:`~amrex.space3d.Array4_double.to_xp`, which is a locally zero-based view of the whole fab, this indexing is in AMReX global index space -- which is what makes it correct under tiling, where a whole-array expression would otherwise be applied once per tile to the entire fab.

``amr.xp`` is the array namespace matching your build: NumPy on CPU, CuPy for CUDA/HIP, dpnp for SYCL.

On-node parallelism
^^^^^^^^^^^^^^^^^^^

In rough order of what to try:

#. **More MPI ranks.** This is the primary parallelism in pyAMReX and usually the best answer.
   Over-decompose with ``ba.max_size(...)`` and run ``mpirun -np <cores> python script.py``.
#. **Let AMReX run the loop**, using its built-in ``MultiFab`` operations where one fits.
#. **The** ``threads=`` **argument**, as above.
   It runs the per-tile kernels on a thread pool, which parallelizes because NumPy and CuPy release the GIL for the array operations a kernel body is made of.
   It only helps if the per-tile work is element-local -- no ghost exchange, no cross-tile dependencies.
#. **The GPU**, by building with CUDA/HIP/SYCL. The same kernel source then runs on the device.

Do not reach for ``multiprocessing``: the field memory lives in one process and AMReX+MPI is not fork-safe.

.. note::

   ``threads=`` is a *separate* thread pool from the one an ``AMREX_OMP=ON`` build uses for AMReX's own kernels.
   Using both oversubscribes the node. Pick one, or set ``OMP_NUM_THREADS=1``.

Tiling
^^^^^^

:py:func:`~amrex.space3d.TilingIfNotGPU` tiles on CPU and never on GPU, like its C++ namesake, but the tile size has **no default** and tiling is opt-in.
Plain ``for mfi in mfab:`` does not tile, and there is no global switch that changes that -- the ``fabarray.mfiter_tile_size`` runtime parameter only sets the size used *when* tiling is requested.

Tiling costs more in Python than in C++, where a tile is nearly free: here each tile is a loop iteration plus an array view per field.
It is worth it in one situation, when you have fewer boxes per rank than threads and want to feed the thread pool -- which is the same job it does for OpenMP threads in C++.
Size tiles so that you get roughly one to a few per thread.
Do not reuse AMReX's C++ default of ``(1024000, 8, 8)``: in Python it produces thousands of tiny work units and runs *slower than serial*.

.. note::

   Two things do not carry over from C++, by design.
   There is no ``ParallelFor``, because device lambdas cannot be written in Python; and there is no OpenMP region.
   Portability in pyAMReX means *array-expression* portability across NumPy/CuPy/dpnp, not *scalar-kernel* portability.
   A kernel with per-cell control flow has to become a masked array expression rather than translating line for line.

GPU streams and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You do not need to synchronize by hand.
AMReX keeps a pool of GPU streams -- four by default, set with the ``amrex.max_gpu_streams`` runtime parameter -- and ``MFIter`` round-robins over them, advancing the stream index on every tile.
``MFIter::Finalize()`` then synchronizes the streams it used, and pyAMReX runs it however the loop is left, including ``break``, ``return`` and exceptions, exactly as ``~MFIter()`` does in C++.

Two consequences are specific to Python kernels.

First, **you do not get the multi-stream overlap that C++ gets.**
The round-robin only helps if the work is launched on AMReX's current stream, which is what ``ParallelFor`` does.
A CuPy kernel launches on *CuPy's* current stream instead, so all tiles of a Python kernel serialize onto that one stream no matter which stream index ``MFIter`` has selected.

Second, when you mix custom kernels with AMReX's own operations, correctness rests on those two stream sets being ordered.
Today they are, because AMReX creates its pool with default flags -- making them *blocking* streams, which implicitly synchronize against the legacy default stream that CuPy uses by default.
That is a coincidence of defaults, not a guarantee: it is lost if you open an explicit ``cupy.cuda.Stream()`` or set ``CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1``.
In that case synchronize yourself between a Python kernel and the next AMReX call.

.. note::

   AMReX can also adopt a caller-supplied stream, via ``amrex::Gpu::setExternalGpuStream()`` and the RAII ``ExternalGpuStreamRegion``, which makes ``numGpuStreams()`` report 1 for as long as it is active.
   Handing AMReX CuPy's stream that way would put both on one stream and remove the reliance on default-stream semantics.
   pyAMReX does not bind this yet.


.. _usage-compute-particles:

Particles
---------

AMReX `Particles <https://amrex-codes.github.io/amrex/docs_html/Particle_Chapter.html>`__ are stored in the `ParticleContainer <https://amrex-codes.github.io/amrex/docs_html/Particle.html#the-particlecontainer>`__ class.

There are a few small differences to the `iteration over a ParticleContainer <https://amrex-codes.github.io/amrex/docs_html/Particle.html#iterating-over-particles>`__ compared to a ``MultiFab``:

* ``ParticleContainer`` is aware of mesh-refinement levels,
* AMReX supports a variety of data layouts for particles, the modern pure SoA + runtime attribute layout and the legacy AoS + SoA + runtime SoA attributes layout.

Here is the general structure for computing on particles:

.. tab-set::

   .. tab-item:: Modern (pure SoA) Layout

      .. tab-set::

         .. tab-item:: Simple

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Simple pti START
               :end-before: # Manual: Pure SoA Compute PC Simple pti END

         .. tab-item:: Detailed

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Detailed START
               :end-before: # Manual: Pure SoA Compute PC Detailed END

         .. tab-item:: Pandas (read-only)

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Pandas START
               :end-before: # Manual: Pure SoA Compute PC Pandas END


   .. tab-item:: Legacy (AoS + SoA) Layout

      .. literalinclude:: ../../../tests/test_particleContainer.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Legacy Compute PC Detailed START
         :end-before: # Manual: Legacy Compute PC Detailed END

For many small CPU and GPU examples on how to compute on particles, see the following test cases:

* .. dropdown:: Examples in ``test_particleContainer.py``

     .. literalinclude:: ../../../tests/test_particleContainer.py
        :language: python3
        :caption: This files is in ``tests/test_particleContainer.py``.

* .. dropdown:: Examples in ``test_aos.py``

     .. literalinclude:: ../../../tests/test_aos.py
        :language: python3
        :caption: This files is in ``tests/test_aos.py``.

* .. dropdown:: Examples in ``test_soa.py``

     .. literalinclude:: ../../../tests/test_soa.py
        :language: python3
        :caption: This files is in ``tests/test_soa.py``.


Other C++ Calls
---------------

pyAMReX exposes many more C++-written and GPU-accelerated AMReX functions for :py:class:`~amrex.space3d.MultiFab` and :ref:`particles <usage-api-particles>` to Python, which can be used to set, reduce, rescale, redistribute, etc. contained data.
Check out the detailed :ref:`API docs for more details <usage-api>` and use further third party libraries at will on the exposed data structures, replacing the hot loops described above.

You can also leave the Python world in pyAMReX and go back to C++ whenever needed.
For :ref:`some applications <usage_run>`, pyAMReX might act as *scriptable glue* that transports fields and particles from one (C++) function to another without recompilation, by exposing the functions and methods to Python.
