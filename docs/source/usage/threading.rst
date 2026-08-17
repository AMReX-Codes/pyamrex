.. _usage-threading:

Threading (Free-Threaded Python)
================================

CPython 3.13 introduced a *free-threaded* build (`PEP 703 <https://peps.python.org/pep-0703/>`__)
in which the global interpreter lock is disabled, so Python threads run in
parallel on separate cores. pyAMReX supports it: the extension modules declare
``Py_mod_gil = Py_MOD_GIL_NOT_USED``, so importing pyAMReX does **not** switch
the GIL back on.

This matters most for the code pyAMReX exists to enable: per-block Python
kernels, in-situ analysis, and data-science glue, which are written in Python
and therefore serialized by the GIL no matter how many blocks a ``MultiFab``
has.

Check that you are on a free-threaded interpreter and that nothing re-enabled
the GIL:

.. code-block:: python

   import sys
   import amrex.space3d as amr

   assert sys._is_gil_enabled() is False

Free-threaded interpreters are named ``python3.13t`` / ``python3.14t``, and
their wheels carry the ``cp313t`` / ``cp314t`` ABI tag. On a regular
(GIL-enabled) build the declaration is inert and everything below still works,
just without the parallelism.


.. _usage-threading-contract:

What is guaranteed
------------------

pyAMReX follows the same rule NumPy does: **the library's own global state is
race-free; your data is yours to synchronize.**

Safe from any number of threads:

* Working on **distinct** objects -- each thread with its own ``MultiFab``,
  ``ParticleContainer``, ``BoxArray``, ``Geometry``, and so on, including
  objects built from the same :py:class:`~amrex.space3d.BoxArray` and
  :py:class:`~amrex.space3d.DistributionMapping`.
* **Reading** the same object concurrently.
* Each thread driving **its own** :py:class:`~amrex.space3d.MFIter`, even over
  the same ``MultiFab``, as long as the threads write to disjoint boxes.
* Creating and releasing zero-copy views (``to_numpy``, ``to_cupy``, ``to_xp``,
  ``__dlpack__``) concurrently.
* Querying :py:class:`~amrex.space3d.ParmParse` concurrently.

Your responsibility, exactly as with a NumPy array:

* **Writing to the same data from two threads.** Two threads writing the same
  box, or writing overlapping regions of one array, is a data race. Partition
  the work, or use a lock.
* Keeping an object alive while another thread uses a view into it. A zero-copy
  view keeps its Python owner alive, but calling ``clear()`` on a ``MultiFab``
  while another thread iterates it is not defensible.

Main thread only:

* :py:func:`~amrex.space3d.initialize` and :py:func:`~amrex.space3d.finalize`.
  They start and stop process-wide state -- the AMReX instance stack, the
  memory arenas, the parameter table, signal and floating-point-exception
  handlers. pyAMReX serializes them so a mistake is not silent corruption, but
  they are not meant to overlap with anything.
* **MPI-collective calls** (reductions such as ``min``/``max``/``norm0``/``sum``
  with ``local=False``, plotfile writes, ``Redistribute`` across ranks) unless
  AMReX was built with ``AMReX_MPI_THREAD_MULTIPLE=ON``. Without it, AMReX
  calls ``MPI_Init`` rather than ``MPI_Init_thread``, and MPI is not safe to
  enter from two threads.
* Process-wide policy switches: ``Config.verbose``, the AMReX error handler,
  ``amrex.throw_exception``.
* **Reseeding the random number generator.**
  :py:meth:`~amrex.space3d.ParticleContainer_2_1_3_1_default.init_random` and
  ``init_random_per_box`` forward to ``amrex::InitRandom``, which reseeds
  *process-global* state -- and on a GPU build frees and reallocates the device
  RNG state array. Seed once before starting worker threads. (Drawing numbers
  is fine concurrently on the host: each thread has its own generator.)
* :cpp:class:`amrex::TinyProfiler` (``tiny_profiler.enabled=1``). Its section
  stack is global and asserts on nesting; use it single-threaded.
* Externally supplied GPU streams (``Gpu::setExternalStream``).
* ``MFItInfo().set_dynamic(True)``. Dynamic MFIter scheduling shares one
  counter across an OpenMP team; two threads each opening their own team would
  share it.


.. _usage-threading-patterns:

Two patterns
------------

**Each thread drives its own MFIter.** Threads pick their boxes by index, so no
two touch the same data:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   import amrex.space3d as amr

   nthreads = 8

   def work(t):
       for mfi in mfab:                       # this thread's own iterator
           if mfi.index % nthreads != t:      # claim every nth box
               continue
           field = mfab.array(mfi).to_xp()
           field[...] = field * 2.0 + 1.0

   with ThreadPoolExecutor(max_workers=nthreads) as pool:
       list(pool.map(work, range(nthreads)))  # list() so errors propagate

**Snapshot the tiles, then run the kernels in a pool.** One serial ``MFIter``
pass collects the per-box views, and only the kernel bodies are threaded. Use
this when the kernel is what costs, and you would rather keep iteration
obviously sequential:

.. code-block:: python

   views = [mfab.array(mfi).to_xp() for mfi in mfab]

   with ThreadPoolExecutor(max_workers=nthreads) as pool:
       list(pool.map(lambda v: v.__setitem__(Ellipsis, v * 2.0 + 1.0), views))

Both scale about the same. The first is more direct; the second composes better
when the tile list is built once and reused.


.. _usage-threading-openmp:

Threads and OpenMP
------------------

Python threads and an ``AMReX_OMP=ON`` build are two separate thread pools over
the same cores. Running both oversubscribes the node and usually ends up slower
than either alone -- pick one:

* **Python threads** (``AMREX_OMP=OFF``, the default for pyAMReX wheels) when
  the hot loop is Python.
* **OpenMP** (``AMREX_OMP=ON``, ``OMP_NUM_THREADS=N``) when the hot loop is
  inside AMReX's own C++ kernels and Python is only orchestrating.

If you must use both, cap them so the product matches the core count, e.g.
``OMP_NUM_THREADS=2`` with four Python threads on eight cores.


.. _usage-threading-numbers:

What to expect
--------------

A per-box kernel written in Python, over a 128³ domain decomposed into 64
boxes, on a 14-core / 20-thread CPU:

=========  =================  ==================
threads    GIL enabled        GIL disabled
=========  =================  ==================
1          1.00x              1.00x
2          1.03x              1.92x
4          1.04x              3.31x
8          1.01x              4.87x
14         0.95x              6.16x
20         0.90x              6.92x
=========  =================  ==================

A NumPy-vectorised kernel over the same data is memory-bandwidth-bound and
scales only weakly either way -- but note that *with* the GIL it actively
degrades past four threads (0.31x at 20 threads), because the threads spend
their time contending for the lock rather than computing.

Reproduce with ``tools/bench_free_threading.py``; the same binary, run twice::

    PYTHON_GIL=0 python tools/bench_free_threading.py --json off.json
    PYTHON_GIL=1 python tools/bench_free_threading.py --json on.json
    python tools/bench_free_threading.py --compare off.json on.json


.. _usage-threading-testing:

Testing your own code
---------------------

``PYTHON_GIL=0`` / ``PYTHON_GIL=1`` (or ``-X gil=0`` / ``-X gil=1``) force the
GIL off or on in a free-threaded interpreter, so you can A/B the same build.
Races are timing-dependent, so run concurrency tests repeatedly --
``pytest --count=50`` with `pytest-repeat
<https://pypi.org/project/pytest-repeat/>`__ -- and prefer assertions on
computed values over "it did not crash".

pyAMReX's own concurrency tests are in ``tests/test_free_threading.py``.
