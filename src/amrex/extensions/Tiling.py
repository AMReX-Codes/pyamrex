"""
This file is part of pyAMReX

Copyright 2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from concurrent.futures import ThreadPoolExecutor

from .Iterator import iterate


def TilingIfNotGPU(amr, tile=None):
    """Return an ``MFItInfo`` that tiles on CPU and never on GPU.

    This is the analogue of C++ ``amrex::TilingIfNotGPU()``, with one
    deliberate difference: ``tile`` has **no default**, so tiling is opt-in.

    In C++ the default tile size is ``FabArrayBase::mfiter_tile_size``
    (``1024000,8,8`` in 3D), which is sized for OpenMP: it hands threads many
    small work units, and the per-tile cost of an ``MFIter`` step is
    negligible. In Python the per-tile cost is a loop iteration plus an array
    view per field, which is not negligible -- with that tile size a
    representative kernel measured *slower than serial*. A large default
    instead would silently do nothing for the typical 16-64^3 box.

    Tiling in Python only pays off when there are fewer boxes than threads and
    you are threading over them (see :py:func:`for_each_tile`), so the caller
    has to say so, and say how big.

    Parameters
    ----------
    amr : module
        The dimensionality-specific pyAMReX module.
    tile : sequence of int, optional
        Tile size, ``AMREX_SPACEDIM`` entries. ``None`` (default) means do not
        tile, i.e. iterate whole boxes. Ignored on GPU.

    Returns
    -------
    amr.MFItInfo
        Info object to hand to ``amr.MFIter``.
    """
    info = amr.MFItInfo()
    if tile is not None and not amr.Config.have_gpu:
        info.enable_tiling(amr.IntVect(*tile))
    return info


def tiles(amr, mfab, tile=None):
    """Iterate the local boxes of ``mfab``, optionally split into tiles.

    Equivalent to ``iter(mfab)`` when ``tile`` is ``None``. Like every
    iteration path in pyAMReX this yields the ``MFIter`` itself, and finalizes
    it however the loop is left.

    Parameters
    ----------
    amr : module
        The dimensionality-specific pyAMReX module.
    mfab : amr.MultiFab or amr.iMultiFab
        The field to iterate.
    tile : sequence of int, optional
        Tile size; see :py:func:`TilingIfNotGPU`.

    Yields
    ------
    amr.MFIter
        The iterator, positioned on the current box or tile.
    """
    return iterate(amr.MFIter(mfab, TilingIfNotGPU(amr, tile)))


def _sync_device(amr, sample):
    """Wait for the array library's stream/queue, on GPU builds.

    Kernels written against ``amr.xp`` launch on CuPy's stream or dpnp's SYCL
    queue, not on the AMReX stream that ``MFIter::Finalize()`` synchronizes, so
    that one has to be waited on separately.

    Parameters
    ----------
    amr : module
        The dimensionality-specific pyAMReX module.
    sample : amr.Array4_*
        An Array4 of a field the kernel wrote. Only SYCL needs it, to find the
        queue the work went to, so the array view is built only in that case.

    Raises
    ------
    ImportError
        If the build's array library (CuPy or dpnp) is not installed. Those are
        optional dependencies of pyAMReX; reaching here means the kernel just
        ran on device arrays, so one of them is necessarily already imported.
    """
    if amr.Config.gpu_backend == "SYCL":
        # dpnp has no module-level synchronize(); the queue lives on the array
        sample.to_xp(copy=False).sycl_queue.wait()
    else:  # CUDA, HIP
        import cupy

        cupy.cuda.get_current_stream().synchronize()


def ix_type(self):
    """The index type (staggering/centering) of this field, as an ``IntVect``.

    The analogue of C++ ``mf.ixType().toIntVect()``, for passing to
    ``mfi.tilebox(...)`` so that the returned box carries the right centering.
    """
    return self.box_array().ix_type().to_IntVect()


def for_each_tile(amr, mfab, *others, tile=None, threads=1):
    """Run the decorated kernel over every box or tile of ``mfab``.

    This is the ``MFIter`` loop expressed as a decorator, so that a portable
    kernel reads in the same order as the equivalent C++ block::

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (MFIter mfi(divE, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox(divE.ixType().toIntVect());
            Array4<Real> const& d  = divE.array(mfi);
            Array4<Real> const& ex = Ex.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                d(i,j,k) = inv2dr * (ex(i-1,j,k) - ex(i+1,j,k));
            });
        }

    becomes::

        @amr.for_each_tile(divE, Ex, tile=(64,) * 3, threads=8)
        def _(bx, d, ex):
            d(bx)[...] = inv2dr * (ex(bx, di=-1) - ex(bx, di=+1))

    The decorator line stands in for the pragma and the ``MFIter`` line, the
    parameter list for the ``tilebox`` and ``array(mfi)`` extractions, and the
    body for the ``ParallelFor`` lambda. The kernel receives the tilebox
    followed by one ``Array4`` per field passed here, in order; index those
    with the box, as ``d(bx)`` or ``ex(bx, di=-1)``.

    The decorated function is called immediately and returned unchanged, so
    naming it ``_`` is conventional but not required.

    On threading: a single serial ``MFIter`` pass collects the per-tile
    arguments before any kernel runs, and this is required rather than an
    optimization. The ``MFIter`` mutates in place on each step and yields
    itself, so its state cannot be handed to a worker to use later; and AMReX
    permits only one live ``MFIter`` at a time
    (``AMREX_ALWAYS_ASSERT(depth == 1)``), so workers cannot drive their own.
    Threading then works because numpy/cupy release the GIL for the array
    operations the kernel body is made of.

    Note that ``threads`` is a *separate* thread pool from the one an
    ``AMREX_OMP=ON`` build uses for AMReX's own kernels. Using both
    oversubscribes the node; pick one.

    Parameters
    ----------
    amr : module
        The dimensionality-specific pyAMReX module.
    mfab : amr.MultiFab or amr.iMultiFab
        Field defining the iteration space; also the first kernel argument.
    *others : amr.MultiFab or amr.iMultiFab
        Further fields, passed to the kernel after ``mfab``. They must share
        ``mfab``'s BoxArray and DistributionMapping.
    tile : sequence of int, optional
        Tile size; see :py:func:`TilingIfNotGPU`.
    threads : int, optional
        Worker threads for the kernel (default 1, i.e. serial). Forced to 1 on
        GPU, where the device provides the parallelism.

    Returns
    -------
    callable
        A decorator that runs the kernel and returns it unchanged.
    """

    def decorate(kernel):
        nthreads = 1 if amr.Config.have_gpu else threads

        # Snapshot per-tile arguments in one serial pass; see docstring.
        ixt = ix_type(mfab)
        tasks = [
            (mfi.tilebox(ixt), mfab.array(mfi), *[o.array(mfi) for o in others])
            for mfi in tiles(amr, mfab, tile)
        ]

        if nthreads == 1:
            for task in tasks:
                kernel(*task)
        else:
            with ThreadPoolExecutor(max_workers=nthreads) as pool:
                # list() so that exceptions raised in a worker propagate here
                list(pool.map(lambda task: kernel(*task), tasks))

        # The MFIter is already finalized at this point -- the snapshot pass
        # above ran it to completion -- so its Gpu::streamSynchronize() came
        # *before* any of these kernels launched and cannot cover them. The
        # kernels also ran on the array library's stream rather than AMReX's,
        # so synchronize that one to make the work complete on return.
        if amr.Config.have_gpu and tasks:
            _sync_device(amr, tasks[0][1])

        return kernel

    return decorate
