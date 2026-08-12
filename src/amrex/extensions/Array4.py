"""
This file is part of pyAMReX

Copyright 2023-2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from .dlpack_helpers import dlpack_to_cupy, dlpack_to_dpnp, reorder, xp_module_name


def array4_to_numpy(self, copy=False, order="F"):
    """
    Provide a NumPy view into an Array4.

    This includes ngrow guard cells of the box.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.Array4_*
        An Array4 class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    np.array
        A NumPy n-dimensional array.
    """
    import numpy as np

    if copy:
        # This supports a device-to-host copy.
        data = self.to_host()
    else:
        # host-accessible memory (CPU, pinned, managed/shared USM): the
        # __array_interface__ exposes the host pointer regardless of the
        # DLPack device tag, unlike np.from_dlpack which is host-device only
        data = np.array(self, copy=False)

    return reorder(data, order)


def array4_to_cupy(self, copy=False, order="F"):
    """
    Provide a CuPy view into an Array4.

    This includes ngrow guard cells of the box.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.Array4_*
        An Array4 class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    cupy.array
        A cupy n-dimensional array.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    return reorder(dlpack_to_cupy(self, copy), order)


def array4_to_dpnp(self, copy=False, order="F"):
    """
    Provide a dpnp view into an Array4.

    This includes ngrow guard cells of the box.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as dpnp.
    The order="C" option will index as z,y,x and may perform better.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.Array4_*
        An Array4 class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    dpnp.array
        A dpnp n-dimensional array.

    Raises
    ------
    ImportError
        Raises an exception if dpnp is not installed
    """
    return reorder(dlpack_to_dpnp(self, copy), order)


def array4_to_xp(self, copy=False, order="F"):
    """
    Provide a NumPy, CuPy or dpnp view into an Array4, depending on amr.Config.have_gpu
    and amr.Config.gpu_backend .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    This includes ngrow guard cells of the box.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.Array4_*
        An Array4 class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    xp.array
        A NumPy, CuPy or dpnp n-dimensional array.
    """
    import inspect

    amr = inspect.getmodule(self)
    return getattr(self, "to_" + xp_module_name(amr))(copy, order)


def array4_call(self, bx, di=0, dj=0, dk=0, comp=0):
    """Provide a view of this Array4 over a Box, in AMReX global indexing.

    This is the array-expression analogue of indexing an ``Array4`` in C++:
    ``a(bx, di=-1)`` corresponds to ``a(i-1, j, k)`` evaluated for every
    ``(i,j,k)`` in ``bx``, so a stencil written in C++ as::

        d(i,j,k) = inv2dr * (ex(i-1,j,k) - ex(i+1,j,k));

    becomes::

        d(bx)[...] = inv2dr * (ex(bx, di=-1) - ex(bx, di=+1))

    Unlike :py:meth:`to_xp`, which is a view of the whole fab in local 0-based
    indexing, ``bx`` here is in AMReX global index space. That is what makes it
    usable under tiling: ``mfi.tilebox()`` names a sub-region of the fab, and
    without restricting to it a whole-array expression would be applied once
    per tile to the entire fab.

    Reading offset cells (``di``/``dj``/``dk``) reaches into the guard cells,
    so ``bx`` grown by the offsets must stay inside the fab; for a valid-region
    ``bx`` that means the field needs enough ghost cells for the stencil.

    Parameters
    ----------
    self : amrex.Array4_*
        An Array4 class in pyAMReX.
    bx : amrex.Box
        Index-space region to view, in AMReX global indices.
    di, dj, dk : int, optional
        Shift the region by this many cells per direction (default 0).
    comp : int, optional
        Component to select (default 0).

    Returns
    -------
    xp.array
        A non-copying NumPy, CuPy or dpnp view of ``bx`` shifted by
        ``(di, dj, dk)``, with ``AMREX_SPACEDIM`` dimensions.
    """
    import inspect

    amr = inspect.getmodule(self)

    arr = self.to_xp(copy=False, order="F")
    lo = amr.lbound(self)
    lo = (lo.x, lo.y, lo.z)
    shift = (di, dj, dk)

    # An Array4 is always 4D (i,j,k,n): unused directions carry extent 1 rather
    # than being dropped. So always index all three spatial axes -- taking only
    # AMREX_SPACEDIM of them would make `comp` land on k in 1D/2D. The unused
    # ones are indexed with a scalar rather than sliced, which drops them, so
    # the result has AMREX_SPACEDIM dimensions in every build.
    dims = amr.Config.spacedim
    slices = tuple(
        slice(
            bx.small_end[d] + shift[d] - lo[d],
            bx.big_end[d] + shift[d] - lo[d] + 1,
        )
        if d < dims
        else 0
        for d in range(3)
    )
    return arr[slices + (comp,)]


def register_Array4_extension(amr):
    """Array4 helper methods"""
    import inspect
    import sys

    # register member functions for every Array4_* type
    for _, Array4_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("Array4_")
        ),
    ):
        Array4_type.to_numpy = array4_to_numpy
        Array4_type.to_cupy = array4_to_cupy
        Array4_type.to_dpnp = array4_to_dpnp
        Array4_type.to_xp = array4_to_xp
        Array4_type.__call__ = array4_call
