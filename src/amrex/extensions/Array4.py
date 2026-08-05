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
