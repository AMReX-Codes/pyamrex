"""
This file is part of pyAMReX

Copyright 2025-2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from .dlpack_helpers import dlpack_to_cupy, dlpack_to_dpnp, reorder, xp_module_name


def smallmatrix_to_numpy(self, copy=False, order="F"):
    """
    Provide a NumPy view into an SmallMatrix.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.SmallMatrix_*
        A SmallMatrix class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    np.array
        A NumPy 2-dimensional array.
    """
    import numpy as np

    # SmallMatrix data is always host-side
    data = np.array(self, copy=copy)

    # TODO: Check self.order == "F" ?
    return reorder(data, order)


def smallmatrix_to_cupy(self, copy=False, order="F"):
    """
    Provide a CuPy view into an SmallMatrix.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.SmallMatrix_*
        A SmallMatrix class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    cupy.array
        A cupy 2-dimensional array.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    # SmallMatrix data is always host-side: importing into CuPy copies
    # to the device, independent of the copy argument
    # TODO: Check self.order == "F" ?
    return reorder(dlpack_to_cupy(self, copy), order)


def smallmatrix_to_dpnp(self, copy=False, order="F"):
    """
    Provide a dpnp copy of a SmallMatrix.

    SmallMatrix data is always host-side: importing into dpnp copies to the
    device, independent of the copy argument.

    Parameters
    ----------
    self : amrex.SmallMatrix_*
        A SmallMatrix class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    dpnp.array
        A dpnp 2-dimensional array.

    Raises
    ------
    ImportError
        Raises an exception if dpnp is not installed
    """
    # TODO: Check self.order == "F" ?
    return reorder(dlpack_to_dpnp(self, copy), order)


def smallmatrix_to_xp(self, copy=False, order="F"):
    """
    Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.SmallMatrix_*
        A SmallMatrix class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    xp.array
        A NumPy or CuPy 2-dimensional array.
    """
    import inspect

    amr = inspect.getmodule(self)
    return getattr(self, "to_" + xp_module_name(amr))(copy, order)


def register_SmallMatrix_extension(amr):
    """SmallMatrix helper methods"""
    import inspect
    import sys

    # register member functions for every Array4_* type
    for _, SmallMatrix_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("SmallMatrix_")
        ),
    ):
        SmallMatrix_type.to_numpy = smallmatrix_to_numpy
        SmallMatrix_type.to_cupy = smallmatrix_to_cupy
        SmallMatrix_type.to_dpnp = smallmatrix_to_dpnp
        SmallMatrix_type.to_xp = smallmatrix_to_xp
