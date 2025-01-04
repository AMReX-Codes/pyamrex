"""
This file is part of pyAMReX

Copyright 2025 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


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

    if copy:
        data = np.array(self, copy=True)
    else:
        data = np.array(self, copy=False)

    # TODO: Check self.order == "F" ?
    if order == "F":
        return data.T
    elif order == "C":
        return data
    else:
        raise ValueError("The order argument must be F or C.")


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
    import cupy as cp

    # TODO: Check self.order == "F" ?
    if order == "F":
        return cp.array(self, copy=copy).T
    elif order == "C":
        return cp.array(self, copy=copy)
    else:
        raise ValueError("The order argument must be F or C.")


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
    return (
        self.to_cupy(copy, order) if amr.Config.have_gpu else self.to_numpy(copy, order)
    )


def register_SmallMatrix_extension(amr):
    """SmallMatrix helper methods"""
    import inspect
    import sys

    # register member functions for every Array4_* type
    for _, SmallMatrix_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("SmallMatrix_"),
    ):
        SmallMatrix_type.to_numpy = smallmatrix_to_numpy
        SmallMatrix_type.to_cupy = smallmatrix_to_cupy
        SmallMatrix_type.to_xp = smallmatrix_to_xp
