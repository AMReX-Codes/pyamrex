"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def array4_to_numpy(self, copy=False, order="F"):
    """
    Provide a Numpy view into an Array4.

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
        A numpy n-dimensional array.
    """
    import numpy as np

    if copy:
        # This supports a device-to-host copy.
        data = self.to_host()
    else:
        data = np.array(self, copy=False)

    if order == "F":
        return data.T
    elif order == "C":
        return data
    else:
        raise ValueError("The order argument must be F or C.")


def array4_to_cupy(self, copy=False, order="F"):
    """
    Provide a Cupy view into an Array4.

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
    import cupy as cp

    if order == "F":
        return cp.array(self, copy=copy).T
    elif order == "C":
        return cp.array(self, copy=copy)
    else:
        raise ValueError("The order argument must be F or C.")


# torch


def register_Array4_extension(amr):
    """Array4 helper methods"""
    import inspect
    import sys

    # register member functions for every Array4_* type
    for _, Array4_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("Array4_"),
    ):
        Array4_type.to_numpy = array4_to_numpy
        Array4_type.to_cupy = array4_to_cupy
