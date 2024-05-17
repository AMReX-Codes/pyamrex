"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def aos_to_numpy(self, copy=False):
    """
    Provide NumPy views into a ArrayOfStructs.

    Parameters
    ----------
    self : amrex.ArrayOfStructs_*
        An ArrayOfStructs class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each lists
        of 1D NumPy arrays.
    """
    import numpy as np

    if self.size() == 0:
        raise ValueError("AoS is empty.")

    if copy:
        # This supports a device-to-host copy.
        #
        # todo: validate of the to_host() returned object
        #       lifetime is always managed correctly by
        #       Python's GC - otherwise copy twice via copy=True
        return np.array(self.to_host(), copy=False)
    else:
        return np.array(self, copy=False)


def aos_to_cupy(self, copy=False):
    """
    Provide CuPy views into a ArrayOfStructs.

    Parameters
    ----------
    self : amrex.ArrayOfStructs_*
        An ArrayOfStructs class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each lists
        of 1D NumPy arrays.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    import cupy as cp

    if self.size() == 0:
        raise ValueError("AoS is empty.")

    return cp.array(self, copy=copy)


def aos_to_xp(self, copy=False):
    """
    Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Parameters
    ----------
    self : amrex.ArrayOfStructs_*
        An ArrayOfStructs class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each lists
        of 1D NumPy or CuPy arrays.
    """
    import inspect

    amr = inspect.getmodule(self)
    return self.to_cupy(copy) if amr.Config.have_gpu else self.to_numpy(copy)


def register_AoS_extension(amr):
    """ArrayOfStructs helper methods"""
    import inspect
    import sys

    # register member functions for every ArrayOfStructs_* type
    for _, AoS_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("ArrayOfStructs_"),
    ):
        AoS_type.to_numpy = aos_to_numpy
        AoS_type.to_cupy = aos_to_cupy
        AoS_type.to_xp = aos_to_xp
