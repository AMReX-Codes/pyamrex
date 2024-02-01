"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from collections import namedtuple


def soa_to_numpy(self, copy=False):
    """
    Provide Numpy views into a StructOfArrays.

    Parameters
    ----------
    self : amrex.StructOfArrays_*
        A StructOfArrays class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each lists
        of 1D numpy arrays.
    """
    SoA_np = namedtuple(type(self).__name__ + "_np", ["real", "int"])

    soa_view = SoA_np([], [])

    if self.size() == 0:
        raise ValueError("SoA is empty.")

    for idx_real in range(self.num_real_comps):
        soa_view.real.append(self.GetRealData(idx_real).to_numpy(copy=copy))

    for idx_int in range(self.num_int_comps):
        soa_view.int.append(self.GetIntData(idx_int).to_numpy(copy=copy))

    return soa_view


def soa_to_cupy(self, copy=False):
    """
    Provide Cupy views into a StructOfArrays.

    Parameters
    ----------
    self : amrex.StructOfArrays_*
        A StructOfArrays class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each lists
        of 1D numpy arrays.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    SoA_cp = namedtuple(type(self).__name__ + "_cp", ["real", "int"])

    soa_view = SoA_cp([], [])

    if self.size() == 0:
        raise ValueError("SoA is empty.")

    for idx_real in range(self.num_real_comps):
        soa_view.real.append(self.GetRealData(idx_real).to_cupy(copy=copy))

    for idx_int in range(self.num_int_comps):
        soa_view.int.append(self.GetIntData(idx_int).to_cupy(copy=copy))

    return soa_view


def register_SoA_extension(amr):
    """StructOfArrays helper methods"""
    import inspect
    import sys

    # register member functions for every StructOfArrays_* type
    for _, SoA_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("StructOfArrays_"),
    ):
        SoA_type.to_numpy = soa_to_numpy
        SoA_type.to_cupy = soa_to_cupy
