"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from collections import namedtuple

from .ParticleComponentNames import soa_int_comps, soa_real_comps


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
        A tuple with real and int components that are each dicts
        of 1D numpy arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.
    """
    if self.size == 0:
        raise ValueError("SoA is empty.")

    SoA_np = namedtuple(type(self).__name__ + "_np", ["real", "int", "idcpu"])

    # note: Python 3.7+ dicts are guaranteed to keep the insertion order,
    #       so users can also access them with .values()[<num>] as in the
    #       unnamed C++ API if they want to
    if self.has_idcpu:
        soa_view = SoA_np({}, {}, self.get_idcpu_data().to_numpy(copy=copy))
    else:
        soa_view = SoA_np({}, {}, None)

    # for the legacy data layout, do not start with x, y, z but with a, b, c, ...
    if self.has_idcpu:
        real_comp_names = soa_real_comps(self.num_real_comps)
    else:
        real_comp_names = soa_real_comps(self.num_real_comps, rotate=False)

    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(
            idx_real
        ).to_numpy(copy=copy)

    int_comp_names = soa_int_comps(self.num_int_comps)
    for idx_int in range(self.num_int_comps):
        soa_view.int[int_comp_names[idx_int]] = self.get_int_data(idx_int).to_numpy(
            copy=copy
        )

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
        A tuple with real and int components that are each dicts
        of 1D numpy arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    if self.size == 0:
        raise ValueError("SoA is empty.")

    SoA_cp = namedtuple(type(self).__name__ + "_cp", ["real", "int", "idcpu"])

    # note: Python 3.7+ dicts are guaranteed to keep the insertion order,
    #       so users can also access them with .values()[<num>] as in the
    #       unnamed C++ API if they want to
    if self.has_idcpu:
        soa_view = SoA_cp({}, {}, self.get_idcpu_data().to_cupy(copy=copy))
    else:
        soa_view = SoA_cp({}, {}, None)

    real_comp_names = soa_real_comps(self.num_real_comps)
    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(idx_real).to_cupy(
            copy=copy
        )

    int_comp_names = soa_int_comps(self.num_int_comps)
    for idx_int in range(self.num_int_comps):
        soa_view.int[int_comp_names[idx_real]] = self.get_int_data(idx_int).to_cupy(
            copy=copy
        )

    if self.has_idcpu:
        soa_view.idcpu = self.get_idcpu_data().to_cupy(copy=copy)

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
