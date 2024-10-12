"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from collections import namedtuple


def soa_to_numpy(self, copy=False):
    """
    Provide NumPy views into a StructOfArrays.

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
        of 1D NumPy arrays. The dictionary key order is the same as
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

    real_comp_names = self.real_names
    if len(real_comp_names) != self.num_real_comps:
        raise ValueError("Missing names for SoA Real components.")
    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(
            idx_real
        ).to_numpy(copy=copy)

    int_comp_names = self.int_names
    if len(int_comp_names) != self.num_int_comps:
        raise ValueError("Missing names for SoA int components.")
    for idx_int in range(self.num_int_comps):
        soa_view.int[int_comp_names[idx_int]] = self.get_int_data(idx_int).to_numpy(
            copy=copy
        )

    return soa_view


def soa_to_cupy(self, copy=False):
    """
    Provide CuPy views into a StructOfArrays.

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
        of 1D NumPy arrays. The dictionary key order is the same as
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

    real_comp_names = self.real_names
    if len(real_comp_names) != self.num_real_comps:
        raise ValueError("Missing names for SoA Real components.")
    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(idx_real).to_cupy(
            copy=copy
        )

    int_comp_names = self.int_names
    if len(int_comp_names) != self.num_int_comps:
        raise ValueError("Missing names for SoA int components.")
    for idx_int in range(self.num_int_comps):
        soa_view.int[int_comp_names[idx_int]] = self.get_int_data(idx_int).to_cupy(
            copy=copy
        )

    return soa_view


def soa_to_xp(self, copy=False):
    """
    Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

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
        of 1D NumPy or CuPy arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.
    """
    import inspect

    amr = inspect.getmodule(self)
    return self.to_cupy(copy) if amr.Config.have_gpu else self.to_numpy(copy)


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
        # converters
        SoA_type.to_numpy = soa_to_numpy
        SoA_type.to_cupy = soa_to_cupy
        SoA_type.to_xp = soa_to_xp
