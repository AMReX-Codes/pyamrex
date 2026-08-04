"""
This file is part of pyAMReX

Copyright 2023-2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from collections import namedtuple

from .dlpack_helpers import xp_module_name


def _soa_convert(self, converter_name, suffix, copy):
    """Convert every SoA component with the given PODVector converter.

    Parameters
    ----------
    self : amrex.StructOfArrays_*
        A StructOfArrays class in pyAMReX
    converter_name : string
        Name of the PODVector conversion method to apply per component,
        e.g., "to_numpy".
    suffix : string
        Suffix for the namedtuple type name, e.g., "np".
    copy : bool
        Copy the data if true, otherwise create views.

    Returns
    -------
    namedtuple
        A tuple with real and int components that are each dicts
        of 1D arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.
    """
    if self.size == 0:
        raise ValueError("SoA is empty.")

    SoA_tuple = namedtuple(type(self).__name__ + "_" + suffix, ["real", "int", "idcpu"])

    def convert(component):
        return getattr(component, converter_name)(copy=copy)

    # note: Python 3.7+ dicts are guaranteed to keep the insertion order,
    #       so users can also access them with .values()[<num>] as in the
    #       unnamed C++ API if they want to
    if self.has_idcpu:
        soa_view = SoA_tuple({}, {}, convert(self.get_idcpu_data()))
    else:
        soa_view = SoA_tuple({}, {}, None)

    real_comp_names = self.real_names
    if len(real_comp_names) != self.num_real_comps:
        raise ValueError("Missing names for SoA Real components.")
    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = convert(self.get_real_data(idx_real))

    int_comp_names = self.int_names
    if len(int_comp_names) != self.num_int_comps:
        raise ValueError("Missing names for SoA int components.")
    for idx_int in range(self.num_int_comps):
        soa_view.int[int_comp_names[idx_int]] = convert(self.get_int_data(idx_int))

    return soa_view


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
    return _soa_convert(self, "to_numpy", "np", copy)


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
        of 1D CuPy arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    return _soa_convert(self, "to_cupy", "cp", copy)


def soa_to_dpnp(self, copy=False):
    """
    Provide dpnp views into a StructOfArrays.

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
        of 1D dpnp arrays. The dictionary key order is the same as
        in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.

    Raises
    ------
    ImportError
        Raises an exception if dpnp is not installed
    """
    return _soa_convert(self, "to_dpnp", "dp", copy)


def soa_to_xp(self, copy=False):
    """
    Provide NumPy, CuPy or dpnp views into a StructOfArrays, depending on
    amr.Config.have_gpu and amr.Config.gpu_backend .

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
        of 1D NumPy, CuPy or dpnp arrays. The dictionary key order is the
        same as in the C++ component order.
        For pure SoA particle layouts, an additional component idcpu
        with global particle indices is populated.
    """
    import inspect

    amr = inspect.getmodule(self)
    return getattr(self, "to_" + xp_module_name(amr))(copy)


def register_SoA_extension(amr):
    """StructOfArrays helper methods"""
    import inspect
    import sys

    # register member functions for every StructOfArrays_* type
    for _, SoA_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("StructOfArrays_")
        ),
    ):
        # converters
        SoA_type.to_numpy = soa_to_numpy
        SoA_type.to_cupy = soa_to_cupy
        SoA_type.to_dpnp = soa_to_dpnp
        SoA_type.to_xp = soa_to_xp
