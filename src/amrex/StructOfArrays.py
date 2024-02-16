"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from collections import namedtuple


def soa_real_comps(self, num_comps, spacedim=3, rotate=True):
    """
    Name the ParticleReal components in SoA.

    Parameters
    ----------
    self : SoA Type
      maybe unused, depending on implementation
    num_comps : int
      number of components to generate names for.
    spacedim : int
      AMReX dimensionality
    rotate : bool = True
      start with "x", "y", "z", "a", "b", ...

    Returns
    -------
    A list of length num_comps with values
    rotate=True (for pure SoA layout):
    - 3D: "x", "y", "z", "a", "b", ... "w", "r0", "r1", ...
    - 2D: "x", "y", "a", "b", ... "w", "r0", "r1", ...
    - 1D: "x", "a", "b", ... "w", "r0", "r1", ...
    rotate=False (for legacy layout):
    - 1D-3D: "a", "b", ... "w", "r0", "r1", ...
    """
    import string

    # x, y, z, a, b, ...
    comp_names = list(string.ascii_lowercase)
    if rotate:
        # rotate x, y, z to be beginning (positions)
        comp_names = comp_names[-3:] + comp_names[:-3]
    else:
        # cut off x, y, z to avoid confusion
        comp_names = comp_names[:-3]

    num_named = len(comp_names)
    if num_comps < num_named:
        comp_names = list(comp_names)[0:num_comps]
    elif num_comps > num_named:
        comp_names.extend(["r" + str(i) for i in range(num_comps - num_named)])

    return comp_names


def soa_int_comps(self, num_comps):
    """
    Name the int components in SoA.

    Parameters
    ----------
    self : SoA Type
      maybe unused, depending on implementation
    num_comps : int
      number of components to generate names for.

    Returns
    -------
    A list of length num_comps with values "i1", "i2", "i3", ...
    """
    comp_names = ["i" + str(i) for i in range(num_comps)]

    return comp_names


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
        real_comp_names = self.soa_real_comps(self.num_real_comps)
    else:
        real_comp_names = self.soa_real_comps(self.num_real_comps, rotate=False)

    for idx_real in range(self.num_real_comps):
        print(idx_real, real_comp_names[idx_real], self.get_real_data(idx_real).size())
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(
            idx_real
        ).to_numpy(copy=copy)

    int_comp_names = self.soa_int_comps(self.num_int_comps)
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

    real_comp_names = self.soa_real_comps(self.num_real_comps)
    for idx_real in range(self.num_real_comps):
        soa_view.real[real_comp_names[idx_real]] = self.get_real_data(idx_real).to_cupy(
            copy=copy
        )

    int_comp_names = self.soa_int_comps(self.num_int_comps)
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
        # name providers
        SoA_type.soa_real_comps = soa_real_comps
        SoA_type.soa_int_comps = soa_int_comps

        # converters
        SoA_type.to_numpy = soa_to_numpy
        SoA_type.to_cupy = soa_to_cupy
