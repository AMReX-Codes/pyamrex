"""
This file is part of pyAMReX

Copyright 2024 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def soa_real_comps(num_comps, spacedim=3, rotate=True):
    """
    Name the ParticleReal components in SoA.

    Parameters
    ----------
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
      3D: "x", "y", "z", "a", "b", ... "w", "r0", "r1", ...
      2D: "x", "y", "a", "b", ... "w", "r0", "r1", ...
      1D: "x", "a", "b", ... "w", "r0", "r1", ...
    rotate=False (for legacy layout):
      1D-3D: "a", "b", ... "w", "r0", "r1", ...
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


def soa_int_comps(num_comps):
    """
    Name the int components in SoA.

    Parameters
    ----------
    num_comps : int
      number of components to generate names for.

    Returns
    -------
    A list of length num_comps with values "i1", "i2", "i3", ...
    """
    comp_names = ["i" + str(i) for i in range(num_comps)]

    return comp_names
