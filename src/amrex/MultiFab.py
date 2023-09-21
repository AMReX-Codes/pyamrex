"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def mf_to_numpy(self, copy=False, order="F"):
    """
    Provide a Numpy view into a MultiFab.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    list of np.array
        A list of numpy n-dimensional arrays, for each local block in the
        MultiFab.
    """
    views = []
    for mfi in self:
        views.append(self.array(mfi).to_numpy(copy, order))

    return views


def mf_to_cupy(self, copy=False, order="F"):
    """
    Provide a Cupy view into a MultiFab.

    Note on the order of indices:
    By default, this is as in AMReX in Fortran contiguous order, indexing as
    x,y,z. This has performance implications for use in external libraries such
    as cupy.
    The order="C" option will index as z,y,x and perform better with cupy.
    https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).
    order : string, optional
        F order (default) or C. C is faster with external libraries.

    Returns
    -------
    list of cupy.array
        A list of cupy n-dimensional arrays, for each local block in the
        MultiFab.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    views = []
    for mfi in self:
        views.append(self.array(mfi).to_cupy(copy, order))

    return views


def register_MultiFab_extension(amr):
    """MultiFab helper methods"""
    import inspect
    import sys

    # register member functions for every MultiFab* type
    for _, MultiFab_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("MultiFab"),
    ):
        MultiFab_type.to_numpy = mf_to_numpy
        MultiFab_type.to_cupy = mf_to_cupy
