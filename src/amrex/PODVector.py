"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def podvector_to_numpy(self, copy=False):
    """
    Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    np.array
        A 1D numpy array.
    """
    import numpy as np

    if self.size() > 0:
        if copy:
            # This supports a device-to-host copy.
            #
            # todo: validate of the to_host() returned object
            #       lifetime is always managed correctly by
            #       Python's GC - otherwise copy twice via copy=True
            return np.array(self.to_host(), copy=False)
        else:
            return np.array(self, copy=False)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_cupy(self, copy=False):
    """
    Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    cupy.array
        A 1D cupy array.

    Raises
    ------
    ImportError
        Raises an exception if cupy is not installed
    """
    import cupy as cp

    if self.size() > 0:
        return cp.array(self, copy=copy)
    else:
        raise ValueError("Vector is empty.")


def register_PODVector_extension(amr):
    """PODVector helper methods"""
    import inspect
    import sys

    # register member functions for every PODVector_* type
    for _, POD_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: inspect.isclass(member)
        and member.__module__ == amr.__name__
        and member.__name__.startswith("PODVector_"),
    ):
        POD_type.to_numpy = podvector_to_numpy
        POD_type.to_cupy = podvector_to_cupy
