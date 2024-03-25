"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from .Iterator import next


def mf_to_numpy(amr, self, copy=False, order="F"):
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
    list of numpy.array
        A list of numpy n-dimensional arrays, for each local block in the
        MultiFab.
    """
    mf = self
    if copy:
        mf = amr.MultiFab(
            self.box_array(),
            self.dm(),
            self.n_comp,
            self.n_grow_vect,
            amr.MFInfo().set_arena(amr.The_Pinned_Arena()),
        )
        amr.dtoh_memcpy(mf, self)

    views = []
    for mfi in mf:
        views.append(mf.array(mfi).to_numpy(copy=False, order=order))

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


def copy_multifab(amr, self):
    """
    TODO

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    """
    #mfinfo = self.MFInfo()
    mf = amr.MultiFab(
        self.box_array(),
        self.dm(),
        self.n_comp,
        self.n_grow_vect,
        amr.MFInfo().set_arena(amr.The_Device_Arena()),  # TODO: use same Arena as used by self
    )
    amr.copy_mfab(
        dst=mf,
        src=self,
        srccomp=0,
        dstcomp=0,
        numcomp=self.n_comp,
        nghost=self.n_grow_vect
    )
    return mf


def register_MultiFab_extension(amr):
    """MultiFab helper methods"""

    # register member functions for the MFIter type
    amr.MFIter.__next__ = next

    # FabArrayBase: iterate as data access in Box index space
    amr.FabArrayBase.__iter__ = lambda fab: amr.MFIter(fab)

    # register member functions for the MultiFab type
    amr.MultiFab.__iter__ = lambda mfab: amr.MFIter(mfab)

    amr.MultiFab.to_numpy = lambda self, copy=False, order="F": mf_to_numpy(
        amr, self, copy, order
    )
    amr.MultiFab.to_numpy.__doc__ = mf_to_numpy.__doc__

    amr.MultiFab.to_cupy = mf_to_cupy

    amr.MultiFab.copy = lambda self: copy_multifab(amr, self)