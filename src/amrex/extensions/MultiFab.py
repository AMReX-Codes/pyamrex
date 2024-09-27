"""
This file is part of pyAMReX

Copyright 2024 AMReX community
Authors: Axel Huebl, David Grote
License: BSD-3-Clause-LBNL
"""

import numpy as np

from .Iterator import next

try:
    from mpi4py import MPI as mpi

    comm_world = mpi.COMM_WORLD
    npes = comm_world.Get_size()
except ImportError:
    npes = 1


def mf_to_numpy(self, copy=False, order="F"):
    """
    Provide a NumPy view into a MultiFab.

    This includes ngrow guard cells of each box.

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
        A list of NumPy n-dimensional arrays, for each local block in the
        MultiFab.
    """
    import inspect

    amr = inspect.getmodule(self)

    mf = self
    if copy:
        mf = amr.MultiFab(
            self.box_array(),
            self.dm(),
            self.n_comp,
            self.n_grow_vect,
            amr.MFInfo().set_arena(amr.The_Pinned_Arena()),
            self.factory,
        )
        amr.dtoh_memcpy(mf, self)

    views = []
    for mfi in mf:
        views.append(mf.array(mfi).to_numpy(copy=False, order=order))

    return views


def mf_to_cupy(self, copy=False, order="F"):
    """
    Provide a CuPy view into a MultiFab.

    This includes ngrow guard cells of each box.

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
        A list of CuPy n-dimensional arrays, for each local block in the
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


def mf_to_xp(self, copy=False, order="F"):
    """
    Provide a NumPy or CuPy view into a MultiFab,
    depending on amr.Config.have_gpu .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    This includes ngrow guard cells of each box.

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
    list of xp.array
        A list of NumPy or CuPy n-dimensional arrays, for each local block in the
        MultiFab.
    """
    import inspect

    amr = inspect.getmodule(self)
    return (
        self.to_cupy(copy, order) if amr.Config.have_gpu else self.to_numpy(copy, order)
    )


def copy_multifab(amr, self):
    """
    Create a copy of this MultiFab, using the same Arena.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX

    Returns
    -------
    amrex.MultiFab
        A copy of this MultiFab.
    """
    mf = amr.MultiFab(
        self.box_array(),
        self.dm(),
        self.n_comp,
        self.n_grow_vect,
        amr.MFInfo().set_arena(self.arena),
        self.factory,
    )
    amr.copy_mfab(
        dst=mf,
        src=self,
        srccomp=0,
        dstcomp=0,
        numcomp=self.n_comp,
        nghost=self.n_grow_vect,
    )
    return mf


def imesh(self, idir, include_ghosts=False):
    """Returns the integer mesh along the specified direction with the appropriate centering.
    This is the location of the data points in grid cell units.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    direction : integer
        Zero based direction number.
        In a typical Cartesian case, 0 would be 'x' direction.
    include_ghosts : bool, default=False
        Whether or not ghost cells are included in the mesh.
    """

    min_box = self.box_array().minimal_box()
    ilo = min_box.small_end[idir]
    ihi = min_box.big_end[idir]

    if include_ghosts:
        # The ghost cells are added to the upper and lower end of the global domain.
        nghosts = self.n_grow_vect
        ilo -= nghosts[idir]
        ihi += nghosts[idir]

    # The centering shift
    ix_type = self.box_array().ix_type()
    if ix_type.node_centered(idir):
        # node centered
        shift = 0.0
    else:
        # cell centered
        shift = 0.5

    return np.arange(ilo, ihi + 1) + shift


def shape(self, include_ghosts=False):
    """Returns the shape of the global array

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    include_ghosts : bool, default=False
        Whether or not ghost cells are included
    """
    min_box = self.box_array().minimal_box()
    result = min_box.size
    if include_ghosts:
        result = result + self.n_grow_vect * 2
    result = list(result) + [self.nComp]
    return tuple(result)


def shape_with_ghosts(self):
    """Returns the shape of the global array including ghost cells

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    """
    return shape(self, include_ghosts=True)


def _get_indices(index, missing):
    """Expand the index list to length three.

    Parameters
    ----------
    index: sequence of length dims
        The indices for each dim

    missing:
        The value used to fill in the extra dimensions added
    """
    return list(index) + (3 - len(index)) * [missing]


def _get_min_indices(self, include_ghosts):
    """Returns the minimum indices, expanded to length 3

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    include_ghosts : bool, default=False
        Whether or not ghost cells are included
    """
    min_box = self.box_array().minimal_box()
    if include_ghosts:
        min_box.grow(self.n_grow_vect)
    return _get_indices(min_box.small_end, 0)


def _get_max_indices(self, include_ghosts):
    """Returns the maximum indices, expanded to length 3.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    include_ghosts : bool, default=False
        Whether or not ghost cells are included
    """
    min_box = self.box_array().minimal_box()
    if include_ghosts:
        min_box.grow(self.n_grow_vect)
    return _get_indices(min_box.big_end, 0)


def _fix_index(self, ii, imax, d, include_ghosts):
    """Handle negative index, wrapping them as needed.
    When ghost cells are included, the indices are
    shifted by the number of ghost cells before being wrapped.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    ii : integer
        The index to be wrapped
    imax : integer
        The maximum value that the index could have
    d : integer
        The direction of the index
    include_ghosts: bool
        Whether or not ghost cells are included
    """
    nghosts = list(_get_indices(self.n_grow_vect, 0)) + [0]
    if include_ghosts:
        ii += nghosts[d]
    if ii < 0:
        ii += imax
    if include_ghosts:
        ii -= nghosts[d]
    return ii


def _find_start_stop(self, ii, imin, imax, d, include_ghosts):
    """Given the input index, calculate the start and stop range of the indices.

    Parameters
    ----------
    ii : None, slice, or integer
        Input index, either None, a slice object, or an integer.
        Note that ii can be negative.
    imin : integer
        The global lowest lower bound in the specified direction.
        This can include the ghost cells.
    imax : integer
        The global highest upper bound in the specified direction.
        This can include the ghost cells.
        This should be the max index + 1.
    d : integer
        The dimension number, 0, 1, 2, or 3 (3 being the components)
    include_ghosts : bool
        Whether or not ghost cells are included

    If ii is a slice, the start and stop values are used directly,
    unless they are None, then the lower or upper bound is used.
    An assertion checks if the indices are within the bounds.
    """
    if ii is None:
        iistart = imin
        iistop = imax
    elif isinstance(ii, slice):
        if ii.start is None:
            iistart = imin
        else:
            iistart = _fix_index(self, ii.start, imax, d, include_ghosts)
        if ii.stop is None:
            iistop = imax
        else:
            iistop = _fix_index(self, ii.stop, imax, d, include_ghosts)
    else:
        ii = _fix_index(self, ii, imax, d, include_ghosts)
        iistart = ii
        iistop = ii + 1
    assert imin <= iistart <= imax, Exception(
        f"Dimension {d+1} lower index is out of bounds"
    )
    assert imin <= iistop <= imax, Exception(
        f"Dimension {d+1} upper index is out of bounds"
    )
    return iistart, iistop


def _get_field(self, mfi, include_ghosts):
    """Return the field at the given mfi.
    If include ghosts is true, return the whole array, otherwise
    return the interior slice that does not include the ghosts.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    mfi : amrex.MFIiter
        Index to the FAB of the MultiFab
    include_ghosts : bool, default=False
        Whether or not ghost cells are included
    """
    # Note that the array will always have 4 dimensions.
    # even when dims < 3.
    # The transpose is taken since the Python array interface to Array4 in
    # self.array(mfi) is in C ordering.
    # Note: transposing creates a view and not a copy.
    import inspect

    amr = inspect.getmodule(self)
    if amr.Config.have_gpu:
        device_arr = self.array(mfi).to_cupy(copy=False, order="F")
    else:
        device_arr = self.array(mfi).to_numpy(copy=False, order="F")
    if not include_ghosts:
        device_arr = device_arr[tuple([slice(ng, -ng) for ng in self.n_grow_vect])]
    return device_arr


def _get_intersect_slice(
    self, mfi, starts, stops, icstart, icstop, include_ghosts, with_internal_ghosts
):
    """Return the slices where the block intersects with the global slice.
    If the block does not intersect, return None.
    This also shifts the block slices by the number of ghost cells in the
    MultiFab arrays since the arrays include the ghost cells.

    Parameters
    ----------
    mfi : MFIter
        The MFIter instance for the current block,
    starts : sequence
        The minimum indices of the global slice.
        These can be negative.
    stops : sequence
        The maximum indices of the global slice.
        These can be negative.
    icstart : integer
        The minimum component index of the global slice.
        These can be negative.
    icstops : integer
        The maximum component index of the global slice.
        These can be negative.
    include_ghosts : bool, default=False
        Whether or not ghost cells are included
    with_internal_ghosts: bool
        Whether the internal ghosts are included in the slices

    Returns
    -------
    block_slices : tuple or None
        The slices of the intersections relative to the block
    global_slices : tuple or None
        The slices of the intersections relative to the global array where the data from individual block will go
    """
    box = mfi.tilebox()
    box_small_end = box.small_end
    box_big_end = box.big_end
    if include_ghosts:
        nghosts = self.mf.n_grow_vect
        box.grow(nghosts)
        if with_internal_ghosts:
            box_small_end = box.small_end
            box_big_end = box.big_end
        else:
            min_box = self.box_array().minimal_box()
            for i in range(len(nghosts)):
                if box_small_end[i] == min_box.small_end[i]:
                    box_small_end[i] -= nghosts[i]
                if box_big_end[i] == min_box.big_end[i]:
                    box_big_end[i] += nghosts[i]

    boxlo = _get_indices(box.small_end, 0)
    ilo = _get_indices(box_small_end, 0)
    ihi = _get_indices(box_big_end, 0)

    # Add 1 to the upper end to be consistent with the slicing notation
    ihi_p1 = [i + 1 for i in ihi]
    i1 = np.maximum(starts, ilo)
    i2 = np.minimum(stops, ihi_p1)

    if np.all(i1 < i2):
        block_slices = []
        global_slices = []
        for i in range(3):
            block_slices.append(slice(i1[i] - boxlo[i], i2[i] - boxlo[i]))
            global_slices.append(slice(i1[i] - starts[i], i2[i] - starts[i]))

        block_slices.append(slice(icstart, icstop))
        global_slices.append(slice(0, icstop - icstart))

        return tuple(block_slices), tuple(global_slices)
    else:
        return None, None


def __getitem__(self, index):
    """Returns slice of the MultiFab using global indexing.
    The shape of the object returned depends on the number of ix, iy and iz specified, which
    can be from none to all three. Note that the values of ix, iy and iz are
    relative to the fortran indexing, meaning that 0 is the lower boundary
    of the whole domain, and in fortran ordering, i.e. [ix,iy,iz].
    This allows negative indexing, though with ghosts cells included, the first n-ghost negative
    indices will refer to the lower guard cells.

    Parameters
    ----------
    index : integer, or sequence of integers or slices, or Ellipsis
        Index of the slice to return
    """
    # Temporary value until fixed
    include_ghosts = False

    # Get the number of dimensions. Is there a cleaner way to do this?
    dims = len(self.n_grow_vect)

    # Note that the index can have negative values (which wrap around) and has 1 added to the upper
    # limit using python style slicing
    if index == Ellipsis:
        index = dims * [slice(None)]
    elif isinstance(index, slice):
        # If only one slice passed in, it was not wrapped in a list
        index = [index]

    if len(index) < dims + 1:
        # Add extra dims to index, including for the component.
        # These are the dims left out and assumed to extend over the full size of the dim
        index = list(index)
        while len(index) < dims + 1:
            index.append(slice(None))
    elif len(index) > dims + 1:
        raise Exception("Too many indices given")

    # Expand the indices to length 3
    ii = _get_indices(index, None)
    ic = index[-1]

    # Global extent. These include the ghost cells when include_ghosts is True
    ixmin, iymin, izmin = _get_min_indices(self, include_ghosts)
    ixmax, iymax, izmax = _get_max_indices(self, include_ghosts)

    # Setup the size of the array to be returned
    ixstart, ixstop = _find_start_stop(self, ii[0], ixmin, ixmax + 1, 0, include_ghosts)
    iystart, iystop = _find_start_stop(self, ii[1], iymin, iymax + 1, 1, include_ghosts)
    izstart, izstop = _find_start_stop(self, ii[2], izmin, izmax + 1, 2, include_ghosts)
    icstart, icstop = _find_start_stop(self, ic, 0, self.n_comp, 3, include_ghosts)

    # Gather the data to be included in a list to be sent to other processes
    starts = [ixstart, iystart, izstart]
    stops = [ixstop, iystop, izstop]
    datalist = []
    for mfi in self:
        block_slices, global_slices = _get_intersect_slice(
            self, mfi, starts, stops, icstart, icstop, include_ghosts, False
        )
        if global_slices is not None:
            # Note that the array will always have 4 dimensions.
            device_arr = _get_field(self, mfi, include_ghosts)
            slice_arr = device_arr[block_slices]
            try:
                # Copy data from host to device using cupy syntax
                slice_arr = slice_arr.get()
            except AttributeError:
                # Array is already a numpy array on the host
                pass
            datalist.append((global_slices, slice_arr))

    # Gather the data from all processors
    if npes == 1:
        all_datalist = [datalist]
    else:
        all_datalist = comm_world.allgather(datalist)

    # Create the array to be returned
    result_shape = (
        max(0, ixstop - ixstart),
        max(0, iystop - iystart),
        max(0, izstop - izstart),
        max(0, icstop - icstart),
    )

    # Now, copy the data into the result array
    result_global = None
    for datalist in all_datalist:
        for global_slices, f_arr in datalist:
            if result_global is None:
                # Delay allocation to here so that the type can be obtained
                result_global = np.zeros(result_shape, dtype=f_arr.dtype)
            result_global[global_slices] = f_arr

    if result_global is None:
        # Something went wrong with the index and no data was found. Return an empty array.
        result_global = np.zeros(0)

    # Remove dimensions of length 1, and if all dimensions
    # are removed, return a scalar (that's what the [()] does)
    return result_global.squeeze()[()]


def __setitem__(self, index, value):
    """Sets slices of a decomposed array using global indexing.
    The shape of the input object depends on the number of arguments specified, which can
    be from none to all three.
    This allows negative indexing, though with ghosts cells included, the first n-ghost negative
    indices will refer to the lower guard cells.

    Parameters
    ----------
    index : integer, or sequence of integers or slices, or Ellipsis
        The slice to set
    value : scalar or array
        Input value to assign to the specified slice of the MultiFab
    """
    # Temporary value until fixed
    include_ghosts = False
    # Get the number of dimensions. Is there a cleaner way to do this?
    dims = len(self.n_grow_vect)

    # Note that the index can have negative values (which wrap around) and has 1 added to the upper
    # limit using python style slicing
    if index == Ellipsis:
        index = tuple(dims * [slice(None)])
    elif isinstance(index, slice):
        # If only one slice passed in, it was not wrapped in a list
        index = [index]

    if len(index) < dims + 1:
        # Add extra dims to index, including for the component.
        # These are the dims left out and assumed to extend over the full size of the dim.
        index = list(index)
        while len(index) < dims + 1:
            index.append(slice(None))
    elif len(index) > dims + 1:
        raise Exception("Too many indices given")

    # Expand the indices to length 3
    ii = _get_indices(index, None)
    ic = index[-1]

    # Global extent. These include the ghost cells when include_ghosts is True
    ixmin, iymin, izmin = _get_min_indices(self, include_ghosts)
    ixmax, iymax, izmax = _get_max_indices(self, include_ghosts)

    # Setup the size of the global array to be set
    ixstart, ixstop = _find_start_stop(self, ii[0], ixmin, ixmax + 1, 0, include_ghosts)
    iystart, iystop = _find_start_stop(self, ii[1], iymin, iymax + 1, 1, include_ghosts)
    izstart, izstop = _find_start_stop(self, ii[2], izmin, izmax + 1, 2, include_ghosts)
    icstart, icstop = _find_start_stop(self, ic, 0, self.n_comp, 3, include_ghosts)

    if isinstance(value, np.ndarray):
        # Expand the shape of the input array to match the shape of the global array
        # (it needs to be 4-D).
        # This converts value to an array if needed, and the [...] grabs a view so
        # that the shape change below doesn't affect value.
        value3d = np.array(value)[...]
        global_shape = list(value3d.shape)
        # The shape of 1 is added for the extra dimensions and when index is an integer
        # (in which case the dimension was not in the input array).
        if not isinstance(ii[0], slice):
            global_shape[0:0] = [1]
        if not isinstance(ii[1], slice):
            global_shape[1:1] = [1]
        if not isinstance(ii[2], slice):
            global_shape[2:2] = [1]
        if not isinstance(ic, slice) or len(global_shape) < 4:
            global_shape[3:3] = [1]
        value3d.shape = global_shape

        if libwarpx.libwarpx_so.Config.have_gpu:
            # check if cupy is available for use
            xp, cupy_status = load_cupy()
            if cupy_status is not None:
                libwarpx.amr.Print(cupy_status)

    starts = [ixstart, iystart, izstart]
    stops = [ixstop, iystop, izstop]
    for mfi in self:
        block_slices, global_slices = _get_intersect_slice(
            self, mfi, starts, stops, icstart, icstop, include_ghosts, True
        )
        if global_slices is not None:
            mf_arr = _get_field(self, mfi, include_ghosts)
            if isinstance(value, np.ndarray):
                # The data is copied from host to device automatically if needed
                mf_arr[block_slices] = value3d[global_slices]
            else:
                mf_arr[block_slices] = value


def register_MultiFab_extension(amr):
    """MultiFab helper methods"""

    # register member functions for the MFIter type
    amr.MFIter.__next__ = next

    # FabArrayBase: iterate as data access in Box index space
    amr.FabArrayBase.__iter__ = lambda fab: amr.MFIter(fab)

    # register member functions for the MultiFab type
    amr.MultiFab.__iter__ = lambda mfab: amr.MFIter(mfab)

    amr.MultiFab.to_numpy = mf_to_numpy
    amr.MultiFab.to_cupy = mf_to_cupy
    amr.MultiFab.to_xp = mf_to_xp

    amr.MultiFab.copy = lambda self: copy_multifab(amr, self)
    amr.MultiFab.copy.__doc__ = copy_multifab.__doc__

    amr.MultiFab.imesh = imesh
    amr.MultiFab.shape = property(shape)
    amr.MultiFab.shape_with_ghosts = property(shape_with_ghosts)
    amr.MultiFab.__getitem__ = __getitem__
    amr.MultiFab.__setitem__ = __setitem__
