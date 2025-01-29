"""
This file is part of pyAMReX

Copyright 2024 AMReX community
Authors: Axel Huebl, David Grote
License: BSD-3-Clause-LBNL
"""

import numpy as np

from .Iterator import next


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
        mf_type = type(self)  # MultiFab or iMultiFab
        mf = mf_type(
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
    mf_type = type(self)  # MultiFab or iMultiFab
    mf = mf_type(
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
    """Returns the minimum indices, expanded to length 3+1

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
    return _get_indices(min_box.small_end, 0) + [0]


def _get_max_indices(self, include_ghosts):
    """Returns the maximum indices, expanded to length 3+1

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
    return _get_indices(min_box.big_end, 0) + [self.n_comp - 1]


def _handle_imaginary_negative_index(index, imin, imax):
    """This convects imaginary and negative indices to the actual value"""
    if isinstance(index, complex):
        assert index.real == 0.0, "Ghost indices must be purely imaginary"
        ii = int(index.imag)
        if ii <= 0:
            result = imin + ii
        else:
            result = imax + ii
    elif index < 0:
        result = index + (imax - imin) + 1
    else:
        result = index
    return result


def _process_index(self, index):
    """Convert the input index into a list of slices"""
    # Get the number of dimensions. Is there a cleaner way to do this?
    dims = len(self.n_grow_vect)

    if index == Ellipsis:
        index = []  # This will be filled in below to cover the valid cells
    elif isinstance(index, slice) or isinstance(index, int):
        # If only one slice or integer passed in, it was not wrapped in a tuple
        index = [index]
    elif isinstance(index, tuple):
        if len(index) == 0:
            # The empty tuple specifies all valid and ghost cells
            index = [index]
        else:
            index = list(index)
            for i in range(len(index)):
                if index[i] == Ellipsis:
                    index = (
                        index[:i]
                        + (dims + 2 - len(index)) * [slice(None)]
                        + index[i + 1 :]
                    )
                    break
    else:
        raise Exception("MultiFab.__getitem__: unexpected index type")

    if len(index) < dims + 1:
        # Add extra dims to index, including for the component.
        # These are the dims left out and assumed to extend over the valid cells
        index = index + (dims + 1 - len(index)) * [slice(None)]
    elif len(index) > dims + 1:
        raise Exception("Too many indices given")

    # Expand index to length 3 + 1
    index = _get_indices(index[:-1], 0) + [index[-1]]

    mins = _get_min_indices(self, False)
    maxs = _get_max_indices(self, False)
    mins_with_ghost = _get_min_indices(self, True)
    maxs_with_ghost = _get_max_indices(self, True)

    # Replace all None's in the slices with the bounds of the valid cells,
    # handle imaginary indices that specify ghost cells, and adjust negative indices
    for i in range(4):
        if isinstance(index[i], slice):
            if index[i].start is None:
                start = mins[i]
            else:
                start = _handle_imaginary_negative_index(
                    index[i].start, mins[i], maxs[i]
                )
            if index[i].stop is None:
                stop = maxs[i] + 1
            else:
                stop = _handle_imaginary_negative_index(index[i].stop, mins[i], maxs[i])
            index[i] = slice(start, stop, index[i].step)
        elif isinstance(index[i], tuple):
            # The slice includes ghosts
            assert len(index[i]) == 0, (
                "Indicator to include all ghost cells must be an empty tuple"
            )
            index[i] = slice(mins_with_ghost[i], maxs_with_ghost[i] + 1)
        else:
            ii = _handle_imaginary_negative_index(index[i], mins[i], maxs[i])
            assert mins_with_ghost[i] <= ii and ii <= maxs_with_ghost[i], (
                "Index out of range"
            )
            index[i] = slice(ii, ii + 1)

    return index


def _get_field(self, mfi):
    """Return the field at the given mfi.
    If the field is on a GPU, a cupy reference to it is returned,
    otherwise a numpy reference.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    mfi : amrex.MFIiter
        Index to the FAB of the MultiFab
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
    return device_arr


def _get_intersect_slice(self, mfi, index, with_internal_ghosts):
    """Return the slices where the block intersects with the global slice.
    If the block does not intersect, return None.
    This also shifts the block slices by the number of ghost cells in the
    MultiFab arrays since the arrays include the ghost cells.

    Parameters
    ----------
    self : amrex.MultiFab
        A MultiFab class in pyAMReX
    mfi : MFIter
        The MFIter instance for the current block,
    index : sequence
        The list indices of the global slice.
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

    # This always include ghost cells since they are included in the MF arrays
    nghosts = self.n_grow_vect
    box.grow(nghosts)
    if with_internal_ghosts:
        box_small_end = box.small_end
        box_big_end = box.big_end
    else:
        # Only expand the box to include the ghost cells at the edge of the domain
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
    i1 = np.maximum([index[0].start, index[1].start, index[2].start], ilo)
    i2 = np.minimum([index[0].stop, index[1].stop, index[2].stop], ihi_p1)

    if np.all(i1 < i2):
        block_slices = [slice(i1[i] - boxlo[i], i2[i] - boxlo[i]) for i in range(3)]
        global_slices = [
            slice(i1[i] - index[i].start, i2[i] - index[i].start) for i in range(3)
        ]

        block_slices.append(index[3])
        global_slices.append(slice(0, index[3].stop - index[3].start))

        return tuple(block_slices), tuple(global_slices)
    else:
        return None, None


def __getitem__(self, index, with_internal_ghosts=False):
    """Returns slice of the MultiFab using global indexing, as a numpy array.
    This uses numpy array indexing, with the indexing relative to the global array.
    The slice ranges can cross multiple blocks and the result will be gathered into a single
    array.

    In an MPI context, this is a global operation. An "allgather" is performed so that the full
    result is returned on all processors.

    Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

    The default range of the indices includes only the valid cells. The ":" index will include all of
    the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
    negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
    The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
    ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

    Parameters
    ----------
    index : the index using numpy style indexing
        Index of the slice to return.
    with_internal_ghosts : bool, optional
        Whether to include internal ghost cells. When true, data from ghost cells may be used that
        overlaps valid cells.
    """
    index4 = _process_index(self, index)

    # Gather the data to be included in a list to be sent to other processes
    datalist = []
    for mfi in self:
        block_slices, global_slices = _get_intersect_slice(
            self, mfi, index4, with_internal_ghosts
        )
        if global_slices is not None:
            # Note that the array will always have 4 dimensions.
            device_arr = _get_field(self, mfi)
            slice_arr = device_arr[block_slices]
            try:
                # Copy data from host to device using cupy syntax
                slice_arr = slice_arr.get()
            except AttributeError:
                # Array is already a numpy array on the host
                pass
            datalist.append((global_slices, slice_arr))

    # Gather the data from all processors
    import inspect

    amr = inspect.getmodule(self)
    if amr.Config.have_mpi:
        npes = amr.ParallelDescriptor.NProcs()
    else:
        npes = 1
    if npes == 1:
        all_datalist = [datalist]
    else:
        try:
            from mpi4py import MPI as mpi

            comm_world = mpi.COMM_WORLD
        except ImportError:
            raise Exception("MultiFab.__getitem__ requires mpi4py")
        all_datalist = comm_world.allgather(datalist)

    # The shape of the array to be returned
    result_shape = tuple([max(0, ii.stop - ii.start) for ii in index4])

    # If the boxes do not fill the domain, then include the internal ghost
    # cells since they may cover internal regions not otherwise covered by valid cells.
    # If the domain is not completely covered, __getitem__ is done twice, the first time
    # including internal ghost cells, and the second time without. The second time is needed
    # to ensure that in places where ghost cells overlap with valid cells, that the data
    # from the valid cells is used.
    # The domain is complete if the number of cells in the box array is the same as
    # the number of cells in the minimal box.
    cell_centered_box_array = self.box_array().enclosed_cells()
    domain_complete = (
        cell_centered_box_array.numPts == cell_centered_box_array.minimal_box().numPts()
    )

    if domain_complete or with_internal_ghosts:
        result_global = None
    else:
        # First get the data including the internal ghost cells
        result_global = self.__getitem__(index, with_internal_ghosts=True)

    # Now, copy the data into the result array
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
    if with_internal_ghosts:
        # Return the data without the squeeze so that the result can be used in the loop
        # above again.
        return result_global
    else:
        return result_global.squeeze()[()]


def __setitem__(self, index, value):
    """Sets the slice of the MultiFab using global indexing.
    This uses numpy array indexing, with the indexing relative to the global array.
    The slice ranges can cross multiple blocks and the value will be distributed accordingly.
    Note that this will apply the value to both valid and ghost cells.

    In an MPI context, this is a local operation. On each processor, the blocks within the slice
    range will be set to the value.

    Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

    The default range of the indices includes only the valid cells. The ":" index will include all of
    the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
    negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
    The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
    ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

    Parameters
    ----------
    index : the index using numpy style indexing
        Index of the slice to return.
    value : scalar or array
        Input value to assign to the specified slice of the MultiFab
    """
    index = _process_index(self, index)

    if isinstance(value, np.ndarray):
        # Expand the shape of the input array to match the shape of the global array
        # (it needs to be 4-D).
        # This converts value to an array if needed, and the [...] grabs a view so
        # that the shape change below doesn't affect value.
        value3d = np.array(value)[...]
        global_shape = list(value3d.shape)
        # The shape of 1 is added for the extra dimensions and when index is an integer
        # (in which case the dimension was not in the input array).
        if (index[0].stop - index[0].start) == 1:
            global_shape[0:0] = [1]
        if (index[1].stop - index[1].start) == 1:
            global_shape[1:1] = [1]
        if (index[2].stop - index[2].start) == 1:
            global_shape[2:2] = [1]
        if (index[3].stop - index[3].start) == 1 or len(global_shape) < 4:
            global_shape[3:3] = [1]
        value3d.shape = global_shape

    for mfi in self:
        block_slices, global_slices = _get_intersect_slice(self, mfi, index, True)
        if global_slices is not None:
            mf_arr = _get_field(self, mfi)
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

    # iMultiFab
    amr.iMultiFab.__iter__ = lambda imfab: amr.MFIter(imfab)

    amr.iMultiFab.to_numpy = mf_to_numpy
    amr.iMultiFab.to_cupy = mf_to_cupy
    amr.iMultiFab.to_xp = mf_to_xp

    amr.iMultiFab.copy = lambda self: copy_multifab(amr, self)
    amr.iMultiFab.copy.__doc__ = copy_multifab.__doc__

    amr.iMultiFab.imesh = imesh
    amr.iMultiFab.shape = property(shape)
    amr.iMultiFab.shape_with_ghosts = property(shape_with_ghosts)
    amr.iMultiFab.__getitem__ = __getitem__
    amr.iMultiFab.__setitem__ = __setitem__
