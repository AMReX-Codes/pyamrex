"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

import os
import warnings

from .Iterator import getitem, next


def iterator(self, *args, level=None):
    """
    Create an iterator over all particle tiles

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A ParticleContainer class in pyAMReX
    args : deprecated positional argument
    level : int | str, optional
        The MR level. Allowed values are [0:self.finest_level+1) and "all".
        If there is more than one MR level, the argument is required.

    Returns
    -------
    amrex.ParIter_*
        Iterator over all particle tiles at the specified level.

    Examples
    --------
    >>> pc.iterator(level="all")
    >>> pc.iterator(level=0)  # only particles on the the coarsest MR level
    """
    # Warn if a second positional argument is provided (ignored argument)
    if len(args) > 0:
        if len(args) == 1 and isinstance(args[0], int) and level is None:
            level = args[0]
        else:
            warnings.warn(
                "The second positional argument to iterator() is deprecated and ignored. "
                "Please update your code to use iterator(self, level=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    has_mr = self.finest_level > 0

    if level is None:
        if has_mr:
            raise ValueError(
                "level must be specified for multi-level ParticleContainers"
            )
        else:
            level = 0

    if level == "all":
        raise ValueError("level='all' is not yet supported for ParticleContainers")
        # TODO: This does not work
        # for lvl in range(self.finest_level + 1):
        #     yield self.Iterator(self, level=lvl)
    elif isinstance(level, int) and level >= 0:
        return self.Iterator(self, level=level)
    else:
        raise ValueError(
            f"level must be an integer in [0:{self.finest_level + 1}) or 'all', but got: {level}"
        )


def pc_to_df(self, local=True, comm=None, root_rank=None):
    """
    Copy all particles into a pandas.DataFrame

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A ParticleContainer class in pyAMReX
    local : bool
        MPI rank-local particles only
    comm : MPI Communicator
        if local is False, the mpi4py communicator to gather with.
        Defaults to the communicator of AMReX.
    root_rank : int
        if local is False, the MPI rank to gather to.
        Defaults to the I/O rank of AMReX (``ParallelDescriptor.IOProcessorNumber()``).

    Returns
    -------
    A concatenated pandas.DataFrame with particles from all levels, one
    column per component (including ``idcpu``) and a row number index.

    Returns None if no particles were found.
    If local=False, then all ranks but the root_rank will return None.

    Notes
    -----
    The particle ids are in the ``idcpu`` column, not in the DataFrame
    index: the index only numbers the rows. ``idcpu`` is unique within one
    snapshot, but not when DataFrames of several steps are combined, e.g.,
    to track particles. Use ``amrex.unpack_ids`` and ``amrex.unpack_cpus``
    to split ``idcpu`` into the id and the cpu number of each particle.
    """
    import pandas as pd

    amr = _amrex_module(self)

    # silently ignore local=False for non-MPI runs
    if not local and not amr.Config.have_mpi:
        local = True

    # create a DataFrame per particle box and append it to the list of
    # local DataFrame(s)
    dfs_local = []
    for lvl in range(self.finest_level + 1):
        for pti in self.const_iterator(level=lvl):
            if pti.size == 0:
                continue

            if self.is_soa_particle:
                soa_view = pti.soa().to_numpy(copy=True)

                next_df = pd.DataFrame()

                next_df["idcpu"] = soa_view.idcpu

                soa_np_real = soa_view.real
                for name, array in soa_np_real.items():
                    next_df[name] = array

                soa_np_int = soa_view.int
                for name, array in soa_np_int.items():
                    next_df[name] = array
            else:
                # AoS
                aos_np = pti.aos().to_numpy(copy=True)
                next_df = pd.DataFrame(aos_np)

                # SoA
                soa_view = pti.soa().to_numpy(copy=True)
                soa_np_real = soa_view.real
                soa_np_int = soa_view.int

                for name, array in soa_np_real.items():
                    next_df[f"SoA_{name}"] = array

                for name, array in soa_np_int.items():
                    next_df[f"SoA_{name}"] = array

            dfs_local.append(next_df)

    # MPI Gather to root rank if requested
    if local:
        if len(dfs_local) == 0:
            df = None
        else:
            df = pd.concat(dfs_local, ignore_index=True)
    else:
        if comm is None:
            comm = _amrex_comm(amr)
        if root_rank is None:
            root_rank = amr.ParallelDescriptor.IOProcessorNumber()
        rank = comm.Get_rank()

        # a list for each rank's list of DataFrame(s)
        df_list_list = comm.gather(dfs_local, root=root_rank)

        if rank == root_rank:
            flattened_list = [df for sublist in df_list_list for df in sublist]

            if len(flattened_list) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(flattened_list, ignore_index=True)
        else:
            df = None

    return df


def _amrex_module(obj):
    """The pyAMReX module (e.g., amrex.space3d) of a pyAMReX object."""
    from .dlpack_helpers import _pyamrex_module

    amr = _pyamrex_module(obj)
    if amr is None:
        raise TypeError(f"{type(obj).__name__} is not a pyAMReX type")
    return amr


def _amrex_comm(amr):
    """The MPI communicator of AMReX, as an mpi4py communicator."""
    from mpi4py import MPI

    return MPI.Comm.f2py(amr.ParallelDescriptor.Communicator())


def _component_dtypes(pc):
    """The NumPy dtype of every component of a pure SoA container, by name."""
    import numpy as np

    amr = _amrex_module(pc)
    real = np.float64 if amr.Config.precision_particles == "DOUBLE" else np.float32
    dtypes = {"idcpu": np.dtype(np.uint64)}
    dtypes.update({name: np.dtype(real) for name in pc.real_soa_names})
    dtypes.update({name: np.dtype(np.int32) for name in pc.int_soa_names})
    return dtypes


def _is_scalar(value):
    """Python and NumPy numbers, and 0-d NumPy, CuPy or dpnp arrays."""
    import numbers

    import numpy as np

    return isinstance(value, (numbers.Number, np.generic)) or (
        getattr(value, "ndim", None) == 0 and hasattr(value, "item")
    )


def _scalar_value(value):
    """A scalar as a Python number (copied to the host for device arrays)."""
    return value.item() if hasattr(value, "item") else value


def _array_module(kind):
    """The array module of a prepared column: numpy, cupy or dpnp."""
    if kind == "cupy":
        import cupy

        return cupy
    if kind == "dpnp":
        import dpnp

        return dpnp
    import numpy

    return numpy


def _to_host(kind, src):
    """Copy a column prepared by PODVector's _prepare_source to a NumPy array."""
    import numpy as np

    if kind == "podvector":
        # AMReX copy through pinned memory (a view for host memory)
        return src.to_numpy(copy=True) if src.size() > 0 else np.empty(0)
    if kind == "cupy":
        import cupy as cp

        return cp.asnumpy(src)
    if kind == "dpnp":
        import dpnp as dp

        return dp.asnumpy(src)
    return src


def _byte_bounds(kind, src):
    """Byte address range ``(lo, hi)`` of a prepared 1-D column or PODVector.

    Returns None for empty data. The addresses come from the array
    interfaces, so they are host or device addresses; they are only compared
    to detect overlapping memory.
    """
    import numpy as np

    if kind == "dpnp":
        iface = src.__sycl_usm_array_interface__
    elif kind == "cupy":
        iface = src.__cuda_array_interface__
    else:
        # NumPy arrays and PODVectors (also with device memory)
        iface = src.__array_interface__
    n = iface["shape"][0]
    if n == 0:
        return None
    itemsize = np.dtype(iface["typestr"]).itemsize
    # as unsigned 64 bit: PODVectors export a signed intptr_t, which is
    # negative for high (e.g., SYCL USM) addresses that dpnp reports unsigned
    start = iface["data"][0] % 2**64
    strides = iface.get("strides")
    stride = strides[0] if strides else itemsize
    if kind == "dpnp":
        # the SYCL USM interface counts the offset and strides in elements
        start += iface.get("offset", 0) * itemsize
        stride = strides[0] * itemsize if strides else itemsize
    last = (n - 1) * stride
    return start + min(0, last), start + max(0, last) + itemsize


def _overlaps(bounds, other_bounds):
    """Whether the byte range bounds overlaps any of other_bounds."""
    if bounds is None:
        return False
    lo, hi = bounds
    return any(b is not None and lo < b[1] and b[0] < hi for b in other_bounds)


def _copy_column(kind, src):
    """A copy of a prepared column, in the same memory."""
    if kind == "podvector":
        from .PODVector import _podvector_base

        return _podvector_base(type(src))(src)
    return src.copy()


def _collect_columns(data, columns):
    """Merge a mapping or DataFrame and keyword columns into one dict."""
    import numpy as np

    merged = {}
    if data is not None:
        if hasattr(data, "keys"):
            names = list(data.keys())
        elif hasattr(data, "columns"):
            names = list(data.columns)
        else:
            raise TypeError(
                "data must be a mapping or a DataFrame of particle columns, "
                f"not {type(data).__name__}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Particle column names must be unique: {names}")
        merged = {name: data[name] for name in names}

        # a DataFrame with the ids as its index, e.g., after df.set_index("idcpu")
        index = getattr(data, "index", None)
        if "idcpu" not in merged and getattr(index, "name", None) == "idcpu":
            merged["idcpu"] = np.asarray(index)

    twice = [name for name in columns if name in merged]
    if twice:
        raise ValueError(
            f"Particle columns given both in data and as keyword arguments: {twice}"
        )
    merged.update(columns)
    return merged


def _has_particle_data(data, columns):
    """Whether data or columns contain particles, i.e., a non-empty array.

    None, scalars and empty mappings, DataFrames or arrays are no particles.
    """
    from .PODVector import _is_podvector

    for value in _collect_columns(data, columns).values():
        if value is None or _is_scalar(value):
            continue
        shape = getattr(value, "shape", None)
        if shape:
            n = shape[0]
        elif _is_podvector(value):
            n = value.size()
        else:
            try:
                n = len(value)
            except TypeError:
                # e.g., a DLPack-only producer: assume it holds particles
                return True
        if n > 0:
            return True
    return False


def _check_integer_range(name, xp, src, dst_dtype):
    """Raise if the values of src do not fit into the integer dst_dtype."""
    import numpy as np

    if src.dtype.kind == "f":
        if not bool(xp.isfinite(src).all()) or not bool((xp.floor(src) == src).all()):
            raise ValueError(
                f"Particle column '{name}': values must be finite integers "
                f"to be stored as {dst_dtype}"
            )
    elif np.can_cast(src.dtype, dst_dtype, "safe"):
        return
    info = np.iinfo(dst_dtype)
    # compare Python numbers: dpnp cannot compare uint64 arrays with
    # negative Python integers
    lo, hi = src.min().item(), src.max().item()
    if lo < info.min or hi > info.max:
        raise ValueError(
            f"Particle column '{name}': values in [{lo}, {hi}] do not fit "
            f"into {dst_dtype}"
        )


def _check_float_range(name, xp, src, dst_dtype):
    """Raise if finite values overflow to infinity in the float dst_dtype."""
    import numpy as np

    if src.dtype.kind != "f" or src.dtype.itemsize <= dst_dtype.itemsize:
        return
    # cheap check first: no copy of the data
    limit = float(np.finfo(dst_dtype).max)
    lo, hi = src.min().item(), src.max().item()
    if -limit <= lo and hi <= limit:
        return
    # infinite or NaN values, or values out of range: check element-wise
    with np.errstate(over="ignore"):
        overflow = xp.isfinite(src) & ~xp.isfinite(src.astype(dst_dtype))
    if bool(overflow.any()):
        raise ValueError(
            f"Particle column '{name}': values overflow the range of {dst_dtype}"
        )


def _check_column(name, kind, src, dst_dtype):
    """Raise if a prepared column cannot be stored without silent data loss.

    Floating point values may be rounded (e.g., to single precision), but not
    overflow; integer values must fit; particle ids are cast to uint64 and
    must be valid. Returns the (possibly converted) ``(kind, src)``.
    """
    from .dlpack_helpers import array_kind
    from .PODVector import _element_type, _have_module, _prepare_array

    if kind == "podvector":
        element = _element_type(type(src))
        if (dst_dtype.kind == "f" and element in ("int", "uint64")) or (
            element == {"f": "real", "i": "int", "u": "uint64"}[dst_dtype.kind]
            and name != "idcpu"
        ):
            # always representable
            return kind, src
        # check the values (e.g., the idcpu valid bits) through a view of
        # the PODVector, or a host copy if CuPy/dpnp is missing
        memory = array_kind(src)
        if memory == "numpy" or _have_module(memory):
            kind, src = _prepare_array(src)
        else:
            kind, src = "numpy", _to_host(kind, src)

    if src.dtype.kind not in "biuf":
        raise TypeError(
            f"Particle column '{name}': cannot store {src.dtype} values as {dst_dtype}"
        )
    if src.shape[0] == 0:
        return kind, src

    xp = _array_module(kind)
    if name == "idcpu":
        if src.dtype.kind not in "iu":
            raise TypeError(
                f"Particle column 'idcpu': expected integers, not {src.dtype}; "
                "floating point numbers cannot hold particle ids exactly"
            )
        # particle ids are 64 bit unsigned integers; signed 64 bit integers
        # are reinterpreted, narrower ones must not be negative (the cast
        # would extend the sign bit into the valid bit)
        if src.dtype.kind == "i" and src.dtype.itemsize < 8 and src.min().item() < 0:
            raise ValueError("Particle column 'idcpu': contains negative values")
        src = src.astype(xp.uint64, copy=False)
        # the leftmost bit marks valid particles; invalid ones are removed
        if not bool(((src >> 63) == 1).all()):
            raise ValueError(
                "Particle column 'idcpu': contains invalid particle ids "
                "(see amrex.make_valid and amrex.pack_ids)"
            )
        return kind, src

    if dst_dtype.kind == "f":
        _check_float_range(name, xp, src, dst_dtype)
    else:
        _check_integer_range(name, xp, src, dst_dtype)
    return kind, src


def _check_scalar(name, value, dst_dtype):
    """Raise if a scalar cannot be stored without silent data loss."""
    import math
    import numbers

    import numpy as np

    if name == "idcpu":
        raise ValueError("Particle column 'idcpu' cannot be a scalar")
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"Particle column '{name}': cannot store {value!r} as {dst_dtype}"
        )
    # Python ints can exceed the float range: they are always finite
    finite = isinstance(value, numbers.Integral) or math.isfinite(value)
    if dst_dtype.kind == "f":
        # a Python float: compares exactly with Python ints of any size
        if finite and abs(value) > float(np.finfo(dst_dtype).max):
            raise ValueError(
                f"Particle column '{name}': {value!r} overflows the range of {dst_dtype}"
            )
        return
    info = np.iinfo(dst_dtype)
    if not finite or value != int(value) or not info.min <= value <= info.max:
        raise ValueError(
            f"Particle column '{name}': {value!r} is not an integer that fits "
            f"into {dst_dtype}"
        )


def _normalize_columns(pc, merged, fill_missing):
    """Validate columns against the pure SoA layout of a particle container.

    Array columns are prepared for PODVector.copy_from and checked, so that
    invalid data raises here, before any particle container is changed.
    Returns the prepared ``name: (kind, src)`` array columns, the number of
    particles, and the scalar columns as ``name: Python number``. No columns
    means no particles.
    """
    from .PODVector import _prepare_source, _source_size

    if fill_missing is not None and not _is_scalar(fill_missing):
        raise TypeError(f"fill_missing must be a scalar, not {fill_missing!r}")
    if not merged:
        return {}, 0, {}

    dtypes = _component_dtypes(pc)
    required = list(pc.real_soa_names) + list(pc.int_soa_names)

    unknown = [name for name in merged if name not in dtypes]
    if unknown:
        hint = ""
        if set(unknown) & {"data", "df"}:
            hint = " Pass a mapping or DataFrame as the first positional argument."
        if "redistribute" in unknown:
            hint += (
                " To add particles without redistributing them, use "
                "distribute='none' (or 'equally' with local=False)."
            )
        raise ValueError(
            f"Unknown particle columns {unknown}; expected {required} "
            f"and optionally 'idcpu'.{hint}"
        )
    missing = [name for name in required if name not in merged]
    if missing and fill_missing is None:
        raise KeyError(
            f"Missing particle columns {missing}; pass them, or set "
            "fill_missing to a value for all missing columns."
        )

    scalars = {
        name: _scalar_value(value)
        for name, value in merged.items()
        if _is_scalar(value)
    }
    scalars.update({name: _scalar_value(fill_missing) for name in missing})
    for name, value in scalars.items():
        _check_scalar(name, value, dtypes[name])

    columns = {}
    for name, value in merged.items():
        if name in scalars:
            continue
        try:
            kind, src = _prepare_source(value)
        except (TypeError, ValueError) as e:
            raise type(e)(f"Particle column '{name}': {e}") from e
        columns[name] = _check_column(name, kind, src, dtypes[name])

    if not columns:
        raise ValueError(
            "At least one particle column must be an array, to define the "
            "number of particles; the others can be scalars."
        )
    lengths = {name: _source_size(*column) for name, column in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Particle columns have unequal lengths: {lengths}")

    return columns, list(lengths.values())[0], scalars


def _split_counts(npart, owners, nranks):
    """Number of particles per rank when splitting npart over the owner ranks.

    Every owner rank gets ``npart // len(owners)`` particles and the first
    ``npart % len(owners)`` owners get one extra, like ImpactX' split_equally.
    Ranks that are not owners get zero particles.
    """
    navg, nleft = divmod(npart, len(owners))
    counts = [0] * nranks
    for k, owner in enumerate(owners):
        counts[owner] = navg + (1 if k < nleft else 0)
    return counts


# the largest count/displacement of an MPI call without MPI-4 large counts
_MAX_MPI_COUNT = 2**31 - 1


def _mpi_large_count():
    """Whether mpi4py and the MPI library support MPI-4 large counts."""
    import mpi4py
    from mpi4py import MPI

    return int(mpi4py.__version__.split(".")[0]) >= 4 and MPI.Get_version() >= (4, 0)


def _scatter_columns(pc, data, columns, fill_missing, comm, root_rank):
    """Scatter the particle columns of root_rank to the ranks that own boxes.

    Collective. The columns are validated and cast to the component types on
    the root; scalar columns are sent as values. Returns this rank's
    ``name: (kind, src)`` array columns, its number of particles, and the
    scalar columns.
    """
    import numpy as np

    rank = comm.Get_rank()
    nranks = comm.Get_size()

    root_error = None
    host_columns = None
    if rank == root_rank:
        try:
            merged = _collect_columns(data, columns)
            prepared, npart, scalars = _normalize_columns(pc, merged, fill_missing)
            dtypes = _component_dtypes(pc)
            # MPI sends host buffers, so device arrays are staged to the host once
            host_columns = {
                name: np.ascontiguousarray(_to_host(kind, src), dtype=dtypes[name])
                for name, (kind, src) in prepared.items()
            }
            if npart > _MAX_MPI_COUNT and not _mpi_large_count():
                raise ValueError(
                    f"Scattering {npart} particles requires MPI-4 large count "
                    "support (mpi4py >= 4 and an MPI-4 library); add the "
                    "particles in batches, or with local=True."
                )
            owners = sorted(set(pc.particle_distribution_map(0).ProcessorMap()))
            meta = {
                "error": None,
                "names": list(host_columns.keys()),
                "dtypes": [col.dtype.str for col in host_columns.values()],
                "counts": _split_counts(npart, owners, nranks),
                "scalars": scalars,
            }
        except Exception as e:
            root_error = e
            meta = {"error": repr(e)}
    else:
        meta = None

    meta = comm.bcast(meta, root=root_rank)
    if meta["error"] is not None:
        if root_error is not None:
            raise root_error
        raise RuntimeError(
            f"add_arrays(local=False) failed on the root rank {root_rank}: {meta['error']}"
        )

    counts = meta["counts"]
    displs = [0] * nranks
    for r in range(1, nranks):
        displs[r] = displs[r - 1] + counts[r - 1]
    npart_local = counts[rank]

    # allocate all receive buffers first and agree on errors (e.g., out of
    # memory), so that no rank is left behind in a Scatterv
    error = None
    local_columns = {}
    try:
        for name, dtype_str in zip(meta["names"], meta["dtypes"]):
            recv = np.empty(npart_local, dtype=np.dtype(dtype_str))
            local_columns[name] = ("numpy", recv)
    except Exception as e:
        error = e
    _raise_on_any_error(comm, error)

    for name, (_, recv) in local_columns.items():
        send = None
        if rank == root_rank:
            send = [host_columns[name], (counts, displs)]
        comm.Scatterv(send, recv, root=root_rank)

    return local_columns, npart_local, meta["scalars"]


def _prepare_insert(pc, columns, npart, scalars):
    """Prepare appending particles to a tile of this rank on level 0.

    The particles go to the first tile of the first box that this MPI rank
    owns on level 0. Everything that can fail on user input happens here or
    before, so that _commit_insert does not. Returns None if there is
    nothing to add.
    """
    import numpy as np

    if npart == 0:
        return None

    amr = _amrex_module(pc)
    rank = amr.ParallelDescriptor.MyProc()

    proc_map = list(pc.particle_distribution_map(0).ProcessorMap())
    if rank not in proc_map:
        raise RuntimeError(
            f"MPI rank {rank} owns no box on level 0 and cannot receive particles. "
            "Pass the particles on a rank that owns a box, or use local=False."
        )
    grid = proc_map.index(rank)
    tile = pc.define_and_return_particle_tile(0, grid, 0)
    soa = tile.get_struct_of_arrays()

    real_names = list(pc.real_soa_names)
    int_names = list(pc.int_soa_names)
    if len(real_names) != soa.num_real_comps or len(int_names) != soa.num_int_comps:
        raise RuntimeError("Particle component names do not match the tile layout.")

    # (component PODVector, name) pairs in the tile layout order; resizing
    # the tile reallocates the data of the PODVectors, not the PODVectors
    destinations = [(soa.get_idcpu_data(), "idcpu")]
    destinations += [(soa.get_real_data(i), name) for i, name in enumerate(real_names)]
    destinations += [(soa.get_int_data(i), name) for i, name in enumerate(int_names)]

    # a column can be a view into this tile, e.g., from pti.soa().to_numpy()
    # or soa.get_real_data(i): resizing the tile in _commit_insert frees its
    # memory before the copy, so copy such columns first. Comparing
    # addresses of different memory spaces can only cause an extra copy.
    destination_bounds = [_byte_bounds("podvector", dst) for dst, _ in destinations]
    columns = {
        name: (
            (kind, _copy_column(kind, src))
            if _overlaps(_byte_bounds(kind, src), destination_bounds)
            else (kind, src)
        )
        for name, (kind, src) in columns.items()
    }

    dtypes = _component_dtypes(pc)
    for name, value in scalars.items():
        columns[name] = ("numpy", np.full(npart, value, dtype=dtypes[name]))

    if "idcpu" not in columns:
        # new ids, unique per rank; the rank as cpu makes them globally unique
        first = type(pc).reserve_particle_ids(npart)
        idcpu = np.zeros(npart, dtype=np.uint64)
        amr.pack_ids(idcpu, np.arange(first, first + npart, dtype=np.int64))
        amr.pack_cpus(idcpu, np.full(npart, rank, dtype=np.int32))
        columns["idcpu"] = ("numpy", idcpu)

    return {
        "tile": tile,
        "destinations": destinations,
        "columns": columns,
        "npart": npart,
        "old_size": tile.size,
    }


def _commit_insert(plan):
    """Append the particles prepared by _prepare_insert to their tile.

    Each component is written like PODVector.copy_from: device-to-device
    for device data into device memory, otherwise through the host with an
    AMReX copy, so CuPy and dpnp are never required. On failure, the tile is
    restored.
    """
    from .PODVector import _copy_prepared

    if plan is None:
        return

    tile = plan["tile"]
    old_size = plan["old_size"]
    tile.resize(old_size + plan["npart"])
    try:
        for dst, name in plan["destinations"]:
            kind, src = plan["columns"][name]
            _copy_prepared(dst, kind, src, old_size)
    except Exception:
        _rollback_insert(plan)
        raise


def _rollback_insert(plan):
    """Remove the particles added by _commit_insert again."""
    if plan is not None:
        plan["tile"].resize(plan["old_size"])


def _check_comm(amr, comm):
    """Raise on all ranks if the ranks of comm do not match the AMReX ranks.

    Collective on comm. AMReX collectives, e.g., redistribute(), use the
    AMReX communicator, so comm must agree on errors for the same ranks.
    """
    from mpi4py import MPI

    mismatch = (
        comm.Get_size() != amr.ParallelDescriptor.NProcs()
        or comm.Get_rank() != amr.ParallelDescriptor.MyProc()
    )
    if comm.allreduce(mismatch, op=MPI.LOR):
        raise ValueError("The ranks of comm must match the AMReX MPI ranks.")


def _raise_on_any_error(comm, error):
    """Collectively raise if any MPI rank had an error.

    The rank(s) with an error re-raise it, all other ranks raise a
    RuntimeError, so no rank continues into later collective calls alone.
    """
    errors = comm.allgather(None if error is None else repr(error))
    if error is not None:
        raise error
    failed = [r for r, e in enumerate(errors) if e is not None]
    if failed:
        raise RuntimeError(
            f"Adding particles failed on MPI rank(s) {failed}: {errors[failed[0]]}"
        )


def pc_add_arrays(
    self,
    data=None,
    /,
    *,
    local=True,
    comm=None,
    root_rank=None,
    distribute="redistribute",
    fill_missing=None,
    **columns,
):
    """
    Add particles from arrays, one per particle component

    This is the counterpart of :py:meth:`to_df`: the column names are the
    same as the ones returned by ``to_df()`` for pure SoA particles.

    Examples
    --------
    >>> pc.add_arrays(x=x, y=y, z=z, w=w)  # one array per component
    >>> pc.add_arrays(x=x, y=y, z=z, w=1.0)  # scalars are broadcast
    >>> pc.add_arrays(df)  # a DataFrame, e.g., from to_df()
    >>> pc.add_arrays({"x": x, ...}, w=1.0)  # a mapping, plus keywords
    >>> pc.add_arrays(x=x, y=y, z=z, fill_missing=0.0)  # others are 0
    >>> pc.add_arrays(df, local=False)  # add df of the I/O rank on all ranks
    >>> pc.add_arrays(local=False)  # ... on the other ranks
    >>> pc.add_arrays(df, local=False, distribute="equally")  # 1/N per rank
    >>> pc.add_arrays(df, distribute="none")  # not collective
    >>> pc.redistribute()  # ... e.g., after adding several batches

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A pure SoA ParticleContainer class in pyAMReX
    data : mapping or DataFrame, optional
        Particle columns by component name, e.g., a dict of arrays or a
        pandas DataFrame. Needed for components whose names are no valid
        Python identifiers or that are also option names of this function.
        A DataFrame index named ``idcpu`` (e.g., after
        ``df.set_index("idcpu")``) provides the ``idcpu`` column.
    local : bool
        If True, every MPI rank adds its own particles, e.g., the chunk of
        a file it read.
        If False, the particles of ``root_rank`` are added; the other ranks
        must pass no particles (None, an empty mapping or DataFrame, or only
        scalars), otherwise all ranks raise a ValueError. This is the
        counterpart of ``to_df(local=False)``.
    comm : MPI Communicator
        The mpi4py communicator used for collective error handling and, if
        local is False, for the scatter. Defaults to the communicator of
        AMReX; its ranks must match the AMReX MPI ranks.
    root_rank : int
        if local is False, the MPI rank that holds the data. Defaults to the
        I/O rank of AMReX (``ParallelDescriptor.IOProcessorNumber()``).
    distribute : str
        Where the new particles go:

        - ``"redistribute"`` (default): call :py:meth:`redistribute`
          afterwards, which moves every particle to the rank, MR level, box
          and tile that owns its position. This is collective. With
          local=False, the root adds all particles and the redistribute
          sends each of them once (if the root owns no box on level 0, they
          are split 1/N over the ranks first).
        - ``"equally"``: only with local=False. Split the particles of the
          root over all MPI ranks that own a box on level 0, without
          :py:meth:`redistribute`: each gets 1/N of them, the first ranks
          one remaining particle more. For particles whose position does
          not decide their rank, e.g., beam particles without space charge.
        - ``"none"``: no :py:meth:`redistribute`; with local=True, this is
          not collective. The particles stay on the adding rank (with
          local=False, the root), until :py:meth:`redistribute` is called,
          e.g., after adding several batches.

        Without redistribute, the particles are in the tile of the first
        box on level 0 that their rank owns.
    fill_missing : scalar, optional
        Value for all components that are not given, except ``idcpu``
        (without an ``idcpu`` column, new ids are always reserved). By
        default, missing components raise a KeyError.
    **columns : array or scalar
        Particle columns by component name, in addition to ``data``.

    Notes
    -----
    All Real and int components of the container (``self.real_soa_names``
    and ``self.int_soa_names``) are required; ``idcpu`` is optional: if
    given, its values are used as-is, otherwise new ids are reserved on the
    adding rank. Unknown columns, and columns given both in ``data`` and as
    keywords, raise a ValueError.

    Columns can be NumPy, CuPy or dpnp arrays, pyAMReX PODVectors,
    array-likes or scalars (broadcast to all particles); at least one must
    be an array. They are copied like :py:meth:`PODVector.copy_from`:
    device-to-device if possible, and never requiring CuPy or dpnp.
    Values are cast to the component types: floating point values may be
    rounded but not overflow; integer values that do not fit, non-integer
    values for int components, and ``idcpu`` values that are not valid
    particle ids raise an error. Integer ``idcpu`` values are cast to
    ``uint64`` (the bits of signed 64 bit integers are kept).

    All MPI ranks must pass the same ``local``, ``distribute``, ``comm``
    and ``root_rank``. Unless local is True and distribute is "none", the
    call is collective; invalid data on any rank then raises on all ranks.
    Reusing an ``idcpu`` column, e.g., when adding the output of
    ``to_df()`` back to the same container, creates duplicate particle ids.

    With local=False and distribute="redistribute" (or "none"), the root
    rank holds all new particles in its (device) memory. If they do not
    fit, add them with ``distribute="equally"`` and call
    :py:meth:`redistribute` afterwards: this splits them over all ranks
    first, at the cost of sending most of them twice. The 1/N split goes
    through host memory, also for device arrays. Legacy AoS particle
    containers are not supported.
    """
    amr = _amrex_module(self)

    if not self.is_soa_particle:
        raise NotImplementedError(
            "add_arrays/add_df only support pure SoA particle containers."
        )

    choices = ("redistribute", "equally", "none")
    if distribute not in choices:
        raise ValueError(f"distribute must be one of {choices}, not {distribute!r}")

    # silently ignore local=False for non-MPI runs: the only rank keeps all
    if not local and not amr.Config.have_mpi:
        local = True
        if distribute == "equally":
            distribute = "none"
    if local and distribute == "equally":
        raise ValueError(
            "distribute='equally' splits the particles of root_rank and needs "
            "local=False; with local=True, every rank keeps its own particles "
            "with distribute='none'."
        )

    redistribute = distribute == "redistribute"
    parallel = amr.Config.have_mpi and amr.ParallelDescriptor.NProcs() > 1
    collective = parallel and (redistribute or not local)
    uses_comm = collective or not local
    if comm is None:
        if uses_comm:
            comm = _amrex_comm(amr)
    elif uses_comm:
        _check_comm(amr, comm)
    if root_rank is None:
        root_rank = amr.ParallelDescriptor.IOProcessorNumber()

    # local=False: the root adds all particles, unless they are split 1/N
    # ("equally"). With redistribute, they are then sent only once; if the
    # root owns no box, they are split 1/N before the redistribute instead.
    root_adds = False
    if not local:
        nranks = comm.Get_size()
        if not 0 <= root_rank < nranks:
            raise ValueError(
                f"root_rank={root_rank} is not a rank of comm (0..{nranks - 1})"
            )
        root_owns_box = root_rank in set(
            self.particle_distribution_map(0).ProcessorMap()
        )
        if distribute == "none" and not root_owns_box:
            raise ValueError(
                f"root_rank={root_rank} owns no box on level 0 and cannot keep "
                "the particles with distribute='none'; use another root_rank, "
                "or distribute='equally' or 'redistribute'."
            )
        root_adds = distribute != "equally" and root_owns_box

    error = None
    plan = None
    if not local and comm.Get_rank() != root_rank:
        # only the particles of the root are added: data on other ranks
        # would be lost silently, e.g., if local=True was meant
        try:
            if _has_particle_data(data, columns):
                raise ValueError(
                    f"add_arrays(local=False) adds the particles of root_rank="
                    f"{root_rank} only, but MPI rank {comm.Get_rank()} passed "
                    "particles, too. To add the particles of every rank, use "
                    "local=True (it redistributes them, unless "
                    "distribute='none'). With local=False, pass no particles "
                    "on the other ranks."
                )
        except Exception as e:
            error = e

    if local or root_adds:
        if local or comm.Get_rank() == root_rank:
            try:
                merged = _collect_columns(data, columns)
                prepared, npart, scalars = _normalize_columns(
                    self, merged, fill_missing
                )
                plan = _prepare_insert(self, prepared, npart, scalars)
            except Exception as e:
                error = e
    else:
        # agree on errors of the other ranks before the scatter
        _raise_on_any_error(comm, error)
        prepared, npart, scalars = _scatter_columns(
            self, data, columns, fill_missing, comm, root_rank
        )
        try:
            plan = _prepare_insert(self, prepared, npart, scalars)
        except Exception as e:
            error = e

    # agree on errors before changing the container and before the
    # collective redistribute, so that no rank is left behind
    if collective:
        _raise_on_any_error(comm, error)
    elif error is not None:
        raise error

    # the data is validated, but agree again in case of, e.g., out of memory
    error = None
    try:
        _commit_insert(plan)
    except Exception as e:
        error = e
    if collective:
        try:
            _raise_on_any_error(comm, error)
        except Exception:
            if error is None:
                _rollback_insert(plan)
            raise
    elif error is not None:
        raise error

    if redistribute:
        self.redistribute()


def pc_add_df(
    self,
    df=None,
    /,
    *,
    local=True,
    comm=None,
    root_rank=None,
    distribute="redistribute",
    fill_missing=None,
    **columns,
):
    """
    Add particles from a DataFrame

    This is the counterpart of :py:meth:`to_df` and the same as
    :py:meth:`add_arrays` with a DataFrame, e.g., a pandas.DataFrame whose
    columns are the particle components, as returned by ``to_df()``.

    Examples
    --------
    >>> df = pc.to_df(local=False)  # gather all particles to the I/O rank
    >>> pc2.add_df(df, local=False)  # add and redistribute them again

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A pure SoA ParticleContainer class in pyAMReX
    df : DataFrame, optional
        Particles to add, one column per component. The index is ignored,
        unless it is named ``idcpu`` and there is no ``idcpu`` column.
    local, comm, root_rank, distribute, fill_missing, **columns :
        See :py:meth:`add_arrays`.

    See Also
    --------
    add_arrays : details, including collective behavior and particle ids.
    """
    return pc_add_arrays(
        self,
        df,
        local=local,
        comm=comm,
        root_rank=root_rank,
        distribute=distribute,
        fill_missing=fill_missing,
        **columns,
    )


def list_particle_species(plotfile):
    """List the particle species stored in a plotfile or checkpoint.

    Particle data lives in sub-directories of a plotfile, named by the writing
    application (e.g. ``"particles"``, ``"electrons"``, ``"particle0"``). This
    scans the plotfile for sub-directories that contain a particle ``Header``
    file and returns their names, ready to be passed as ``particle_dir`` to
    :py:func:`read_particles` or :py:class:`amrex.ParticleHeader`.

    Parameters
    ----------
    plotfile : str
        Path to the plotfile / checkpoint directory.

    Returns
    -------
    list of str
        Names of the particle species sub-directories, sorted alphabetically.
    """
    import os

    species = []
    for entry in sorted(os.scandir(plotfile), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        header = os.path.join(entry.path, "Header")
        if not os.path.isfile(header):
            continue
        try:
            with open(header) as f:
                first_token = f.readline().strip()
        except OSError:
            continue
        # particle Headers start with a version string like
        # "Version_Two_Dot_One_double"; the mesh Header sits at the top level
        # and mesh level directories (Level_*) contain no Header file
        if first_token.startswith("Version_"):
            species.append(entry.name)
    return species


def read_particles(
    amr, plotfile, particle_dir="particles", communicate=True, container=None
):
    """Read AMReX particle data from a plotfile or checkpoint.

    The on-disk layout (number and names of the real/int components, the
    precision and whether the file is a checkpoint) is discovered via
    :py:class:`amrex.ParticleHeader`, which uses the same AMReX C++ header reader
    as ``ParticleContainer::Restart``. A polymorphic, pure Struct-of-Arrays
    ParticleContainer is then configured with matching runtime components and
    restarted from the file - so no prior knowledge of the container's
    compile-time layout is required.

    Parameters
    ----------
    amr : module
        The dimension-specific pyAMReX module (e.g. ``amrex.space3d``).
    plotfile : str
        Path to the plotfile / checkpoint directory.
    particle_dir : str, optional
        Name of the particle sub-directory inside ``plotfile``
        (default: ``"particles"``).
    communicate : bool, optional
        Whether the added runtime components participate in redistribution
        (default: ``True``).
    container : ParticleContainer, optional
        An existing, geometry-defined container to restart into. If ``None``
        (default), the geometry is recovered from the plotfile's
        :py:class:`amrex.PlotFileData` and a fresh polymorphic pure-SoA,
        single-level container is created. Particles from all levels in the
        file are read; in the auto-created container they are all placed on
        level 0 (positions are preserved, the MR level assignment is not).
        Since plotfiles do not record periodicity, the auto-created geometry
        is non-periodic. Application *checkpoints* store their geometry in an
        application-specific format, so reading those requires passing a
        ``container``.

    Returns
    -------
    ParticleContainer
        The populated particle container. Iterate the data via, e.g.,
        ``for pti in pc.iterator(level=0): soa = pti.soa()``.

    Raises
    ------
    FileNotFoundError
        If there is no particle ``Header`` under ``plotfile/particle_dir``, or
        if ``container`` is ``None`` and there is no top-level plotfile
        ``Header`` to recover the AMR geometry from.
    ValueError
        If ``container`` is ``None`` and ``plotfile`` is an application
        checkpoint, whose top-level ``Header`` does not describe the geometry.
    """
    particle_header = os.path.join(plotfile, particle_dir, "Header")
    if not os.path.isfile(particle_header):
        raise FileNotFoundError(
            f"read_particles: no particle Header found at '{particle_header}'. "
            f"Check that '{plotfile}' is an AMReX plotfile/checkpoint directory "
            f"and that it contains particle output under '{particle_dir}/'."
        )
    header = amr.ParticleHeader.read(plotfile, particle_dir)

    pc = container
    if pc is None:
        # recover the AMR geometry from the plotfile metadata. Conforming
        # particle plotfiles always contain a top-level plotfile Header: AMReX
        # writes a dummy MultiFab for pure-particle outputs to ensure that.
        plotfile_header = os.path.join(plotfile, "Header")
        if not os.path.isfile(plotfile_header):
            raise FileNotFoundError(
                f"read_particles: no plotfile Header found at "
                f"'{plotfile_header}', so the AMR geometry cannot be "
                "recovered. To read anyway, pass an existing, geometry-defined "
                "'container'."
            )
        with open(plotfile_header) as f:
            first_line = f.readline().strip()
        if first_line.startswith("CheckPointVersion"):
            raise ValueError(
                f"read_particles: '{plotfile}' is an application checkpoint, "
                "not a plotfile; its top-level Header does not describe the "
                "AMR geometry. Pass an existing, geometry-defined 'container' "
                "to read into instead."
            )
        plt = amr.PlotFileData(plotfile)
        prob_domain = plt.probDomain(0)
        domain_box = amr.Box(prob_domain.small_end, prob_domain.big_end)
        real_box = amr.RealBox(plt.probLo(), plt.probHi())
        geom = amr.Geometry(
            domain_box, real_box, plt.coordSys(), [0] * amr.Config.spacedim
        )

        pc_type_name = f"ParticleContainer_pureSoA_{amr.Config.spacedim}_0_polymorphic"
        try:
            pc_type = getattr(amr, pc_type_name)
        except AttributeError as e:
            raise AttributeError(
                f"pyAMReX was built without the pure-SoA container '{pc_type_name}'."
            ) from e
        # a single-level container defined on the coarsest level: Restart reads
        # particles from every level in the file and Redistribute() then places
        # them all on level 0
        pc = pc_type(geom, plt.DistributionMap(0), plt.boxArray(0))
        # the polymorphic allocator needs an arena before any tile allocation
        pc.arena = amr.The_Arena()

    # configure the runtime SoA components to match the on-disk layout. For pure
    # SoA particles the AMREX_SPACEDIM positions are compile-time components and
    # header.{real,int}_comp_names list exactly the runtime components to add.
    for name in header.real_comp_names:
        if not pc.has_real_comp(name):
            pc.add_real_comp(name, communicate)
    for name in header.int_comp_names:
        if not pc.has_int_comp(name):
            pc.add_int_comp(name, communicate)

    pc.restart_checkpoint(plotfile, particle_dir, header.is_checkpoint)
    return pc


def register_ParticleContainer_extension(amr):
    """ParticleContainer helper methods"""
    import inspect
    import sys

    # register member functions for every Par(Const)Iter* type
    for _, ParIter_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and (
                member.__name__.startswith("ParIter")
                or member.__name__.startswith("ParConstIter")
            )
        ),
    ):
        ParIter_type.__next__ = next
        ParIter_type.__iter__ = lambda self: self
        ParIter_type.__getitem__ = getitem

    # register member functions for every ParticleContainer_* type
    for _, ParticleContainer_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("ParticleContainer_")
        ),
    ):
        ParticleContainer_type.iterator = iterator
        ParticleContainer_type.const_iterator = (
            iterator  # TODO: simplified, code duplication
        )
        ParticleContainer_type.to_df = pc_to_df
        ParticleContainer_type.add_arrays = pc_add_arrays
        ParticleContainer_type.add_df = pc_add_df
