"""
This file is part of pyAMReX

Copyright 2023-2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

from .dlpack_helpers import dlpack_to_cupy, dlpack_to_dpnp, xp_module_name


def podvector_to_numpy(self, copy=False):
    """
    Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    np.array
        A 1D NumPy array.
    """
    import numpy as np

    if self.size() > 0:
        if copy:
            # This supports a device-to-host copy.
            #
            # The to_host() returned object is a temporary, and
            # np.array using the __array_interface__ protocol does
            # not keep it alive automatically unless it is stored
            # in an actual variable (tmp).
            tmp = self.to_host()
            ret = np.array(tmp, copy=False)
            assert ret.base is tmp
            return ret
        else:
            # host-accessible memory (CPU, pinned, managed/shared USM): the
            # __array_interface__ exposes the host pointer regardless of the
            # DLPack device tag, unlike np.from_dlpack which is host-device only
            return np.array(self, copy=False)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_cupy(self, copy=False):
    """
    Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

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
    if self.size() > 0:
        return dlpack_to_cupy(self, copy)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_dpnp(self, copy=False):
    """
    Provide a dpnp view into a PODVector (e.g., RealVector, IntVector).

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    dpnp.array
        A 1D dpnp array.

    Raises
    ------
    ImportError
        Raises an exception if dpnp is not installed
    """
    if self.size() > 0:
        return dlpack_to_dpnp(self, copy)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_xp(self, copy=False):
    """
    Provide a NumPy, CuPy or dpnp view into a PODVector (e.g., RealVector,
    IntVector), depending on amr.Config.have_gpu and amr.Config.gpu_backend .

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    copy : bool, optional
        Copy the data if true, otherwise create a view (default).

    Returns
    -------
    xp.array
        A 1D NumPy, CuPy or dpnp array.
    """
    import inspect

    amr = inspect.getmodule(self)
    return getattr(self, "to_" + xp_module_name(amr))(copy)


def _podvector_base(cls):
    """The pyAMReX PODVector class of ``cls`` (or of one of its bases).

    Returns None if ``cls`` is not a PODVector. Looking through the MRO
    also supports Python subclasses of the pyAMReX classes.
    """
    for base in cls.__mro__:
        if base.__name__.startswith("PODVector_"):
            return base
    return None


def _element_type(cls):
    """The element type name of a PODVector class, e.g., "real" or "int"."""
    return _podvector_base(cls).__name__.split("_")[1]


def _is_podvector(arr):
    return _podvector_base(type(arr)) is not None


def _check_1d(ndim, shape):
    if ndim != 1:
        raise ValueError(f"expected a 1-D array, but got shape {tuple(shape)}")


def _device_to_host(arr, missing):
    """Producer-side device-to-host copy via DLPack, without CuPy or dpnp.

    This is the fallback when the array module for a device array is not
    installed. It needs a DLPack producer that supports copies to the host
    (DLPack >= 1.0 ``dl_device``/``copy``), e.g., pyAMReX PODVectors.
    """
    import numpy as np

    try:
        return np.from_dlpack(arr, device="cpu", copy=True)
    except Exception as e:
        raise TypeError(
            f"Copying from {type(arr).__name__} requires {missing}, or a DLPack "
            f"producer that supports copies to the host (NumPy >= 2.1)."
        ) from e


def _prepare_source(arr):
    """Normalize array data for a copy into a PODVector.

    Returns ``(kind, src)`` with ``kind`` one of "podvector", "numpy",
    "cupy" or "dpnp" and ``src`` a 1-D PODVector, NumPy, CuPy or dpnp array.
    CuPy and dpnp are only imported if the data lives on such a device; if
    they are not installed (or the device is not supported by either), the
    data is copied to the host by its DLPack producer instead.
    """
    if arr is None:
        raise TypeError("expected array data, but got None")

    if _is_podvector(arr):
        return "podvector", arr

    return _prepare_array(arr)


def _prepare_array(arr):
    """Normalize array data (including a PODVector viewed as an array).

    See _prepare_source.
    """

    from .dlpack_helpers import array_kind

    kind = array_kind(arr)
    if kind == "cupy":
        try:
            import cupy as cp
        except ImportError:
            kind, src = "numpy", _device_to_host(arr, "CuPy")
        else:
            from .dlpack_helpers import cupy_from_dlpack

            # from_dlpack orders the producer's work before the current stream
            src = (
                cupy_from_dlpack(arr) if hasattr(arr, "__dlpack__") else cp.asarray(arr)
            )
    elif kind == "dpnp":
        try:
            import dpnp as dp
        except ImportError:
            kind, src = "numpy", _device_to_host(arr, "dpnp")
        else:
            # dp.asarray would copy, from_dlpack is a view
            src = arr if isinstance(arr, dp.ndarray) else dp.from_dlpack(arr)
    elif kind == "dlpack":
        kind, src = "numpy", _device_to_host(arr, "CuPy or dpnp")
    else:
        src = _host_array(arr)

    _check_1d(src.ndim, src.shape)
    return kind, src


def _host_array(arr):
    """A NumPy view of host data, ready to be read.

    DLPack producers are imported with ``numpy.from_dlpack``: unlike the
    array interface, the DLPack export synchronizes pending device work on
    host-accessible device memory (e.g., pinned or managed memory written by
    an AMReX kernel). Other data (lists, older producers) uses
    ``numpy.asarray``.
    """
    import numpy as np

    if hasattr(arr, "__dlpack__") and not isinstance(arr, np.ndarray):
        try:
            return np.from_dlpack(arr)
        except (BufferError, TypeError, ValueError, RuntimeError):
            if not (hasattr(arr, "__array__") or hasattr(arr, "__array_interface__")):
                raise
    return np.asarray(arr)


def _source_size(kind, src):
    return src.size() if kind == "podvector" else src.shape[0]


def _have_module(name):
    """Whether an optional module (e.g., "cupy") can be imported."""
    import importlib

    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _check_range(self, offset, n):
    if offset < 0 or offset + n > self.size():
        raise IndexError(
            f"copy_from: cannot copy {n} elements at offset {offset} into a "
            f"PODVector of size {self.size()}; resize it first"
        )


def _copy_path(src_kind, dst_kind, src_is_podvector, same_element_type, have_module):
    """Pick how to copy data into a PODVector.

    Parameters
    ----------
    src_kind, dst_kind : str
        Memory of the source and the destination, as by
        :func:`dlpack_helpers.array_kind`: "numpy" (host), "cupy" or "dpnp".
    src_is_podvector : bool
        Whether the source is a pyAMReX PODVector.
    same_element_type : bool
        Whether a PODVector source has the element type of the destination.
    have_module : bool
        Whether the array module of the source memory (CuPy or dpnp) can be
        imported. Always true for CuPy and dpnp array sources.

    Returns
    -------
    str
        - "amrex": a direct AMReX copy between any two memory spaces.
        - "device": a device-to-device copy (and cast) with CuPy or dpnp.
        - "host": through host memory; the final copy (host-to-host or
          host-to-device) is done by AMReX, so this needs neither CuPy nor
          dpnp for host data.
    """
    if src_is_podvector and same_element_type:
        return "amrex"
    if src_kind in ("cupy", "dpnp") and src_kind == dst_kind and have_module:
        return "device"
    return "host"


def _copy_prepared(self, kind, src, offset):
    """Copy a source from _prepare_source into ``self[offset:]``.

    The copy path is picked by :func:`_copy_path`.
    """
    from .dlpack_helpers import array_kind, dlpack_to_cupy, dlpack_to_dpnp

    n = _source_size(kind, src)
    _check_range(self, offset, n)
    if n == 0:
        return

    src_is_podvector = kind == "podvector"
    src_kind = array_kind(src) if src_is_podvector else kind
    dst_kind = array_kind(self)
    same_element_type = src_is_podvector and _element_type(type(src)) == _element_type(
        type(self)
    )
    # only import CuPy/dpnp for PODVectors if a device copy is possible
    have_module = not src_is_podvector or (
        not same_element_type
        and src_kind in ("cupy", "dpnp")
        and src_kind == dst_kind
        and _have_module(src_kind)
    )
    path = _copy_path(
        src_kind, dst_kind, src_is_podvector, same_element_type, have_module
    )

    if path == "amrex":
        self.copy_from_podvector(src, offset)
        return

    if src_is_podvector:
        # element type cast: continue with the PODVector as an array
        if path == "device" or src_kind == "numpy":
            # device view, or a synchronized host view
            kind, src = _prepare_array(src)
        else:
            # device memory: AMReX device-to-host copy into pinned memory
            # (needs neither CuPy/dpnp nor a recent NumPy)
            kind, src = "numpy", src.to_numpy(copy=True)

    if path == "device":
        if kind == "cupy":
            import cupy as cp

            dlpack_to_cupy(self)[offset : offset + n] = src
            # the write is asynchronous on the current CuPy stream
            cp.cuda.get_current_stream().synchronize()
        else:
            import dpnp as dp

            dst = dlpack_to_dpnp(self)[offset : offset + n]
            if src.sycl_queue != dst.sycl_queue:
                src = dp.asarray(src, sycl_queue=dst.sycl_queue)
            dst[...] = src.astype(dst.dtype, copy=False)
            dst.sycl_queue.wait()
        return

    # path == "host"
    if kind == "cupy":
        import cupy as cp

        src = cp.asnumpy(src)
    elif kind == "dpnp":
        import dpnp as dp

        src = dp.asnumpy(src)
    self.copy_from_numpy(src, offset)


def podvector_copy_from(self, arr, offset=0):
    """
    Copy array data into this PODVector at an offset.

    Writes ``self[offset:offset + len(arr)]``; the vector is not resized.
    Values are cast to the vector's element type.

    The fastest available copy is used: a direct AMReX copy for PODVector
    input, and a device-to-device copy for CuPy or dpnp input into device
    memory. Otherwise, the data is copied through the host with AMReX, so
    CuPy and dpnp are never required: they are only imported if the data
    is on such a device.

    Parameters
    ----------
    self : amrex.PODVector_*
        A PODVector class in pyAMReX
    arr : array_like or PODVector
        1-D data: a PODVector, a NumPy, CuPy or dpnp array, any DLPack
        producer, or an array-like such as a list.
    offset : int, optional
        First element of this vector to write (default: 0).

    Raises
    ------
    IndexError
        If the data does not fit into this vector at the offset.
    ValueError
        If the data is not 1-D.
    TypeError
        If the data is None or on an unsupported device.

    Notes
    -----
    The source must not overlap with the written part of this vector, e.g.,
    it must not be a view into it.
    """
    import operator

    offset = operator.index(offset)
    # check the bounds before any (possibly expensive) conversion
    if _is_podvector(arr):
        _check_range(self, offset, arr.size())
    elif getattr(arr, "ndim", None) == 1:
        _check_range(self, offset, arr.shape[0])
    kind, src = _prepare_source(arr)
    _copy_prepared(self, kind, src, offset)


def podvector_from_array(cls, arr, copy=True):
    """
    Create a PODVector from array data.

    The element type and allocator are those of ``cls`` (e.g.,
    ``DeviceVector_real``, ``PODVector_int_std``); values are cast to the
    element type. See :meth:`copy_from` for the supported inputs and how
    the data is copied: CuPy and dpnp are never required.

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like or PODVector
        1-D data: a PODVector, a NumPy, CuPy or dpnp array, any DLPack
        producer, or an array-like such as a list.
    copy : bool or None, optional
        Like ``numpy.asarray``: if True (default), always copy. If None,
        return ``arr`` itself if it already is a ``cls`` instance, otherwise
        copy. If False, return ``arr`` itself and raise ValueError if a copy
        would be needed.

    Returns
    -------
    PODVector
        A ``cls`` instance.

    Raises
    ------
    ValueError
        If the data is not 1-D, or if ``copy=False`` and a copy is needed.
    TypeError
        If the data is None or on an unsupported device.
    """
    if copy not in (True, False, None):
        raise ValueError(f"from_array: copy must be True, False or None, not {copy!r}")
    copy = None if copy is None else bool(copy)

    if isinstance(arr, cls) and copy is not True:
        return arr
    if copy is False:
        raise ValueError(
            f"from_array: a copy is needed to create a {cls.__name__} from "
            f"{type(arr).__name__}; use copy=True or copy=None"
        )

    kind, src = _prepare_source(arr)
    pv = cls(_source_size(kind, src))
    _copy_prepared(pv, kind, src, 0)
    return pv


def podvector_from_cupy(cls, arr):
    """
    Create a new PODVector from a CuPy array (or array-like).

    Always copies the data into a newly allocated PODVector.
    Works for every allocator type. Equivalent to :meth:`from_array`,
    but requires CuPy. Array-likes on the host are copied without a detour
    through the device.

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data, convertible to a CuPy array.

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.
    """
    import cupy  # noqa: F401  (documented requirement)

    return podvector_from_array(cls, arr)


def podvector_from_dpnp(cls, arr):
    """
    Create a new PODVector from a dpnp array (or array-like).

    Always copies the data into a newly allocated PODVector.
    Works for every allocator type. Equivalent to :meth:`from_array`,
    but requires dpnp. Array-likes on the host are copied without a detour
    through the device.

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data, convertible to a dpnp array.

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.
    """
    import dpnp  # noqa: F401  (documented requirement)

    return podvector_from_array(cls, arr)


def podvector_from_xp(cls, arr):
    """
    Create a new PODVector from a NumPy, CuPy or dpnp array.

    Always copies the data into a newly allocated PODVector, like
    :meth:`from_array`, which supports any of these inputs regardless of
    the build. Unlike :meth:`to_xp`, a zero-copy view is not possible here
    because PODVector always owns its memory through its allocator.

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data (NumPy, CuPy or dpnp array).

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.
    """
    return podvector_from_array(cls, arr)


def register_PODVector_extension(amr):
    """PODVector helper methods"""
    import inspect
    import sys

    # register member functions for every PODVector_* type
    for _, POD_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("PODVector_")
        ),
    ):
        # instance methods: PODVector -> array
        POD_type.to_numpy = podvector_to_numpy
        POD_type.to_cupy = podvector_to_cupy
        POD_type.to_dpnp = podvector_to_dpnp
        POD_type.to_xp = podvector_to_xp
        POD_type.copy_from = podvector_copy_from

        # class methods: array -> PODVector
        # (from_numpy is provided in C++ as a static method)
        POD_type.from_cupy = classmethod(podvector_from_cupy)
        POD_type.from_dpnp = classmethod(podvector_from_dpnp)
        POD_type.from_xp = classmethod(podvector_from_xp)
        POD_type.from_array = classmethod(podvector_from_array)
