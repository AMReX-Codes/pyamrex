"""
This file is part of pyAMReX

Shared building blocks for the ``to_numpy``/``to_cupy``/``to_dpnp``/``to_xp``
conversion helpers of the per-class extension modules
(Array4, MultiFab, PODVector, SmallMatrix, StructOfArrays).

All converters exchange data through the standardized DLPack protocol
(``__dlpack__``/``__dlpack_device__``) implemented by the pyAMReX C++
classes. CuPy and dpnp remain optional dependencies: they are imported
lazily, only when a conversion to them is actually requested.

Copyright 2025-2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

# DLPack device types (DLDeviceType values) with host-side memory
kDLCPU = 1
kDLCUDAHost = 3
kDLROCMHost = 11
# DLPack device types (DLDeviceType values) with device-side memory
kDLCUDA = 2
kDLROCM = 10
kDLCUDAManaged = 13
kDLOneAPI = 14


def reorder(data, order):
    """Apply pyAMReX's index order convention to a C-indexed array view.

    pyAMReX data (e.g., Array4) is exported in C index order (z, y, x).
    ``order="F"`` returns the transposed view, indexing as x, y, z like
    AMReX; ``order="C"`` returns the view unchanged.
    """
    if order == "F":
        # full reversal of axes (x, y, z, n <- n, z, y, x); .T is deprecated
        # for non-2D dpnp arrays, so use an explicit transpose
        return data.transpose(tuple(range(data.ndim - 1, -1, -1)))
    elif order == "C":
        return data
    else:
        raise ValueError("The order argument must be F or C.")


def dlpack_to_numpy(self, copy=False):
    """Import a pyAMReX object into NumPy via DLPack.

    ``copy=False`` returns a zero-copy view (host-accessible data only);
    ``copy=True`` returns an isolated copy, transferring device data to the
    host as needed.
    """
    import numpy as np

    if copy:
        device_type, _ = self.__dlpack_device__()
        if device_type in (kDLCPU, kDLCUDAHost, kDLROCMHost):
            return np.from_dlpack(self).copy()
        # device data: producer-side device-to-host copy
        # (requires NumPy >= 2.1 for the device/copy arguments)
        return np.from_dlpack(self, device="cpu", copy=True)
    return np.from_dlpack(self)


def _pyamrex_module(obj):
    """The pyAMReX module (e.g., amrex.space3d) of a pyAMReX object, or None.

    Found through the MRO, so this also works for Python subclasses and for
    subclasses bound by application codes.
    """
    import sys

    for cls in type(obj).__mro__:
        if cls.__module__.startswith("amrex."):
            return sys.modules[cls.__module__]
    return None


def _is_pyamrex(obj):
    """Whether obj is a pyAMReX object (or a Python subclass of one)."""
    return _pyamrex_module(obj) is not None


class _SynchronizedExport:
    """Export a pyAMReX object via DLPack with ``stream=None``.

    pyAMReX's exporter fully synchronizes its stream for ``stream=None``, so
    the data is ready on any consumer stream. This is needed for CUDA managed
    memory: DLPack reports it as ``(kDLCUDAManaged, 0)``, so CuPy requests the
    synchronization for a stream of device 0, but then uses the data on the
    current stream of its current device, which is another one if AMReX runs
    on another GPU.
    """

    def __init__(self, obj):
        self._obj = obj

    def __dlpack_device__(self):
        return self._obj.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        return self._obj.__dlpack__(stream=None, **kwargs)


def cupy_from_dlpack(obj):
    """``cupy.from_dlpack``, synchronized for pyAMReX managed memory.

    See _SynchronizedExport.
    """
    import cupy as cp

    if _is_pyamrex(obj) and int(obj.__dlpack_device__()[0]) == kDLCUDAManaged:
        obj = _SynchronizedExport(obj)
    return cp.from_dlpack(obj)


def dlpack_to_cupy(self, copy=False):
    """Import a pyAMReX object into CuPy via DLPack.

    Device data is imported zero-copy (or as an isolated device-side copy
    for ``copy=True``). Host-side data is always copied to the device,
    since a cross-device view is not possible.
    """
    import cupy as cp

    device_type, _ = self.__dlpack_device__()
    if device_type in (kDLCPU, kDLCUDAHost, kDLROCMHost):
        # host-accessible memory (plain host or CUDA/ROCm pinned): CuPy's
        # from_dlpack rejects the pinned-host device types, so stage a
        # host-to-device copy via NumPy
        import numpy as np

        # cp.asarray does an asynchronous host-to-device copy; keep the host
        # view (and thus the producer) alive and synchronize before returning
        # so the copy finishes reading the source before it can be modified
        host_view = np.from_dlpack(self)
        arr = cp.asarray(host_view)
        cp.cuda.get_current_stream().synchronize()
        return arr
    # device data: zero-copy import, then a consumer-side copy if requested.
    # We do not pass copy= to cp.from_dlpack: CuPy >= 14 forwards its current
    # stream to __dlpack__, which the exporter rejects together with copy=True
    # (a producer-made copy requires stream=None).
    arr = cupy_from_dlpack(self)
    if not copy:
        return arr
    result = arr.copy()
    # ensure the copy has finished reading the source before the DLPack view
    # (and thus the producer) is released
    cp.cuda.get_current_stream().synchronize()
    return result


def dlpack_to_dpnp(self, copy=False):
    """Import a pyAMReX object into dpnp via DLPack.

    SYCL USM data is imported zero-copy (or as an isolated copy for
    ``copy=True``). Host-side data is always copied to the device, since
    a cross-device view is not possible.
    """
    import dpnp as dp

    device_type, _ = self.__dlpack_device__()
    if device_type in (kDLCPU, kDLCUDAHost, kDLROCMHost):
        # host-accessible memory: stage a host-to-device copy via NumPy
        import numpy as np

        # dp.asarray does an asynchronous host-to-device copy; keep the host
        # view (and thus the producer) alive and synchronize before returning
        # so the copy finishes reading the source before it can be modified
        host_view = np.from_dlpack(self)
        arr = dp.asarray(host_view)
        arr.sycl_queue.wait()
        return arr
    # device data: zero-copy import, then a consumer-side copy if requested.
    # We do not pass copy= to dpnp.from_dlpack (a producer-made copy requires
    # stream=None, and importing one crashes dpnp 0.20/dpctl 0.22).
    arr = dp.from_dlpack(self)
    if not copy:
        return arr
    result = arr.copy()
    # ensure the copy has finished reading the source before the DLPack view
    # (and thus the producer) is released
    result.sycl_queue.wait()
    return result


def xp_module_name(amr):
    """The array module matching the AMReX build, as portable
    NumPy/CuPy/dpnp short-hand ``xp``:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Parameters
    ----------
    amr :
        The amrex.space*d module of the object to convert.

    Returns
    -------
    str
        "numpy", "cupy" or "dpnp".
    """
    if amr.Config.have_gpu:
        if amr.Config.gpu_backend == "SYCL":
            return "dpnp"
        else:  # CUDA, HIP
            return "cupy"
    return "numpy"


def array_kind(arr):
    """Classify array data by the array module that can read it.

    The classification is duck-typed, so neither CuPy nor dpnp is imported.
    The DLPack device (``__dlpack_device__``) is checked first because it is
    the most reliable way to tell host from device memory: e.g., pyAMReX host
    containers also expose ``__cuda_array_interface__``. Without DLPack, an
    ``__array_interface__`` means host memory, and the CUDA array interface
    alone means device memory.

    Parameters
    ----------
    arr :
        Array data, e.g., a NumPy, CuPy or dpnp array, a DLPack producer
        or an array-like such as a list.

    Returns
    -------
    str
        "numpy" for host memory and array-likes, "cupy" for CUDA and ROCm
        device memory, "dpnp" for SYCL (oneAPI) device memory, and "dlpack"
        for other DLPack devices.
    """
    dlpack_device = getattr(arr, "__dlpack_device__", None)
    if dlpack_device is not None:
        device_type = int(dlpack_device()[0])
        if device_type in (kDLCPU, kDLCUDAHost, kDLROCMHost):
            return "numpy"
        if device_type in (kDLCUDA, kDLROCM, kDLCUDAManaged):
            return "cupy"
        if device_type == kDLOneAPI:
            return "dpnp"
        return "dlpack"
    if hasattr(arr, "__array_interface__"):
        return "numpy"
    if hasattr(arr, "__cuda_array_interface__"):
        return "cupy"
    return "numpy"
