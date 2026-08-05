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
    arr = cp.from_dlpack(self)
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
