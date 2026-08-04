# -*- coding: utf-8 -*-
#
# SYCL / dpnp DLPack round-trip tests. These are skipped unless pyAMReX was
# built with the SYCL backend (Intel GPUs).

import numpy as np
import pytest

import amrex.space3d as amr

pytestmark = pytest.mark.skipif(
    amr.Config.gpu_backend != "SYCL", reason="Requires AMReX_GPU_BACKEND=SYCL"
)


def test_array4_dlpack_dpnp(mfab_device):
    dp = pytest.importorskip("dpnp")

    # device pointer is reported as a oneAPI device
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        device_type, device_id = arr.__dlpack_device__()
        assert device_type == int(amr.DLDeviceType.kDLOneAPI)
        assert device_id >= 0

        # AMReX -> dpnp: zero-copy view on the device
        marr = arr.to_dpnp()
        assert marr.sycl_device.is_gpu
        marr[...] = 5.0

    # mutations through the view are visible in the MultiFab
    for mfi in mfab_device:
        marr = mfab_device.array(mfi).to_dpnp()
        assert float(marr.min()) == 5.0 == float(marr.max())

    # device-side copy: isolated from the source
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        c_marr = arr.to_dpnp(copy=True)
        c_marr[...] = 6.0
        assert float(dp.from_dlpack(arr).max()) == 5.0
        break


def test_array4_dlpack_dpnp_to_host(mfab_device):
    pytest.importorskip("dpnp")

    # np.from_dlpack(..., device="cpu") triggers a producer-side
    # device-to-host copy (staged through pinned USM memory)
    mfab_device.set_val(7.0)
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        host = np.from_dlpack(arr, device="cpu")
        assert np.all(host == 7.0)

        with pytest.raises(BufferError):
            arr.__dlpack__(dl_device=(int(amr.DLDeviceType.kDLCPU), 0), copy=False)
        break


def test_multifab_to_xp_dpnp(mfab_device):
    pytest.importorskip("dpnp")

    mfab_device.set_val(2.0)
    views = mfab_device.to_xp()
    assert len(views) > 0
    for v in views:
        assert type(v).__module__.startswith("dpnp")
        assert float(v.max()) == 2.0
