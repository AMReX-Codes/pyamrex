# -*- coding: utf-8 -*-

import sys

import numpy as np
import pytest

import amrex.space3d as amr


def test_podvector_init():
    podv = amr.PODVector_real_std()
    print(podv.__array_interface__)
    # podv[0] = 1
    # podv[2] = 3
    assert podv.size() == 0
    podv.push_back(1)
    podv.push_back(2)
    assert podv.size() == 2 and podv[1] == 2
    podv.pop_back()
    assert podv.size() == 1
    podv.push_back(2.14)
    assert not podv.empty()
    podv.push_back(3.1)
    podv[2] = 5
    assert podv.size() == 3 and podv[2] == 5
    podv.clear()
    assert podv.size() == 0
    assert podv.empty()


def test_array_interface():
    podv = amr.PODVector_int_std()
    podv.push_back(1)
    podv.push_back(2)
    podv.push_back(1)
    podv.push_back(5)
    arr = podv.to_numpy()
    print(arr)

    # podv[2] = 3
    arr[2] = 3
    print(arr)
    print(podv)
    assert arr[2] == podv[2] == 3

    podv[1] = 5
    assert arr[1] == podv[1] == 5


def test_from_numpy():
    import numpy as np

    # basic roundtrip (cast to the vector's element type so the test is
    # precision-agnostic, e.g. single-precision builds)
    arr = np.array([1.0, 2.5, 3.7, 4.0], dtype=np.float64)
    podv = amr.DeviceVector_real.from_numpy(arr)
    assert podv.size() == 4
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))

    # from_numpy creates a copy, not a view
    arr[0] = 999.0
    assert podv[0] != 999.0

    # empty array
    empty = np.array([], dtype=np.float64)
    podv_empty = amr.DeviceVector_real.from_numpy(empty)
    assert podv_empty.size() == 0

    # from list (array-like)
    podv_list = amr.DeviceVector_real.from_numpy([10.0, 20.0])
    assert podv_list.size() == 2
    assert podv_list[1] == 20.0


def test_from_numpy_normalizes_input():
    import numpy as np

    # non-contiguous (strided) input is made contiguous on the host
    base = np.array([1.0, 9.0, 2.0, 9.0, 3.0, 9.0], dtype=np.float64)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    podv = amr.DeviceVector_real.from_numpy(strided)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], result.dtype))

    # mismatched dtype is cast to the vector's element type
    ints = np.array([4, 5, 6], dtype=np.int32)
    podv2 = amr.DeviceVector_real.from_numpy(ints)
    result2 = podv2.to_numpy(copy=True)
    np.testing.assert_array_equal(result2, np.array([4.0, 5.0, 6.0], result2.dtype))


def test_to_device_empty():
    podv = amr.PODVector_int_std()
    device = podv.to_device()
    assert isinstance(device, amr.DeviceVector_int)
    assert device.size() == 0
    assert device.empty()


def test_to_device_from_host_vector():
    import numpy as np

    values = np.array([1, -2, 5, 8], dtype=np.int32)
    podv = amr.PODVector_int_std.from_numpy(values)
    device = podv.to_device()

    assert isinstance(device, amr.DeviceVector_int)
    assert device.size() == values.size
    result = device.to_numpy(copy=True)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    podv[0] = 99
    np.testing.assert_array_equal(
        device.to_numpy(copy=True), values.astype(result.dtype)
    )


def test_to_device_from_device_vector():
    import numpy as np

    values = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    podv = amr.DeviceVector_real.from_numpy(values)
    device = podv.to_device()

    assert isinstance(device, amr.DeviceVector_real)
    assert device.size() == values.size
    result = device.to_numpy(copy=True)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    podv[1] = 7.0
    np.testing.assert_array_equal(
        device.to_numpy(copy=True), values.astype(result.dtype)
    )


def test_podvector_dlpack():
    import numpy as np

    podv = amr.PODVector_int_std()
    for v in [1, 2, 1, 5]:
        podv.push_back(v)

    # host memory
    assert podv.__dlpack_device__() == (int(amr.DLDeviceType.kDLCPU), 0)

    # zero-copy view
    view = np.from_dlpack(podv)
    assert view.ndim == 1
    assert view.shape == (4,)
    view[2] = 3
    assert podv[2] == 3

    # isolated copy
    copied = np.from_dlpack(podv, copy=True)
    copied[0] = 42
    assert podv[0] == 1


def test_podvector_dlpack_keeps_alive(assert_keeps_python_alive):
    import numpy as np

    podv = amr.PODVector_real_std()
    podv.push_back(1.0)
    view = assert_keeps_python_alive(podv, lambda: np.from_dlpack(podv))
    view[0] = 2.0
    assert podv[0] == 2.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_device():
    cp = pytest.importorskip("cupy")
    import numpy as np

    values = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    podv = amr.NonManagedDeviceVector_real.from_numpy(values)

    device_type, _ = podv.__dlpack_device__()
    assert device_type in (
        int(amr.DLDeviceType.kDLCUDA),
        int(amr.DLDeviceType.kDLROCM),
        int(amr.DLDeviceType.kDLOneAPI),
    )

    # zero-copy view on the device
    marr = cp.from_dlpack(podv)
    cp.testing.assert_array_equal(marr, cp.asarray(values.astype(marr.dtype)))
    marr[0] = 7.0
    assert podv[0] == 7.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_pinned():
    import numpy as np

    # pinned memory is host-accessible: NumPy can view it directly
    pinned = amr.HostVector_real()
    pinned.push_back(1.0)
    pinned.push_back(2.0)

    view = np.from_dlpack(pinned)
    np.testing.assert_array_equal(view, np.array([1.0, 2.0], dtype=view.dtype))
    view[0] = 3.0
    assert pinned[0] == 3.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_managed_host_view():
    # managed/shared (unified) memory is host-accessible even though its DLPack
    # device type is a device type (kDLCUDAManaged / kDLOneAPI / kDLROCM). A CPU
    # request must therefore be zero-copy: copy=False must succeed, not raise.
    import numpy as np

    mv = amr.ManagedVector_real()
    for v in [1.0, 2.0, 3.0]:
        mv.push_back(v)

    host = np.from_dlpack(mv, device="cpu", copy=False)
    np.testing.assert_array_equal(host, np.array([1.0, 2.0, 3.0]))

    # zero-copy: writing through the host view modifies the source
    host[0] = 9.0
    assert mv[0] == 9.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_pinned_to_cupy():
    # pinned host memory is advertised as kDLCUDAHost/kDLROCMHost, which CuPy's
    # from_dlpack rejects; to_cupy() must stage a host-to-device copy instead
    cp = pytest.importorskip("cupy")
    import numpy as np

    pinned = amr.HostVector_real()
    for v in [1.0, 2.0, 3.0]:
        pinned.push_back(v)

    marr = pinned.to_cupy()
    cp.testing.assert_array_equal(marr, cp.asarray(np.array([1.0, 2.0, 3.0])))

    # the host-to-device staging copy must be synchronized before returning:
    # modifying the source afterwards must not change the (snapshot) result
    pinned[0] = 42.0
    cp.testing.assert_array_equal(marr, cp.asarray(np.array([1.0, 2.0, 3.0])))


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
@pytest.mark.parametrize(
    "ctor_name",
    ["NonManagedDeviceVector_real", "ManagedVector_real", "HostVector_real"],
)
def test_podvector_dlpack_empty_device_type(ctor_name):
    # an empty vector has a null data pointer; its DLPack device must be
    # classified from the allocator's arena kind so it does NOT change once
    # the vector holds data (previously an empty device/managed/pinned vector
    # was mislabeled kDLCPU and then flipped after the first push_back)
    gpu = (
        int(amr.DLDeviceType.kDLCUDA),
        int(amr.DLDeviceType.kDLROCM),
        int(amr.DLDeviceType.kDLOneAPI),
    )

    ctor = getattr(amr, ctor_name)
    empty = ctor()
    assert empty.size() == 0
    empty_dev = empty.__dlpack_device__()

    empty.push_back(1.0)
    assert empty.__dlpack_device__() == empty_dev, (
        "device type changed after allocation"
    )

    # a non-host-accessible allocator must report a GPU device even when empty
    if ctor_name == "NonManagedDeviceVector_real":
        assert empty_dev[0] in gpu


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_from_numpy_device_only():
    # device-only allocator: the host-to-device copy must work without CuPy
    import numpy as np

    arr = np.array([1.0, 2.5, 3.7, 4.0], dtype=np.float64)
    podv = amr.NonManagedDeviceVector_real.from_numpy(arr)
    assert podv.size() == 4
    # read back through an AMReX device-to-host copy (no CuPy either)
    result = podv.to_host().to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))


def test_from_xp():
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0])
    podv = amr.DeviceVector_real.from_xp(arr)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_from_cp():
    cp = pytest.importorskip("cupy")

    arr = cp.array([1.0, 2.5, 3.7, 4.0], dtype=cp.float64)
    podv = amr.DeviceVector_real.from_cupy(arr)
    assert podv.size() == 4
    result = podv.to_cupy()
    cp.testing.assert_array_equal(result, arr.astype(result.dtype))

    arr[0] = 999.0
    assert podv[0] != 999.0

    empty = cp.array([], dtype=cp.float64)
    podv_empty = amr.DeviceVector_real.from_cupy(empty)
    assert podv_empty.size() == 0

    podv_list = amr.DeviceVector_real.from_cupy([10.0, 20.0])
    assert podv_list.size() == 2
    result = podv_list.to_cupy()
    cp.testing.assert_array_equal(result, cp.array([10.0, 20.0], dtype=result.dtype))


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_host_from_cp():
    cp = pytest.importorskip("cupy")
    import numpy as np

    arr = cp.array([1.0, 2.5, 3.7, 4.0], dtype=cp.float64)
    podv = amr.HostVector_real.from_cupy(arr)
    assert podv.size() == 4
    np.testing.assert_array_equal(podv.to_numpy(), cp.asnumpy(arr))

    arr[0] = 999.0
    assert podv[0] != 999.0


ALLOCATORS = ["pinned", "arena", "std", "polymorphic"]
if amr.Config.have_gpu:
    ALLOCATORS += ["device", "managed", "async"]


def _host_values(podv):
    """Copy a PODVector of any allocator to a NumPy array"""
    return podv.to_host().to_numpy(copy=True) if podv.size() > 0 else []


class _ForbidImport:
    """A meta path finder that fails the test when a module is imported"""

    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise AssertionError(f"{fullname} must not be imported here")
        return None


@pytest.fixture
def without_cupy_dpnp(monkeypatch):
    """Make CuPy and dpnp unavailable: ``import cupy`` raises ImportError"""
    for name in ("cupy", "dpnp", "dpctl"):
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def forbid_cupy_dpnp(monkeypatch):
    """Fail if CuPy or dpnp are imported"""
    names = ("cupy", "dpnp", "dpctl")
    for name in list(sys.modules):
        if name.split(".")[0] in names:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_ForbidImport(names)] + sys.meta_path)


def _device_xp():
    """The array module of the GPU backend, or skip"""
    if not amr.Config.have_gpu:
        pytest.skip("requires AMReX GPU support")
    if amr.Config.gpu_backend == "SYCL":
        return pytest.importorskip("dpnp")
    return pytest.importorskip("cupy")


def test_from_array_none():
    with pytest.raises(TypeError):
        amr.DeviceVector_real.from_array(None)


def test_from_array_numpy_and_list(forbid_cupy_dpnp):
    arr = np.array([1.0, 2.5, 3.7, 4.0])
    podv = amr.DeviceVector_real.from_array(arr)
    assert isinstance(podv, amr.DeviceVector_real)
    assert podv.size() == 4
    result = _host_values(podv)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))

    # from_array creates a copy, not a view
    arr[0] = 999.0
    assert podv[0] != 999.0

    # array-likes: list and tuple
    podv_list = amr.DeviceVector_real.from_array([10.0, 20.0])
    assert podv_list.size() == 2
    assert podv_list[1] == 20.0
    podv_tuple = amr.DeviceVector_real.from_array((10.0, 20.0, 30.0))
    assert podv_tuple.size() == 3
    assert podv_tuple[2] == 30.0

    # empty
    assert amr.DeviceVector_real.from_array([]).size() == 0


def test_from_array_dtype_cast():
    # mismatched dtype is cast to the vector's element type
    ints = np.array([4, 5, 6], dtype=np.int32)
    podv = amr.DeviceVector_real.from_array(ints)
    result = _host_values(podv)
    np.testing.assert_array_equal(result, np.array([4.0, 5.0, 6.0], result.dtype))

    # strided input
    strided = np.arange(10, dtype=np.int64)[::3]
    podv = amr.PODVector_int_std.from_array(strided)
    np.testing.assert_array_equal(podv.to_numpy(), [0, 3, 6, 9])


def test_from_array_not_1d():
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array(np.float64(1.0))


def test_from_array_copy_semantics():
    src = amr.DeviceVector_real.from_array([1.0, 2.0, 3.0])

    # copy=True (default): always a new, independent vector
    copied = amr.DeviceVector_real.from_array(src)
    assert copied is not src
    src[0] = 42.0
    assert copied[0] == 1.0

    # copy=None: like numpy.asarray, no copy if already the right type
    assert amr.DeviceVector_real.from_array(src, copy=None) is src
    assert amr.PODVector_real_pinned.from_array(src, copy=None) is not src

    # copy=False: never copy
    assert amr.DeviceVector_real.from_array(src, copy=False) is src
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array([1.0, 2.0], copy=False)


@pytest.mark.parametrize("dst_alloc", ALLOCATORS)
@pytest.mark.parametrize("src_alloc", ALLOCATORS)
def test_from_array_podvector(dst_alloc, src_alloc, forbid_cupy_dpnp):
    """Same element type: direct AMReX copies between any memory spaces"""
    values = np.array([1, -2, 5, 8], dtype=np.int32)
    src = getattr(amr, f"PODVector_int_{src_alloc}").from_numpy(values)
    dst_type = getattr(amr, f"PODVector_int_{dst_alloc}")

    dst = dst_type.from_array(src)
    assert isinstance(dst, dst_type)
    np.testing.assert_array_equal(_host_values(dst), values)

    # the copy is independent of the source
    src[0] = 99
    np.testing.assert_array_equal(_host_values(dst), values)


@pytest.mark.parametrize("dst_alloc", ALLOCATORS)
@pytest.mark.parametrize("src_alloc", ALLOCATORS)
def test_from_array_podvector_cast(dst_alloc, src_alloc):
    """Different element type: the result has the element type of cls"""
    values = np.array([1, -2, 5, 8], dtype=np.int32)
    src = getattr(amr, f"PODVector_int_{src_alloc}").from_numpy(values)
    dst_type = getattr(amr, f"PODVector_real_{dst_alloc}")

    dst = dst_type.from_array(src)
    assert isinstance(dst, dst_type)
    result = _host_values(dst)
    np.testing.assert_array_equal(result, values.astype(result.dtype))


@pytest.mark.parametrize("src_alloc", ALLOCATORS)
def test_from_array_podvector_cast_without_cupy_dpnp(src_alloc, without_cupy_dpnp):
    """Casting device data without CuPy or dpnp falls back to a host copy"""
    values = np.array([1, -2, 5, 8], dtype=np.int32)
    src = getattr(amr, f"PODVector_int_{src_alloc}").from_numpy(values)
    dst = amr.DeviceVector_real.from_array(src)
    result = _host_values(dst)
    np.testing.assert_array_equal(result, values.astype(result.dtype))


def test_from_array_pyamrex_host_objects(forbid_cupy_dpnp):
    """pyAMReX host containers also expose __cuda_array_interface__"""
    vec = amr.Vector_Real([1.0, 2.0, 3.0])
    podv = amr.DeviceVector_real.from_array(vec)
    result = _host_values(podv)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], result.dtype))


def test_from_array_dlpack_only_producer(forbid_cupy_dpnp):
    """Host data that only implements the DLPack protocol"""

    class DLPackOnly:
        def __init__(self, arr):
            self._arr = arr

        def __dlpack__(self, **kwargs):
            return self._arr.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return self._arr.__dlpack_device__()

    values = np.array([3.0, 1.0, 2.0])
    podv = amr.DeviceVector_real.from_array(DLPackOnly(values))
    result = _host_values(podv)
    np.testing.assert_array_equal(result, values.astype(result.dtype))


@pytest.mark.parametrize("dst_alloc", ALLOCATORS)
def test_copy_from_offset(dst_alloc, forbid_cupy_dpnp):
    dst = getattr(amr, f"PODVector_int_{dst_alloc}").from_array([0] * 6)

    # array into a slice
    dst.copy_from(np.array([1, 2], dtype=np.int64), offset=1)
    np.testing.assert_array_equal(_host_values(dst), [0, 1, 2, 0, 0, 0])

    # PODVector into a slice, from every memory space
    for i, src_alloc in enumerate(ALLOCATORS):
        values = [10 * i + 7, 10 * i + 8]
        src = getattr(amr, f"PODVector_int_{src_alloc}").from_array(values)
        dst.copy_from(src, offset=4)
        np.testing.assert_array_equal(_host_values(dst), [0, 1, 2, 0] + values)
    dst.copy_from([7, 8], offset=4)

    # does not fit: nothing is written
    with pytest.raises(IndexError):
        dst.copy_from([5, 5], offset=5)
    with pytest.raises(IndexError):
        dst.copy_from(amr.PODVector_int_std.from_array([5, 5]), offset=5)
    with pytest.raises(IndexError):
        dst.copy_from([5], offset=-1)
    with pytest.raises(IndexError):
        dst.copy_from_numpy(np.array([5, 5], dtype=np.int32), 5)
    with pytest.raises(IndexError):
        dst.copy_from_numpy(np.array([5], dtype=np.int32), -1)
    with pytest.raises(IndexError):
        dst.copy_from_podvector(amr.PODVector_int_std.from_array([5]), -1)
    np.testing.assert_array_equal(_host_values(dst), [0, 1, 2, 0, 7, 8])

    # empty input is a no-op
    dst.copy_from([], offset=6)


def test_copy_from_self():
    podv = amr.PODVector_int_std.from_array([1, 2, 3, 4])

    # copying a vector onto itself is a no-op
    podv.copy_from(podv)
    np.testing.assert_array_equal(podv.to_numpy(), [1, 2, 3, 4])

    # the whole source is copied, so it does not fit at a non-zero offset
    with pytest.raises(IndexError):
        podv.copy_from(podv, offset=1)


def test_from_array_numpy_without_cupy_dpnp(without_cupy_dpnp):
    """Host data into any memory space never needs CuPy or dpnp"""
    for alloc in ALLOCATORS:
        podv = getattr(amr, f"PODVector_real_{alloc}").from_array([1.0, 2.0])
        result = _host_values(podv)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], result.dtype))


@pytest.mark.parametrize("dst_alloc", ALLOCATORS)
def test_from_array_device_array(dst_alloc):
    """CuPy/dpnp input into every memory space, with a cast"""
    xp = _device_xp()
    values = np.array([1, -2, 5, 8], dtype=np.int32)
    arr = xp.asarray(values)
    dst_type = getattr(amr, f"PODVector_real_{dst_alloc}")

    dst = dst_type.from_array(arr)
    assert isinstance(dst, dst_type)
    result = _host_values(dst)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    # strided device input
    dst = dst_type.from_array(xp.asarray(values)[::2])
    result = _host_values(dst)
    np.testing.assert_array_equal(result, values[::2].astype(result.dtype))

    # empty device input
    assert dst_type.from_array(xp.asarray(values)[:0]).size() == 0

    # into a slice
    dst = dst_type.from_array(np.zeros(6))
    dst.copy_from(arr, offset=2)
    result = _host_values(dst)
    np.testing.assert_array_equal(
        result, np.concatenate([[0, 0], values]).astype(result.dtype)
    )


def test_from_array_device_array_not_1d():
    xp = _device_xp()
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array(xp.zeros((2, 3), dtype=xp.float32))
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array(xp.zeros((2, 1), dtype=xp.float32))


@pytest.mark.skipif(
    not amr.Config.have_gpu or amr.Config.gpu_backend == "SYCL",
    reason="requires AMReX CUDA or HIP support",
)
def test_from_array_cupy_non_default_stream():
    """Data written on a non-default CuPy stream is complete in the vector"""
    cp = pytest.importorskip("cupy")

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        arr = cp.zeros(1 << 20, dtype=cp.float32)
        for _ in range(50):
            arr += 1
        podv = amr.DeviceVector_real.from_array(arr)
    result = podv.to_host().to_numpy(copy=True)
    assert (result == 50).all()


@pytest.mark.parametrize("alloc", ALLOCATORS)
def test_copy_from_uint64(alloc, forbid_cupy_dpnp):
    """Full 64 bit values, e.g., particle idcpu"""
    values = np.array([0, 1, 2**63 + 5, 2**64 - 1], dtype=np.uint64)
    dst = getattr(amr, f"PODVector_uint64_{alloc}")(6)
    dst.copy_from(values, offset=2)
    np.testing.assert_array_equal(_host_values(dst)[2:], values)


def test_array_kind():
    from amrex.extensions.dlpack_helpers import array_kind

    class HostWithCAI:
        """A host container that also exposes the CUDA array interface"""

        __array_interface__ = np.zeros(1).__array_interface__
        __cuda_array_interface__ = {}

    class CAIOnly:
        __cuda_array_interface__ = {}

    class OtherDevice:
        def __dlpack_device__(self):
            return (7, 0)  # kDLVulkan

    assert array_kind([1, 2]) == "numpy"
    assert array_kind(np.zeros(2)) == "numpy"
    assert array_kind(amr.Vector_Real([1.0])) == "numpy"
    assert array_kind(HostWithCAI()) == "numpy"
    assert array_kind(CAIOnly()) == "cupy"
    assert array_kind(OtherDevice()) == "dlpack"


def test_from_array_other_dlpack_device(forbid_cupy_dpnp):
    """Unknown DLPack devices are copied to the host by the producer"""

    class OtherDevice:
        def __init__(self, arr):
            self._arr = arr

        def __dlpack__(self, **kwargs):
            return self._arr.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return (7, 0)  # kDLVulkan

    values = np.array([3.0, 1.0, 2.0])
    try:
        podv = amr.DeviceVector_real.from_array(OtherDevice(values))
    except TypeError:
        # NumPy < 2.1 cannot request a copy to the host
        assert np.lib.NumpyVersion(np.__version__) < "2.1.0"
        return
    result = _host_values(podv)
    np.testing.assert_array_equal(result, values.astype(result.dtype))


def test_from_array_copy_argument():
    podv = amr.DeviceVector_real.from_array([1.0])
    assert amr.DeviceVector_real.from_array(podv, copy=np.False_) is podv
    with pytest.raises(ValueError):
        amr.DeviceVector_real.from_array(podv, copy="no")


KINDS = ["numpy", "cupy", "dpnp"]


@pytest.mark.parametrize("src_kind", KINDS)
@pytest.mark.parametrize("dst_kind", KINDS)
@pytest.mark.parametrize("src_is_podvector", [False, True])
@pytest.mark.parametrize("same_element_type", [False, True])
@pytest.mark.parametrize("have_module", [False, True])
def test_copy_path(
    src_kind, dst_kind, src_is_podvector, same_element_type, have_module
):
    """The copy path for every source and destination memory"""
    from amrex.extensions.PODVector import _copy_path

    if src_is_podvector and same_element_type:
        # same element type: AMReX copies between any memory spaces
        expected = "amrex"
    elif src_kind == dst_kind == "cupy" and have_module:
        expected = "device"
    elif src_kind == dst_kind == "dpnp" and have_module:
        expected = "device"
    else:
        # host data, different memory, or no CuPy/dpnp
        expected = "host"

    path = _copy_path(
        src_kind, dst_kind, src_is_podvector, same_element_type, have_module
    )
    assert path == expected


@pytest.mark.skipif(
    not amr.Config.have_gpu or amr.Config.gpu_backend == "SYCL",
    reason="requires AMReX CUDA or HIP support",
)
def test_from_array_cai_only_producer():
    """Device data that only implements the CUDA array interface"""
    cp = pytest.importorskip("cupy")

    class CAIOnly:
        def __init__(self, arr):
            self._arr = arr

        @property
        def __cuda_array_interface__(self):
            return self._arr.__cuda_array_interface__

    arr = cp.arange(4, dtype=cp.float32)
    podv = amr.DeviceVector_real.from_array(CAIOnly(arr))
    result = podv.to_host().to_numpy(copy=True)
    np.testing.assert_array_equal(result, np.arange(4, dtype=result.dtype))


def _cuda_cupy():
    """CuPy with a CUDA device, independent of the AMReX build, or skip"""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    return cp


@pytest.mark.parametrize("dst_alloc", ALLOCATORS)
def test_from_array_cupy_any_build(dst_alloc):
    """CuPy input on any build, including CPU-only pyAMReX builds"""
    cp = _cuda_cupy()
    values = np.array([1, -2, 5, 8], dtype=np.int32)
    dst_type = getattr(amr, f"PODVector_real_{dst_alloc}")

    dst = dst_type.from_array(cp.asarray(values))
    result = _host_values(dst)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    dst = dst_type.from_array(np.zeros(6))
    dst.copy_from(cp.asarray(values), offset=2)
    result = _host_values(dst)
    np.testing.assert_array_equal(
        result, np.concatenate([[0, 0], values]).astype(result.dtype)
    )


def test_from_array_subclass(forbid_cupy_dpnp):
    """Python subclasses of the pyAMReX PODVector classes"""

    class MyIntVector(amr.PODVector_int_std):
        pass

    base = amr.PODVector_int_std.from_array([1, 2, 3])

    # as the target type
    mine = MyIntVector.from_array(base)
    assert isinstance(mine, MyIntVector)
    np.testing.assert_array_equal(mine.to_numpy(), [1, 2, 3])

    # as the source, same and different element type
    np.testing.assert_array_equal(
        amr.PODVector_int_std.from_array(mine).to_numpy(), [1, 2, 3]
    )
    result = _host_values(amr.DeviceVector_real.from_array(mine))
    np.testing.assert_array_equal(result, np.array([1, 2, 3], result.dtype))

    base.copy_from(mine, offset=0)
    assert MyIntVector.from_array(mine, copy=None) is mine


def test_synchronized_export():
    """pyAMReX managed memory is exported to CuPy with stream=None"""
    from amrex.extensions.dlpack_helpers import _SynchronizedExport

    class Producer:
        def __dlpack_device__(self):
            return (13, 0)  # kDLCUDAManaged

        def __dlpack__(self, **kwargs):
            self.kwargs = kwargs
            return "capsule"

    producer = Producer()
    export = _SynchronizedExport(producer)
    assert export.__dlpack_device__() == (13, 0)
    assert export.__dlpack__(stream=1234, max_version=(1, 1)) == "capsule"
    assert producer.kwargs == {"stream": None, "max_version": (1, 1)}


@pytest.mark.skipif(
    not amr.Config.have_gpu or amr.Config.gpu_backend != "CUDA",
    reason="requires AMReX CUDA support",
)
def test_managed_cupy_non_default_stream():
    """CuPy data into and out of managed memory on a non-blocking stream"""
    cp = pytest.importorskip("cupy")

    values = np.arange(1 << 16, dtype=np.int32)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        arr = cp.asarray(values)
        arr += 1
        # managed destination: a CuPy view of AMReX memory
        dst = amr.PODVector_real_managed.from_array(np.zeros(values.size + 2))
        dst.copy_from(arr, offset=2)
        # managed source with a cast: a CuPy view of AMReX memory
        src = amr.PODVector_int_managed.from_array(values)
        cast = amr.PODVector_real_device.from_array(src)
        view = src.to_cupy()
        view_sum = int(view.sum())

    result = _host_values(dst)
    np.testing.assert_array_equal(result[2:], (values + 1).astype(result.dtype))
    result = _host_values(cast)
    np.testing.assert_array_equal(result, values.astype(result.dtype))
    assert view_sum == int(values.sum())
