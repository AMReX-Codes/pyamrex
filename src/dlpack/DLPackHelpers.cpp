/* Copyright 2025-2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "DLPackHelpers.H"

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>

#include <cstring>
#include <memory>
#include <optional>


namespace
{
    using namespace pyAMReX::dlpack;

    /** Owns everything the exported tensor refers to.
     *
     * Referenced from DLManagedTensor(Versioned)::manager_ctx and destroyed
     * by the DLPack deleter, which the consumer calls exactly once.
     */
    struct TensorHolder
    {
        //! owned reference keeping the producing Python object alive
        PyObject* producer = nullptr;
        std::vector<std::int64_t> shape;
        std::vector<std::int64_t> strides;
        //! frees a producer-made copy of the data, if any
        std::function<void()> free_data;
        //! exactly one of these two is handed to the consumer
        DLManagedTensorVersioned versioned{};
        DLManagedTensor legacy{};
    };

    void destroy_holder (TensorHolder* holder)
    {
        if (holder == nullptr) { return; }
        if (holder->free_data) { holder->free_data(); }
        if (holder->producer != nullptr && Py_IsInitialized()) {
            // the consumer may run the DLPack deleter on a thread that does
            // not hold the GIL
            PyGILState_STATE state = PyGILState_Ensure();
            Py_DECREF(holder->producer);
            PyGILState_Release(state);
        }
        delete holder;
    }

    void deleter_versioned (DLManagedTensorVersioned* self)
    {
        destroy_holder(static_cast<TensorHolder*>(self->manager_ctx));
    }

    void deleter_legacy (DLManagedTensor* self)
    {
        destroy_holder(static_cast<TensorHolder*>(self->manager_ctx));
    }

    void capsule_destructor_versioned (PyObject* capsule)
    {
        // consumers rename the capsule to "used_dltensor_versioned" once
        // consumed and take over the deleter call; nothing to do then
        if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
            auto* mt = static_cast<DLManagedTensorVersioned*>(
                PyCapsule_GetPointer(capsule, "dltensor_versioned"));
            if (mt != nullptr && mt->deleter != nullptr) { mt->deleter(mt); }
        }
    }

    void capsule_destructor_legacy (PyObject* capsule)
    {
        if (PyCapsule_IsValid(capsule, "dltensor")) {
            auto* mt = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(capsule, "dltensor"));
            if (mt != nullptr && mt->deleter != nullptr) { mt->deleter(mt); }
        }
    }

    std::int64_t numel (std::vector<std::int64_t> const& shape)
    {
        std::int64_t n = 1;
        for (auto const extent : shape) { n *= extent; }
        return n;
    }

    bool is_compact_row_major (DLPackInfo const& info)
    {
        if (info.strides.empty()) { return true; }
        std::int64_t expected = 1;
        for (auto d = info.shape.size(); d-- > 0; ) {
            if (info.shape[d] > 1 && info.strides[d] != expected) { return false; }
            expected *= info.shape[d];
        }
        return true;
    }

    //! can the host access this memory in place?
    bool is_host_accessible (DLDeviceType device_type)
    {
        return device_type == kDLCPU ||
               device_type == kDLCUDAHost ||
               device_type == kDLROCMHost ||
               device_type == kDLCUDAManaged;
    }

    bool want_versioned_capsule (py::object const& max_version)
    {
        if (max_version.is_none()) { return false; }
        if (!(py::isinstance<py::tuple>(max_version) ||
              py::isinstance<py::list>(max_version)))
        {
            throw py::type_error(
                "__dlpack__: max_version must be None or a "
                "(major, minor) tuple of int");
        }
        auto const seq = py::reinterpret_borrow<py::sequence>(max_version);
        if (py::len(seq) < 2) {
            throw py::type_error(
                "__dlpack__: max_version must be None or a "
                "(major, minor) tuple of int");
        }
        auto const major = py::cast<long>(py::int_(seq[0]));
        return major >= 1;
    }

    std::optional<DLDevice> parse_dl_device (py::object const& dl_device)
    {
        if (dl_device.is_none()) { return std::nullopt; }
        if (!(py::isinstance<py::tuple>(dl_device) ||
              py::isinstance<py::list>(dl_device)))
        {
            throw py::type_error(
                "__dlpack__: dl_device must be None or a "
                "(device_type, device_id) tuple");
        }
        auto const seq = py::reinterpret_borrow<py::sequence>(dl_device);
        if (py::len(seq) != 2) {
            throw py::type_error(
                "__dlpack__: dl_device must be None or a "
                "(device_type, device_id) tuple");
        }
        DLDevice device{};
        // py::int_ conversion accepts both plain int and IntEnum members
        device.device_type = static_cast<DLDeviceType>(
            py::cast<std::int32_t>(py::int_(seq[0])));
        device.device_id = py::cast<std::int32_t>(py::int_(seq[1]));
        return device;
    }

    std::optional<bool> parse_copy (py::object const& copy)
    {
        if (copy.is_none()) { return std::nullopt; }
        if (!py::isinstance<py::bool_>(copy)) {
            throw py::type_error("__dlpack__: copy must be None, True or False");
        }
        return py::cast<bool>(copy);
    }

    //! CUDA/ROCm-family memory: consumers (e.g., CuPy) pass their stream
    //! also for pinned and managed tensors
    bool stream_capable (DLDevice const& device)
    {
        return device.device_type == kDLCUDA ||
               device.device_type == kDLCUDAHost ||
               device.device_type == kDLCUDAManaged ||
               device.device_type == kDLROCM ||
               device.device_type == kDLROCMHost;
    }

    /** Validate the `stream` argument against the exported device.
     *
     * Raises on illegal stream/device combinations per the DLPack Python
     * spec (e.g., a non-None stream for host/SYCL data, or the reserved
     * stream values for CUDA/ROCm). Performs no synchronization, so it is
     * safe to call on both the zero-copy and the copy path.
     *
     * @param is_copy the caller will hand over a producer-made copy, which is
     *                synchronized on the AMReX stream before hand-off and
     *                cannot run on a consumer-provided stream; stream=None is
     *                then required (any other value raises)
     */
    void check_stream (py::object const& stream, DLDevice const& device,
                       bool is_copy = false)
    {
        if (!stream_capable(device)) {
            if (!stream.is_none()) {
                throw py::value_error(
                    "__dlpack__: 'stream' must be None for this device type");
            }
            return;
        }
        if (is_copy) {
            // A producer-made copy is issued and synchronized on the AMReX
            // stream. DLPack requires a copy to run on the consumer-provided
            // stream, which we cannot honor; require stream=None so the copy
            // is synchronized and safe to use on any consumer stream.
            if (!stream.is_none()) {
                throw py::buffer_error(
                    "__dlpack__: copy=True requires stream=None; a "
                    "producer-made copy is synchronized before hand-off and "
                    "cannot be executed on a consumer-provided stream");
            }
            return;
        }
        if (stream.is_none()) { return; }
        if (!py::isinstance<py::int_>(stream)) {
            throw py::type_error("__dlpack__: stream must be None or int");
        }
        auto const s = py::cast<std::intptr_t>(stream);
        if (s < -1) {
            // DLPack permits stream >= -1; smaller values would otherwise be
            // reinterpreted as a (garbage) stream handle
            throw py::value_error(
                "__dlpack__: stream must be None or an int >= -1");
        }
        if (s == -1) { return; }  // consumer requests no synchronization
#if defined(AMREX_USE_CUDA)
        if (s == 0) {
            throw py::value_error(
                "__dlpack__: stream=0 is ambiguous on CUDA per the DLPack "
                "spec; use None, 1, or 2");
        }
#elif defined(AMREX_USE_HIP)
        if (s == 1 || s == 2) {
            throw py::value_error(
                "__dlpack__: stream=1 and stream=2 are not supported on ROCm "
                "per the DLPack spec; use None, 0, or a stream handle");
        }
#endif
    }

    /** Make the exported data ready for the consumer, per the DLPack stream
     *  semantics of the exported device.
     *
     * Only for the zero-copy path: a copy already fully synchronizes the
     * producer stream, so its callers use check_stream() instead.
     */
    void handle_stream (py::object const& stream, DLDevice const& device)
    {
        check_stream(stream, device);

        if (!stream_capable(device)) {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP) || defined(AMREX_USE_SYCL)
            // host or SYCL export: make all pending device work on the data
            // visible to any consumer
            amrex::Gpu::streamSynchronize();
#endif
            return;
        }

        // stream=None: the consumer did not name a stream (it may even read
        // from the host, e.g., NumPy on pinned/managed memory) -> make the
        // data ready everywhere
        if (stream.is_none()) {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
            amrex::Gpu::streamSynchronize();
#endif
            return;
        }

#if defined(AMREX_USE_CUDA)
        auto const s = py::cast<std::intptr_t>(stream);
        if (s == -1) { return; }  // consumer requests no synchronization
        cudaStream_t const consumer =
            (s == 1) ? cudaStreamLegacy :
            (s == 2) ? cudaStreamPerThread :
                       reinterpret_cast<cudaStream_t>(s);  // NOLINT(performance-no-int-to-ptr)
        cudaStream_t const producer = amrex::Gpu::gpuStream();
        if (consumer == producer) { return; }
        // make the consumer stream wait for pending producer work without
        // blocking the host
        cudaEvent_t event;
        AMREX_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        AMREX_CUDA_SAFE_CALL(cudaEventRecord(event, producer));
        AMREX_CUDA_SAFE_CALL(cudaStreamWaitEvent(consumer, event, 0));
        // CUDA defers the destruction until the event completed
        AMREX_CUDA_SAFE_CALL(cudaEventDestroy(event));
#elif defined(AMREX_USE_HIP)
        auto const s = py::cast<std::intptr_t>(stream);
        if (s == -1) { return; }  // consumer requests no synchronization
        hipStream_t const consumer =
            (s == 0) ? hipStream_t{nullptr} :  // default stream
                       reinterpret_cast<hipStream_t>(s);  // NOLINT(performance-no-int-to-ptr)
        hipStream_t const producer = amrex::Gpu::gpuStream();
        if (consumer == producer) { return; }
        hipEvent_t event;
        AMREX_HIP_SAFE_CALL(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        AMREX_HIP_SAFE_CALL(hipEventRecord(event, producer));
        AMREX_HIP_SAFE_CALL(hipStreamWaitEvent(consumer, event, 0));
        AMREX_HIP_SAFE_CALL(hipEventDestroy(event));
#endif
    }

    /** Copy the tensor data into a fresh allocation of the same memory space
     *  (or to the host for device-to-host requests).
     *
     * Updates info.data/device/strides and returns the matching deallocator.
     */
    std::function<void()> make_copy (DLPackInfo& info, bool to_host)
    {
        if (!is_compact_row_major(info)) {
            throw py::buffer_error(
                "__dlpack__: copying a non-contiguous tensor is not supported");
        }
        auto const n = numel(info.shape);
        auto const nbytes = static_cast<std::size_t>(n) * info.itemsize;
        info.strides.clear();  // a compact copy needs no strides

        if (n == 0) {
            info.data = nullptr;
            if (to_host) { info.device = DLDevice{kDLCPU, 0}; }
            return {};
        }

        if (info.device.device_type == kDLCPU) {
            // plain host copy
            void* buf = ::operator new(nbytes);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP) || defined(AMREX_USE_SYCL)
            amrex::Gpu::streamSynchronize();  // pending device writers
#endif
            std::memcpy(buf, info.data, nbytes);
            info.data = buf;
            return [buf] () { ::operator delete(buf); };
        }

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP) || defined(AMREX_USE_SYCL)
        if (to_host) {
            // device-to-host transfer, staged into pinned (USM) host memory:
            // copies into pageable host memory are not supported on all
            // backends (e.g., SYCL Level Zero)
            amrex::Arena* arena = amrex::The_Pinned_Arena();
            void* buf = arena->alloc(nbytes);
            amrex::Gpu::dtoh_memcpy_async(buf, info.data, nbytes);
            amrex::Gpu::streamSynchronize();
            info.data = buf;
            info.device = DLDevice{kDLCPU, 0};
            return [arena, buf] () { arena->free(buf); };
        }
#else
        amrex::ignore_unused(to_host);
#endif

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP) || defined(AMREX_USE_SYCL)
        // same-device copy, preserving the memory space
        amrex::Arena* arena = nullptr;
        switch (info.device.device_type) {
            case kDLCUDAHost:
            case kDLROCMHost:
                arena = amrex::The_Pinned_Arena();
                break;
            case kDLCUDAManaged:
                arena = amrex::The_Managed_Arena();
                break;
            default:  // kDLCUDA, kDLROCM, kDLOneAPI
                arena = amrex::The_Device_Arena();
                break;
        }
        void* buf = arena->alloc(nbytes);
        if (is_host_accessible(info.device.device_type)) {
            amrex::Gpu::streamSynchronize();  // pending device writers
            std::memcpy(buf, info.data, nbytes);
        } else {
            // synchronize the copy before returning: the fresh buffer and the
            // source must both be safe for the deleter to free (buf) and for
            // the producer to be released (source), and the data is then ready
            // on any consumer stream without running the copy on it
            amrex::Gpu::dtod_memcpy_async(buf, info.data, nbytes);
            amrex::Gpu::streamSynchronize();
        }
        info.data = buf;
        return [arena, buf] () { arena->free(buf); };
#else
        throw py::buffer_error(
            "__dlpack__: cannot copy memory of a device type this build "
            "does not support");
#endif
    }
}

namespace pyAMReX::dlpack
{
    DLDevice detect_device_from_pointer ([[maybe_unused]] void const* ptr,
                                         bool* host_accessible)
    {
        DLDevice device {kDLCPU, 0};
        // plain host memory (and the GPU-less default) is host-accessible;
        // set false only for pure device memory below
        bool host_ok = true;

#if defined(AMREX_USE_CUDA)
        cudaPointerAttributes attr;
        cudaError_t const err = cudaPointerGetAttributes(&attr, ptr);
        if (err == cudaSuccess) {
            if (attr.type == cudaMemoryTypeDevice) {
                device = DLDevice{kDLCUDA, attr.device};
                host_ok = false;
            } else if (attr.type == cudaMemoryTypeManaged) {
                // DLPack convention (see dlpack.h): device_id is 0 for
                // vanilla CPU, pinned and managed memory
                device = DLDevice{kDLCUDAManaged, 0};  // host-accessible
            } else if (attr.type == cudaMemoryTypeHost) {
                device = DLDevice{kDLCUDAHost, 0};  // pinned: host-accessible
            }
            // cudaMemoryTypeUnregistered: plain host memory -> kDLCPU
        } else {
            (void)cudaGetLastError();  // clear the error state
        }
#elif defined(AMREX_USE_HIP)
        hipPointerAttribute_t attr;
        hipError_t const err = hipPointerGetAttributes(&attr, ptr);
        if (err == hipSuccess) {
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR < 6)
            auto const type = attr.memoryType;
#else
            auto const type = attr.type;
#endif
            if (type == hipMemoryTypeDevice) {
                device = DLDevice{kDLROCM, attr.device};
                host_ok = false;
            } else if (type == hipMemoryTypeUnified ||
                       type == hipMemoryTypeManaged) {
                // DLPack has no managed type for ROCm; the device can access
                // it, and so can the host (unified/managed)
                device = DLDevice{kDLROCM, attr.device};  // host-accessible
            } else if (type == hipMemoryTypeHost) {
                device = DLDevice{kDLROCMHost, 0};  // pinned: host-accessible
            }
        } else {
            (void)hipGetLastError();  // clear the error state
        }
#elif defined(AMREX_USE_SYCL)
        try {
            auto const& context = amrex::Gpu::Device::syclContext();
            auto const usm_type = sycl::get_pointer_type(ptr, context);

            if (usm_type == sycl::usm::alloc::device ||
                usm_type == sycl::usm::alloc::shared)
            {
                device.device_type = kDLOneAPI;
                device.device_id = 0;
                // device USM is device-only; shared USM is host-accessible
                host_ok = (usm_type == sycl::usm::alloc::shared);
                auto const dev = sycl::get_pointer_device(ptr, context);
                auto const devices = context.get_devices();
                for (std::size_t i = 0; i < devices.size(); ++i) {
                    if (devices[i] == dev) {
                        device.device_id = static_cast<std::int32_t>(i);
                        break;
                    }
                }
            }
            // sycl::usm::alloc::host (pinned): host-accessible everywhere,
            // keep kDLCPU so host consumers (NumPy et al.) can view it
            // sycl::usm::alloc::unknown: plain host memory -> kDLCPU
        } catch (sycl::exception const&) {
            // not a USM pointer: keep kDLCPU
        }
#endif

        if (host_accessible) { *host_accessible = host_ok; }
        return device;
    }

    DLDevice device_from_arena ([[maybe_unused]] amrex::Arena const* arena,
                                bool* host_accessible)
    {
        // mirror detect_device_from_pointer() from the arena's kind, so an
        // empty container reports the same device as it will once allocated
        // (isManaged / isDevice / isPinned are mutually exclusive)
        if (host_accessible) {
            // managed, pinned and plain-host arenas are host-accessible;
            // ask the arena directly so this stays correct for custom arenas
            *host_accessible = arena->isHostAccessible();
        }
#if defined(AMREX_USE_CUDA)
        if (arena->isManaged()) { return DLDevice{kDLCUDAManaged, 0}; }
        if (arena->isPinned())  { return DLDevice{kDLCUDAHost, 0}; }
        if (arena->isDevice())  { return DLDevice{kDLCUDA, amrex::Gpu::Device::deviceId()}; }
#elif defined(AMREX_USE_HIP)
        // DLPack has no managed type for ROCm; managed maps to the device
        if (arena->isManaged()) { return DLDevice{kDLROCM, amrex::Gpu::Device::deviceId()}; }
        if (arena->isPinned())  { return DLDevice{kDLROCMHost, 0}; }
        if (arena->isDevice())  { return DLDevice{kDLROCM, amrex::Gpu::Device::deviceId()}; }
#elif defined(AMREX_USE_SYCL)
        // shared (managed) and device USM are kDLOneAPI; host USM is re-badged
        // to kDLCPU (as in detect_device_from_pointer) so host consumers work.
        // device_id is the context-relative index: AMReX builds the SYCL
        // context with only the selected device, so it is 0 (matching the
        // pointer path), not the global Gpu::Device::deviceId()
        if (arena->isManaged()) { return DLDevice{kDLOneAPI, 0}; }
        if (arena->isDevice())  { return DLDevice{kDLOneAPI, 0}; }
#endif
        return DLDevice{kDLCPU, 0};
    }

    py::capsule make_dlpack_capsule (
        py::object producer,
        DLPackInfo info,
        bool versioned,
        bool copied,
        std::function<void()> free_data)
    {
        auto holder = std::make_unique<TensorHolder>();
        holder->producer = producer ? producer.release().ptr() : nullptr;
        holder->shape = std::move(info.shape);
        holder->strides = std::move(info.strides);
        holder->free_data = std::move(free_data);

        DLTensor tensor{};
        // zero-size tensors must expose a null data pointer per the spec
        tensor.data = (numel(holder->shape) == 0) ? nullptr : info.data;
        tensor.device = info.device;
        tensor.ndim = static_cast<std::int32_t>(holder->shape.size());
        tensor.dtype = info.dtype;
        tensor.shape = holder->shape.data();
        tensor.strides = holder->strides.empty() ? nullptr : holder->strides.data();
        tensor.byte_offset = 0;

        TensorHolder* h = holder.release();
        try {
            if (versioned) {
                auto& mt = h->versioned;
                mt.version.major = DLPACK_MAJOR_VERSION;
                mt.version.minor = DLPACK_MINOR_VERSION;
                mt.manager_ctx = h;
                mt.deleter = &deleter_versioned;
                mt.flags = 0;
                // a fresh copy is solely owned by the consumer: writable
                if (info.read_only && !copied) {
                    mt.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
                }
                if (copied) { mt.flags |= DLPACK_FLAG_BITMASK_IS_COPIED; }
                mt.dl_tensor = tensor;
                return py::capsule(&mt, "dltensor_versioned",
                                   &capsule_destructor_versioned);
            } else {
                auto& mt = h->legacy;
                mt.manager_ctx = h;
                mt.deleter = &deleter_legacy;
                // the legacy struct cannot communicate read_only; like
                // __array_interface__, such exports are marked writable
                mt.dl_tensor = tensor;
                return py::capsule(&mt, "dltensor", &capsule_destructor_legacy);
            }
        } catch (...) {
            destroy_holder(h);
            throw;
        }
    }

    py::capsule dlpack_export (
        py::object producer,
        DLPackInfo info,
        py::object const& stream,
        py::object const& max_version,
        py::object const& dl_device,
        py::object const& copy)
    {
        bool const versioned = want_versioned_capsule(max_version);
        auto const requested_device = parse_dl_device(dl_device);
        auto const copy_request = parse_copy(copy);

        bool const same_device =
            !requested_device ||
            (requested_device->device_type == info.device.device_type &&
             requested_device->device_id == info.device.device_id);

        // cross-device export: only (kDLCPU, 0) requests are supported.
        // CPU is always device 0, so a non-zero id is an unsupported device.
        bool to_host = false;
        if (!same_device) {
            if (requested_device->device_type != kDLCPU ||
                requested_device->device_id != 0) {
                throw py::buffer_error(
                    "__dlpack__: exporting to a different device is only "
                    "supported for dl_device=(kDLCPU, 0)");
            }
            if (info.host_accessible) {
                // pinned / managed / shared (unified) memory: the host can use
                // it in place, so re-badge as CPU without copying. This uses
                // the pointer/arena-derived flag, since the DLPack device type
                // collapses host-accessible shared USM into kDLOneAPI/kDLROCM.
                info.device = DLDevice{kDLCPU, 0};
            } else if (copy_request.has_value() && !copy_request.value()) {
                throw py::buffer_error(
                    "__dlpack__: dl_device requests a device-to-host "
                    "transfer, which requires a copy, but copy=False was "
                    "passed");
            } else {
                to_host = true;
            }
        }

        bool const want_copy =
            to_host || (copy_request.has_value() && copy_request.value());

        std::function<void()> free_data;
        if (want_copy) {
            // validate the stream against the DESTINATION device: a
            // device-to-host copy hands over CPU memory (stream must be None),
            // a same-device copy keeps the producer's GPU device
            DLDevice const dst_device = to_host ? DLDevice{kDLCPU, 0} : info.device;
            check_stream(stream, dst_device, /*is_copy=*/true);
            // make_copy fully synchronizes, so the fresh buffer is complete and
            // the source is no longer read: it is safe both to release the
            // producer now and for the deleter to free the copy later. The copy
            // requires stream=None (enforced above), so the synchronized data
            // is ready for any subsequent consumer-stream use.
            free_data = make_copy(info, to_host);
            // the copy fully belongs to the consumer: no need to keep the
            // producer object alive
            producer = py::object();
        } else {
            handle_stream(stream, info.device);
        }

        return make_dlpack_capsule(std::move(producer), std::move(info),
                                   versioned, want_copy, std::move(free_data));
    }
}


void init_DLPack (py::module& m)
{
    py::native_enum<DLDeviceType>(m, "DLDeviceType", "enum.IntEnum")
        .value("kDLCPU", DLDeviceType::kDLCPU)
        .value("kDLCUDA", DLDeviceType::kDLCUDA)
        .value("kDLCUDAHost", DLDeviceType::kDLCUDAHost)
        .value("kDLOpenCL", DLDeviceType::kDLOpenCL)
        .value("kDLVulkan", DLDeviceType::kDLVulkan)
        .value("kDLMetal", DLDeviceType::kDLMetal)
        .value("kDLVPI", DLDeviceType::kDLVPI)
        .value("kDLROCM", DLDeviceType::kDLROCM)
        .value("kDLROCMHost", DLDeviceType::kDLROCMHost)
        .value("kDLExtDev", DLDeviceType::kDLExtDev)
        .value("kDLCUDAManaged", DLDeviceType::kDLCUDAManaged)
        .value("kDLOneAPI", DLDeviceType::kDLOneAPI)
        .value("kDLWebGPU", DLDeviceType::kDLWebGPU)
        .value("kDLHexagon", DLDeviceType::kDLHexagon)
        .value("kDLMAIA", DLDeviceType::kDLMAIA)
        .export_values()
        .finalize()
    ;
}
