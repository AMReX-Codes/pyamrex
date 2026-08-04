/* Copyright 2025-2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"
#include "dlpack.h"

#include <complex>
#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace pyAMReX
{
    // helper type traits, shared between the array interfaces and DLPack
    template <typename T>
    struct get_value_type { using value_type = T; };
    template <typename T>
    struct get_value_type<std::complex<T>> { using value_type = T; };
    template <typename T>
    using get_value_type_t = typename get_value_type<T>::value_type;

    //! helper to check if a container element type T is non-const
    template <typename T>
    constexpr bool is_not_const ()
    {
        return std::is_same_v<
            std::remove_cv_t<
                T
            >,
            T
        > &&
        std::is_same_v<
            std::remove_cv_t<
                get_value_type_t<T>
            >,
            get_value_type_t<T>
        >;
    }
}

namespace pyAMReX::dlpack
{
    /** Does the element type T have a DLPack representation?
     *
     * Containers instantiated for element types without a DLPack
     * representation (e.g., long double or particle structs) use this trait
     * to skip binding __dlpack__/__dlpack_device__ at compile time.
     */
    template <typename T>
    inline constexpr bool has_dlpack_dtype_v =
        (std::is_arithmetic_v<std::remove_cv_t<T>> &&
         !std::is_same_v<std::remove_cv_t<T>, long double>) ||
        std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
        std::is_same_v<std::remove_cv_t<T>, std::complex<double>>;

    /** DLPack element type descriptor for T */
    template <typename T>
    constexpr DLDataType get_dlpack_dtype ()
    {
        using V = std::remove_cv_t<T>;
        static_assert(has_dlpack_dtype_v<V>,
            "Type has no DLPack representation; guard callers with "
            "pyAMReX::dlpack::has_dlpack_dtype_v");

        DLDataType dtype{};
        dtype.lanes = 1;
        dtype.bits = sizeof(V) * 8;
        if constexpr (std::is_same_v<V, bool>) {
            // per common array library convention, bool is stored in 8 bits
            dtype.code = kDLBool;
        }
        else if constexpr (std::is_floating_point_v<V>) {
            dtype.code = kDLFloat;
        }
        else if constexpr (std::is_integral_v<V>) {
            dtype.code = std::is_signed_v<V> ? kDLInt : kDLUInt;
        }
        else {  // std::complex<float>, std::complex<double>
            dtype.code = kDLComplex;
        }
        return dtype;
    }

    /** Classify the memory space a pointer refers to.
     *
     * Distinguishes plain host memory (kDLCPU) from device (kDLCUDA/kDLROCM/
     * kDLOneAPI), pinned (kDLCUDAHost/kDLROCMHost) and CUDA managed
     * (kDLCUDAManaged) memory, including the device id.
     *
     * Note: the underlying pointer attribute queries are not free. Containers
     * that know their memory space at compile time (e.g., purely host-side
     * classes) can fill DLPackInfo::device directly instead.
     */
    DLDevice detect_device_from_pointer (void const* ptr);

    /** Everything a container must provide to export a DLPack tensor */
    struct DLPackInfo
    {
        //! start of the tensor data
        void* data = nullptr;
        //! memory location of the data
        DLDevice device {kDLCPU, 0};
        //! element type
        DLDataType dtype {};
        //! extent per dimension, in C index order (slowest varying first)
        std::vector<std::int64_t> shape;
        //! stride per dimension in elements (not bytes);
        //! empty means compact, C ("row-major") ordered
        std::vector<std::int64_t> strides;
        //! disallow writes through the exported tensor
        //! (communicated via versioned capsules only)
        bool read_only = false;
        //! bytes per element
        std::size_t itemsize = 0;
    };

    /** Build a DLPack PyCapsule around an existing (or copied) allocation.
     *
     * The capsule keeps `producer` alive until the consumer calls the DLPack
     * deleter (pass an empty object for producer-owned copies) and owns the
     * shape/strides storage. `free_data` is invoked from the deleter to
     * release a producer-made data copy, if any.
     *
     * @param producer  Python object owning the exported memory (or empty)
     * @param info      tensor description
     * @param versioned emit DLManagedTensorVersioned ("dltensor_versioned",
     *                  DLPack >= 1.0) instead of the legacy DLManagedTensor
     *                  ("dltensor")
     * @param copied    the data is a fresh copy solely owned by the consumer
     * @param free_data deallocates the copied data buffer (empty if none)
     */
    py::capsule make_dlpack_capsule (
        py::object producer,
        DLPackInfo info,
        bool versioned,
        bool copied = false,
        std::function<void()> free_data = {}
    );

    /** Implementation of the full __dlpack__ keyword-argument protocol.
     *
     * Validates and applies `stream` (producer-consumer synchronization),
     * `max_version` (versioned vs. legacy capsule), `dl_device` (incl.
     * device-to-host transfers) and `copy` per the DLPack Python
     * specification, then builds the capsule via make_dlpack_capsule.
     *
     * https://dmlc.github.io/dlpack/latest/python_spec.html
     */
    py::capsule dlpack_export (
        py::object producer,
        DLPackInfo info,
        py::object const& stream,
        py::object const& max_version,
        py::object const& dl_device,
        py::object const& copy
    );

    inline constexpr char const dlpack_doc[] = R"doc(
        Export the data as a DLPack capsule for zero-copy tensor exchange.

        https://dmlc.github.io/dlpack/latest/python_spec.html

        Parameters
        ----------
        stream: int, optional
            Consumer stream (CUDA/ROCm device data only) the data shall be
            made ready for, e.g., by an event wait. -1 requests no
            synchronization. Must be None for host-side data.
        max_version: tuple of int, optional
            Highest DLPack version the consumer supports. None or a major
            version below 1 yields a legacy "dltensor" capsule, otherwise a
            "dltensor_versioned" capsule (DLPack 1.1) is returned.
        dl_device: tuple of (DLDeviceType, int), optional
            Requested device of the returned tensor. Besides the producer
            device, (kDLCPU, 0) is supported: zero-copy for host-accessible
            memory, otherwise a device-to-host copy (unless copy=False).
        copy: bool, optional
            True always copies (the copy is solely owned by the consumer);
            False never copies (raises BufferError if a copy is needed);
            None (default) copies only if required.

        Returns
        -------
        A PyCapsule wrapping a DLPack managed tensor.

        Raises
        ------
        BufferError
            If an unsupported device transfer or copy is requested.

        Notes
        -----
        Capsules that own a copy of device data (copy=True on GPU data or
        device-to-host transfers) allocate from an AMReX Arena and must be
        freed (i.e., the consuming array garbage collected) before
        amrex.finalize().

        Legacy "dltensor" capsules cannot communicate read-only tensors;
        like __array_interface__, such exports are marked writable.
        )doc";

    inline constexpr char const dlpack_device_doc[] = R"doc(
        DLPack device of the data, as a (device_type, device_id) tuple of int.

        https://dmlc.github.io/dlpack/latest/python_spec.html
        )doc";

    /** Bind __dlpack__ and __dlpack_device__ on a pybind11 class.
     *
     * The class-specific part is reduced to `info_fn`, a callable
     * `DLPackInfo(Class const&)` describing the exported tensor. The
     * keyword-argument protocol, capsule creation, producer keep-alive and
     * copy/synchronization handling are shared across all containers.
     */
    template <class PyClass, class InfoFn>
    void add_dlpack (PyClass& cl, InfoFn info_fn)
    {
        using Class = typename PyClass::type;

        cl.def("__dlpack__",
            [info_fn](py::object self, py::object stream, py::object max_version,
                      py::object dl_device, py::object copy)
            {
                // non-const: writable tensors are mutable views of the data
                DLPackInfo info = info_fn(py::cast<Class&>(self));
                return dlpack_export(std::move(self), std::move(info),
                                     stream, max_version, dl_device, copy);
            },
            py::kw_only(),
            py::arg("stream") = py::none(),
            py::arg("max_version") = py::none(),
            py::arg("dl_device") = py::none(),
            py::arg("copy") = py::none(),
            dlpack_doc
        );

        cl.def("__dlpack_device__",
            [info_fn](Class& self)
            {
                DLPackInfo const info = info_fn(self);
                return std::make_tuple(
                    static_cast<std::int32_t>(info.device.device_type),
                    info.device.device_id
                );
            },
            dlpack_device_doc
        );
    }
}
