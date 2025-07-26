/*!
*  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 *
 * Source: https://github.com/dmlc/dlpack/blob/v1.1/include/dlpack/dlpack.h
 */
#ifndef AMREX_DLPACK_H_
#define AMREX_DLPACK_H_

#include <AMReX_Extension.H>
#include <AMReX_Gpu.H>

#include <stdexcept>
#include <type_traits>

extern "C" {

#include <stdint.h>
#include <stddef.h>

/*!
 * \brief The DLPack version.
 *
 * A change in major version indicates that we have changed the
 * data layout of the ABI - DLManagedTensorVersioned.
 *
 * A change in minor version indicates that we have added new
 * code, such as a new device type, but the ABI is kept the same.
 *
 * If an obtained DLPack tensor has a major version that disagrees
 * with the version number specified in this header file
 * (i.e. major != DLPACK_MAJOR_VERSION), the consumer must call the deleter
 * (and it is safe to do so). It is not safe to access any other fields
 * as the memory layout will have changed.
 *
 * In the case of a minor version mismatch, the tensor can be safely used as
 * long as the consumer knows how to interpret all fields. Minor version
 * updates indicate the addition of enumeration values.
 */
typedef struct {
  /*! \brief DLPack major version. */
  uint32_t major = 1;
  /*! \brief DLPack minor version. */
  uint32_t minor = 1;
} DLPackVersion;

/*!
 * \brief The device type in DLDevice.
 */
typedef enum : int32_t {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*!
   * \brief Pinned CUDA CPU memory by cudaMallocHost
   */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics. */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*!
   * \brief Pinned ROCm CPU memory allocated by hipMallocHost
   */
  kDLROCMHost = 11,
  /*!
   * \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
  /*!
   * \brief CUDA managed/unified memory allocated by cudaMallocManaged
   */
  kDLCUDAManaged = 13,
  /*!
   * \brief Unified shared memory allocated on a oneAPI non-partititioned
   * device. Call to oneAPI runtime is required to determine the device
   * type, the USM allocation type and the sycl context it is bound to.
   */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard. */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
  /*! \brief Microsoft MAIA devices */
  kDLMAIA = 17,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*!
   * \brief The device index.
   * For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*! \brief signed integer */
  kDLInt = 0U,
  /*! \brief unsigned integer */
  kDLUInt = 1U,
  /*! \brief IEEE floating point */
  kDLFloat = 2U,
  /*!
   * \brief Opaque handle type, reserved for testing purposes.
   * Frameworks need to agree on the handle data type for the exchange to be well-defined.
   */
  kDLOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kDLBfloat = 4U,
  /*!
   * \brief complex number
   * (C/C++/Python layout: compact struct per complex number)
   */
  kDLComplex = 5U,
  /*! \brief boolean */
  kDLBool = 6U,
  /*! \brief FP8 data types */
  kDLFloat8_e3m4 = 7U,
  kDLFloat8_e4m3 = 8U,
  kDLFloat8_e4m3b11fnuz = 9U,
  kDLFloat8_e4m3fn = 10U,
  kDLFloat8_e4m3fnuz = 11U,
  kDLFloat8_e5m2 = 12U,
  kDLFloat8_e5m2fnuz = 13U,
  kDLFloat8_e8m0fnu = 14U,
  /*! \brief FP6 data types
   * Setting bits != 6 is currently unspecified, and the producer must ensure it is set
   * while the consumer must stop importing if the value is unexpected.
   */
  kDLFloat6_e2m3fn = 15U,
  kDLFloat6_e3m2fn = 16U,
  /*! \brief FP4 data types
   * Setting bits != 4 is currently unspecified, and the producer must ensure it is set
   * while the consumer must stop importing if the value is unexpected.
   */
  kDLFloat4_e2m1fn = 17U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 *   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
 *   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
 *   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
 *
 *  When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
 *  for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
 */
typedef struct {
  /*!
   * \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
  uint8_t code;
  /*!
   * \brief Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /*! \brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*!
   * \brief The data pointer points to the allocated data. This will be CUDA
   * device pointer or cl_mem handle in OpenCL. It may be opaque on some device
   * types. This pointer is always aligned to 256 bytes as in CUDA. The
   * `byte_offset` field should be used to point to the beginning of the data.
   *
   * Note that as of Nov 2021, multiple libraries (CuPy, PyTorch, TensorFlow,
   * TVM, perhaps others) do not adhere to this 256 byte aligment requirement
   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
   * (after which this note will be updated); at the moment it is recommended
   * to not rely on the data pointer being correctly aligned.
   *
   * For given DLTensor, the size of memory required to store the contents of
   * data is calculated as follows:
   *
   * \code{.c}
   * static inline size_t GetDataSize(const DLTensor* t) {
   *   size_t size = 1;
   *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * \endcode
   *
   * Note that if the tensor is of size zero, then the data pointer should be
   * set to `NULL`.
   */
  void* data;
  /*! \brief The device of the tensor */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*!
   * \brief strides of the tensor (in number of elements, not bytes)
   *  can be NULL, indicating tensor is compact and row-majored.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;

// bit masks used in the DLManagedTensorVersioned

/*! \brief bit mask to indicate that the tensor is read only. */
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)

/*!
 * \brief bit mask to indicate that the tensor is a copy made by the producer.
 *
 * If set, the tensor is considered solely owned throughout its lifetime by the
 * consumer, until the producer-provided deleter is invoked.
 */
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

/*
 * \brief bit mask to indicate that whether a sub-byte type is packed or padded.
 *
 * The default for sub-byte types (ex: fp4/fp6) is assumed packed. This flag can
 * be set by the producer to signal that a tensor of sub-byte type is padded.
 */
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2UL)

/*!
 * \brief A versioned and managed C Tensor object, manage memory of DLTensor.
 *
 * This data structure is intended to facilitate the borrowing of DLTensor by
 * another framework. It is not meant to transfer the tensor. When the borrowing
 * framework doesn't need the tensor, it should call the deleter to notify the
 * host that the resource is no longer needed.
 *
 * \note This is the current standard DLPack exchange data structure.
 */
struct DLManagedTensorVersioned {
  /*!
   * \brief The API and ABI version of the current managed Tensor
   */
  DLPackVersion version;
  /*!
   * \brief the context of the original host framework.
   *
   * Stores DLManagedTensorVersioned is used in the
   * framework. It can also be NULL.
   */
  void *manager_ctx;
  /*!
   * \brief Destructor.
   *
   * This should be called to destruct manager_ctx which holds the DLManagedTensorVersioned.
   * It can be NULL if there is no way for the caller to provide a reasonable
   * destructor. The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensorVersioned *self);
  /*!
   * \brief Additional bitmask flags information about the tensor.
   *
   * By default the flags should be set to 0.
   *
   * \note Future ABI changes should keep everything until this field
   *       stable, to ensure that deleter can be correctly called.
   *
   * \sa DLPACK_FLAG_BITMASK_READ_ONLY
   * \sa DLPACK_FLAG_BITMASK_IS_COPIED
   */
  uint64_t flags;
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
};

} // extern "C"

namespace pyAMReX::dlpack
{

    template<typename T>
    AMREX_INLINE
    DLDataType get_dlpack_dtype ()
    {
        using V = std::decay_t<T>;
        DLDataType dtype{};

        if constexpr (std::is_same_v<V, float>) {
            dtype.code = kDLFloat;
            dtype.bits = 32;
            dtype.lanes = 1;
        }
        else if constexpr (std::is_same_v<V, double>) {
            dtype.code = kDLFloat;
            dtype.bits = 64;
            dtype.lanes = 1;
        }
        else if constexpr (std::is_same_v<V, int32_t>) {
            dtype.code = kDLInt;
            dtype.bits = 32;
            dtype.lanes = 1;
        }
        else if constexpr (std::is_same_v<V, int64_t>) {
            dtype.code = kDLInt;
            dtype.bits = 64;
            dtype.lanes = 1;
        }
        else if constexpr (std::is_same_v<V, uint32_t>) {
            dtype.code = kDLUInt;
            dtype.bits = 32;
            dtype.lanes = 1;
        }
        else if constexpr (std::is_same_v<V, uint64_t>) {
            dtype.code = kDLUInt;
            dtype.bits = 64;
            dtype.lanes = 1;
        }
        else {
            throw std::runtime_error("Unsupported dtype for DLPack");
        }

        return dtype;
    }

    AMREX_INLINE
    DLDevice detect_device_from_pointer ([[maybe_unused]] const void* ptr)
    {
        DLDevice device{ kDLCPU, 0 };

#ifdef AMREX_USE_CUDA
        // Check if data is on GPU by checking if pointer is in CUDA memory
        // note: cudaPointerGetAttributes is quite expensive, remove and
        //       assume device-side if need be.
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            device.device_type = kDLCUDA;
            device.device_id = attr.device;
        }
#elif defined(AMREX_USE_HIP)
        // Check if data is on GPU by checking if pointer is in HIP memory
        // note: hipPointerGetAttributes is quite expensive, remove and
        //       assume device-side if need be.
        hipPointerAttribute_t attr;
        hipError_t err = hipPointerGetAttributes(&attr, ptr);
        if (err == hipSuccess && attr.memoryType == hipMemoryTypeDevice) {
            device.device_type = kDLROCM;
            device.device_id = attr.device;
        }

#elif defined(AMREX_USE_DPCPP)
        // try {
            // Get the SYCL context and queue from AMReX
            auto const& queue = amrex::Gpu::Device::streamQueue();
            auto const& context = queue.get_context();

            // Try to get pointer attributes using SYCL USM queries
            auto usm_type = sycl::get_pointer_type(ptr, context);

            if (usm_type == sycl::usm::alloc::device ||
                usm_type == sycl::usm::alloc::shared) {
                device.device_type = kDLOneAPI;

                // Try to get the actual device from the pointer
                try {
                    auto device_ptr = sycl::get_pointer_device(ptr, context);
                    device.device_id = 0;  // Default to first device

                    auto devices = context.get_devices();
                    for (size_t i = 0; i < devices.size(); ++i) {
                        if (devices[i] == device_ptr) {
                            device.device_id = static_cast<int32_t>(i);
                            break;
                        }
                    }
                } catch (const sycl::exception&) {
                    // If we can't determine the specific device, default to 0
                    device.device_id = 0;
                }
            } else if (usm_type == sycl::usm::alloc::host) {
                // Host USM allocation - still oneAPI but accessible from host
                device.device_type = kDLOneAPI;
                device.device_id = 0;
            }
            // If usm_type is sycl::usm::alloc::unknown, it might be regular CPU memory
            // In that case, we keep the default CPU device type set above

        /*
        }
        catch (const sycl::exception&) {
            // If SYCL queries fail, assume it's regular CPU memory
            // device remains as kDLCPU, 0 (set at function start)
        } catch (...) {
            // Handle any other exceptions gracefully
            // device remains as kDLCPU, 0 (set at function start)
        }
        */
#endif

        return device;
    }

} // namespace pyAMReX::dlpack

#endif // AMREX_DLPACK_H_
