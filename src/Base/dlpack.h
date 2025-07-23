#ifndef AMREX_DLPACK_H_
#define AMREX_DLPACK_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Device type codes
#define kDLCPU 1
#define kDLCUDA 2
#define kDLCUDAHost 3
#define kDLOpenCL 4
#define kDLVulkan 7
#define kDLMetal 8
#define kDLVPI 9
#define kDLROCM 10
#define kDLROCMHost 11
#define kDLExtDev 12

// Data type codes
#define kDLInt 0
#define kDLUInt 1
#define kDLFloat 2

// Device context
typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

// Data type
typedef struct {
    uint8_t code;   // kDLFloat=2, kDLInt=0, kDLUInt=1
    uint8_t bits;   // number of bits, e.g., 32, 64
    uint16_t lanes; // number of lanes (for vector types)
} DLDataType;

// Tensor structure
typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    int64_t* shape;
    int64_t* strides; // in elements, not bytes; can be NULL for compact
    uint64_t byte_offset;
    DLDataType dtype;
} DLTensor;

// Managed tensor with deleter
struct DLManagedTensor;
typedef void (*DLManagedTensorDeleter)(struct DLManagedTensor* self);

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    DLManagedTensorDeleter deleter;
} DLManagedTensor;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AMREX_DLPACK_H_
