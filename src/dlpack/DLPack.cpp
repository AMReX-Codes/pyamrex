#include "pyAMReX.H"

#include "dlpack.h"


void init_DLPack(py::module& m)
{
    using namespace amrex;

    // register types only if not already present, e.g., from another library
    // that also implements DLPack bindings and exposes the types

    // TODO: py::type pyDLDeviceType = py::type::of<DLDeviceType>();
    bool pyDLDeviceType = false;
    if (!pyDLDeviceType) {
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
        ;
    }

}
