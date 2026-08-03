/* Copyright 2021-2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX.H>
#include <AMReX_SIMD.H>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>


namespace amrex {
   struct Config {};
}

namespace
{
    using ConfigValue = std::variant<
        bool,
        int,
        std::string,
        std::optional<std::string>
    >;
    struct ConfigEntry
    {
        ConfigValue value;
        char const * doc;
    };
    using ConfigMap = std::map<std::string, ConfigEntry>;

    std::string config_repr (ConfigMap const & config)
    {
        std::size_t name_width = 0;
        for (auto const & entry : config)
        {
            if (entry.first.size() > name_width)
            {
                name_width = entry.first.size();
            }
        }

        std::string repr = "amrex.Config:";
        for (auto const & [name, entry] : config)
        {
            repr += "\n    " + name;
            repr.append(name_width - name.size(), ' ');
            repr += " = ";
            repr += py::repr(py::cast(entry.value)).cast<std::string>();
        }
        return repr;
    }
}

void init_Config (py::module& m)
{
    using namespace amrex;

    std::optional<std::string> gpu_backend;
#ifdef AMREX_USE_CUDA
    gpu_backend = "CUDA";
#elif defined(AMREX_USE_HIP)
    gpu_backend = "HIP";
#elif defined(AMREX_USE_DPCPP)
    gpu_backend = "SYCL";
#endif

    std::shared_ptr<ConfigMap const> const config = std::make_shared<ConfigMap>(
        ConfigMap{
            {"amrex_version", {
                Version(),
                "AMReX library version"}},

            {"gpu_backend", {
                gpu_backend,
                "GPU backend ('CUDA', 'HIP' or 'SYCL'), None without GPU support"}},

            {"have_eb", {
#ifdef AMREX_USE_EB
                true,
#else
                false,
#endif
                "Build supports embedded boundaries"}},

            {"have_gpu", {
#ifdef AMREX_USE_GPU
                true,
#else
                false,
#endif
                "Build supports GPUs"}},

            {"have_mpi", {
#ifdef AMREX_USE_MPI
                true,
#else
                false,
#endif
                "Build supports MPI"}},

            {"have_omp", {
#ifdef AMREX_USE_OMP
                true,
#else
                false,
#endif
                "Build supports OpenMP"}},

            {"have_simd", {
#ifdef AMREX_USE_SIMD
                true,
#else
                false,
#endif
                "Build supports explicit SIMD vectorization"}},

            {"precision", {
#ifdef AMREX_USE_FLOAT
                std::string{"SINGLE"},
#else
                std::string{"DOUBLE"},
#endif
                "Floating point precision of amrex::Real ('SINGLE' or 'DOUBLE')"}},

            {"precision_particles", {
#ifdef AMREX_SINGLE_PRECISION_PARTICLES
                std::string{"SINGLE"},
#else
                std::string{"DOUBLE"},
#endif
                "Floating point precision of amrex::ParticleReal ('SINGLE' or 'DOUBLE')"}},

            {"simd_size", {
                static_cast<int>(amrex::simd::native_simd_size_real),
                "Number of amrex::Real elements in a native SIMD vector"}},

            {"spacedim", {
                AMREX_SPACEDIM,
                "Number of spatial dimensions (AMREX_SPACEDIM)"}}
        }
    );

    py::dict config_metaclass_namespace;
    config_metaclass_namespace["__module__"] = m.attr("__name__");
    config_metaclass_namespace["__repr__"] = py::cpp_function(
        [config]() {
            return config_repr(*config);
        }
    );
    py::object const amrex_class = m.attr("AMReX");
    py::object const pybind11_metaclass = py::type::of(amrex_class);
    py::object const config_metaclass = py::type::of(pybind11_metaclass)(
        "ConfigMeta",
        py::make_tuple(pybind11_metaclass),
        config_metaclass_namespace
    );

    py::class_<Config> pyAMReXConfig(
        m, "Config", py::metaclass(config_metaclass)
    );
    for (auto const & kv : *config)
    {
        std::string const & name = kv.first;
        ConfigEntry const & entry = kv.second;
        pyAMReXConfig.def_property_readonly_static(
            name.c_str(),
            [config, name](py::object const &) {
                return config->at(name).value;
            },
            entry.doc
        );
    }
    pyAMReXConfig.def_static(
        "to_dict",
        [config]() {
            py::dict d;
            for (auto const & [name, entry] : *config)
            {
                d[name.c_str()] = entry.value;
            }
            return d;
        },
        "Return the AMReX build configuration as a dictionary."
    );

    // runtime-mutable option, not part of the static build configuration
    pyAMReXConfig.def_property_static(
        "verbose",
        [](py::object const &) { return Verbose(); },
        [](py::object const &, int const v) { SetVerbose(v); },
        "Verbosity level of AMReX outputs"
    );
}
