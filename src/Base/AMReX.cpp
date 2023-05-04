#include <AMReX_Config.H>
#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <string>

namespace py = nanobind;
using namespace amrex;

namespace amrex {
   struct Config {};
}

void init_AMReX(py::module_& m)
{
    py::class_<AMReX>(m, "AMReX")
        .def_static("empty", &AMReX::empty)
        .def_static("size", &AMReX::size)
        .def_static("erase", &AMReX::erase)
        .def_static("top", &AMReX::top,
                    py::return_value_policy::reference)
        ;

    py::class_<Config>(m, "Config")
        .def_prop_rw_static(
            "amrex_version",
            [](py::object) { return Version(); },
            "AMReX library version")
        .def_prop_rw_static(
            "spacedim",
            [](py::object) { return AMREX_SPACEDIM; })
        .def_prop_rw_static(
            "verbose",
            [](py::object) { return Verbose(); },
            [](py::object, const int v) { SetVerbose(v); })
        .def_prop_rw_static(
            "have_mpi",
            [](py::object){
#ifdef AMREX_USE_MPI
                return true;
#else
                return false;
#endif
            })
        .def_prop_rw_static(
            "have_gpu",
            [](py::object){
#ifdef AMREX_USE_GPU
                return true;
#else
                return false;
#endif
            })
        .def_prop_rw_static(
            "have_omp",
            [](py::object){
#ifdef AMREX_USE_OMP
                return true;
#else
                return false;
#endif
            })
        .def_prop_rw_static(
            "gpu_backend",
            [](py::object){
#ifdef AMREX_USE_CUDA
                return "CUDA";
#elif defined(AMREX_USE_HIP)
                return "HIP";
#elif defined(AMREX_USE_DPCPP)
                return "SYCL";
#else
                return py::none();
#endif
            })
        ;

    m.def("initialize",
          [](const py::list args) {
              Vector<std::string> cargs{"amrex"};
              Vector<char*> argv;

              // Populate the "command line"
              for (const auto& v: args)
                  cargs.push_back(v.cast<std::string>());
              for (auto& v: cargs)
                  argv.push_back(&v[0]);
              int argc = argv.size();

              // note: +1 since there is an extra char-string array element,
              //       that ANSII C requires to be a simple NULL entry
              //       https://stackoverflow.com/a/39096006/2719194
              argv.push_back(NULL);
              char** tmp = argv.data();

              const bool build_parm_parse = (cargs.size() > 1);
              // TODO: handle version with MPI
              return Initialize(argc, tmp, build_parm_parse);
          }, py::return_value_policy::reference,
          "Initialize AMReX library");

    constexpr auto run_gc = []() {
        // explicitly run the garbage collector, so deleted objects
        // get freed.
        // This is a convenience helper/bandage for making work with Python
        // garbage collectors in various implementations more easy.
        // https://github.com/AMReX-Codes/pyamrex/issues/81
        auto m_gc = py::module_::import_("gc");
        auto collect = m_gc.attr("collect");
        collect();
    };

    m.def("finalize",
          [run_gc]() {
              run_gc();
              amrex::Finalize();
          });
    m.def("finalize",
          [run_gc](AMReX* pamrex) {
              run_gc();
              amrex::Finalize(pamrex);
          });
}
