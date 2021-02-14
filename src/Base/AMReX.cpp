#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

#include <string>

namespace py = pybind11;
using namespace amrex;

void init_AMReX(py::module& m)
{
    py::class_<AMReX>(m, "AMReX")
        .def_static("empty", &AMReX::empty)
        .def_static("size", &AMReX::size)
        .def_static("erase", &AMReX::erase)
        .def_static("top", &AMReX::top,
                    py::return_value_policy::reference)
        ;

    m.def("initialize",
          [](const py::list args) {
              Vector<std::string> cargs{"pyamrex"};
              Vector<char*> argv{&cargs.back()[0]};

              // Populate the "command line"
              for (const auto& v: args) {
                  cargs.push_back(v.cast<std::string>());
                  argv.push_back(&cargs.back()[0]);
              }

              int argc = argv.size();
              char** tmp = argv.data();
              const bool build_parm_parse = (cargs.size() > 1);
              // TODO: handle version with MPI
              return Initialize(argc, tmp, build_parm_parse);
          }, py::return_value_policy::reference,
          "Initialize AMReX library");

    m.def("finalize",
          py::overload_cast<>(&Finalize));
    m.def("finalize", py::overload_cast<AMReX*>(&Finalize));

    m.def("space_dim", [](){ return AMREX_SPACEDIM; });
}
