/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_MPMD.H>

#ifdef AMREX_USE_MPI

void init_MPMD(py::module &m) {
    using namespace amrex;

    // Several functions here are copied from AMReX.cpp
    m.def("MPMD_Initialize",
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

              return MPMD::Initialize(argc, tmp);
          }, py::return_value_policy::reference);

    // This is AMReX::Initialize when MPMD exists 
    m.def("initialize_when_MPMD",
          [](const py::list args, py::object app_comm_py) {
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
              MPI_Comm app_comm = app_comm_py.cast<MPI_Comm>();

              return Initialize(argc, tmp, build_parm_parse,app_comm);
          }, py::return_value_policy::reference);

    constexpr auto run_gc = []() {
        // explicitly run the garbage collector, so deleted objects
        // get freed.
        // This is a convenience helper/bandage for making work with Python
        // garbage collectors in various implementations more easy.
        // https://github.com/AMReX-Codes/pyamrex/issues/81
        auto m_gc = py::module::import("gc");
        auto collect = m_gc.attr("collect");
        collect();
    };
    m.def("MPMD_Finalize",
          [run_gc]() {
              run_gc();
              MPMD::Finalize();
          });
    m.def("MPMD_Initialized",&MPMD::Initialized);
    m.def("MPMD_MyProc",&MPMD::MyProc);
    m.def("MPMD_NProcs",&MPMD::NProcs);
    m.def("MPMD_MyProgId",&MPMD::MyProgId);

    // Binding MPMD::Copier class
    py::class_< MPMD::Copier >(m, "MPMD_Copier")
        //! Construct an MPMD::Copier
        .def(py::init< BoxArray const&, DistributionMapping const& >())
        // Copier function to send data
        .def("send",&MPMD::Copier::send<FArrayBox>)
        // Copier function to receive data
        .def("recv",&MPMD::Copier::recv<FArrayBox>)
    ;

}

#endif
