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
#include <AMReX_ParallelDescriptor.H>

#ifdef AMREX_USE_MPI
#include <mpi.h>

/** mpi4py communicator wrapper
 *
 * refs:
 * - https://github.com/mpi4py/mpi4py/blob/3.0.0/src/mpi4py/libmpi.pxd#L35-L36
 * - https://github.com/mpi4py/mpi4py/blob/3.0.0/src/mpi4py/MPI.pxd#L100-L105
 * - installed: include/mpi4py/mpi4py.MPI.h
 */
struct pyAMReX_PyMPICommObject
{
    PyObject_HEAD MPI_Comm ob_mpi;
    unsigned int flags;
};
using pyAMReX_PyMPIIntracommObject = pyAMReX_PyMPICommObject;


void init_MPMD(py::module &m) {
    using namespace amrex;

    // Several functions here are copied from AMReX.cpp
    m.def("MPMD_Initialize_without_split",
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
              MPMD::Initialize_without_split(argc, tmp);
          });

    // This is AMReX::Initialize when MPMD exists
    m.def("initialize_when_MPMD",
          [](const py::list args, py::object &app_comm_py) {
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

              //! TODO perform mpi4py import test and check min-version
              //!       careful: double MPI_Init risk? only import mpi4py.MPI?
              //!       required C-API init? probably just checks:
              //! refs:
              //! -
              //! https://bitbucket.org/mpi4py/mpi4py/src/3.0.0/demo/wrap-c/helloworld.c
              //! - installed: include/mpi4py/mpi4py.MPI_api.h
              //auto m_mpi4py = py::module::import("mpi4py");
              //amrex::ignore_unused(m_mpi4py);

              if (app_comm_py.ptr() == Py_None)
                  throw std::runtime_error(
                      "MPMD: MPI communicator cannot be None.");
              if (app_comm_py.ptr() == nullptr)
                  throw std::runtime_error(
                      "MPMD: MPI communicator is a nullptr.");

              // check type string to see if this is mpi4py
              //   __str__ (pretty)
              //   __repr__ (unambiguous)
              //   mpi4py: <mpi4py.MPI.Intracomm object at 0x7f998e6e28d0>
              //   pyMPI:  ... (TODO)
              py::str const comm_pystr = py::repr(app_comm_py);
              std::string const comm_str = comm_pystr.cast<std::string>();
              if (comm_str.substr(0, 12) != std::string("<mpi4py.MPI."))
                  throw std::runtime_error(
                      "MPMD: comm is not an mpi4py communicator: " +
                      comm_str);
              // only checks same layout, e.g. an `int` in `PyObject` could
              // pass this
              if (!py::isinstance<py::class_<pyAMReX_PyMPIIntracommObject> >(
                      py::type::of(app_comm_py)))
                  // TODO add mpi4py version from above import check to error
                  // message
                  throw std::runtime_error(
                      "MPMD: comm has unexpected type layout in " +
                      comm_str +
                      " (Mismatched MPI at compile vs. runtime? "
                      "Breaking mpi4py release?)");

              // todo other possible implementations:
              // - pyMPI (inactive since 2008?): import mpi; mpi.WORLD

              // reimplementation of mpi4py's:
              // MPI_Comm* mpiCommPtr = PyMPIComm_Get(app_comm_py.ptr());
              MPI_Comm *mpiCommPtr =
                  &((pyAMReX_PyMPIIntracommObject *)(app_comm_py.ptr()))->ob_mpi;

              if (PyErr_Occurred())
                  throw std::runtime_error(
                      "MPMD: MPI communicator access error.");
              if (mpiCommPtr == nullptr)
              {
                  throw std::runtime_error(
                      "MPMD: MPI communicator cast failed. "
                      "(Mismatched MPI at compile vs. runtime?)");
              }

              return Initialize(argc, tmp, build_parm_parse, *mpiCommPtr);
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
    m.def("MPMD_AppNum",&MPMD::AppNum);
    m.def("MPMD_MyProgId",&MPMD::MyProgId);

    // Binding MPMD::Copier class
    py::class_< MPMD::Copier >(m, "MPMD_Copier")
        //! Construct an MPMD::Copier without BoxArray and DistributionMApping
        .def(py::init <bool>())
        //! Construct an MPMD::Copier with BoxArray and DistributionMApping
        .def(py::init< BoxArray const&, DistributionMapping const&,bool>(),
             py::arg("ba"),py::arg("dm"),py::arg("send_ba")=false)
        // Copier function to send data
        .def("send",&MPMD::Copier::send<FArrayBox>)
        // Copier function to receive data
        .def("recv",&MPMD::Copier::recv<FArrayBox>)
        // Copier's BoxArray
        .def("box_array",&MPMD::Copier::boxArray,
                py::return_value_policy::reference_internal)
        // Copier's DistributionMapping
        .def("distribution_map",&MPMD::Copier::DistributionMap,
                py::return_value_policy::reference_internal)
    ;

}

#endif
