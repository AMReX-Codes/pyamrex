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


void init_MPMD(nb::module_ &m) {
    using namespace amrex;

    // Several functions here are copied from AMReX.cpp
    m.def("MPMD_Initialize_without_split",
          [](const nb::list args) {
              Vector<std::string> cargs{"amrex"};
              Vector<char*> argv;

              // Populate the "command line"
              for (const auto& v: args)
                  cargs.push_back(nb::cast<std::string>(v));
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
          [](const nb::list args, nb::object &app_comm_py) {
              Vector<std::string> cargs{"amrex"};
              Vector<char*> argv;

              // Populate the "command line"
              for (const auto& v: args)
                  cargs.push_back(nb::cast<std::string>(v));
              for (auto& v: cargs)
                  argv.push_back(&v[0]);
              int argc = argv.size();

              // note: +1 since there is an extra char-string array element,
              //       that ANSII C requires to be a simple NULL entry
              //       https://stackoverflow.com/a/39096006/2719194
              argv.push_back(NULL);
              char** tmp = argv.data();

              const bool build_parm_parse = (cargs.size() > 1);

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
              nb::str const comm_pystr = nb::repr(app_comm_py);
              std::string const comm_str = nb::cast<std::string>(comm_pystr);
              if (comm_str.substr(0, 12) != std::string("<mpi4py.MPI."))
                  throw std::runtime_error(
                      "MPMD: comm is not an mpi4py communicator: " +
                      comm_str);
              nb::object const mpi4py_intracomm =
                  nb::module_::import_("mpi4py.MPI").attr("Intracomm");
              if (!nb::isinstance(app_comm_py, mpi4py_intracomm))
                  throw std::runtime_error(
                      "MPMD: comm is not an mpi4py.MPI.Intracomm: " +
                      comm_str);

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
          }, nb::rv_policy::reference);

    constexpr auto run_gc = []() {
        // explicitly run the garbage collector, so deleted objects
        // get freed.
        // This is a convenience helper/bandage for making work with Python
        // garbage collectors in various implementations more easy.
        // https://github.com/AMReX-Codes/pyamrex/issues/81
        auto m_gc = nb::module_::import_("gc");
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
    nb::class_< MPMD::Copier >(m, "MPMD_Copier")
        //! Construct an MPMD::Copier without BoxArray and DistributionMApping
        .def(nb::init <bool>())
        //! Construct an MPMD::Copier with BoxArray and DistributionMApping
        .def(nb::init< BoxArray const&, DistributionMapping const&,bool>(),
             nb::arg("ba"),nb::arg("dm"),nb::arg("send_ba")=false)
        // Copier function to send data
        .def("send",&MPMD::Copier::send<FArrayBox>)
        // Copier function to receive data
        .def("recv",&MPMD::Copier::recv<FArrayBox>)
        // Copier's BoxArray
        .def("box_array",&MPMD::Copier::boxArray,
                nb::rv_policy::reference_internal)
        // Copier's DistributionMapping
        .def("distribution_map",&MPMD::Copier::DistributionMap,
                nb::rv_policy::reference_internal)
    ;

}

#endif
