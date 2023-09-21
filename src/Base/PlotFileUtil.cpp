/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>

#include <sstream>
#include <string>

namespace py = pybind11;
using namespace amrex;

void init_PlotFileUtil(py::module& m)
{
    m.def("write_single_level_plotfile",
          &amrex::WriteSingleLevelPlotfile,
          "Writes single level plotfile",
          py::arg("plotfilename"),
          py::arg("mf"),
          py::arg("varnames"),
          py::arg("geom"),
          py::arg("time"),
          py::arg("level_step"),
          py::arg("versionName")="HyperCLaw-V1.1",
          py::arg("levelPrefix")="Level_",
          py::arg("mfPrefix")="Cell",
          py::arg_v("extra_dirs", Vector<std::string>(), "list[str]")
    );
}
