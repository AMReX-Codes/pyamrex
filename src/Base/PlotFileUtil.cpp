/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <sstream>
#include <string>

namespace py = pybind11;
using namespace amrex;

void init_PlotFileUtil(py::module &m) {
  m.def("write_single_level_plotfile", &amrex::WriteSingleLevelPlotfile,
        "Writes single level plotfile", py::arg("plotfilename"), py::arg("mf"),
        py::arg("varnames"), py::arg("geom"), py::arg("time"),
        py::arg("level_step"), py::arg("versionName") = "HyperCLaw-V1.1",
        py::arg("levelPrefix") = "Level_", py::arg("mfPrefix") = "Cell",
        py::arg_v("extra_dirs", Vector<std::string>(), "list[str]"));

  py::class_<PlotFileData>(m, "PlotFileData")
      .def(py::init<std::string const &>(), &PlotFileData())

      .def("spaceDim", &PlotFileData::spaceDim)
      .def("time", &PlotFileData::time)
      .def("finestLevel", &PlotFileData::finestLevel)
      .def("refRatio", &PlotFileData::refRatio)
      .def("levelStep", &PlotFileData::levelStep)
      .def("boxArray", &PlotFileData::boxArray)
      .def("DistributionMap", &PlotFileData::DistributionMap)
      .def("syncDistributionMap", &PlotFileData::syncDistributionMap)
      .def("coordSys", &PlotFileData::coordSys)
      .def("probDomain", &PlotFileData::probDomain)
      .def("probSize", &PlotFileData::probSize)
      .def("probLo", &PlotFileData::probLo)
      .def("probHi", &PlotFileData::probHi)
      .def("cellSize", &PlotFileData::cellSize)
      .def("varNames", &PlotFileData::varNames)
      .def("nComp", &PlotFileData::nComp)
      .def("nGrowVect", &PlotFileData::nGrowVect)

      // MultiFab get (int level) noexcept { return m_impl->get(level); }
      .def("get", &PlotFileData::get)

      // MultiFab get (int level, std::string const& varname) noexcept { return
      // m_impl->get(level, varname); }
      .def("get", &PlotFileData::get);
}
