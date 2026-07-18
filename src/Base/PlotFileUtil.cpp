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

namespace nb = nanobind;
using namespace amrex;

void init_PlotFileUtil(nb::module_ &m) {
  m.def("write_single_level_plotfile", &amrex::WriteSingleLevelPlotfile,
        "Writes single level plotfile", nb::arg("plotfilename"), nb::arg("mf"),
        nb::arg("varnames"), nb::arg("geom"), nb::arg("time"),
        nb::arg("level_step"), nb::arg("versionName") = "HyperCLaw-V1.1",
        nb::arg("levelPrefix") = "Level_", nb::arg("mfPrefix") = "Cell",
        nb::arg("extra_dirs") = Vector<std::string>());

  nb::class_<PlotFileData>(m, "PlotFileData")
      // explicitly provide constructor argument types
      .def(nb::init<std::string const&>())

      .def("spaceDim", &PlotFileData::spaceDim)
      .def("time", &PlotFileData::time)
      .def("finestLevel", &PlotFileData::finestLevel)
      .def("refRatio", &PlotFileData::refRatio)
      .def("levelStep", &PlotFileData::levelStep)
      .def("boxArray", &PlotFileData::boxArray)
      .def("DistributionMap", &PlotFileData::DistributionMap)
      .def("syncDistributionMap", nb::overload_cast<PlotFileData const&>(&PlotFileData::syncDistributionMap))
      .def("syncDistributionMap", nb::overload_cast<int, PlotFileData const&>(&PlotFileData::syncDistributionMap))

      .def("coordSys", &PlotFileData::coordSys)
      .def("probDomain", &PlotFileData::probDomain)
      .def("probSize", &PlotFileData::probSize)
      .def("probLo", &PlotFileData::probLo)
      .def("probHi", &PlotFileData::probHi)
      .def("cellSize", &PlotFileData::cellSize)
      .def("varNames", &PlotFileData::varNames)
      .def("nComp", &PlotFileData::nComp)
      .def("nGrowVect", &PlotFileData::nGrowVect)

      .def("get", nb::overload_cast<int>(&PlotFileData::get))
      .def("get", nb::overload_cast<int, std::string const&>(&PlotFileData::get));
}
