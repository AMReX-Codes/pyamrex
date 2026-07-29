/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace amrex;

void init_PlotFileUtil(nb::module_ &m) {
  m.def("write_single_level_plotfile", &amrex::WriteSingleLevelPlotfile,
        "Writes single level plotfile", nb::arg("plotfilename"), nb::arg("mf"),
        nb::arg("varnames"), nb::arg("geom"), nb::arg("time"),
        nb::arg("level_step"), nb::arg("versionName") = "HyperCLaw-V1.1",
        nb::arg("levelPrefix") = "Level_", nb::arg("mfPrefix") = "Cell",
        nb::arg("extra_dirs") = Vector<std::string>());

  m.def("write_multi_level_plotfile",
        [](std::string const & plotfilename,
           std::vector<MultiFab const *> const & mf,
           std::vector<std::string> const & varnames,
           std::vector<Geometry> const & geom,
           Real time,
           std::vector<int> const & level_steps,
           std::vector<IntVect> const & ref_ratio,
           std::string const & versionName,
           std::string const & levelPrefix,
           std::string const & mfPrefix,
           std::vector<std::string> const & extra_dirs)
        {
            auto const nlevels = static_cast<int>(mf.size());
            if (static_cast<int>(geom.size()) != nlevels ||
                static_cast<int>(level_steps.size()) != nlevels ||
                static_cast<int>(ref_ratio.size()) < nlevels - 1)
            {
                throw std::invalid_argument(
                    "write_multi_level_plotfile: mf, geom and level_steps need "
                    "one entry per level and ref_ratio one per coarse level");
            }
            WriteMultiLevelPlotfile(
                plotfilename, nlevels,
                Vector<MultiFab const *>(mf.begin(), mf.end()),
                Vector<std::string>(varnames.begin(), varnames.end()),
                Vector<Geometry>(geom.begin(), geom.end()),
                time,
                Vector<int>(level_steps.begin(), level_steps.end()),
                Vector<IntVect>(ref_ratio.begin(), ref_ratio.end()),
                versionName, levelPrefix, mfPrefix,
                Vector<std::string>(extra_dirs.begin(), extra_dirs.end()));
        },
        "Writes a multi-level plotfile: one MultiFab, Geometry and level step "
        "per level, and one refinement ratio per coarse level.",
        nb::arg("plotfilename"), nb::arg("mf"), nb::arg("varnames"),
        nb::arg("geom"), nb::arg("time"), nb::arg("level_steps"),
        nb::arg("ref_ratio"), nb::arg("versionName") = "HyperCLaw-V1.1",
        nb::arg("levelPrefix") = "Level_", nb::arg("mfPrefix") = "Cell",
        nb::arg("extra_dirs") = std::vector<std::string>());

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
