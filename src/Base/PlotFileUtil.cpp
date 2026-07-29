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

namespace py = pybind11;
using namespace amrex;

void init_PlotFileUtil(py::module &m) {
  m.def("write_single_level_plotfile", &amrex::WriteSingleLevelPlotfile,
        "Writes single level plotfile", py::arg("plotfilename"), py::arg("mf"),
        py::arg("varnames"), py::arg("geom"), py::arg("time"),
        py::arg("level_step"), py::arg("versionName") = "HyperCLaw-V1.1",
        py::arg("levelPrefix") = "Level_", py::arg("mfPrefix") = "Cell",
        py::arg_v("extra_dirs", Vector<std::string>(), "list[str]"));

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
        py::arg("plotfilename"), py::arg("mf"), py::arg("varnames"),
        py::arg("geom"), py::arg("time"), py::arg("level_steps"),
        py::arg("ref_ratio"), py::arg("versionName") = "HyperCLaw-V1.1",
        py::arg("levelPrefix") = "Level_", py::arg("mfPrefix") = "Cell",
        py::arg_v("extra_dirs", std::vector<std::string>(), "list[str]"));

  py::class_<PlotFileData>(m, "PlotFileData")
      // explicitly provide constructor argument types
      .def(py::init<std::string const&>())

      .def("spaceDim", &PlotFileData::spaceDim)
      .def("time", &PlotFileData::time)
      .def("finestLevel", &PlotFileData::finestLevel)
      .def("refRatio", &PlotFileData::refRatio)
      .def("levelStep", &PlotFileData::levelStep)
      .def("boxArray", &PlotFileData::boxArray)
      .def("DistributionMap", &PlotFileData::DistributionMap)
      .def("syncDistributionMap", py::overload_cast<PlotFileData const&>(&PlotFileData::syncDistributionMap))
      .def("syncDistributionMap", py::overload_cast<int, PlotFileData const&>(&PlotFileData::syncDistributionMap))

      .def("coordSys", &PlotFileData::coordSys)
      .def("probDomain", &PlotFileData::probDomain)
      .def("probSize", &PlotFileData::probSize)
      .def("probLo", &PlotFileData::probLo)
      .def("probHi", &PlotFileData::probHi)
      .def("cellSize", &PlotFileData::cellSize)
      .def("varNames", &PlotFileData::varNames)
      .def("nComp", &PlotFileData::nComp)
      .def("nGrowVect", &PlotFileData::nGrowVect)

      .def("get", py::overload_cast<int>(&PlotFileData::get))
      .def("get", py::overload_cast<int, std::string const&>(&PlotFileData::get));
}
