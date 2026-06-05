/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <sstream>
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
        [](std::string const& plotfilename, int nlevels,
           std::vector<MultiFab*> const& mf,
           std::vector<std::string> const& varnames,
           std::vector<Geometry> const& geom, Real time,
           std::vector<int> const& level_steps,
           std::vector<IntVect> const& ref_ratio,
           std::string const& versionName, std::string const& levelPrefix,
           std::string const& mfPrefix,
           std::vector<std::string> const& extra_dirs)
        {
            Vector<const MultiFab*> c_mf;
            c_mf.reserve(mf.size());
            for (auto const* ptr : mf) { c_mf.push_back(ptr); }
            WriteMultiLevelPlotfile(
                plotfilename, nlevels, c_mf,
                Vector<std::string>(varnames.begin(), varnames.end()),
                Vector<Geometry>(geom.begin(), geom.end()), time,
                Vector<int>(level_steps.begin(), level_steps.end()),
                Vector<IntVect>(ref_ratio.begin(), ref_ratio.end()),
                versionName, levelPrefix, mfPrefix,
                Vector<std::string>(extra_dirs.begin(), extra_dirs.end()));
        },
        "Writes a multi-level plotfile", py::arg("plotfilename"),
        py::arg("nlevels"), py::arg("mf"), py::arg("varnames"),
        py::arg("geom"), py::arg("time"), py::arg("level_steps"),
        py::arg("ref_ratio"), py::arg("versionName") = "HyperCLaw-V1.1",
        py::arg("levelPrefix") = "Level_", py::arg("mfPrefix") = "Cell",
        py::arg_v("extra_dirs", std::vector<std::string>(), "list[str]"));

  m.def("level_path", &amrex::LevelPath,
        "return the path of the Level directory, e.g., Level_5",
        py::arg("level"), py::arg("levelPrefix") = "Level_");
  m.def("multifab_header_path", &amrex::MultiFabHeaderPath,
        "return the path of the MultiFab to write to the header, "
        "e.g., Level_5/Cell",
        py::arg("level"), py::arg("levelPrefix") = "Level_",
        py::arg("mfPrefix") = "Cell");
  m.def("level_full_path", &amrex::LevelFullPath,
        "return the full path of the Level directory, e.g., "
        "plt00005/Level_5",
        py::arg("level"), py::arg("plotfilename"),
        py::arg("levelPrefix") = "Level_");
  m.def("multifab_file_full_prefix", &amrex::MultiFabFileFullPrefix,
        "return the full path MultiFab prefix, e.g., plt00005/Level_5/Cell",
        py::arg("level"), py::arg("plotfilename"),
        py::arg("levelPrefix") = "Level_", py::arg("mfPrefix") = "Cell");
  m.def("pre_build_director_hierarchy", &amrex::PreBuildDirectorHierarchy,
        "prebuild a hierarchy of directories. dirName is built first; if "
        "dirName exists, it is renamed. Then dirName/Level_0 .. "
        "dirName/Level_{nSubDirs-1} are built. If callBarrier is true, "
        "ParallelDescriptor::Barrier() is called after all directories "
        "are built; ParallelDescriptor::IOProcessor() creates the "
        "directories",
        py::arg("dirName"), py::arg("subDirPrefix"), py::arg("nSubDirs"),
        py::arg("callBarrier"));

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
