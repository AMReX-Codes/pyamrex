/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrMesh.H>
#ifdef AMREX_PARTICLES
#   include <AMReX_AmrParGDB.H>
#endif

namespace py = pybind11;
using namespace amrex;


void init_AmrCore(py::module &m) {
    py::class_< AmrCore, AmrMesh >(m, "AmrCore")
        .def("__repr__",
            [](AmrCore const &) {
                return "<amrex.AmrCore>";
            }
        )

        /* Note: cannot be constructed due to purely virtual functions
        .def(py::init< >())
        .def(py::init<
                const RealBox&,
                int,
                const Vector<int>&,
                int,
                Vector<IntVect> const&,
                Array<int,AMREX_SPACEDIM> const&
             >(),
             py::arg("rb"), py::arg("max_level_in"), py::arg("n_cell_in"), py::arg("coord"), py::arg("ref_ratios"), py::arg("is_per"))
        .def(py::init< Geometry const&, AmrInfo const& >(),
             py::arg("level_0_geom"), py::arg("amr_info"))
        //AmrCore (AmrCore&& rhs)
        */

#ifdef AMREX_PARTICLES
        .def("GetParGDB", &AmrCore::GetParGDB)
#endif

        .def("InitFromScratch", &AmrCore::InitFromScratch)
        .def("regrid", &AmrCore::regrid)
        .def("printGridSummary", &AmrCore::printGridSummary)
    ;
}
