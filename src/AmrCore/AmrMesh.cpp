/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_AmrMesh.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


void init_AmrMesh(py::module &m) {
    py::class_< AmrInfo >(m, "AmrInfo")
        .def("__repr__",
            [](AmrInfo const & amr_info) {
                std::stringstream s;
                s << amr_info.max_level;
                return "<amrex.AmrInfo of max_level '" + s.str() + "'>";
            }
        )

        .def(py::init< >())
    ;

    py::class_< AmrMesh /*, AmrInfo*/ >(m, "AmrMesh")
        .def("__repr__",
            [](AmrMesh const &) {
                return "<amrex.AmrMesh>";
            }
        )

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

        .def("Verbose", &AmrMesh::Verbose)
        .def("max_level", &AmrMesh::maxLevel)
        .def("finest_level", &AmrMesh::finestLevel)
        .def("ref_ratio", py::overload_cast< >(&AmrMesh::refRatio, py::const_))
        .def("ref_ratio", py::overload_cast< int >(&AmrMesh::refRatio, py::const_))
        
    ;
}
