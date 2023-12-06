/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_AmrMesh.H>

#include <sstream>


void init_AmrMesh(py::module &m)
{
    using namespace amrex;

    py::class_< AmrInfo >(m, "AmrInfo")
        .def("__repr__",
            [](AmrInfo const & amr_info) {
                std::stringstream s;
                s << amr_info.max_level;
                return "<amrex.AmrInfo of max_level '" + s.str() + "'>";
            }
        )

        .def(py::init< >())

        .def_readwrite("verbose", &AmrInfo::verbose)
        .def_readwrite("max_level", &AmrInfo::max_level)

        // note: https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#making-opaque-types
        //.def_readwrite("ref_ratio", &AmrInfo::ref_ratio)
        //.def_readwrite("blocking_factor", &AmrInfo::blocking_factor)
        //.def_readwrite("max_grid_size", &AmrInfo::max_grid_size)
        //.def_readwrite("n_error_buf", &AmrInfo::n_error_buf)
        .def("ref_ratio", [](AmrInfo const & amr_info, int lev){ return amr_info.ref_ratio.at(lev); })
        .def("blocking_factor", [](AmrInfo const & amr_info, int lev){ return amr_info.blocking_factor.at(lev); })
        .def("max_grid_size", [](AmrInfo const & amr_info, int lev){ return amr_info.max_grid_size.at(lev); })
        .def("n_error_buf", [](AmrInfo const & amr_info, int lev){ return amr_info.n_error_buf.at(lev); })

        .def_readwrite("grid_eff", &AmrInfo::grid_eff)
        .def_readwrite("n_proper", &AmrInfo::n_proper)
        .def_readwrite("use_fixed_upto_level", &AmrInfo::use_fixed_upto_level)
        .def_readwrite("use_fixed_coarse_grids", &AmrInfo::use_fixed_coarse_grids)
        .def_readwrite("refine_grid_layout", &AmrInfo::refine_grid_layout)
        .def_readwrite("refine_grid_layout_dims", &AmrInfo::refine_grid_layout_dims)
        .def_readwrite("check_input", &AmrInfo::check_input)
        .def_readwrite("use_new_chop", &AmrInfo::use_new_chop)
        .def_readwrite("iterate_on_new_grids", &AmrInfo::iterate_on_new_grids)

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

        .def_property_readonly("verbose", &AmrMesh::Verbose)
        .def_property_readonly("max_level", &AmrMesh::maxLevel)
        .def_property_readonly("finest_level", &AmrMesh::finestLevel)
        .def("ref_ratio", py::overload_cast< >(&AmrMesh::refRatio, py::const_))
        .def("ref_ratio", py::overload_cast< int >(&AmrMesh::refRatio, py::const_))
    ;
}
