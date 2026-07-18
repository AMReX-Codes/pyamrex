/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_AmrMesh.H>

#include <sstream>


void init_AmrMesh(nb::module_ &m)
{
    using namespace amrex;

    nb::class_< AmrInfo >(m, "AmrInfo")
        .def("__repr__",
            [](AmrInfo const & amr_info) {
                std::stringstream s;
                s << amr_info.max_level;
                return "<amrex.AmrInfo of max_level '" + s.str() + "'>";
            }
        )

        .def(nb::init< >())

        .def_rw("verbose", &AmrInfo::verbose)
        .def_rw("max_level", &AmrInfo::max_level)

        // These Vector members are exposed through indexed accessors to avoid
        // copying their opaque value types.
        //.def_rw("ref_ratio", &AmrInfo::ref_ratio)
        //.def_rw("blocking_factor", &AmrInfo::blocking_factor)
        //.def_rw("max_grid_size", &AmrInfo::max_grid_size)
        //.def_rw("n_error_buf", &AmrInfo::n_error_buf)
        .def("ref_ratio", [](AmrInfo const & amr_info, int lev){ return amr_info.ref_ratio.at(lev); })
        .def("blocking_factor", [](AmrInfo const & amr_info, int lev){ return amr_info.blocking_factor.at(lev); })
        .def("max_grid_size", [](AmrInfo const & amr_info, int lev){ return amr_info.max_grid_size.at(lev); })
        .def("n_error_buf", [](AmrInfo const & amr_info, int lev){ return amr_info.n_error_buf.at(lev); })

        .def_rw("grid_eff", &AmrInfo::grid_eff)
        .def_rw("n_proper", &AmrInfo::n_proper)
        .def_rw("use_fixed_upto_level", &AmrInfo::use_fixed_upto_level)
        .def_rw("use_fixed_coarse_grids", &AmrInfo::use_fixed_coarse_grids)
        .def_rw("refine_grid_layout", &AmrInfo::refine_grid_layout)
        .def_rw("refine_grid_layout_dims", &AmrInfo::refine_grid_layout_dims)
        .def_rw("check_input", &AmrInfo::check_input)
        .def_rw("use_new_chop", &AmrInfo::use_new_chop)
        .def_rw("iterate_on_new_grids", &AmrInfo::iterate_on_new_grids)

    ;

    nb::class_< AmrMesh /*, AmrInfo*/ >(m, "AmrMesh")
        .def("__repr__",
            [](AmrMesh const &) {
                return "<amrex.AmrMesh>";
            }
        )

        .def(nb::init< >())
        .def(nb::init<
                const RealBox&,
                int,
                const Vector<int>&,
                int,
                Vector<IntVect> const&,
                Array<int,AMREX_SPACEDIM> const&
             >(),
             nb::arg("rb"), nb::arg("max_level_in"), nb::arg("n_cell_in"), nb::arg("coord"), nb::arg("ref_ratios"), nb::arg("is_per"))

        .def_prop_ro("verbose", &AmrMesh::Verbose)
        .def_prop_ro("max_level", &AmrMesh::maxLevel)
        .def_prop_ro("finest_level", &AmrMesh::finestLevel)
        .def("ref_ratio", nb::overload_cast< >(&AmrMesh::refRatio, nb::const_))
        .def("ref_ratio", nb::overload_cast< int >(&AmrMesh::refRatio, nb::const_))

        .def("geom",
             nb::overload_cast< int >(&AmrMesh::Geom, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("lev"),
             "Return the Geometry stored for AMR level lev.")
        .def("set_geometry", &AmrMesh::SetGeometry,
             nb::arg("lev"), nb::arg("geom_in"),
             "Replace the Geometry stored for AMR level lev.")
    ;
}
