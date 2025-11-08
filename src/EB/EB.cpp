/* Copyright 2022 The AMReX Community
 *
 * Authors: Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_EB2.H>


void init_EBFabFactory (py::module& m);

void init_EB (py::module& m)
{
    using namespace amrex;

    m.def(
        "EB2_Build",
        [] (Geometry const& geom, int required_coarsening_level, int max_coarsening_level,
            int ngrow, bool build_coarse_level_by_coarsening, bool extend_domain_face,
            int num_coarsen_opt)
        {
            EB2::Build(geom, required_coarsening_level, max_coarsening_level, ngrow,
                       build_coarse_level_by_coarsening, extend_domain_face, num_coarsen_opt);
        },
        py::arg("geom"), py::arg("required_coarsening_level"), py::arg("max_coarsening_level"),
        py::arg("ngrow") = 4, py::arg("build_coarse_level_by_coarsening") = true,
        py::arg("extend_domain_face") = EB2::ExtendDomainFace(),
        py::arg("num_coarsen_opt") = EB2::NumCoarsenOpt(),
        "EB generation"
    );

    init_EBFabFactory(m);
}
