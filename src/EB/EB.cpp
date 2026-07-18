/* Copyright 2022 The AMReX Community
 *
 * Authors: Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_EB2.H>


void init_EBFabFactory (nb::module_& m);

void init_EB (nb::module_& m)
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
        nb::arg("geom"), nb::arg("required_coarsening_level"), nb::arg("max_coarsening_level"),
        nb::arg("ngrow") = 4, nb::arg("build_coarse_level_by_coarsening") = true,
        nb::arg("extend_domain_face") = EB2::ExtendDomainFace(),
        nb::arg("num_coarsen_opt") = EB2::NumCoarsenOpt(),
        "EB generation"
    );

    init_EBFabFactory(m);
}
