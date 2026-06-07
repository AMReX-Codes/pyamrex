/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>


void init_BCUtil(py::module& m)
{
    using namespace amrex;

    m.def("fill_domain_boundary",
          &FillDomainBoundary,
          py::arg("phi"), py::arg("geom"), py::arg("bc"),
          R"pbdoc(Fill cell-centered physical-domain ghost cells.

This fills non-periodic ghost cells outside the physical domain for
BCType.foextrap, BCType.hoextrap, BCType.hoextrapcc,
BCType.reflect_even, and BCType.reflect_odd. It intentionally leaves
BCType.ext_dir and BCType.ext_dir_cc unchanged; fill those values from
application code, for example with PhysBCFunctUser.

Args:
    phi: MultiFab to modify in place. All components are processed.
    geom: Geometry defining the physical domain and periodic directions.
    bc: Vector_BCRec with one record per component in phi.

Notes:
    This function fills physical-domain ghost cells only. For multi-box
    MultiFabs, call phi.fill_boundary() separately when interior or
    periodic ghost cells also need to be valid.
)pbdoc"
    );
}
