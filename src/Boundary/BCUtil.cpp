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
          "Fill cell-centered data outside the physical domain "
          "(excluding periodic boundaries). It only fills "
          "BCType.foextrap, BCType.hoextrap, BCType.hoextrapcc, "
          "BCType.reflect_even, and BCType.reflect_odd. It does not fill "
          "BCType.ext_dir and BCType.ext_dir_cc (i.e., external "
          "Dirichlet). For BCType.ext_dir and BCType.ext_dir_cc, fill the "
          "ghost cells from Python, e.g., via a PhysBCFunctUser callback."
    );
}
