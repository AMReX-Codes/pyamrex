/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <vector>


void init_MLABecLaplacian(py::module& m)
{
    using namespace amrex;

    py::class_<MLABecLaplacian, MLCellABecLap>(m, "MLABecLaplacian",
            "An ABec Laplacian operator: "
            "(alpha * a - beta * del dot b grad) phi")
        .def(py::init(
                 [](std::vector<Geometry> const & geom,
                    std::vector<BoxArray> const & grids,
                    std::vector<DistributionMapping> const & dmap,
                    LPInfo const & info)
                 {
                     return std::make_unique<MLABecLaplacian>(
                         Vector<Geometry>(geom.begin(), geom.end()),
                         Vector<BoxArray>(grids.begin(), grids.end()),
                         Vector<DistributionMapping>(dmap.begin(),
                                                     dmap.end()),
                         info);
                 }),
             py::arg("geom"), py::arg("grids"), py::arg("dmap"),
             py::arg_v("info", LPInfo(), "LPInfo()"),
             "Construct an ABec Laplacian operator on a hierarchy of "
             "levels (one entry per level for a composite solve; a "
             "single entry for a level-by-level solve).")

        .def("set_scalars",
             [](MLABecLaplacian & linop, Real a, Real b)
             { linop.setScalars(a, b); },
             py::arg("a"), py::arg("b"),
             "Set the scalars alpha and beta.")
        .def("set_a_coeffs",
             [](MLABecLaplacian & linop, int amrlev, MultiFab const & alpha)
             { linop.setACoeffs(amrlev, alpha); },
             py::arg("amrlev"), py::arg("alpha"),
             "Set the cell-centered coefficients a of an AMR level "
             "(the data is copied).")
        .def("set_a_coeffs",
             [](MLABecLaplacian & linop, int amrlev, Real alpha)
             { linop.setACoeffs(amrlev, alpha); },
             py::arg("amrlev"), py::arg("alpha"),
             "Set the coefficient a of an AMR level to a constant.")
        .def("set_b_coeffs",
             [](MLABecLaplacian & linop, int amrlev,
                std::vector<MultiFab*> const & beta)
             {
                 if (beta.size() != AMREX_SPACEDIM) {
                     throw py::value_error(
                         "beta must have exactly AMREX_SPACEDIM entries "
                         "(one face-centered MultiFab per direction)");
                 }
                 Array<MultiFab const*, AMREX_SPACEDIM> b;
                 for (int d = 0; d < AMREX_SPACEDIM; ++d) { b[d] = beta[d]; }
                 linop.setBCoeffs(amrlev, b);
             },
             py::arg("amrlev"), py::arg("beta"),
             "Set the face-centered coefficients b of an AMR level, one "
             "MultiFab per direction (the data is copied).")
        .def("set_b_coeffs",
             [](MLABecLaplacian & linop, int amrlev, Real beta)
             { linop.setBCoeffs(amrlev, beta); },
             py::arg("amrlev"), py::arg("beta"),
             "Set the coefficient b of an AMR level to a constant.")
    ;
}
