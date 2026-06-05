/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLNodeLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <vector>


void init_MLNodeLaplacian(py::module& m)
{
    using namespace amrex;

    py::class_<MLNodeLinOp, MLLinOp>(m, "MLNodeLinOp",
        "Base class of the nodal linear operators");

    py::class_<MLNodeLaplacian, MLNodeLinOp>(m, "MLNodeLaplacian",
            "A nodal Laplacian operator: del dot (sigma grad) phi, with "
            "phi and the right hand side on nodes and sigma at cell "
            "centers")
        .def(py::init(
                 [](std::vector<Geometry> const & geom,
                    std::vector<BoxArray> const & grids,
                    std::vector<DistributionMapping> const & dmap,
                    LPInfo const & info)
                 {
                     return std::make_unique<MLNodeLaplacian>(
                         Vector<Geometry>(geom.begin(), geom.end()),
                         Vector<BoxArray>(grids.begin(), grids.end()),
                         Vector<DistributionMapping>(dmap.begin(),
                                                     dmap.end()),
                         info);
                 }),
             py::arg("geom"), py::arg("grids"), py::arg("dmap"),
             py::arg_v("info", LPInfo(), "LPInfo()"),
             "Construct a nodal Laplacian operator on a hierarchy of "
             "levels (one entry per level for a composite solve; a "
             "single entry for a level-by-level solve).")

        .def("set_sigma",
             [](MLNodeLaplacian & linop, int amrlev, MultiFab const & sigma)
             { linop.setSigma(amrlev, sigma); },
             py::arg("amrlev"), py::arg("sigma"),
             "Set the cell-centered coefficients sigma of an AMR level "
             "(the data is copied).")
        .def("set_normalization_threshold",
             &MLNodeLaplacian::setNormalizationThreshold, py::arg("t"))
    ;
}
