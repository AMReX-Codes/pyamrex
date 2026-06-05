/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <vector>


void init_MLPoisson(py::module& m)
{
    using namespace amrex;

    py::class_<MLPoisson, MLCellABecLap>(m, "MLPoisson",
            "A Poisson operator: del dot grad phi")
        .def(py::init(
                 [](std::vector<Geometry> const & geom,
                    std::vector<BoxArray> const & grids,
                    std::vector<DistributionMapping> const & dmap,
                    LPInfo const & info)
                 {
                     return std::make_unique<MLPoisson>(
                         Vector<Geometry>(geom.begin(), geom.end()),
                         Vector<BoxArray>(grids.begin(), grids.end()),
                         Vector<DistributionMapping>(dmap.begin(),
                                                     dmap.end()),
                         info);
                 }),
             py::arg("geom"), py::arg("grids"), py::arg("dmap"),
             py::arg_v("info", LPInfo(), "LPInfo()"),
             "Construct a Poisson operator on a hierarchy of levels "
             "(one entry per level for a composite solve; a single "
             "entry for a level-by-level solve).")
    ;
}
