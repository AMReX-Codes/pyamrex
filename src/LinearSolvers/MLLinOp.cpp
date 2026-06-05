/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MLCellABecLap.H>
#include <AMReX_MLCellLinOp.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>

#include <array>


void init_MLLinOp(py::module& m)
{
    using namespace amrex;

    py::native_enum<LinOpBCType>(m, "LinOpBCType", "enum.IntEnum",
            "Boundary condition types of linear operators "
            "(AMReX_LO_BCTYPES.H)")
        .value("interior", LinOpBCType::interior)
        .value("Dirichlet", LinOpBCType::Dirichlet)
        .value("Neumann", LinOpBCType::Neumann)
        .value("reflect_odd", LinOpBCType::reflect_odd)
        .value("Marshak", LinOpBCType::Marshak)
        .value("SanchezPomraning", LinOpBCType::SanchezPomraning)
        .value("inflow", LinOpBCType::inflow)
        .value("inhomogNeumann", LinOpBCType::inhomogNeumann)
        .value("Robin", LinOpBCType::Robin)
        .value("symmetry", LinOpBCType::symmetry)
        .value("Periodic", LinOpBCType::Periodic)
        .value("bogus", LinOpBCType::bogus)
        .finalize()
    ;

    py::native_enum<BottomSolver>(m, "BottomSolver", "enum.IntEnum",
            "The solver used on the coarsest level of the multigrid "
            "hierarchy")
        .value("Default", BottomSolver::Default)
        .value("smoother", BottomSolver::smoother)
        .value("bicgstab", BottomSolver::bicgstab)
        .value("cg", BottomSolver::cg)
        .value("bicgcg", BottomSolver::bicgcg)
        .value("cgbicg", BottomSolver::cgbicg)
        .value("hypre", BottomSolver::hypre)
        .value("petsc", BottomSolver::petsc)
        .finalize()
    ;

    py::native_enum<LinOpEnumType::Location>(m, "Location", "enum.IntEnum",
            "The location of data, e.g., for fluxes returned by MLMG")
        .value("FaceCenter", LinOpEnumType::Location::FaceCenter)
        .value("FaceCentroid", LinOpEnumType::Location::FaceCentroid)
        .value("CellCenter", LinOpEnumType::Location::CellCenter)
        .value("CellCentroid", LinOpEnumType::Location::CellCentroid)
        .finalize()
    ;

    py::class_<LPInfo>(m, "LPInfo",
            "Information and parameters for the construction of linear "
            "operators")
        .def(py::init<>())
        .def_readwrite("do_agglomeration", &LPInfo::do_agglomeration)
        .def_readwrite("do_consolidation", &LPInfo::do_consolidation)
        .def_readwrite("do_semicoarsening", &LPInfo::do_semicoarsening)
        .def_readwrite("agg_grid_size", &LPInfo::agg_grid_size)
        .def_readwrite("con_grid_size", &LPInfo::con_grid_size)
        .def_readwrite("has_metric_term", &LPInfo::has_metric_term)
        .def_readwrite("max_coarsening_level", &LPInfo::max_coarsening_level)
        .def_readwrite("max_semicoarsening_level",
                       &LPInfo::max_semicoarsening_level)
        .def("set_agglomeration", &LPInfo::setAgglomeration, py::arg("x"))
        .def("set_consolidation", &LPInfo::setConsolidation, py::arg("x"))
        .def("set_semicoarsening", &LPInfo::setSemicoarsening, py::arg("x"))
        .def("set_agglomeration_grid_size",
             &LPInfo::setAgglomerationGridSize, py::arg("x"))
        .def("set_consolidation_grid_size",
             &LPInfo::setConsolidationGridSize, py::arg("x"))
        .def("set_metric_term", &LPInfo::setMetricTerm, py::arg("x"))
        .def("set_max_coarsening_level",
             &LPInfo::setMaxCoarseningLevel, py::arg("n"))
        .def("set_max_semicoarsening_level",
             &LPInfo::setMaxSemicoarseningLevel, py::arg("n"))
        .def("set_semicoarsening_direction",
             &LPInfo::setSemicoarseningDirection, py::arg("n"))
    ;

    // the linear operator class hierarchy (MultiFab instantiations);
    // the bases are abstract: no Python constructors
    py::class_<MLLinOp>(m, "MLLinOp",
            "Base class of the linear operators that MLMG can solve")
        .def("set_verbose", &MLLinOp::setVerbose, py::arg("v"))
        .def("set_max_order", &MLLinOp::setMaxOrder, py::arg("o"))
        .def("set_domain_bc",
             [](MLLinOp & linop,
                std::array<LinOpBCType, AMREX_SPACEDIM> const & lobc,
                std::array<LinOpBCType, AMREX_SPACEDIM> const & hibc)
             {
                 Array<LinOpBCType, AMREX_SPACEDIM> lo, hi;
                 for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                     lo[d] = lobc[d];
                     hi[d] = hibc[d];
                 }
                 linop.setDomainBC(lo, hi);
             },
             py::arg("lobc"), py::arg("hibc"),
             "Set the boundary conditions at the domain boundaries, per "
             "side and direction.")
        .def("set_level_bc",
             [](MLLinOp & linop, int amrlev, MultiFab const * levelbcdata)
             { linop.setLevelBC(amrlev, levelbcdata); },
             py::arg("amrlev"), py::arg("levelbcdata").none(true),
             // the linop stores a copy of the boundary values; no
             // keep_alive needed
             "Set the boundary values (on inhomogeneous Dirichlet or "
             "Neumann boundaries) of an AMR level. The ghost cells of "
             "levelbcdata that lie on the physical domain boundary are "
             "used; pass None for homogeneous boundaries.")
        .def("set_coarse_fine_bc",
             [](MLLinOp & linop, MultiFab const * crse, int crse_ratio)
             { linop.setCoarseFineBC(crse, crse_ratio); },
             py::arg("crse").none(true), py::arg("crse_ratio"),
             // the linop stores the raw pointer: keep the coarse data
             // alive as long as the linop lives
             py::keep_alive<1, 2>(),
             "For a solve on a level (or levels) above the coarsest AMR "
             "level: provide the coarse data at the coarse/fine boundary "
             "below the solve's coarsest level.")
        .def("set_coarse_fine_bc",
             [](MLLinOp & linop, MultiFab const * crse,
                IntVect const & crse_ratio)
             { linop.setCoarseFineBC(crse, crse_ratio); },
             py::arg("crse").none(true), py::arg("crse_ratio"),
             py::keep_alive<1, 2>())
    ;

    py::class_<MLCellLinOp, MLLinOp>(m, "MLCellLinOp",
        "Base class of the cell-centered linear operators");
    py::class_<MLCellABecLap, MLCellLinOp>(m, "MLCellABecLap",
        "Base class of the cell-centered linear operators of the form "
        "(A alpha - B div (beta grad)) phi");
}
