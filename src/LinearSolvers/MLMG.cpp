/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Array.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <vector>


namespace
{
    using namespace amrex;

    Vector<MultiFab*> to_vector (std::vector<MultiFab*> const & in)
    {
        return Vector<MultiFab*>(in.begin(), in.end());
    }

    Vector<MultiFab const*> to_const_vector (std::vector<MultiFab*> const & in)
    {
        Vector<MultiFab const*> out;
        out.reserve(in.size());
        for (auto const * ptr : in) { out.push_back(ptr); }
        return out;
    }

    /** per-level lists of per-direction (face) MultiFabs */
    Vector<Array<MultiFab*, AMREX_SPACEDIM>>
    to_face_vector (std::vector<std::vector<MultiFab*>> const & in)
    {
        Vector<Array<MultiFab*, AMREX_SPACEDIM>> out(int(in.size()));
        for (std::size_t lev = 0; lev < in.size(); ++lev) {
            if (in[lev].size() != AMREX_SPACEDIM) {
                throw py::value_error(
                    "expected AMREX_SPACEDIM face MultiFabs (one per "
                    "direction) per level");
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                out[int(lev)][d] = in[lev][d];
            }
        }
        return out;
    }
}


void init_MLMG(py::module& m)
{
    using namespace amrex;

    py::class_<MLMG>(m, "MLMG",
            "A multi-level multigrid solver for linear operators")
        .def(py::init<MLLinOp&>(), py::arg("linop"),
             // MLMG stores a reference to the operator
             py::keep_alive<1, 2>())

        .def("set_verbose", &MLMG::setVerbose, py::arg("v"))
        .def("set_max_iter", &MLMG::setMaxIter, py::arg("n"))
        .def("set_max_fmg_iter", &MLMG::setMaxFmgIter, py::arg("n"))
        .def("set_fixed_iter", &MLMG::setFixedIter, py::arg("nit"))
        .def("set_bottom_verbose", &MLMG::setBottomVerbose, py::arg("v"))
        .def("set_bottom_solver", &MLMG::setBottomSolver, py::arg("s"))
        .def("set_bottom_max_iter", &MLMG::setBottomMaxIter, py::arg("n"))
        .def("set_bottom_tolerance", &MLMG::setBottomTolerance,
             py::arg("tol"))
        .def("set_pre_smooth", &MLMG::setPreSmooth, py::arg("n"))
        .def("set_post_smooth", &MLMG::setPostSmooth, py::arg("n"))

        .def("solve",
             [](MLMG & mlmg, std::vector<MultiFab*> const & sol,
                std::vector<MultiFab*> const & rhs,
                Real tol_rel, Real tol_abs)
             {
                 return mlmg.solve(to_vector(sol), to_const_vector(rhs),
                                   tol_rel, tol_abs);
             },
             py::arg("sol"), py::arg("rhs"),
             py::arg("tol_rel"), py::arg("tol_abs"),
             "Solve the linear system (one MultiFab per level, the "
             "level(s) the linear operator was built on). sol provides "
             "the initial guess and returns the solution; returns the "
             "final residual norm.")

        .def("get_grad_solution",
             [](MLMG & mlmg,
                std::vector<std::vector<MultiFab*>> const & grad_sol,
                LinOpEnumType::Location loc)
             { mlmg.getGradSolution(to_face_vector(grad_sol), loc); },
             py::arg("grad_sol"),
             py::arg_v("loc", LinOpEnumType::Location::FaceCenter,
                       "Location.FaceCenter"),
             "After a solve, compute grad(sol) into per-level lists of "
             "per-direction face MultiFabs.")

        .def("get_fluxes",
             [](MLMG & mlmg,
                std::vector<std::vector<MultiFab*>> const & fluxes,
                LinOpEnumType::Location loc)
             { mlmg.getFluxes(to_face_vector(fluxes), loc); },
             py::arg("fluxes"),
             py::arg_v("loc", LinOpEnumType::Location::FaceCenter,
                       "Location.FaceCenter"),
             "After a solve, compute the fluxes -b grad(sol) into "
             "per-level lists of per-direction face MultiFabs.")

        .def("get_fluxes_cc",
             [](MLMG & mlmg, std::vector<MultiFab*> const & fluxes,
                LinOpEnumType::Location loc)
             { mlmg.getFluxes(to_vector(fluxes), loc); },
             py::arg("fluxes"),
             py::arg_v("loc", LinOpEnumType::Location::CellCenter,
                       "Location.CellCenter"),
             "After a solve, compute the fluxes -b grad(sol) into "
             "cell-centered MultiFabs with AMREX_SPACEDIM components.")

        .def("comp_residual",
             [](MLMG & mlmg, std::vector<MultiFab*> const & res,
                std::vector<MultiFab*> const & sol,
                std::vector<MultiFab*> const & rhs)
             {
                 mlmg.compResidual(to_vector(res), to_vector(sol),
                                   to_const_vector(rhs));
             },
             py::arg("res"), py::arg("sol"), py::arg("rhs"),
             "Compute the residual res = rhs - L(sol).")
    ;
}
