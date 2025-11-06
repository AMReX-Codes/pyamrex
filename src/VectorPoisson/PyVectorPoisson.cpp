#include "PyVectorPoisson.H"
#include "VectorPoissonSolver.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

namespace py = pybind11;
using namespace VectorPoisson3D;

void init_VectorPoisson3D(py::module& m)
{
    // BoundaryHandler
    py::class_<BoundaryHandler>(m, "BoundaryHandler")
        .def(py::init<>())
        .def_readwrite("lobc", &BoundaryHandler::lobc)
        .def_readwrite("hibc", &BoundaryHandler::hibc);

    // VectorPoissonSolver
    py::class_<VectorPoissonSolver>(m, "VectorPoissonSolver")
        .def(py::init<const amrex::Geometry&,
                      const amrex::BoxArray&,
                      const amrex::DistributionMapping&,
                      const BoundaryHandler&>(),
             py::arg("geom"),
             py::arg("grids"),
             py::arg("dmap"),
             py::arg("bc_handler"))
        .def("solve", 
             [](VectorPoissonSolver& self,
                py::list A_list,
                py::list J_list,
                py::object mask_obj,
                amrex::Real relative_tol,
                amrex::Real absolute_tol,
                int max_iter,
                int verbose) {
                 
                 if (py::len(A_list) != 3) {
                     throw std::runtime_error("A must be a list of exactly 3 MultiFabs");
                 }
                 if (py::len(J_list) != 3) {
                     throw std::runtime_error("J must be a list of exactly 3 MultiFabs");
                 }
                 
                 amrex::Array<amrex::MultiFab*, 3> A;
                 amrex::Array<amrex::MultiFab*, 3> J;
                 
                 for (int i = 0; i < 3; ++i) {
                     A[i] = py::cast<amrex::MultiFab*>(A_list[i]);
                     J[i] = py::cast<amrex::MultiFab*>(J_list[i]);
                 }
                 
                 // Handle optional mask parameter
                 amrex::iMultiFab* mask = nullptr;
                 if (!mask_obj.is_none()) {
                     mask = py::cast<amrex::iMultiFab*>(mask_obj);
                 }
                 
                 self.solve(A, J, mask, relative_tol, absolute_tol, max_iter, verbose);
             },
             py::arg("A"),
             py::arg("J"),
             py::arg("mask") = py::none(),
             py::arg("relative_tol") = 1.0e-10,
             py::arg("absolute_tol") = 0.0,
             py::arg("max_iter") = 100,
             py::arg("verbose") = 2,
             R"(Solve the vector Poisson equation for RZ geometry.
             
             Parameters
             ----------
             A : list of MultiFab
                 Solution vector potential [A_r, A_theta, A_z]
             J : list of MultiFab
                 Source current density [J_r, J_theta, J_z]
             mask : iMultiFab or None, optional
                 Overset mask where 1 = solve, 0 = fixed value.
                 If provided, set fixed values in A before calling solve.
             relative_tol : float
                 Relative tolerance for solver
             absolute_tol : float
                 Absolute tolerance for solver
             max_iter : int
                 Maximum number of iterations
             verbose : int
                 Verbosity level
             )")
        .def("getNumIters", &VectorPoissonSolver::getNumIters,
             py::arg("component"),
             "Get number of iterations for a component")
        .def("getResidual", &VectorPoissonSolver::getResidual,
             py::arg("component"),
             "Get final residual for a component");
}