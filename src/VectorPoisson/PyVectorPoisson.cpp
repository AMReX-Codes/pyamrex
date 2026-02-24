#include "PyVectorPoisson.H"
#include "VectorPoissonSolver.H"
#include "VectorPoissonSolverNodal.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

namespace py = pybind11;
using namespace VectorPoisson3D;

// Helper to extract Array<MultiFab*, 3> from a Python list, shared by both solvers
static amrex::Array<amrex::MultiFab*, 3>
mfab_list_to_array(py::list lst, const char* name)
{
    if (py::len(lst) != 3) {
        throw std::runtime_error(
            std::string(name) + " must be a list of exactly 3 MultiFabs");
    }
    amrex::Array<amrex::MultiFab*, 3> arr;
    for (int i = 0; i < 3; ++i) {
        arr[i] = py::cast<amrex::MultiFab*>(lst[i]);
    }
    return arr;
}

// Helper to extract optional mask, shared by both solvers
static amrex::iMultiFab*
extract_mask(py::object mask_obj)
{
    if (mask_obj.is_none()) { return nullptr; }
    return py::cast<amrex::iMultiFab*>(mask_obj);
}

void init_VectorPoisson3D(py::module& m)
{
    // -------------------------------------------------------------------------
    // BoundaryHandler
    // -------------------------------------------------------------------------
    py::class_<BoundaryHandler>(m, "BoundaryHandler")
        .def(py::init<bool>(),
             py::arg("periodic_axial") = false,
             "Initialize boundary handler for vector Poisson equation.\n\n"
             "Parameters\n"
             "----------\n"
             "periodic_axial : bool, optional\n"
             "    If True, set axial (z) boundaries to periodic else Neumann. Default is False.\n")
        .def_readwrite("lobc", &BoundaryHandler::lobc)
        .def_readwrite("hibc", &BoundaryHandler::hibc);

    // -------------------------------------------------------------------------
    // VectorPoissonSolver (cell-centered, MLABecLaplacian)
    // -------------------------------------------------------------------------
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
                 auto A = mfab_list_to_array(A_list, "A");
                 auto J = mfab_list_to_array(J_list, "J");
                 self.solve(A, J, extract_mask(mask_obj),
                            relative_tol, absolute_tol, max_iter, verbose);
             },
             py::arg("A"),
             py::arg("J"),
             py::arg("mask") = py::none(),
             py::arg("relative_tol") = 1.0e-10,
             py::arg("absolute_tol") = 0.0,
             py::arg("max_iter") = 100,
             py::arg("verbose") = 2,
             R"(Solve the cell-centered vector Poisson equation for RZ geometry.

             Uses MLABecLaplacian. A, J, and mask (if provided) must be
             cell-centered MultiFabs.

             Parameters
             ----------
             A : list of MultiFab
                 Solution vector potential [A_r, A_theta, A_z]. Cell-centered.
             J : list of MultiFab
                 Source current density [J_r, J_theta, J_z]. Cell-centered.
             mask : iMultiFab or None, optional
                 Overset mask (cell-centered) where 1 = solve, 0 = fixed value.
                 If provided, set fixed values in A before calling solve.
             relative_tol : float
                 Relative tolerance for solver.
             absolute_tol : float
                 Absolute tolerance for solver.
             max_iter : int
                 Maximum number of iterations.
             verbose : int
                 Verbosity level.
             )")
        .def("getNumIters", &VectorPoissonSolver::getNumIters,
             py::arg("component"),
             "Get number of iterations for a given component (0=r, 1=theta, 2=z).")
        .def("getResidual", &VectorPoissonSolver::getResidual,
             py::arg("component"),
             "Get final residual for a given component (0=r, 1=theta, 2=z).");

    // -------------------------------------------------------------------------
    // VectorPoissonSolverNodal (nodal, MLEBNodeFDLaplacian)
    // -------------------------------------------------------------------------
    py::class_<VectorPoissonSolverNodal>(m, "VectorPoissonSolverNodal")
        .def(py::init<const amrex::Geometry&,
                      const amrex::BoxArray&,
                      const amrex::DistributionMapping&,
                      const BoundaryHandler&>(),
             py::arg("geom"),
             py::arg("grids"),
             py::arg("dmap"),
             py::arg("bc_handler"),
             R"(Initialize the nodal vector Poisson solver.

             Parameters
             ----------
             geom : Geometry
                 AMReX Geometry object.
             grids : BoxArray
                 Cell-centered BoxArray. The solver converts to nodal internally.
             dmap : DistributionMapping
                 AMReX DistributionMapping.
             bc_handler : BoundaryHandler
                 Boundary condition handler.

             Notes
             -----
             A, J, and mask passed to solve() must be defined on the nodal
             BoxArray: grids.surroundingNodes() (all directions in 2D RZ).
             )")
        .def("solve",
             [](VectorPoissonSolverNodal& self,
                py::list A_list,
                py::list J_list,
                py::object mask_obj,
                amrex::Real relative_tol,
                amrex::Real absolute_tol,
                int max_iter,
                int verbose) {
                 auto A = mfab_list_to_array(A_list, "A");
                 auto J = mfab_list_to_array(J_list, "J");
                 self.solve(A, J, extract_mask(mask_obj),
                            relative_tol, absolute_tol, max_iter, verbose);
             },
             py::arg("A"),
             py::arg("J"),
             py::arg("mask") = py::none(),
             py::arg("relative_tol") = 1.0e-10,
             py::arg("absolute_tol") = 0.0,
             py::arg("max_iter") = 100,
             py::arg("verbose") = 2,
             R"(Solve the nodal vector Poisson equation for RZ geometry.

             Uses MLEBNodeFDLaplacian with setRZ(True). The 1/r² geometric
             correction is applied to A_r and A_theta only; A_z receives no
             correction term.

             Parameters
             ----------
             A : list of MultiFab
                 Solution vector potential [A_r, A_theta, A_z]. Must be NODAL.
             J : list of MultiFab
                 Source current density [J_r, J_theta, J_z]. Must be NODAL.
             mask : iMultiFab or None, optional
                 Overset mask (nodal) where 1 = solve, 0 = fixed value.
                 If provided, set fixed values in A before calling solve.
             relative_tol : float
                 Relative tolerance for solver.
             absolute_tol : float
                 Absolute tolerance for solver.
             max_iter : int
                 Maximum number of iterations.
             verbose : int
                 Verbosity level.

             Notes
             -----
             A, J, and mask must be allocated on grids.surroundingNodes().
             Passing cell-centered MultiFabs will produce incorrect results.
             )")
        .def("getNumIters", &VectorPoissonSolverNodal::getNumIters,
             py::arg("component"),
             "Get number of iterations for a given component (0=r, 1=theta, 2=z).")
        .def("getResidual", &VectorPoissonSolverNodal::getResidual,
             py::arg("component"),
             "Get final residual for a given component (0=r, 1=theta, 2=z).");
}