#include "PyVectorPoisson.H"
#include "VectorPoissonSolver.H"
#include "VectorPoissonSolverNodal.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#endif

namespace py = pybind11;
using namespace VectorPoisson3D;

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

static amrex::iMultiFab*
extract_mask(py::object mask_obj)
{
    if (mask_obj.is_none()) { return nullptr; }
    return py::cast<amrex::iMultiFab*>(mask_obj);
}

void init_VectorPoisson3D(py::module& m)
{
    py::enum_<amrex::LinOpBCType>(m, "LinOpBCType")
        .value("interior", amrex::LinOpBCType::interior)
        .value("Dirichlet", amrex::LinOpBCType::Dirichlet)
        .value("Neumann", amrex::LinOpBCType::Neumann)
        .value("reflect_odd", amrex::LinOpBCType::reflect_odd)
        .value("Marshak", amrex::LinOpBCType::Marshak)
        .value("SanchezPomraning", amrex::LinOpBCType::SanchezPomraning)
        .value("inflow", amrex::LinOpBCType::inflow)
        .value("inhomogNeumann", amrex::LinOpBCType::inhomogNeumann)
        .value("Robin", amrex::LinOpBCType::Robin)
        .value("symmetry", amrex::LinOpBCType::symmetry)
        .value("Periodic", amrex::LinOpBCType::Periodic)
        .value("bogus", amrex::LinOpBCType::bogus);

    // ================================================================
    // Cell-centered boundary handler
    // ================================================================
    py::class_<BoundaryHandler>(m, "BoundaryHandler")
        .def(py::init<>(),
             "Initialize boundary handler with default RZ BCs.\n\n"
             "Defaults:\n"
             "- lo-r: (A_r/A_theta Dirichlet, A_z Neumann)\n"
             "- hi-r: (A_r Neumann, A_theta/A_z Dirichlet)\n"
             "- lo-z/hi-z: Neumann\n")
        .def(py::init<
             const amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3>&,
             const amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3>&>(),
             py::arg("lobc"),
             py::arg("hibc"),
             R"(Initialize boundary handler from per-side BC arrays.

             Parameters
             ----------
             lobc : list[list[LinOpBCType]]
                 Low-side BCs indexed as [component][dimension].
             hibc : list[list[LinOpBCType]]
                 High-side BCs indexed as [component][dimension].
             )")
        .def_readwrite("lobc", &BoundaryHandler::lobc)
        .def_readwrite("hibc", &BoundaryHandler::hibc);

    // ================================================================
    // Cell-centered solver
    // ================================================================
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

    // ================================================================
    // Nodal boundary handler
    // ================================================================
    py::class_<NodalBoundaryHandler>(m, "NodalBoundaryHandler")
        .def(py::init<bool>(),
             py::arg("is_cartesian") = false,
             R"(Initialize nodal boundary handler with default RZ/cartesian BCs.

             Parameters
             ----------
             is_cartesian : bool, optional
                 If True, initialize all boundaries as Dirichlet. If False,
                 initialize with default RZ settings:
                 - lo-r: Neumann
                 - hi-r: (A_r Neumann, A_theta/A_z Dirichlet)
                 - lo-z/hi-z: Neumann
             )")
        .def(py::init<
             const amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3>&,
             const amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3>&>(),
             py::arg("lobc"),
             py::arg("hibc"),
             R"(Initialize nodal boundary handler from per-side BC arrays.

             Parameters
             ----------
             lobc : list[list[LinOpBCType]]
                 Low-side BCs indexed as [component][dimension].
             hibc : list[list[LinOpBCType]]
                 High-side BCs indexed as [component][dimension].
             )")
        .def_readwrite("lobc", &NodalBoundaryHandler::lobc)
        .def_readwrite("hibc", &NodalBoundaryHandler::hibc);

    // ================================================================
    // Nodal solver
    // ================================================================
    auto solver_cls = py::class_<VectorPoissonSolverNodal>(m, "VectorPoissonSolverNodal")
        .def(py::init(
             [](const amrex::Geometry& geom,
                const amrex::BoxArray& grids,
                const amrex::DistributionMapping& dmap,
                const NodalBoundaryHandler& bc_handler,
                bool is_rz,
                bool eb_enabled
#ifdef AMREX_USE_EB
                , py::object eb_factory_obj
#endif
                ) {
#ifdef AMREX_USE_EB
                 const amrex::EBFArrayBoxFactory* eb_factory = nullptr;
                 if (!eb_factory_obj.is_none()) {
                     eb_factory = py::cast<const amrex::EBFArrayBoxFactory*>(eb_factory_obj);
                 }
                 return std::make_unique<VectorPoissonSolverNodal>(
                     geom, grids, dmap, bc_handler, is_rz, eb_enabled,
                     eb_factory);
#else
                 return std::make_unique<VectorPoissonSolverNodal>(
                     geom, grids, dmap, bc_handler, is_rz, eb_enabled);
#endif
             }),
             py::arg("geom"),
             py::arg("grids"),
             py::arg("dmap"),
             py::arg("bc_handler"),
             py::arg("is_rz") = true,
             py::arg("eb_enabled") = false
#ifdef AMREX_USE_EB
             , py::arg("eb_factory") = py::none()
#endif
             , R"(Create a nodal vector Poisson solver.

             Parameters
             ----------
             geom : Geometry
                 AMReX Geometry object describing the domain.
             grids : BoxArray
                 Cell-centered AMReX BoxArray for the level.
             dmap : DistributionMapping
                 AMReX DistributionMapping for the level.
             bc_handler : NodalBoundaryHandler
                 Boundary conditions for each vector component.
             is_rz : bool, optional
                 Use RZ (cylindrical) coordinates. Default is True.
             eb_enabled : bool, optional
                 Enable embedded boundary support. Default is False.
             eb_factory : EBFArrayBoxFactory or None, optional
                 EB factory (required if eb_enabled is True). Default is None.
             )"
        )
        .def("solve",
             [](VectorPoissonSolverNodal& self,
                py::list A_list,
                py::list J_list,
                amrex::Real relative_tol,
                amrex::Real absolute_tol,
                int max_iter,
                int verbose) {
                 auto A = mfab_list_to_array(A_list, "A");
                 auto J = mfab_list_to_array(J_list, "J");
                 self.solve(A, J, relative_tol, absolute_tol, max_iter, verbose);
             },
             py::arg("A"),
             py::arg("J"),
             py::arg("relative_tol") = 1.0e-10,
             py::arg("absolute_tol") = 0.0,
             py::arg("max_iter") = 100,
             py::arg("verbose") = 2,
             R"(Solve the nodal vector Poisson equation for RZ geometry.

             Uses MLEBNodeFDLaplacian. When is_rz is True, setRZ(True) is
             called and the 1/r² geometric correction (setAlpha) is applied
             to A_r and A_theta only; A_z receives no correction term.

             Parameters
             ----------
             A : list of MultiFab
                 Solution vector potential [A_r, A_theta, A_z]. Must be NODAL.
             J : list of MultiFab
                 Source current density [J_r, J_theta, J_z]. Must be NODAL.
             relative_tol : float
                 Relative tolerance for solver.
             absolute_tol : float
                 Absolute tolerance for solver.
             max_iter : int
                 Maximum number of iterations.
             verbose : int
                 Verbosity level.
             )")
        .def("getNumIters",
             py::overload_cast<int>(&VectorPoissonSolverNodal::getNumIters, py::const_),
             py::arg("component"),
             "Get number of iterations for a given component (0=r, 1=theta, 2=z).")
        .def("getResidual",
             py::overload_cast<int>(&VectorPoissonSolverNodal::getResidual, py::const_),
             py::arg("component"),
             "Get final residual for a given component (0=r, 1=theta, 2=z).");

    // ================================================================
    // CoilSpec — nested under VectorPoissonSolverNodal
    // ================================================================
    py::class_<VectorPoissonSolverNodal::CoilSpec>(solver_cls, "CoilSpec")
        .def(py::init<>())
        .def_readwrite("z_lo", &VectorPoissonSolverNodal::CoilSpec::z_lo)
        .def_readwrite("z_hi", &VectorPoissonSolverNodal::CoilSpec::z_hi)
        .def_readwrite("r1c",  &VectorPoissonSolverNodal::CoilSpec::r1c)
        .def_readwrite("r2c",  &VectorPoissonSolverNodal::CoilSpec::r2c)
        .def_readwrite("drc",  &VectorPoissonSolverNodal::CoilSpec::drc)
        .def_readwrite("psi",  &VectorPoissonSolverNodal::CoilSpec::psi);

    solver_cls.def("setEBCoils", &VectorPoissonSolverNodal::setEBCoils,
             py::arg("component"), py::arg("coils"),
             R"(Set EB Dirichlet values for coil regions.

             Parameters
             ----------
             component : int
                 Vector component (0=r, 1=theta, 2=z).
             coils : list of CoilSpec
                 List of coil specifications with geometry and prescribed psi values.
             )");
}
