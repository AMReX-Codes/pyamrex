#include "VectorPoissonSolverNodal.H"
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <cmath>

namespace VectorPoisson3D {

VectorPoissonSolverNodal::VectorPoissonSolverNodal(
    const amrex::Geometry& geom,
    const amrex::BoxArray& grids,
    const amrex::DistributionMapping& dmap,
    const BoundaryHandler& bc_handler)
    : m_geom(geom)
    , m_grids(grids)
    , m_dmap(dmap)
    , m_bc_handler(bc_handler)
{
    m_num_iters.fill(0);
    m_final_residual.fill(0.0);
}

void VectorPoissonSolverNodal::solve(
    const amrex::Array<amrex::MultiFab*, 3>& A,
    const amrex::Array<amrex::MultiFab*, 3>& J,
    const amrex::iMultiFab* mask,
    amrex::Real relative_tol,
    amrex::Real absolute_tol,
    int max_iter,
    int verbose)
{
    using namespace amrex;

    constexpr Real mu0 = 1.25663706212e-6;

    // Check if current is zero everywhere
    Real max_comp_J = 0.0;
    for (int adim = 0; adim < 3; ++adim) {
        Real norm_J = J[adim]->norm0(0, 0, false, false);
        max_comp_J = std::max(max_comp_J, norm_J);
    }

    const bool always_use_bnorm = (max_comp_J > 0);
    if (!always_use_bnorm && absolute_tol == 0.0) {
        absolute_tol = 1.0e-6;
        if (verbose > 0) {
            Print() << "Warning: Max norm of J is 0, setting absolute tolerance to "
                    << absolute_tol << "\n";
        }
    }

    // LPInfo for nodal solver:
    //   - No setMetricTerm: RZ metric is handled by setRZ(true) on the operator
    //   - setMaxCoarseningLevel(0): consistent with cell-centered solver
    LPInfo info;
    info.setMaxCoarseningLevel(0);

    Array<int, 3> num_iters;
    Array<Real, 3> final_residual;

    for (int adim = 0; adim < 3; ++adim) {

        Real norm_J = J[adim]->norm0(0, 0, false, false);
        if (norm_J == 0.0 && adim != 1) {
            if (verbose > 0) {
                Print() << "Component " << adim << " has zero source, skipping...\n";
            }
            num_iters[adim] = 0;
            final_residual[adim] = 0.0;
            continue;
        }

        if (verbose > 0) {
            Print() << "Solving for A component " << adim << " (nodal)...\n";
        }

        // Create the nodal linear operator.
        // m_grids is cell-centered; MLEBNodeFDLaplacian converts to nodal internally.
        // A[adim], J[adim], and mask must already be on the nodal BoxArray.
        MLEBNodeFDLaplacian linop;
        if (mask != nullptr) {
            // Overset mask: mask=1 means solve, mask=0 means fixed value
            linop.define({m_geom}, {m_grids}, {m_dmap}, info, {}, {mask});
        } else {
            linop.define({m_geom}, {m_grids}, {m_dmap}, info);
        }

        // Uniform sigma = 1: the cylindrical metric (1/r d/dr r d/dr + d²/dz²)
        // is handled entirely by setRZ(true) below, not by a variable sigma
        linop.setSigma({AMREX_D_DECL(1.0, 1.0, 1.0)});

        // Enable cylindrical (RZ) stencil for the divergence operator
        linop.setRZ(true);

        // Add the 1/r² geometric correction term for A_r and A_theta only.
        // A_z has no such term: hat{z} is a Cartesian direction with no
        // Christoffel symbol contribution. Applying it to A_z would be
        // physically incorrect — this is the bug in WarpX PR #6516.
        if (adim != 2) {
            linop.setAlpha(1.0);
        }

        // Set domain BC types
        linop.setDomainBC(m_bc_handler.lobc[adim], m_bc_handler.hibc[adim]);

        // Communicate boundary values; sets Dirichlet BCs and initial guess
        // at domain boundaries. For the overset case, fixed values at mask=0
        // nodes are enforced automatically by the operator.
        A[adim]->FillBoundary(m_geom.periodicity());
        linop.setLevelBC(0, A[adim]);

        // Create nodal RHS = μ₀ J
        // J[adim] must be a nodal MultiFab — same BoxArray as A[adim]
        MultiFab rhs(J[adim]->boxArray(), J[adim]->DistributionMap(), 1, 0);
        MultiFab::Copy(rhs, *J[adim], 0, 0, 1, 0);
        rhs.mult(mu0);

        // Solve
        MLMG mlmg(linop);
        mlmg.setVerbose(verbose);
        mlmg.setMaxIter(max_iter);

        final_residual[adim] = mlmg.solve({A[adim]}, {&rhs}, relative_tol, absolute_tol);
        num_iters[adim] = mlmg.getNumIters();

        // Fill boundary for consistency
        A[adim]->FillBoundary(m_geom.periodicity());

        if (verbose > 0) {
            Print() << "  Component " << adim << " converged in " << num_iters[adim]
                    << " iterations with residual = " << final_residual[adim] << "\n";
        }
    }

    m_num_iters = num_iters;
    m_final_residual = final_residual;
}

} // namespace VectorPoisson3D