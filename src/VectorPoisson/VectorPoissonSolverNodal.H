#ifndef VECTOR_POISSON_SOLVER_NODAL_H
#define VECTOR_POISSON_SOLVER_NODAL_H

#include "VectorPoissonSolver.H"
#include <AMReX_MLEBNodeFDLaplacian.H>

namespace VectorPoisson3D {

/**
 * @brief Nodal vector Poisson solver for RZ geometry using MLEBNodeFDLaplacian
 *
 * Solves the vector Laplacian in cylindrical coordinates:
 *   -∇²_cyl A_r   + (1/r²) A_r   = μ₀ J_r     (adim = 0)
 *   -∇²_cyl A_θ   + (1/r²) A_θ   = μ₀ J_θ     (adim = 1)
 *   -∇²_cyl A_z                   = μ₀ J_z     (adim = 2)
 *
 * where ∇²_cyl = (1/r)∂/∂r(r∂/∂r) + ∂²/∂z²
 *
 * The 1/r² geometric correction is applied via setAlpha(1) with setRZ(true),
 * but only for adim = 0 and adim = 1. A_z has no such correction term.
 *
 * @note A, J, and mask must all be defined on NODAL BoxArrays:
 *       ba_nodal = grids.surroundingNodes() (all directions in 2D RZ).
 *       Passing cell-centered MultiFabs will produce incorrect results.
 */
class VectorPoissonSolverNodal {
public:
    VectorPoissonSolverNodal(const amrex::Geometry& geom,
                             const amrex::BoxArray& grids,
                             const amrex::DistributionMapping& dmap,
                             const BoundaryHandler& bc_handler);

    /**
     * @brief Solve the nodal vector Poisson equation
     * @param A Solution: vector potential [A_r, A_theta, A_z] — must be NODAL MultiFabs
     * @param J Source: current density [J_r, J_theta, J_z]   — must be NODAL MultiFabs
     * @param mask Optional overset mask (1 = solve, 0 = fixed) — must be NODAL iMultiFab
     * @param relative_tol Relative tolerance for solver
     * @param absolute_tol Absolute tolerance for solver
     * @param max_iter Maximum number of iterations
     * @param verbose Verbosity level
     *
     * @note When using mask:
     *       - Set mask = 1 where you want to solve for the value
     *       - Set mask = 0 where you want to fix the value
     *       - Before calling solve(), set the desired fixed values in A at mask = 0 locations
     */
    void solve(const amrex::Array<amrex::MultiFab*, 3>& A,
               const amrex::Array<amrex::MultiFab*, 3>& J,
               const amrex::iMultiFab* mask = nullptr,
               amrex::Real relative_tol = 1.0e-10,
               amrex::Real absolute_tol = 0.0,
               int max_iter = 100,
               int verbose = 2);

    int getNumIters(int component) const { return m_num_iters[component]; }
    amrex::Real getResidual(int component) const { return m_final_residual[component]; }

private:
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids;          // cell-centered; linop handles nodal conversion internally
    amrex::DistributionMapping m_dmap;
    BoundaryHandler m_bc_handler;

    amrex::Array<int, 3> m_num_iters;
    amrex::Array<amrex::Real, 3> m_final_residual;
};

} // namespace VectorPoisson3D

#endif