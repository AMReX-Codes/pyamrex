#ifndef VECTOR_POISSON_SOLVER_3D_H
#define VECTOR_POISSON_SOLVER_3D_H

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_Array.H>

namespace VectorPoisson3D {

/**
 * @brief Boundary condition handler for vector Poisson equation
 */
struct BoundaryHandler {
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> lobc;
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> hibc;

    BoundaryHandler(bool periodic_axial = false) {
        for (int adim = 0; adim < 3; ++adim) {
            lobc[adim][0] = amrex::LinOpBCType::Neumann;  // r=0
            hibc[adim][0] = amrex::LinOpBCType::Dirichlet; // r=rmax
            if (periodic_axial) {
                lobc[adim][1] = amrex::LinOpBCType::Periodic;  // z=zmin
                hibc[adim][1] = amrex::LinOpBCType::Periodic;  // z=zmax
            } else {
                lobc[adim][1] = amrex::LinOpBCType::Neumann;  // z=zmin
                hibc[adim][1] = amrex::LinOpBCType::Neumann;  // z=zmax
            }
        }
    }
};

/**
 * @brief Vector Poisson solver for RZ geometry using MLABecLaplacian
 * 
 * Solves: (1/r²)A_θ - ∇²_cyl A_θ = μ₀J_θ in cylindrical coordinates
 * where ∇²_cyl = (1/r)∂/∂r(r∂/∂r) + ∂²/∂z²
 */
class VectorPoissonSolver {
public:
    VectorPoissonSolver(const amrex::Geometry& geom,
                       const amrex::BoxArray& grids,
                       const amrex::DistributionMapping& dmap,
                       const BoundaryHandler& bc_handler);
    
    /**
     * @brief Solve the vector Poisson equation
     * @param A Solution: vector potential [A_r, A_theta, A_z]
     * @param J Source: current density [J_r, J_theta, J_z]
     * @param mask Optional overset mask (1 = solve, 0 = fixed value). If nullptr, no masking.
     * @param relative_tol Relative tolerance for solver
     * @param absolute_tol Absolute tolerance for solver
     * @param max_iter Maximum number of iterations
     * @param verbose Verbosity level
     * 
     * @note When using mask:
     *       - Set mask = 1 where you want to solve for the value
     *       - Set mask = 0 where you want to fix the value
     *       - Before calling solve(), set the desired fixed values in A at locations where mask = 0
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
    amrex::BoxArray m_grids;
    amrex::DistributionMapping m_dmap;
    BoundaryHandler m_bc_handler;
    
    amrex::Array<int, 3> m_num_iters;
    amrex::Array<amrex::Real, 3> m_final_residual;
};

} // namespace VectorPoisson3D

#endif