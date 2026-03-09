#ifndef VECTOR_POISSON_SOLVER_NODAL_H
#define VECTOR_POISSON_SOLVER_NODAL_H

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#endif

#include <memory>

namespace VectorPoisson3D {

/**
 * @brief Boundary condition handler for the nodal vector Poisson solver.
 *
 * Separate from BoundaryHandler because the nodal solver supports
 * different operator types and the constructor offers an additional
 * axial_dirichlet option.
 */
struct NodalBoundaryHandler
{
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> lobc;
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> hibc;

    /**
     * @brief Construct with default RZ boundary conditions.
     *
     * @param periodic_axial  If true, axial (z) direction uses periodic BCs.
     * @param axial_dirichlet If true (and not periodic), axial BCs are Dirichlet
     *                        instead of Neumann.
     */
    NodalBoundaryHandler (bool periodic_axial = false, bool axial_dirichlet = false,
                        bool is_cartesian = false)
    {
        if (is_cartesian) {
            // Cartesian: Dirichlet on all boundaries
            for (int adim = 0; adim < 3; ++adim) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    lobc[adim][d] = amrex::LinOpBCType::Dirichlet;
                    hibc[adim][d] = amrex::LinOpBCType::Dirichlet;
                }
            }
        } else {
            // RZ geometry
            for (int adim = 0; adim < 3; ++adim) {
                lobc[adim][0] = amrex::LinOpBCType::Neumann;
                hibc[adim][0] = amrex::LinOpBCType::Dirichlet;

                if (periodic_axial) {
                    lobc[adim][1] = amrex::LinOpBCType::Periodic;
                    hibc[adim][1] = amrex::LinOpBCType::Periodic;
                } else if (axial_dirichlet) {
                    lobc[adim][1] = amrex::LinOpBCType::Dirichlet;
                    hibc[adim][1] = amrex::LinOpBCType::Dirichlet;
                } else {
                    lobc[adim][1] = amrex::LinOpBCType::Neumann;
                    hibc[adim][1] = amrex::LinOpBCType::Neumann;
                }
            }
        }
    }
};

/**
 * @brief Nodal vector Poisson solver for RZ geometry using MLEBNodeFDLaplacian.
 */
class VectorPoissonSolverNodal
{
public:
    VectorPoissonSolverNodal (
        const amrex::Geometry& geom,
        const amrex::BoxArray& grids,
        const amrex::DistributionMapping& dmap,
        const NodalBoundaryHandler& bc_handler,
        bool is_rz = true,
        bool eb_enabled = false
#ifdef AMREX_USE_EB
        , const amrex::EBFArrayBoxFactory* eb_factory = nullptr
#endif
    );

    void solve (
        const amrex::Array<amrex::MultiFab*, 3>& A,
        const amrex::Array<amrex::MultiFab*, 3>& J,
        amrex::Real relative_tol = 1.0e-10,
        amrex::Real absolute_tol = 0.0,
        int max_iter = 100,
        int verbose = 2);

    [[nodiscard]] int getNumIters (int component) const { return m_num_iters[component]; }
    [[nodiscard]] amrex::Real getResidual (int component) const { return m_final_residual[component]; }
    [[nodiscard]] amrex::Array<int, 3> getNumIters () const { return m_num_iters; }
    [[nodiscard]] amrex::Array<amrex::Real, 3> getFinalResidual () const { return m_final_residual; }
    [[nodiscard]] amrex::Array<amrex::MLMG*, 3> getMLMG ();

private:
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids;
    amrex::DistributionMapping m_dmap;
    NodalBoundaryHandler m_bc_handler;

    bool m_is_rz = true;
    bool m_eb_enabled = false;

#ifdef AMREX_USE_EB
    const amrex::EBFArrayBoxFactory* m_eb_factory = nullptr;
#endif

    amrex::Array<int, 3> m_num_iters{};
    amrex::Array<amrex::Real, 3> m_final_residual{};

    amrex::Array<std::unique_ptr<amrex::MLEBNodeFDLaplacian>, 3> m_linop;
    amrex::Array<std::unique_ptr<amrex::MLMG>, 3> m_mlmg;
};

} // namespace VectorPoisson3D

#endif