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

struct NodalBoundaryHandler
{
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> lobc;
    amrex::Array<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>, 3> hibc;

    NodalBoundaryHandler (bool periodic_axial = false, bool axial_dirichlet = false,
                          bool is_cartesian = false)
    {
        if (is_cartesian) {
            for (int adim = 0; adim < 3; ++adim) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    lobc[adim][d] = amrex::LinOpBCType::Dirichlet;
                    hibc[adim][d] = amrex::LinOpBCType::Dirichlet;
                }
            }
        } else {
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

class VectorPoissonSolverNodal
{
public:
    struct CoilSpec {
        amrex::Real z_lo, z_hi;
        amrex::Real r1c;              // inner radius at z_lo
        amrex::Real r2c;              // inner radius at z_hi
        amrex::Real drc;              // radial thickness (constant)
        amrex::Real psi;              // flux value for this coil
    };

    struct EBCoilFunctor
    {
        const CoilSpec* coils;
        int ncoils;

        AMREX_GPU_DEVICE
        amrex::Real operator() (AMREX_D_DECL(amrex::Real r, amrex::Real z, amrex::Real /*unused*/)) const
        {
    #if (AMREX_SPACEDIM >= 2)
            for (int ic = 0; ic < ncoils; ++ic) {
                if (z >= coils[ic].z_lo && z <= coils[ic].z_hi) {
                    amrex::Real dz_coil = coils[ic].z_hi - coils[ic].z_lo;
                    amrex::Real tau = (dz_coil > amrex::Real(0.0))
                        ? (z - coils[ic].z_lo) / dz_coil
                        : amrex::Real(0.0);
                    amrex::Real r_inner = coils[ic].r1c
                        + (coils[ic].r2c - coils[ic].r1c) * tau;
                    amrex::Real r_outer = r_inner + coils[ic].drc;
                    if (r >= r_inner && r <= r_outer && r > amrex::Real(0.0)) {
                        return coils[ic].psi / r;
                    }
                }
            }
    #else
            amrex::ignore_unused(r, coils, ncoils);
    #endif
            return amrex::Real(0.0);
        }
    };

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

    void setEBCoils(int component, const std::vector<CoilSpec>& coils);

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

    std::array<std::vector<CoilSpec>, 3> m_coils;
    std::array<amrex::Gpu::DeviceVector<CoilSpec>, 3> m_d_coils;
};

} // namespace VectorPoisson3D

#endif