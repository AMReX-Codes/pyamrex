#include "VectorPoissonSolverNodal.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>

namespace VectorPoisson3D {

VectorPoissonSolverNodal::VectorPoissonSolverNodal (
    const amrex::Geometry& geom,
    const amrex::BoxArray& grids,
    const amrex::DistributionMapping& dmap,
    const NodalBoundaryHandler& bc_handler,
    bool is_rz,
    bool eb_enabled
#ifdef AMREX_USE_EB
    , const amrex::EBFArrayBoxFactory* eb_factory
#endif
)
    : m_geom(geom)
    , m_grids(grids)
    , m_dmap(dmap)
    , m_bc_handler(bc_handler)
    , m_is_rz(is_rz)
    , m_eb_enabled(eb_enabled)
#ifdef AMREX_USE_EB
    , m_eb_factory(eb_factory)
#endif
{
    m_num_iters.fill(0);
    m_final_residual.fill(0.0);

#ifdef AMREX_USE_EB
    if (m_eb_enabled) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_eb_factory != nullptr,
            "VectorPoissonSolverNodal: eb_factory must be provided when eb_enabled is true");
    }
#else
    if (m_eb_enabled) {
        amrex::Abort(
            "VectorPoissonSolverNodal: eb_enabled=true but AMReX was not built with EB support");
    }
#endif
}

void VectorPoissonSolverNodal::solve (
    const amrex::Array<amrex::MultiFab*, 3>& A,
    const amrex::Array<amrex::MultiFab*, 3>& J,
    amrex::Real relative_tol,
    amrex::Real absolute_tol,
    int max_iter,
    int verbose)
{
    using namespace amrex;

    constexpr Real mu0 = 1.25663706212e-6;

    // ----------------------------------------------------------------
    // Determine if current is zero everywhere (across all components)
    // ----------------------------------------------------------------
    Real max_comp_J = 0.0;
    for (int adim = 0; adim < 3; ++adim) {
        Real norm_J = J[adim]->norm0(0, 0, false, false);
        max_comp_J = std::max(max_comp_J, norm_J);
    }
    ParallelDescriptor::ReduceRealMax(max_comp_J);

    const bool always_use_bnorm = (max_comp_J > 0.0);
    if (!always_use_bnorm && absolute_tol == 0.0) {
        absolute_tol = 1.0e-6;
        if (verbose > 0) {
            Print() << "VectorPoissonSolverNodal: Max norm of J is 0, "
                    << "setting absolute tolerance to " << absolute_tol << "\n";
        }
    }

    // ----------------------------------------------------------------
    // Set up LPInfo
    // ----------------------------------------------------------------
    LPInfo info;

    const auto dx = m_geom.CellSizeArray();

    if (!m_eb_enabled && !m_is_rz) {
        // Semi-coarsening helps when cell sizes are very different across directions
        int max_semicoarsening_level = 0;
        int semicoarsening_direction = -1;

        auto min_it = std::min_element(dx.begin(), dx.begin() + AMREX_SPACEDIM);
        auto max_it = std::max_element(dx.begin(), dx.begin() + AMREX_SPACEDIM);
        const int min_dir = static_cast<int>(std::distance(dx.begin(), min_it));
        const int max_dir = static_cast<int>(std::distance(dx.begin(), max_it));
        amrex::ignore_unused(min_dir);

        if (dx[max_dir] > dx[min_dir]) {
            semicoarsening_direction = max_dir;
            max_semicoarsening_level =
                static_cast<int>(std::log2(dx[max_dir] / dx[min_dir]));
        }

        if (max_semicoarsening_level > 0) {
            info.setSemicoarsening(true);
            info.setMaxSemicoarseningLevel(max_semicoarsening_level);
            info.setSemicoarseningDirection(semicoarsening_direction);
        }
    }

    if (m_is_rz) {
        info.setMaxCoarseningLevel(0);  // dont coarsen
    }


    // ----------------------------------------------------------------
    // Reset linear operator and MLMG objects
    // ----------------------------------------------------------------
    for (int adim = 0; adim < 3; ++adim) {
        m_linop[adim].reset();
        m_mlmg[adim].reset();
    }

    // ----------------------------------------------------------------
    // Define linear operators for each component
    // ----------------------------------------------------------------
    for (int adim = 0; adim < 3; ++adim) {
        m_linop[adim] = std::make_unique<MLEBNodeFDLaplacian>();

        if (m_eb_enabled) {
#ifdef AMREX_USE_EB
            m_linop[adim]->define(
                {m_geom}, {m_grids}, {m_dmap}, info, {m_eb_factory});
#endif
        } else {
            m_linop[adim]->define(
                {m_geom}, {m_grids}, {m_dmap}, info);
        }

        // Isotropic Laplacian: sigma = 1 in all directions
        m_linop[adim]->setSigma({AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt)});

        // Homogeneous Dirichlet on embedded boundaries
#ifdef AMREX_USE_EB
        if (m_eb_enabled) {
            m_linop[adim]->setEBDirichlet(0.0_rt);
        }
#endif

        // In RZ geometry, the vector Laplacian for r and theta components is
        //   ∇²_cyl A_i - A_i / r²
        // MLEBNodeFDLaplacian with setRZ(true) handles the cylindrical metric,
        // and setAlpha(1.0) adds the -1/r² diagonal term.
        // The z component does NOT get the alpha term.
        if (m_is_rz) {
            m_linop[adim]->setRZ(true);
            if (adim != 2) {
                // adim 0 = A_r, adim 1 = A_theta: need the -1/r² term
                m_linop[adim]->setAlpha(1.0_rt);
            }
        }

        // Domain boundary conditions
        m_linop[adim]->setDomainBC(
            m_bc_handler.lobc[adim], m_bc_handler.hibc[adim]);
    }

    // ----------------------------------------------------------------
    // Solve each component
    // ----------------------------------------------------------------
    for (int adim = 0; adim < 3; ++adim) {

        Real norm_J = J[adim]->norm0(0, 0, false, false);

        // Skip components with zero source, except theta (adim==1) which may
        // have a BC-driven solution
        if (norm_J == 0.0 && adim != 1) {
            if (verbose > 0) {
                Print() << "VectorPoissonSolverNodal: Component " << adim
                        << " has zero source, skipping...\n";
            }
            m_num_iters[adim] = 0;
            m_final_residual[adim] = 0.0;
            continue;
        }

        if (verbose > 0) {
            Print() << "VectorPoissonSolverNodal: Solving for A component "
                    << adim << "...\n";
        }

        // RHS = -μ₀ J
        // MLEBNodeFDLaplacian solves  ∇²φ - α/r² φ = rhs
        // We want  ∇²A = -μ₀ J,  so  rhs = -μ₀ J
        MultiFab rhs(J[adim]->boxArray(), J[adim]->DistributionMap(),
                     1, J[adim]->nGrowVect());
        MultiFab::Copy(rhs, *J[adim], 0, 0, 1, rhs.nGrowVect());
        rhs.mult(-mu0);

        if (m_is_rz && adim != 2) {
            const Real rlo = m_geom.ProbLo(0);
            if (rlo == 0.0_rt) {
                for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                    const Box& bx = mfi.validbox();
                    if (bx.smallEnd(0) == 0) {
                        Array4<Real> const& rhsarr = rhs.array(mfi);
                        const auto lo = lbound(bx);
                        const auto hi = ubound(bx);
                        amrex::ParallelFor(hi.y - lo.y + 1,
                        [=] AMREX_GPU_DEVICE (int idx) {
                            rhsarr(0, lo.y + idx, 0) = 0.0_rt;
                        });
                    }
                }
            }
        }


        // Create MLMG solver
        m_mlmg[adim] = std::make_unique<MLMG>(*m_linop[adim]);
        m_mlmg[adim]->setVerbose(verbose);
        m_mlmg[adim]->setMaxIter(max_iter);

        m_mlmg[adim]->setBottomSolver(MLMG::BottomSolver::bicgstab);
        m_mlmg[adim]->setBottomVerbose(0);
        m_mlmg[adim]->setBottomMaxIter(200);
        m_mlmg[adim]->setBottomTolerance(1.0e-4);

        m_mlmg[adim]->setConvergenceNormType(
            always_use_bnorm ? MLMGNormType::bnorm : MLMGNormType::greater);

        // Solve
        m_final_residual[adim] = m_mlmg[adim]->solve(
            {A[adim]}, {&rhs}, relative_tol, absolute_tol);
        m_num_iters[adim] = m_mlmg[adim]->getNumIters();

        // Synchronize ghost cells
        A[adim]->FillBoundary(m_geom.periodicity());

        if (verbose > 0) {
            Print() << "  Component " << adim << " converged in "
                    << m_num_iters[adim] << " iterations with residual = "
                    << m_final_residual[adim] << "\n";
        }
    }
}

amrex::Array<amrex::MLMG*, 3> VectorPoissonSolverNodal::getMLMG ()
{
    return {m_mlmg[0].get(), m_mlmg[1].get(), m_mlmg[2].get()};
}

} // namespace VectorPoisson3D