#include "VectorPoissonSolver.H"
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <cmath>

namespace VectorPoisson3D {

VectorPoissonSolver::VectorPoissonSolver(
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

void VectorPoissonSolver::solve(
    const amrex::Array<amrex::MultiFab*, 3>& A,
    const amrex::Array<amrex::MultiFab*, 3>& J,
    const amrex::iMultiFab* mask,  // NEW: 1 = solve, 0 = fixed
    amrex::Real relative_tol,
    amrex::Real absolute_tol,
    int max_iter,
    int verbose)
{
    using namespace amrex;
    
    constexpr Real mu0 = 1.25663706212e-6;
    
    const auto dx = m_geom.CellSizeArray();
    const auto prob_lo = m_geom.ProbLoArray();
    
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
    
    LPInfo info;
    info.setMaxCoarseningLevel(0);
    info.setMetricTerm(true);
    
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
            Print() << "Solving for A component " << adim << "...\n";
        }
        
        if (adim == 1) {
            
            // Create the linear operator with overset mask if provided
            MLABecLaplacian linop;
            if (mask != nullptr) {
                // Use overset mask constructor: mask=1 means solve, mask=0 means fixed
                linop.define({m_geom}, {m_grids}, {m_dmap}, {mask}, info);
            } else {
                // Standard constructor without mask
                linop.define({m_geom}, {m_grids}, {m_dmap}, info);
            }
            
            linop.setScalars(1.0, 1.0);
            
            // Create 'a' coefficient = 1/r²
            MultiFab acoef(m_grids, m_dmap, 1, 0);
            
            for (MFIter mfi(acoef); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                Array4<Real> const& a = acoef.array(mfi);
                
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real r = prob_lo[0] + (i + 0.5) * dx[0];
                    if (r > 1e-10) {
                        a(i,j,k) = 1.0 / (r * r);
                    } else {
                        a(i,j,k) = 0.0;
                    }
                });
            }
            
            linop.setACoeffs(0, acoef);
            
            // Set 'b' coefficient = 1.0
            Array<MultiFab,AMREX_SPACEDIM> face_bcoef;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                BoxArray ba_face = m_grids;
                ba_face.surroundingNodes(idim);
                face_bcoef[idim].define(ba_face, m_dmap, 1, 0);
                face_bcoef[idim].setVal(1.0);
            }
            linop.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));
            
            // Set domain BC types
            linop.setDomainBC(m_bc_handler.lobc[adim], m_bc_handler.hibc[adim]);
            
            // If using mask, set the fixed values in A before solving
            // The overset mask will automatically handle them
            A[adim]->FillBoundary(m_geom.periodicity());
            linop.setLevelBC(0, A[adim]);
            
            // Create RHS = μ₀J_θ
            MultiFab rhs(J[adim]->boxArray(), J[adim]->DistributionMap(), 1, 0);
            MultiFab::Copy(rhs, *J[adim], 0, 0, 1, 0);
            rhs.mult(mu0);
            
            // Solve - the overset mask automatically enforces fixed values
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
            
        } else {
            A[adim]->setVal(0.0);
            num_iters[adim] = 0;
            final_residual[adim] = 0.0;
        }
    }
    
    m_num_iters = num_iters;
    m_final_residual = final_residual;
}

} // namespace VectorPoisson3D