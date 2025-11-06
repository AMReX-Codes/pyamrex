#!/usr/bin/env python3
"""Test VectorPoisson solver in RZ geometry with manufactured solution"""

import numpy as np
import matplotlib.pyplot as plt
import amrex.space2d as amr


def load_cupy():
    """Load cupy if available, otherwise use numpy"""
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            xp = cp
            amr.Print("Note: found and will use cupy")
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Trying managed memory in numpy...")
            import numpy as np
            xp = np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            xp = np
    else:
        import numpy as np
        xp = np
        amr.Print("Note: found and will use numpy")
    return xp


def A_exact_rz(r, z):
    """
    Exact solution for A_theta in RZ geometry
    A_theta = r * sin(pi*r) * sin(pi*z)
    
    This satisfies:
    - Homogeneous Dirichlet BCs: A_theta(0,z) = 0, A_theta(1,z) = 0, A_theta(r,0) = 0, A_theta(r,1) = 0
    - Regular at r=0: A_theta ~ r as r->0
    """
    return r * np.sin(np.pi * r) * np.sin(np.pi * z)


def source_exact_rz(r, z):
    """
    Compute RHS for manufactured solution: -∇²_cyl A_theta + (1/r²)A_theta = f
    where ∇²_cyl = (1/r)∂/∂r(r∂/∂r) + ∂²/∂z²
    
    For A_theta = r sin(πr) sin(πz):
    
    ∂A/∂r = sin(πr)sin(πz) + πr cos(πr)sin(πz)
    r∂A/∂r = r sin(πr)sin(πz) + πr² cos(πr)sin(πz)
    ∂/∂r(r∂A/∂r) = sin(πr)sin(πz) + πr cos(πr)sin(πz) 
                    + 2πr cos(πr)sin(πz) - π²r² sin(πr)sin(πz)
                  = sin(πr)sin(πz) + 3πr cos(πr)sin(πz) - π²r² sin(πr)sin(πz)
    (1/r)∂/∂r(r∂A/∂r) = (1/r)sin(πr)sin(πz) + 3π cos(πr)sin(πz) - π²r sin(πr)sin(πz)
    
    ∂²A/∂z² = -π²r sin(πr)sin(πz)
    
    ∇²_cyl A = (1/r)sin(πr)sin(πz) + 3π cos(πr)sin(πz) - π²r sin(πr)sin(πz) - π²r sin(πr)sin(πz)
             = (1/r)sin(πr)sin(πz) + 3π cos(πr)sin(πz) - 2π²r sin(πr)sin(πz)
    
    (1/r²)A = (1/r)sin(πr)sin(πz)
    
    f = -∇²_cyl A + (1/r²)A
      = -(1/r)sin(πr)sin(πz) - 3π cos(πr)sin(πz) + 2π²r sin(πr)sin(πz) + (1/r)sin(πr)sin(πz)
      = -3π cos(πr)sin(πz) + 2π²r sin(πr)sin(πz)
    
    Divide by μ₀ for the actual source (J_theta)
    """
    mu0 = 1.25663706212e-6
    pi = np.pi
    
    f = -3.0*pi*np.cos(pi*r)*np.sin(pi*z) + 2.0*pi*pi*r*np.sin(pi*r)*np.sin(pi*z)
    return f / mu0


def extract_2d_data(mf, geom, component=0):
    """Extract 2D data from a MultiFab (GPU-safe)"""
    
    # Use cupy if available for GPU data transfer
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            xp = np
    else:
        xp = np
    
    dx = geom.data().CellSize()
    prob_lo = geom.data().ProbLo()
    domain = geom.domain
    
    # Get domain size (cell-centered)
    nx = domain.big_end[0] - domain.small_end[0] + 1
    ny = domain.big_end[1] - domain.small_end[1] + 1
    
    # Create output array on CPU
    data = np.zeros((ny, nx))
    
    # Loop through boxes and fill data
    for mfi in mf:
        bx = mfi.validbox()
        arr = mf.array(mfi)
        
        lo = bx.small_end
        hi = bx.big_end
        
        # Get array view
        arr_view = xp.array(arr, copy=False)
        
        # Copy entire array to CPU if needed
        if xp.__name__ == 'cupy':
            arr_cpu = cp.asnumpy(arr_view)
        else:
            arr_cpu = np.array(arr_view, copy=False)
        
        # arr_cpu shape is [ncomp, k, nz_total, nr_total] for 2D
        ng = mf.n_grow_vect
        ngr = ng[0]
        ngz = ng[1]
        
        # Fill data array
        for j in range(lo[1], hi[1]+1):
            jj = j - lo[1] + ngz
            for i in range(lo[0], hi[0]+1):
                ii = i - lo[0] + ngr
                data[j, i] = float(arr_cpu[0, 0, jj, ii])
    
    # Create coordinate arrays (cell-centered)
    r = prob_lo[0] + (np.arange(nx) + 0.5) * dx[0]
    z = prob_lo[1] + (np.arange(ny) + 0.5) * dx[1]
    
    return r, z, data


def plot_solution_rz(A, geom, filename='solution_rz.png'):
    """Plot RZ solution with comparison to exact solution"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    component_names = ['Ar', 'Aθ', 'Az']
    
    for comp in range(3):
        r, z, data = extract_2d_data(A[comp], geom)
        R, Z = np.meshgrid(r, z)
        
        # Contour plot
        ax = axes[0, comp]
        levels = 20
        vmax = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0
        cf = ax.contourf(R, Z, data, levels=levels, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('r (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'{component_names[comp]}')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label=component_names[comp])
        
        # Line plots
        ax = axes[1, comp]
        
        if comp == 1:  # A_theta only
            # Plot along r at mid-z
            z_mid = len(z) // 2
            ax.plot(r, data[z_mid, :], 'b-', linewidth=2, label='Numerical')
            
            # Analytical solution
            z_val = z[z_mid]
            A_exact_vec = A_exact_rz(r, z_val)
            ax.plot(r, A_exact_vec, 'r--', linewidth=2, label='Exact')
            
            ax.set_xlabel('r (m)')
            ax.set_ylabel(component_names[comp])
            ax.set_title(f'{component_names[comp]} at z={z_val:.3f}')
            
        else:
            # For Ar and Az (should be zero)
            z_mid = len(z) // 2
            ax.plot(r, data[z_mid, :], 'b-', linewidth=2, label='Numerical')
            ax.plot(r, np.zeros_like(r), 'r--', linewidth=2, label='Exact')
            ax.set_xlabel('r (m)')
            ax.set_ylabel(component_names[comp])
            ax.set_title(f'{component_names[comp]} at z={z[z_mid]:.3f}')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


def plot_line_cuts(A, geom, filename='line_cuts_rz.png'):
    """Plot line cuts comparing numerical and exact solutions"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    r, z, data = extract_2d_data(A[1], geom)  # Only A_theta
    
    # Plot along r at z=0.5
    z_mid = len(z) // 2
    z_val = z[z_mid]
    
    ax = axes[0]
    ax.plot(r, data[z_mid, :], 'b-', linewidth=2, label='Numerical')
    ax.plot(r, A_exact_rz(r, z_val), 'r--', linewidth=2, label='Exact')
    ax.set_xlabel('r (m)')
    ax.set_ylabel('Aθ')
    ax.set_title(f'Aθ vs r at z={z_val:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot along z at r=0.5
    r_mid = len(r) // 2
    r_val = r[r_mid]
    
    ax = axes[1]
    ax.plot(z, data[:, r_mid], 'b-', linewidth=2, label='Numerical')
    ax.plot(z, A_exact_rz(r_val, z), 'r--', linewidth=2, label='Exact')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Aθ')
    ax.set_title(f'Aθ vs z at r={r_val:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved line cuts to {filename}")
    plt.close()


def compute_error(A, geom):
    """Compute L1, L2, and Linf errors"""
    
    r, z, data = extract_2d_data(A[1], geom)
    R, Z = np.meshgrid(r, z)
    
    exact = A_exact_rz(R, Z)
    error = data - exact
    
    dx = geom.data().CellSize()
    
    # L1 error
    L1 = np.sum(np.abs(error)) * dx[0] * dx[1]
    
    # L2 error
    L2 = np.sqrt(np.sum(error**2) * dx[0] * dx[1])
    
    # Linf error
    Linf = np.max(np.abs(error))
    
    return L1, L2, Linf


# ============= Main Test Script =============

if __name__ == "__main__":
    # Initialize AMReX
    amr.initialize([])
    
    # Load cupy or numpy
    xp = load_cupy()
    
    print("=== Testing VectorPoisson Solver (RZ) ===")
    print("Using manufactured solution: A_theta = r*sin(πr)*sin(πz)")
    print("This satisfies homogeneous Dirichlet BCs and is regular at r=0")
    
    # Domain setup: r ∈ [0, 1], z ∈ [0, 1]
    n_cell_r = 128
    n_cell_z = 128
    max_grid_size = 32
    
    # Create domain box (2D: r, z)
    domain = amr.Box(amr.IntVect(0, 0), amr.IntVect(n_cell_r-1, n_cell_z-1))
    
    # Create geometry (RZ: coord=1)
    rb = amr.RealBox([0., 0.], [1., 1.])
    is_per = [0, 0]
    coord = 1  # RZ geometry
    geom = amr.Geometry(domain, rb, coord, is_per)
    
    print(f"Domain: {domain}")
    print(f"Geometry: RZ (coord=1)")
    
    # Create BoxArray and DistributionMapping
    ba = amr.BoxArray(domain)
    ba.max_size(max_grid_size)
    dm = amr.DistributionMapping(ba)
    
    print(f"Number of boxes: {ba.size}")
    
    # Create cell-centered MultiFabs for A (solution) and J (source)
    ncomp = 1
    nghost = 1
    
    A = [amr.MultiFab(ba, dm, ncomp, nghost) for _ in range(3)]
    J = [amr.MultiFab(ba, dm, ncomp, nghost) for _ in range(3)]
    
    # Initialize to zero
    for i in range(3):
        A[i].set_val(0.0)
        J[i].set_val(0.0)
    
    print("Cell-centered MultiFabs created successfully")
    
    # Fill J_theta with manufactured solution source
    print("\nFilling source term...")
    dx = geom.data().CellSize()
    prob_lo = geom.data().ProbLo()
    
    ng = J[1].n_grow_vect
    ngr = ng[0]
    ngz = ng[1]
    
    for mfi in J[1]:
        bx = mfi.validbox()
        J_arr = xp.array(J[1].array(mfi), copy=False)
        
        lo = bx.small_end
        hi = bx.big_end
        
        nr = hi[0] - lo[0] + 1
        nz = hi[1] - lo[1] + 1
        
        r = prob_lo[0] + (xp.arange(lo[0], hi[0]+1) + 0.5) * dx[0]
        z = prob_lo[1] + (xp.arange(lo[1], hi[1]+1) + 0.5) * dx[1]
        
        # Compute source term
        R, Z = xp.meshgrid(r, z, indexing='ij')
        J_theta = source_exact_rz(R.T, Z.T)  # Transpose for correct indexing
        
        J_arr[0, 0, ngz:ngz+nz, ngr:ngr+nr] = J_theta
    
    print("Source term filled from manufactured solution")
    
    # Create boundary handler
    bc_handler = amr.BoundaryHandler()
    print("Boundary conditions: Homogeneous Dirichlet on all boundaries")
    
    # Create solver
    print("\nCreating VectorPoissonSolver...")
    solver = amr.VectorPoissonSolver(geom, ba, dm, bc_handler)
    
    # Solve
    print("\nSolving...")
    solver.solve(A, J, relative_tol=1.0e-10, absolute_tol=0.0, max_iter=100, verbose=2)
    
    print("\nSolver statistics:")
    for i in range(3):
        print(f"  Component {i}: {solver.getNumIters(i)} iterations, residual = {solver.getResidual(i):.6e}")
    
    # Compute error
    print("\nComputing error...")
    L1, L2, Linf = compute_error(A, geom)
    
    print("\nError norms for A_theta:")
    print(f"  L1   error: {L1:.6e}")
    print(f"  L2   error: {L2:.6e}")
    print(f"  Linf error: {Linf:.6e}")
    
    # Create plots
    print("\n=== Creating plots ===")
    plot_solution_rz(A, geom, 'solution_rz.png')
    plot_line_cuts(A, geom, 'line_cuts_rz.png')
    
    print("\n=== Test Complete ===")
    
    # Cleanup for GPU
    del A
    del J
    
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
    
    # Finalize
    amr.finalize()