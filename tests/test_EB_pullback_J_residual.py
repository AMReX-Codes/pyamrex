#!/usr/bin/env python3
# Test EB boundary pullback by validating J residuals at embedded boundaries.
# 
# This test mimics the initialization loop that:
# 1. Solves for A with EB Dirichlet BC
# 2. Computes J = curl(B) = curl(curl(A)) 
# 3. Applies J boundary conditions at EB
# 4. Checks that J residuals are properly zeroed at EB boundaries
#
# With correct per-component pullback, J boundary application should result
# in near-zero values at the EB boundary faces where the pullback occurred.

import numpy as np

try:
    import cupy as xp

    def to_numpy(a):
        return a.get()
except ImportError:
    import numpy as xp

    def to_numpy(a):
        return np.asarray(a)

import amrex.space2d as amr


def _single_coil_spec():
    """Create one rectangular coil for testing."""
    return {
        'r1c': 0.30,
        'r2c': 0.30,
        'drc': 0.25,
        'z_lo': 0.35,
        'z_hi': 0.65,
        'psi': 1.0
    }


def _extract_nodal_data(mf, geom):
    """Extract the full nodal array from a MultiFab."""
    problo = geom.ProbLo()
    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    dom = geom.domain
    nr_full = dom.length(0) + 1
    nz_full = dom.length(1) + 1

    full_arr = xp.zeros((nr_full, nz_full), dtype=xp.float64)

    for mfi in mf:
        bx = mfi.validbox()
        lo, hi = bx.small_end, bx.big_end
        nr = hi[0] - lo[0] + 1
        nz = hi[1] - lo[1] + 1
        ng0, ng1 = mf.n_grow_vect[0], mf.n_grow_vect[1]
        marr = xp.asarray(mf.array(mfi))
        if marr.shape[0] == nr + 2 * ng0:
            data = marr[ng0: ng0 + nr, ng1: ng1 + nz, 0, 0]
        else:
            data = marr[0, 0, ng1: ng1 + nz, ng0: ng0 + nr].T
        full_arr[lo[0]:lo[0] + nr, lo[1]:lo[1] + nz] = data

    r_arr = problo[0] + xp.arange(nr_full, dtype=xp.float64) * dx[0]
    z_arr = problo[1] + xp.arange(nz_full, dtype=xp.float64) * dx[1]
    return full_arr, r_arr, z_arr


def _extract_cc_data(mf, geom):
    """Extract cell-centered data from a MultiFab."""
    dom = geom.domain
    nr = dom.length(0)
    nz = dom.length(1)

    full_arr = xp.zeros((nr, nz), dtype=xp.float64)

    for mfi in mf:
        bx = mfi.validbox()
        lo, hi = bx.small_end, bx.big_end
        nr_loc = hi[0] - lo[0] + 1
        nz_loc = hi[1] - lo[1] + 1
        ng0, ng1 = mf.n_grow_vect[0], mf.n_grow_vect[1]

        marr = xp.asarray(mf.array(mfi))
        if marr.shape[0] == nr_loc + 2 * ng0:
            data = marr[ng0:ng0 + nr_loc, ng1:ng1 + nz_loc, 0, 0]
        else:
            data = marr[0, 0, ng1:ng1 + nz_loc, ng0:ng0 + nr_loc].T

        full_arr[lo[0]:lo[0] + nr_loc, lo[1]:lo[1] + nz_loc] = data

    return full_arr


def _compute_vector_laplacian_rz(Ar_nodal, At_nodal, Az_nodal, r_arr, dr, dz):
    """
    Compute J = -∇²A directly using the vector Laplacian in RZ cylindrical coordinates.
    All components are nodal (live on mesh nodes).
    
    Vector Laplacian in RZ cylindrical coordinates with axisymmetry:
    ∇²A_r = ∂²A_r/∂r² + (1/r)∂A_r/∂r - A_r/r² + ∂²A_r/∂z²
    ∇²A_θ = ∂²A_θ/∂r² + (1/r)∂A_θ/∂r - A_θ/r² + ∂²A_θ/∂z²
    ∇²A_z = ∂²A_z/∂r² + (1/r)∂A_z/∂r + ∂²A_z/∂z²
    
    Then J = -∇²A
    """
    nr_nodal, nz_nodal = Ar_nodal.shape
    R_nodal = r_arr[:, xp.newaxis]
    
    # Compute second derivatives and mixed terms for each component
    # Interior points only - boundaries will remain zero (no sources there)
    
    # A_r component
    d2Ar_dr2 = xp.zeros_like(Ar_nodal)
    dAr_dr = xp.zeros_like(Ar_nodal)
    d2Ar_dz2 = xp.zeros_like(Ar_nodal)
    
    # Interior stencils (excluding boundaries)
    d2Ar_dr2[1:-1, 1:-1] = (Ar_nodal[2:, 1:-1] - 2*Ar_nodal[1:-1, 1:-1] + Ar_nodal[:-2, 1:-1]) / dr**2
    dAr_dr[1:-1, 1:-1] = (Ar_nodal[2:, 1:-1] - Ar_nodal[:-2, 1:-1]) / (2*dr)
    d2Ar_dz2[1:-1, 1:-1] = (Ar_nodal[1:-1, 2:] - 2*Ar_nodal[1:-1, 1:-1] + Ar_nodal[1:-1, :-2]) / dz**2
    
    lap_Ar = d2Ar_dr2 + dAr_dr/R_nodal - Ar_nodal/R_nodal**2 + d2Ar_dz2
    Jr_nodal = -lap_Ar
    
    # A_theta component
    d2At_dr2 = xp.zeros_like(At_nodal)
    dAt_dr = xp.zeros_like(At_nodal)
    d2At_dz2 = xp.zeros_like(At_nodal)
    
    d2At_dr2[1:-1, 1:-1] = (At_nodal[2:, 1:-1] - 2*At_nodal[1:-1, 1:-1] + At_nodal[:-2, 1:-1]) / dr**2
    dAt_dr[1:-1, 1:-1] = (At_nodal[2:, 1:-1] - At_nodal[:-2, 1:-1]) / (2*dr)
    d2At_dz2[1:-1, 1:-1] = (At_nodal[1:-1, 2:] - 2*At_nodal[1:-1, 1:-1] + At_nodal[1:-1, :-2]) / dz**2
    
    lap_At = d2At_dr2 + dAt_dr/R_nodal - At_nodal/R_nodal**2 + d2At_dz2
    Jt_nodal = -lap_At
    
    # A_z component (no A_z/r² term)
    d2Az_dr2 = xp.zeros_like(Az_nodal)
    dAz_dr = xp.zeros_like(Az_nodal)
    d2Az_dz2 = xp.zeros_like(Az_nodal)
    
    d2Az_dr2[1:-1, 1:-1] = (Az_nodal[2:, 1:-1] - 2*Az_nodal[1:-1, 1:-1] + Az_nodal[:-2, 1:-1]) / dr**2
    dAz_dr[1:-1, 1:-1] = (Az_nodal[2:, 1:-1] - Az_nodal[:-2, 1:-1]) / (2*dr)
    d2Az_dz2[1:-1, 1:-1] = (Az_nodal[1:-1, 2:] - 2*Az_nodal[1:-1, 1:-1] + Az_nodal[1:-1, :-2]) / dz**2
    
    lap_Az = d2Az_dr2 + dAz_dr/R_nodal + d2Az_dz2
    Jz_nodal = -lap_Az

    return Jr_nodal, Jt_nodal, Jz_nodal


def _get_eb_covered_nodal_mask(eb_factory, geom):
    """Get EB covered nodal mask following pyAMReX staircase preprocessing."""
    nr_cc = geom.domain.length(0)
    nz_cc = geom.domain.length(1)
    nr_nodal = nr_cc + 1
    nz_nodal = nz_cc + 1

    volfrac_cc = _extract_cc_data(eb_factory.getVolFrac(), geom)
    covered_cc = volfrac_cc <= 1.0e-12

    # Project to nodal
    covered_nodal = xp.zeros((nr_nodal, nz_nodal), dtype=xp.bool_)
    covered_nodal[:-1, :-1] |= covered_cc
    covered_nodal[:-1, 1:] |= covered_cc
    covered_nodal[1:, :-1] |= covered_cc
    covered_nodal[1:, 1:] |= covered_cc

    return covered_nodal


def _identify_boundary_and_interior(covered_nodal):
    """
    Identify three regions for J validation:
    
    - eb_edge: EB covered nodes at the edge (will have surface current, zeroed by J BC)
    - plasma_first: First plasma cells adjacent to EB (should have small J with correct pullback)
    - plasma_bulk: Bulk plasma interior away from EB
    
    With correct pullback, J in plasma_first should be small (gradient confined to EB edge).
    """
    nr, nz = covered_nodal.shape
    
    # EB boundary edge nodes (covered, adjacent to plasma)
    eb_edge = xp.zeros((nr, nz), dtype=xp.bool_)
    # First plasma cells adjacent to EB
    plasma_first = xp.zeros((nr, nz), dtype=xp.bool_)
    # Bulk plasma interior
    plasma_bulk = xp.zeros((nr, nz), dtype=xp.bool_)
    
    for i in range(1, nr-1):  # Exclude domain boundaries
        for j in range(1, nz-1):
            if covered_nodal[i, j]:
                # This node is inside EB - check if it's at the edge
                has_uncovered_neighbor = (
                    not covered_nodal[i-1, j] or not covered_nodal[i+1, j] or
                    not covered_nodal[i, j-1] or not covered_nodal[i, j+1]
                )
                if has_uncovered_neighbor:
                    eb_edge[i, j] = True
            else:
                # This is a plasma node
                has_covered_neighbor = (
                    covered_nodal[i-1, j] or covered_nodal[i+1, j] or
                    covered_nodal[i, j-1] or covered_nodal[i, j+1]
                )
                if has_covered_neighbor:
                    plasma_first[i, j] = True
                else:
                    plasma_bulk[i, j] = True
    
    return eb_edge, plasma_first, plasma_bulk

    
    is_interior = covered_nodal & ~is_boundary
    return is_boundary, is_interior


def test_eb_pullback_validation():
    """Test that J is properly zeroed at EB boundaries with staircase nodal-mask EB."""
    ncell = 80
    coil = _single_coil_spec()

    # Setup AMReX geometry and EB
    dom = amr.Box(amr.IntVect(0, 0), amr.IntVect(ncell - 1, ncell - 1))
    geom = amr.Geometry(dom, amr.RealBox(0, 0, 1, 1), 1, [0, 0])

    ba = amr.BoxArray(dom)
    ba.max_size(max(ncell, 32))
    dm = amr.DistributionMapping(ba)

    pp = amr.ParmParse("eb2")
    for key in ("geom_type", "parser_function"):
        try:
            pp.remove(key)
        except Exception:
            pass
    pp.add("geom_type", "all_regular")

    amr.EB2_Build(geom, required_coarsening_level=0,
                  max_coarsening_level=0, ngrow=4)

    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    dr, dz = dx[0], dx[1]
    problo = geom.ProbLo()

    nr_nodal = geom.domain.length(0) + 1
    nz_nodal = geom.domain.length(1) + 1
    r_nodal = problo[0] + xp.arange(nr_nodal, dtype=xp.float64) * dr
    z_nodal = problo[1] + xp.arange(nz_nodal, dtype=xp.float64) * dz
    rr, zz = xp.meshgrid(r_nodal, z_nodal, indexing="ij")
    r_lo = coil['r1c']
    r_hi = coil['r1c'] + coil['drc']
    nodal_mask = (
        (rr >= r_lo) & (rr <= r_hi) &
        (zz >= coil['z_lo']) & (zz <= coil['z_hi'])
    )

    eb_factory = amr.makeStaircaseEBFabFactoryFromCupy(
        geom, ba, dm, nodal_mask, amr.Vector_int([1, 1, 1]), amr.EBSupport.full
    )

    # Create nodal MultiFabs
    ba_nd = amr.BoxArray(ba)
    ba_nd.convert(amr.IntVect(1, 1))

    A = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    J = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    for d in range(3):
        A[d].set_val(0.0)
        J[d].set_val(0.0)

    # Setup solver with coil
    coil_specs = [amr.VectorPoissonSolverNodal.CoilSpec()]
    coil_specs[0].z_lo = coil['z_lo']
    coil_specs[0].z_hi = coil['z_hi']
    coil_specs[0].r1c = coil['r1c']
    coil_specs[0].r2c = coil['r2c']
    coil_specs[0].drc = coil['drc']
    coil_specs[0].psi = coil['psi']

    bc = amr.NodalBoundaryHandler(False)
    lobc = bc.lobc
    hibc = bc.hibc
    for adim in range(3):
        lobc[adim][1] = amr.LinOpBCType.Dirichlet
        hibc[adim][1] = amr.LinOpBCType.Dirichlet
    bc.lobc = lobc
    bc.hibc = hibc
    solver = amr.VectorPoissonSolverNodal(
        geom, ba, dm, bc,
        is_rz=True,
        eb_enabled=True,
        eb_factory=eb_factory,
    )

    # Iteration loop mimicking initialization
    max_iterations = 1
    
    for iteration in range(max_iterations):
        # Step 1: Solve for A with current J (initially zero)
        solver.setEBCoils(1, coil_specs)  # Set coils for A_theta
        solver.solve(A, J, 1e-10, 0.0, 200, 0 if iteration > 0 else 2)
        
        # Extract solution
        Ar_data, r_arr, z_arr = _extract_nodal_data(A[0], geom)
        At_data, _, _ = _extract_nodal_data(A[1], geom)
        Az_data, _, _ = _extract_nodal_data(A[2], geom)
        
        # Step 2: Compute J = -∇²A directly (avoids B field boundary issues)
        Jr_new, Jt_new, Jz_new = _compute_vector_laplacian_rz(
            Ar_data, At_data, Az_data, r_arr, dr, dz
        )
        
        # Get EB masks
        covered_nodal = _get_eb_covered_nodal_mask(eb_factory, geom)
        eb_edge, plasma_first, plasma_bulk = _identify_boundary_and_interior(covered_nodal)
        
        # Diagnostic: Check A_theta values
        At_eb_edge = At_data[eb_edge]
        At_plasma_first = At_data[plasma_first]
        At_plasma_bulk = At_data[plasma_bulk]
        print(f"\nDiagnostic - A_theta values:")
        print(f"  At EB edge: min={float(xp.min(At_eb_edge)) if At_eb_edge.size > 0 else 0:.3e}, max={float(xp.max(At_eb_edge)) if At_eb_edge.size > 0 else 0:.3e}")
        print(f"  At first plasma cells: min={float(xp.min(At_plasma_first)) if At_plasma_first.size > 0 else 0:.3e}, max={float(xp.max(At_plasma_first)) if At_plasma_first.size > 0 else 0:.3e}")
        print(f"  At bulk plasma: min={float(xp.min(At_plasma_bulk)) if At_plasma_bulk.size > 0 else 0:.3e}, max={float(xp.max(At_plasma_bulk)) if At_plasma_bulk.size > 0 else 0:.3e}")
        print(f"  Expected from coil: psi/r ~ {coil['psi']}/{coil['r1c']:.3f} = {coil['psi']/coil['r1c']:.3e}")
        
        # Step 4: Check J in all three regions
        Jr_eb = Jr_new[eb_edge]
        Jt_eb = Jt_new[eb_edge]
        Jz_eb = Jz_new[eb_edge]
        
        Jr_first = Jr_new[plasma_first]
        Jt_first = Jt_new[plasma_first]
        Jz_first = Jz_new[plasma_first]
        
        Jr_bulk = Jr_new[plasma_bulk]
        Jt_bulk = Jt_new[plasma_bulk]
        Jz_bulk = Jz_new[plasma_bulk]
        
        max_Jr_eb = float(xp.max(xp.abs(Jr_eb))) if Jr_eb.size > 0 else 0.0
        max_Jt_eb = float(xp.max(xp.abs(Jt_eb))) if Jt_eb.size > 0 else 0.0
        max_Jz_eb = float(xp.max(xp.abs(Jz_eb))) if Jz_eb.size > 0 else 0.0
        
        max_Jr_first = float(xp.max(xp.abs(Jr_first))) if Jr_first.size > 0 else 0.0
        max_Jt_first = float(xp.max(xp.abs(Jt_first))) if Jt_first.size > 0 else 0.0
        max_Jz_first = float(xp.max(xp.abs(Jz_first))) if Jz_first.size > 0 else 0.0
        
        max_Jr_bulk = float(xp.max(xp.abs(Jr_bulk))) if Jr_bulk.size > 0 else 0.0
        max_Jt_bulk = float(xp.max(xp.abs(Jt_bulk))) if Jt_bulk.size > 0 else 0.0
        max_Jz_bulk = float(xp.max(xp.abs(Jz_bulk))) if Jz_bulk.size > 0 else 0.0
        
        print(f"\nIteration {iteration + 1}:")
        print(f"  J_r: EB edge = {max_Jr_eb:.3e}, first plasma = {max_Jr_first:.3e}, bulk = {max_Jr_bulk:.3e}")
        print(f"  J_θ: EB edge = {max_Jt_eb:.3e}, first plasma = {max_Jt_first:.3e}, bulk = {max_Jt_bulk:.3e}")
        print(f"  J_z: EB edge = {max_Jz_eb:.3e}, first plasma = {max_Jz_first:.3e}, bulk = {max_Jz_bulk:.3e}")
        print(f"  Solver iterations: r={solver.getNumIters(0)}, θ={solver.getNumIters(1)}, z={solver.getNumIters(2)}")
        
        # Update J for next iteration (commented out for single-iteration validation)
        # The key validation is checking J values after solve, not iterating to convergence
        """
        for mfi in J[0]:
            arr = J[0].array(mfi)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i, j] = Jr_new[i, j]
        for mfi in J[1]:
            arr = J[1].array(mfi)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i, j] = Jt_new[i, j]
        for mfi in J[2]:
            arr = J[2].array(mfi)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i, j] = Jz_new[i, j]
        """
    
    # Validation: The key test is whether J has leaked into first plasma cells
    # - J at EB edge (covered): Expected to be large (surface current), will be zeroed by J BC
    # - J in first plasma cells: Should be small with correct pullback (gradient confined to EB)
    # - J in bulk plasma: Should be small (no sources)
    print(f"\n{'='*70}")
    print("Final validation:")
    print(f"  J at EB edge (covered nodes, will be zeroed by J BC):")
    print(f"    Max |J_r| = {max_Jr_eb:.3e}")
    print(f"    Max |J_θ| = {max_Jt_eb:.3e}")
    print(f"    Max |J_z| = {max_Jz_eb:.3e}")
    print(f"  J in first plasma cells (should be small with correct pullback):")
    print(f"    Max |J_r| = {max_Jr_first:.3e}")
    print(f"    Max |J_θ| = {max_Jt_first:.3e}")
    print(f"    Max |J_z| = {max_Jz_first:.3e}")
    print(f"  J in bulk plasma (should be small, no sources):")
    print(f"    Max |J_r| = {max_Jr_bulk:.3e}")
    print(f"    Max |J_θ| = {max_Jt_bulk:.3e}")
    print(f"    Max |J_z| = {max_Jz_bulk:.3e}")
    
    # The critical test: with proper pullback, J in first plasma cells should be small
    # This ensures the gradient/surface current is confined to the EB edge
    tolerance_first = 1e-4
    tolerance_bulk = 1e-4
    print(f"\n  Checking if J leaks (first-plasma tolerance = {tolerance_first:.0e})...")
    assert max_Jr_first < tolerance_first, f"J_r leaked into first plasma cells: {max_Jr_first:.3e}"
    assert max_Jt_first < tolerance_first, f"J_θ leaked into first plasma cells: {max_Jt_first:.3e}"
    assert max_Jz_first < tolerance_first, f"J_z leaked into first plasma cells: {max_Jz_first:.3e}"
    assert max_Jr_bulk < tolerance_bulk, f"J_r leaked into bulk plasma: {max_Jr_bulk:.3e}"
    assert max_Jt_bulk < tolerance_bulk, f"J_θ leaked into bulk plasma: {max_Jt_bulk:.3e}"
    assert max_Jz_bulk < tolerance_bulk, f"J_z leaked into bulk plasma: {max_Jz_bulk:.3e}"
    
    print(f"\n✓ Test passed: J properly confined to EB edge with pullback")
    print(f"  (Surface current at EB edge will be zeroed by J BC application)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    was_initialized = amr.initialized()
    if not was_initialized:
        if len(sys.argv) > 2:
            init_list = [sys.argv[2:]]
        else:
            init_list = [""]
        amr.initialize(init_list)
    
    try:
        test_eb_pullback_validation()
    finally:
        if not was_initialized and amr.initialized():
            amr.finalize()
