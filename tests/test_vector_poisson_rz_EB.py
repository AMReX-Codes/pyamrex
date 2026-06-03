#!/usr/bin/env python3

"""
Self-convergence test: VectorPoissonSolverNodal with EB in 2D RZ.
=================================================================

WHAT THIS TEST DOES
-------------------
This test verifies that the VectorPoissonSolverNodal correctly solves
the vector Poisson equation in RZ (cylindrical) coordinates when an
embedded boundary (EB) is present inside the domain.

PHYSICAL SETUP
--------------
Domain: [0, 1] x [0, 1] in (r, z) coordinates.
EB geometry: A sphere (circle in 2D cross-section) centered at (r, z) = (0.5, 0.5)
             with radius 0.25. The fluid is OUTSIDE the sphere — the interior
             of the sphere is "covered" (removed from the computational domain).

             +-------------------+
             |                   |
             |     FLUID         |
             |       ___         |
             |      /   \        |  z
             |     | EB  |       |  ^
             |      \___/        |  |
             |                   |  +---> r
             |     FLUID         |
             +-------------------+

Boundary conditions:
  - r = 0 (axis): A_theta = 0  (symmetry axis, enforced by is_rz=True)
  - r = 1:        A_theta = 0  (Dirichlet)
  - z = 0:        A_theta = 0  (Dirichlet, via axial_dirichlet=True)
  - z = 1:        A_theta = 0  (Dirichlet)
  - EB surface:   A_theta = 0  (homogeneous Dirichlet on embedded boundary)

Source term: J_theta = 1 A/m^2 everywhere (EB solver ignores covered cells).

EQUATION SOLVED
---------------
The solver solves the RZ vector Poisson equation for A_theta:

    d^2 A     1  dA     A     d^2 A
    ----- + --- ---- - --- + ------ = -mu_0 * J_theta
    dr^2     r  dr     r^2    dz^2

WHY SELF-CONVERGENCE?
---------------------
We don't have an analytical solution for this problem. Instead, we verify
correctness using self-convergence: solve at multiple resolutions, interpolate
each coarse solution onto the next finer grid, and measure the L2 difference.
For a 2nd-order scheme, the difference should decrease by ~4x each time we
double the resolution (order ≈ 2).

EXPECTED RESULTS
----------------
  - Self-convergence order ~2.0 in the fluid bulk
  - A_theta = 0 on all boundaries and inside the sphere
  - A_theta > 0 in the fluid region
"""

import argparse

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


def sphere_distance(r, z, cr=0.5, cz=0.5, radius=0.25):
    """Signed distance from EB sphere surface (positive = outside/fluid)."""
    return xp.sqrt((r - cr) ** 2 + (z - cz) ** 2) - radius


def extract_nodal_data(mf, geom):
    """Extract the full nodal array and coordinate arrays from a MultiFab."""
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
            data = marr[ng0 : ng0 + nr, ng1 : ng1 + nz, 0, 0]
        else:
            data = marr[0, 0, ng1 : ng1 + nz, ng0 : ng0 + nr].T
        full_arr[lo[0] : lo[0] + nr, lo[1] : lo[1] + nz] = data

    r_arr = problo[0] + xp.arange(nr_full, dtype=xp.float64) * dx[0]
    z_arr = problo[1] + xp.arange(nz_full, dtype=xp.float64) * dx[1]
    return full_arr, r_arr, z_arr


def interpolate_to_fine(A_coarse, r_c, z_c, r_f, z_f):
    """Bilinear interpolation of coarse nodal solution onto the fine nodal grid."""
    nr_f, nz_f = len(r_f), len(z_f)
    A_fine_interp = xp.zeros((nr_f, nz_f), dtype=xp.float64)

    dr_c = float(r_c[1] - r_c[0])
    dz_c = float(z_c[1] - z_c[0])
    r0 = float(r_c[0])
    z0 = float(z_c[0])
    nr_c = len(r_c)
    nz_c = len(z_c)

    for i in range(nr_f):
        ri = float(r_f[i])
        ic = min(max(int((ri - r0) / dr_c), 0), nr_c - 2)
        wr = (ri - float(r_c[ic])) / dr_c
        for j in range(nz_f):
            zj = float(z_f[j])
            jc = min(max(int((zj - z0) / dz_c), 0), nz_c - 2)
            wz = (zj - float(z_c[jc])) / dz_c
            A_fine_interp[i, j] = (
                (1 - wr) * (1 - wz) * A_coarse[ic, jc]
                + wr * (1 - wz) * A_coarse[ic + 1, jc]
                + (1 - wr) * wz * A_coarse[ic, jc + 1]
                + wr * wz * A_coarse[ic + 1, jc + 1]
            )
    return A_fine_interp


def run_solve(ncell):
    """Run the EB solve at a given resolution and return solution + coords."""
    dom = amr.Box(amr.IntVect(0, 0), amr.IntVect(ncell - 1, ncell - 1))
    geom = amr.Geometry(dom, amr.RealBox(0, 0, 1, 1), 1, [0, 0])

    ba = amr.BoxArray(dom)
    ba.max_size(max(ncell, 32))
    dm = amr.DistributionMapping(ba)

    amr.EB2_Build(geom, required_coarsening_level=0, max_coarsening_level=0, ngrow=4)

    eb_factory = amr.makeEBFabFactory(
        geom, ba, dm, amr.Vector_int([1, 1, 1]), amr.EBSupport.full
    )

    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    dr, dz = dx[0], dx[1]

    ba_nd = amr.BoxArray(ba)
    ba_nd.convert(amr.IntVect(1, 1))

    A = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    J = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    for d in range(3):
        A[d].set_val(0.0)
        J[d].set_val(0.0)
    J[1].set_val(1.0)

    bc = amr.NodalBoundaryHandler(periodic_axial=False, axial_dirichlet=True)
    solver = amr.VectorPoissonSolverNodal(
        geom,
        ba,
        dm,
        bc,
        is_rz=True,
        eb_enabled=True,
        eb_factory=eb_factory,
    )
    solver.solve(A, J, 1e-12, 0.0, 200, 2)

    print(
        f"  Converged in {solver.getNumIters(1)} iterations, "
        f"residual = {solver.getResidual(1):.2e}"
    )

    A_data, r_arr, z_arr = extract_nodal_data(A[1], geom)
    return A_data, r_arr, z_arr, dr, dz


def plot_solution(solutions, resolutions, plot_file="EB_convergence.png"):
    """Plot A_theta at each resolution."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    n_res = len(resolutions)
    fig, axes = plt.subplots(1, n_res, figsize=(5 * n_res, 4.5))
    if n_res == 1:
        axes = [axes]

    for idx in range(n_res):
        ax = axes[idx]
        A_data, r_arr, z_arr, dr, dz = solutions[idx]
        A_np = to_numpy(A_data)
        r_np = to_numpy(r_arr)
        z_np = to_numpy(z_arr)

        rr, zz = np.meshgrid(r_np, z_np, indexing="ij")
        dist = np.sqrt((rr - 0.5) ** 2 + (zz - 0.5) ** 2) - 0.25
        A_plot = np.where(dist < -0.5 * max(dr, dz), np.nan, A_np)

        pcm = ax.pcolormesh(r_np, z_np, A_plot.T, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label=r"$A_\theta$", shrink=0.8)
        ax.add_patch(
            Circle(
                (0.5, 0.5),
                0.25,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
                linestyle="--",
                label="EB boundary",
            )
        )
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(f"$A_\\theta$ ({resolutions[idx]}x{resolutions[idx]})")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\n  Solution plots saved to {plot_file}")
    plt.close()


def plot_convergence_diff(
    solutions, resolutions, errors, plot_file="EB_convergence_diff.png"
):
    """Plot the difference between successive resolutions."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("  matplotlib not available, skipping convergence diff plots.")
        return

    n_diffs = len(resolutions) - 1
    if n_diffs < 1:
        return

    fig, axes = plt.subplots(1, n_diffs, figsize=(5 * n_diffs, 4.5))
    if n_diffs == 1:
        axes = [axes]

    for idx in range(n_diffs):
        ax = axes[idx]
        A_c, r_c, z_c, dr_c, dz_c = solutions[idx]
        A_f, r_f, z_f, dr_f, dz_f = solutions[idx + 1]

        A_c_interp = interpolate_to_fine(A_c, r_c, z_c, r_f, z_f)
        diff = to_numpy(A_f - A_c_interp)
        r_np = to_numpy(r_f)
        z_np = to_numpy(z_f)

        rr, zz = np.meshgrid(r_np, z_np, indexing="ij")
        dist = np.sqrt((rr - 0.5) ** 2 + (zz - 0.5) ** 2) - 0.25
        diff_plot = np.where(dist < -0.5 * max(dr_f, dz_f), np.nan, diff)

        vmax = np.nanmax(np.abs(diff_plot))
        pcm = ax.pcolormesh(
            r_np,
            z_np,
            diff_plot.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(pcm, ax=ax, label=r"$\Delta A_\theta$", shrink=0.8)
        ax.add_patch(
            Circle(
                (0.5, 0.5),
                0.25,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(
            f"Diff {resolutions[idx]}→{resolutions[idx + 1]}\nL2={errors[idx]:.2e}"
        )
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"  Convergence diff plots saved to {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="EB convergence test for VectorPoissonSolverNodal in RZ"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate solution and convergence plots"
    )
    args = parser.parse_args()

    resolutions = [32, 64, 128, 256]
    solutions = []

    for ncell in resolutions:
        print(f"\n--- Resolution {ncell} x {ncell} ---")
        solutions.append(run_solve(ncell))

    # --- Self-convergence analysis ---
    print("\n" + "=" * 50)
    print("Self-Convergence (Richardson Extrapolation)")
    print("=" * 50)

    errors = []
    for i in range(len(resolutions) - 1):
        A_c, r_c, z_c, dr_c, dz_c = solutions[i]
        A_f, r_f, z_f, dr_f, dz_f = solutions[i + 1]

        A_c_interp = interpolate_to_fine(A_c, r_c, z_c, r_f, z_f)

        rr, zz = xp.meshgrid(r_f, z_f, indexing="ij")
        fluid_mask = sphere_distance(rr, zz) > 3.0 * max(dr_c, dz_c)

        diff = xp.where(fluid_mask, (A_f - A_c_interp) ** 2, 0.0)
        n_pts = int(xp.sum(fluid_mask))
        l2_diff = float(xp.sqrt(xp.sum(diff) / max(n_pts, 1)))

        errors.append(l2_diff)
        print(
            f"  {resolutions[i]:4d} vs {resolutions[i + 1]:4d}: "
            f"L2 diff = {l2_diff:.6e}  ({n_pts} nodes)"
        )

    print()
    for i in range(len(errors)):
        line = (
            f"  {resolutions[i]:4d}->{resolutions[i + 1]:4d}: L2 diff = {errors[i]:.6e}"
        )
        if i > 0:
            ratio = errors[i - 1] / errors[i]
            order = np.log2(ratio)
            line += f"  ratio = {ratio:.2f}  order = {order:.2f}"
        print(line)

    if len(errors) >= 2:
        ratio = errors[-2] / errors[-1]
        order = np.log2(ratio)
        if order > 1.5:
            print(f"\n✓ PASS: Self-convergence order {order:.2f} > 1.5")
        else:
            print(f"\n✗ FAIL: Self-convergence order {order:.2f} <= 1.5")

    if args.plot:
        print("\nGenerating plots...")
        plot_solution(solutions, resolutions)
        plot_convergence_diff(solutions, resolutions, errors)


if __name__ == "__main__":
    amr.initialize(
        [
            "",
            "eb2.geom_type=sphere",
            "eb2.sphere_center=0.5 0.5",
            "eb2.sphere_radius=0.25",
            "eb2.sphere_has_fluid_inside=0",
        ]
    )
    try:
        main()
    finally:
        amr.finalize()
