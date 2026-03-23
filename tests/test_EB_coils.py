#!/usr/bin/env python3

"""
Two-coil EB test: VectorPoissonSolverNodal with two EB boxes in 2D RZ.
======================================================================

Two rectangular coil cross-sections with different prescribed
psi values on their EB surfaces:
  - Coil 1: r in [0.30, 0.70], z in [0.60, 0.80], psi = 1.0
  - Coil 2: r in [0.30, 0.70], z in [0.20, 0.40], psi = 0.5

No current source — the solution is driven entirely by EB Dirichlet BCs.
The C++ functor sets A_theta = psi / r at each EB node.
"""
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

# Coil geometry (rectangular: r1c == r2c)
COILS = [
    {'r1c': 0.30, 'r2c': 0.30, 'drc': 0.4, 'z_lo': 0.60, 'z_hi': 0.80, 'psi': 1.0},
    {'r1c': 0.30, 'r2c': 0.30, 'drc': 0.4, 'z_lo': 0.20, 'z_hi': 0.40, 'psi': 0.5},
]


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
            data = marr[ng0: ng0 + nr, ng1: ng1 + nz, 0, 0]
        else:
            data = marr[0, 0, ng1: ng1 + nz, ng0: ng0 + nr].T
        full_arr[lo[0]:lo[0] + nr, lo[1]:lo[1] + nz] = data

    r_arr = problo[0] + xp.arange(nr_full, dtype=xp.float64) * dx[0]
    z_arr = problo[1] + xp.arange(nz_full, dtype=xp.float64) * dx[1]
    return full_arr, r_arr, z_arr


def run_solve(ncell):
    """Run the two-coil EB solve."""
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

    nodal_mask = xp.zeros((nr_nodal, nz_nodal), dtype=bool)
    for c in COILS:
        r_lo, r_hi = coil_r_bounds(c)
        nodal_mask |= (
            (rr >= r_lo) & (rr <= r_hi) &
            (zz >= c["z_lo"]) & (zz <= c["z_hi"])
        )

    eb_factory = amr.makeStaircaseEBFabFactoryFromCupy(
        geom, ba, dm, nodal_mask, amr.Vector_int([1, 1, 1]), amr.EBSupport.full
    )

    ba_nd = amr.BoxArray(ba)
    ba_nd.convert(amr.IntVect(1, 1))

    A = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    J = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    for d in range(3):
        A[d].set_val(0.0)
        J[d].set_val(0.0)

    coil_specs = []
    for c in COILS:
        cs = amr.VectorPoissonSolverNodal.CoilSpec()
        cs.z_lo = c['z_lo']
        cs.z_hi = c['z_hi']
        cs.r1c = c['r1c']
        cs.r2c = c['r2c']
        cs.drc = c['drc']
        cs.psi = c['psi']
        coil_specs.append(cs)

    bc = amr.NodalBoundaryHandler(periodic_axial=False, axial_dirichlet=True)
    solver = amr.VectorPoissonSolverNodal(
        geom, ba, dm, bc,
        is_rz=True,
        eb_enabled=True,
        eb_factory=eb_factory,
    )

    solver.setEBCoils(1, coil_specs)
    solver.solve(A, J, 1e-12, 0.0, 200, 2)

    print(f"  Component 1 (theta): converged in {solver.getNumIters(1)} iterations, "
          f"residual = {solver.getResidual(1):.2e}")

    A_data, r_arr, z_arr = extract_nodal_data(A[1], geom)
    return A_data, r_arr, z_arr, dr, dz


def coil_r_bounds(c):
    """Return (r_lo, r_hi) for a rectangular coil."""
    return c['r1c'], c['r1c'] + c['drc']


def plot_solution(A_data, r_arr, z_arr, dr, dz,
                  plot_file="EB_two_coil_solution.png"):
    """Plot A_theta solution with two coil outlines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    A_np = to_numpy(A_data)
    r_np = to_numpy(r_arr)
    z_np = to_numpy(z_arr)

    rr, zz = np.meshgrid(r_np, z_np, indexing="ij")

    margin = 0.5 * max(dr, dz)
    inside = np.zeros_like(rr, dtype=bool)
    for c in COILS:
        r_lo, r_hi = coil_r_bounds(c)
        inside |= ((rr > r_lo + margin) & (rr < r_hi - margin) &
                    (zz > c['z_lo'] + margin) & (zz < c['z_hi'] - margin))
    A_plot = np.where(inside, np.nan, A_np)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    pcm = ax.pcolormesh(r_np, z_np, A_plot.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label=r"$A_\theta$", shrink=0.8)

    colors = ["red", "cyan"]
    for ic, c in enumerate(COILS):
        r_lo, r_hi = coil_r_bounds(c)
        ax.add_patch(Rectangle(
            (r_lo, c['z_lo']),
            r_hi - r_lo, c['z_hi'] - c['z_lo'],
            fill=False, edgecolor=colors[ic], linewidth=1.5,
            linestyle="--",
            label=f"Coil {ic+1} ($\\psi$={c['psi']})"))

    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_title(r"$A_\theta$ — Two-coil EB Dirichlet (no source)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {plot_file}")
    plt.close()


def plot_axial_slice(A_data, r_arr, z_arr, dr, dz,
                     plot_file="EB_two_coil_axial_slice.png"):
    """Plot A_theta along the axis r=0.5 (midpoint of coil radial extent)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    A_np = to_numpy(A_data)
    r_np = to_numpy(r_arr)
    z_np = to_numpy(z_arr)

    r_mid = 0.5
    imid = np.argmin(np.abs(r_np - r_mid))
    A_slice = A_np[imid, :]

    margin = 0.5 * dz
    mask = np.ones_like(z_np, dtype=bool)
    for c in COILS:
        mask[(z_np > c['z_lo'] + margin) & (z_np < c['z_hi'] - margin)] = False

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(z_np[mask], A_slice[mask], "b-", linewidth=1.5,
            label=rf"$A_\theta(r={r_mid}, z)$")

    colors = ["red", "cyan"]
    for ic, c in enumerate(COILS):
        ax.axvline(c['z_lo'], color=colors[ic], linestyle="--", alpha=0.5,
                   label=f"Coil {ic+1} edges")
        ax.axvline(c['z_hi'], color=colors[ic], linestyle="--", alpha=0.5)
        ax.axhline(c['psi'] / r_mid, color=colors[ic], linestyle=":",
                   alpha=0.3, label=f"$\\psi_{ic+1}/r$ = {c['psi']/r_mid:.2f}")

    ax.set_xlabel("z")
    ax.set_ylabel(r"$A_\theta$")
    ax.set_title(rf"Axial slice at $r = {r_mid}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {plot_file}")
    plt.close()


def main():
    ncell = 256
    print(f"\n--- Two-coil EB test: {ncell} x {ncell} ---")
    A_data, r_arr, z_arr, dr, dz = run_solve(ncell)

    print("\nGenerating plots...")
    plot_solution(A_data, r_arr, z_arr, dr, dz)
    plot_axial_slice(A_data, r_arr, z_arr, dr, dz)


if __name__ == "__main__":
    amr.initialize([""])
    try:
        main()
    finally:
        amr.finalize()
