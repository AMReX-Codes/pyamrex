#!/usr/bin/env python3
"""Test VectorPoissonSolverNodal in 2D RZ — A_theta convergence study.

Manufactured solution: A_theta = r * sin(pi*r) * sin(pi*z) on [0,1]^2.

The vector Laplacian for A_theta in cylindrical coordinates is:
  L[A] = (1/r) d/dr(r dA/dr) + d²A/dz² - A/r²

For A = r sin(πr) sin(πz):
  L[A] = [3π cos(πr) - 2π²r sin(πr)] sin(πz)

Source: J_theta = -L[A] / μ₀

Expected: second-order convergence.
"""

import argparse

import numpy as np
import pytest

try:
    import cupy as xp
except ImportError:
    import numpy as xp

# RZ is a 2D concept; this test requires the 2D module. It is skipped when the
# 2D module cannot be loaded (e.g. the test suite already registered the 3D one).
amr = pytest.importorskip("amrex.space2d", exc_type=ImportError)

PI = np.pi
MU0 = 1.25663706212e-6


def fill_nodal_multifab(mf, geom, fill_func):
    """Fill a nodal MultiFab using a callable f(r, z) -> array."""
    problo = geom.ProbLo()
    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    for mfi in mf:
        bx = mfi.validbox()
        lo, hi = bx.small_end, bx.big_end
        nr, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
        ii, jj = xp.meshgrid(
            xp.arange(nr, dtype=xp.float64) + lo[0],
            xp.arange(nz, dtype=xp.float64) + lo[1],
            indexing="ij",
        )
        vals = fill_func(problo[0] + ii * dx[0], problo[1] + jj * dx[1])
        marr = xp.asarray(mf.array(mfi))
        ng0, ng1 = mf.n_grow_vect[0], mf.n_grow_vect[1]
        if marr.shape[0] == nr + 2 * ng0:
            marr[ng0 : ng0 + nr, ng1 : ng1 + nz, 0, 0] = vals
        else:
            marr[0, 0, ng1 : ng1 + nz, ng0 : ng0 + nr] = vals.T


def extract_nodal_data(mf, geom):
    """Extract full nodal 2D array and coordinate vectors from a MultiFab."""
    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    problo = geom.ProbLo()
    dom = geom.domain
    nr_nodes = dom.length(0) + 1
    nz_nodes = dom.length(1) + 1
    data = np.zeros((nr_nodes, nz_nodes))
    for mfi in mf:
        bx = mfi.validbox()
        lo, hi = bx.small_end, bx.big_end
        lnr, lnz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
        ng0, ng1 = mf.n_grow_vect[0], mf.n_grow_vect[1]
        marr = xp.asarray(mf.array(mfi))
        if marr.shape[0] == lnr + 2 * ng0:
            block = marr[ng0 : ng0 + lnr, ng1 : ng1 + lnz, 0, 0]
        else:
            block = marr[0, 0, ng1 : ng1 + lnz, ng0 : ng0 + lnr].T
        if hasattr(block, "get"):
            block = block.get()
        data[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1] = block
    r = problo[0] + np.arange(nr_nodes) * dx[0]
    z = problo[1] + np.arange(nz_nodes) * dx[1]
    return r, z, data


def compute_error(numerical, exact, geom):
    """Compute L_inf and L2 error between two nodal MultiFabs."""
    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    linf, l2sum = 0.0, 0.0
    for mfi in numerical:
        bx = mfi.validbox()
        lo, hi = bx.small_end, bx.big_end
        nr, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
        ng0, ng1 = numerical.n_grow_vect[0], numerical.n_grow_vect[1]
        num_arr = xp.asarray(numerical.array(mfi))
        ex_arr = xp.asarray(exact.array(mfi))
        if num_arr.shape[0] == nr + 2 * ng0:
            n = num_arr[ng0 : ng0 + nr, ng1 : ng1 + nz, 0, 0]
            e = ex_arr[ng0 : ng0 + nr, ng1 : ng1 + nz, 0, 0]
        else:
            n = num_arr[0, 0, ng1 : ng1 + nz, ng0 : ng0 + nr].T
            e = ex_arr[0, 0, ng1 : ng1 + nz, ng0 : ng0 + nr].T
        diff = xp.abs(n - e)
        linf = max(linf, float(xp.max(diff)))
        l2sum += float(xp.sum(diff**2))
    return linf, np.sqrt(l2sum * dx[0] * dx[1])


# ---------- Manufactured solution ----------


def atheta_exact(r, z):
    """A_theta = r sin(pi r) sin(pi z)."""
    return r * xp.sin(PI * r) * xp.sin(PI * z)


def jtheta_source(r, z):
    """J_theta such that vector-Lap(A_theta) - A_theta/r^2 = -mu0 J_theta.

    L[A] = [3 pi cos(pi r) - 2 pi^2 r sin(pi r)] sin(pi z)
    J = -L[A] / mu0
    """
    lap = (3 * PI * xp.cos(PI * r) - 2 * PI**2 * r * xp.sin(PI * r)) * xp.sin(PI * z)
    return -lap / MU0


# ---------- Solver wrapper ----------


def run_test(ncell, verbose=0):
    """Solve for A_theta at a given resolution. Return (geom, A, L_inf, L2)."""
    dom = amr.Box(amr.IntVect(0, 0), amr.IntVect(ncell - 1, ncell - 1))
    geom = amr.Geometry(dom, amr.RealBox(0, 0, 1, 1), 1, [0, 0])  # coord=1 (RZ)
    ba = amr.BoxArray(dom)
    ba.max_size(ncell)
    dm = amr.DistributionMapping(ba)
    ba_nd = amr.BoxArray(ba)
    ba_nd.convert(amr.IntVect(1, 1))

    A = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    J = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    for d in range(3):
        A[d].set_val(0.0)
        J[d].set_val(0.0)

    fill_nodal_multifab(J[1], geom, jtheta_source)

    bc = amr.NodalBoundaryHandler(False)
    lobc = bc.lobc
    hibc = bc.hibc
    for adim in range(3):
        lobc[adim][1] = amr.LinOpBCType.Dirichlet
        hibc[adim][1] = amr.LinOpBCType.Dirichlet
    bc.lobc = lobc
    bc.hibc = hibc
    solver = amr.VectorPoissonSolverNodal(geom, ba, dm, bc, is_rz=True)
    solver.solve(A, J, 1e-10, 0.0, 200, verbose)

    A_exact = amr.MultiFab(ba_nd, dm, 1, 1)
    A_exact.set_val(0.0)
    fill_nodal_multifab(A_exact, geom, atheta_exact)

    linf, l2 = compute_error(A[1], A_exact, geom)
    return geom, A, A_exact, linf, l2


# ---------- Plotting ----------


def plot_solution(A, A_exact_mf, geom, filename="solution_rz_atheta.png"):
    """Plot numerical A_theta, exact, and error."""
    import matplotlib.pyplot as plt

    r, z, num_data = extract_nodal_data(A[1], geom)
    _, _, exact_data = extract_nodal_data(A_exact_mf, geom)
    error_data = num_data - exact_data
    R, Z = np.meshgrid(r, z, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    titles = [r"Numerical $A_\theta$", r"Exact $A_\theta$", "Error"]
    datasets = [num_data, exact_data, error_data]

    for ax, data, title in zip(axes, datasets, titles):
        vmax = np.max(np.abs(data)) or 1.0
        cf = ax.contourf(R, Z, data, levels=30, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


def plot_line_cuts(A, geom, filename="line_cuts_rz_atheta.png"):
    """Plot line cuts at z=0.5 and r=0.5 comparing numerical and exact."""
    import matplotlib.pyplot as plt

    r, z, data = extract_nodal_data(A[1], geom)
    iz_mid = len(z) // 2
    ir_mid = len(r) // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A_theta vs r at z = 0.5
    ax = axes[0]
    exact_r = r * np.sin(PI * r) * np.sin(PI * z[iz_mid])
    ax.plot(r, data[:, iz_mid], "b-", lw=2, label="Numerical")
    ax.plot(r, exact_r, "r--", lw=2, label="Exact")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$A_\theta$")
    ax.set_title(rf"$A_\theta$ vs r at z={z[iz_mid]:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # A_theta vs z at r = 0.5
    ax = axes[1]
    exact_z = r[ir_mid] * np.sin(PI * r[ir_mid]) * np.sin(PI * z)
    ax.plot(z, data[ir_mid, :], "b-", lw=2, label="Numerical")
    ax.plot(z, exact_z, "r--", lw=2, label="Exact")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$A_\theta$")
    ax.set_title(rf"$A_\theta$ vs z at r={r[ir_mid]:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


def plot_convergence(resolutions, errors, filename="convergence_rz_atheta.png"):
    """Plot convergence rates on a log-log scale."""
    import matplotlib.pyplot as plt

    h = [1.0 / n for n in resolutions]
    linfs = [e[0] for e in errors]
    l2s = [e[1] for e in errors]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.loglog(h, linfs, "bo-", lw=2, ms=8, label=r"$L_\infty$ error")
    ax.loglog(h, l2s, "rs-", lw=2, ms=8, label=r"$L_2$ error")

    # Reference second-order slope
    h_ref = np.array([h[0], h[-1]])
    scale = linfs[0] / h_ref[0] ** 2
    ax.loglog(
        h_ref,
        scale * h_ref**2,
        "k--",
        lw=1.5,
        alpha=0.5,
        label=r"$\mathcal{O}(h^2)$ reference",
    )

    ax.set_xlabel("h = 1/N")
    ax.set_ylabel("Error")
    ax.set_title(r"Convergence: $A_\theta = r\sin(\pi r)\sin(\pi z)$ in RZ")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


# ---------- Main ----------


def test_nodal_vector_poisson_rz():
    """A_theta in RZ converges at second order for the manufactured solution."""
    resolutions = [16, 32, 64]
    errors = [run_test(n)[3:5] for n in resolutions]

    order_linf = np.log(errors[-2][0] / errors[-1][0]) / np.log(2)
    order_l2 = np.log(errors[-2][1] / errors[-1][1]) / np.log(2)
    assert order_linf > 1.8, f"L_inf convergence order {order_linf:.2f} <= 1.8"
    assert order_l2 > 1.8, f"L2 convergence order {order_l2:.2f} <= 1.8"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate solution, line-cut, and convergence plots",
    )
    args = parser.parse_args()

    print("=== RZ A_theta convergence test ===")
    print(r"Manufactured solution: A_theta = r sin(pi r) sin(pi z)")
    print(r"Operator: Lap_cyl(A) - A/r^2 = -mu0 J")
    print()

    resolutions = [16, 32, 64, 128, 256]
    errors = []
    last_geom, last_A, last_exact = None, None, None

    for n in resolutions:
        verbose = 2 if n == resolutions[0] else 0
        geom, A, A_exact, linf, l2 = run_test(n, verbose=verbose)
        errors.append((linf, l2))
        last_geom, last_A, last_exact = geom, A, A_exact

    print()
    print(
        f"{'N':>6s}  {'L_inf':>12s}  {'L2':>12s}  {'L_inf order':>12s}  {'L2 order':>10s}"
    )
    print("-" * 60)
    for i, (n, (linf, l2)) in enumerate(zip(resolutions, errors)):
        if i == 0:
            print(f"{n:6d}  {linf:12.6e}  {l2:12.6e}  {'---':>12s}  {'---':>10s}")
        else:
            r = n / resolutions[i - 1]
            oi = np.log(errors[i - 1][0] / linf) / np.log(r)
            o2 = np.log(errors[i - 1][1] / l2) / np.log(r)
            print(f"{n:6d}  {linf:12.6e}  {l2:12.6e}  {oi:12.4f}  {o2:10.4f}")

    # Check convergence order
    final_order_linf = np.log(errors[-2][0] / errors[-1][0]) / np.log(2)
    final_order_l2 = np.log(errors[-2][1] / errors[-1][1]) / np.log(2)
    print()
    print(f"Final convergence order (L_inf): {final_order_linf:.2f}")
    print(f"Final convergence order (L2):    {final_order_l2:.2f}")

    if final_order_linf > 1.8:
        print("\n✓ PASS: Second-order convergence achieved for A_theta in RZ")
    else:
        print("\n✗ FAIL: Expected second-order convergence")

    if args.plot:
        plot_solution(last_A, last_exact, last_geom)
        plot_line_cuts(last_A, last_geom)
        plot_convergence(resolutions, errors)


if __name__ == "__main__":
    amr.initialize([])
    try:
        main()
    finally:
        amr.finalize()
