#!/usr/bin/env python3
"""
Time-varying coil flux EB solver: VectorPoissonSolverNodal in 2D RZ.

Loads coil geometry from coils_data.pkl and flux signals from
signals_vacuum.pkl, solves at multiple time snapshots, and produces
a time-sequence plot of A_theta.
"""
import pickle
import numpy as np
from scipy.signal import savgol_filter

try:
    import cupy as xp

    def to_numpy(a):
        if isinstance(a, np.ndarray):
            return a
        return a.get()
except ImportError:
    import numpy as xp

    def to_numpy(a):
        return np.asarray(a)

import amrex.space2d as amr


def load_coils(pkl_file="coils_params/coils_data.pkl", override_dict=None):
    """Load coil geometry from pickle file, filtering to only used coils."""
    with open(pkl_file, "rb") as f:
        coils_data = pickle.load(f)

    if override_dict is None:
        override_dict = {}

    nc = len(coils_data['coil'])
    coils_list = []
    for ic in range(nc):
        name = str(coils_data['coil'][ic])
        if "ED" in name and name not in override_dict:
            continue

        coils_list.append({
            'name': name,
            'r1c': float(coils_data['r1c'][ic]),
            'r2c': float(coils_data['r2c'][ic]),
            'drc': float(coils_data['drc'][ic]),
            'z_lo': float(coils_data['zc'][ic]),
            'z_hi': float(coils_data['zc'][ic] + coils_data['dzc'][ic]),
        })

    return coils_list


def load_signals(signals_file="coils_params/signals_vacuum.pkl"):
    """Load flux signals and time vector."""
    with open(signals_file, "rb") as f:
        signals = pickle.load(f)

    time_vector = np.array(signals['t'], dtype=np.float64)
    flux_signals = {}
    for key, val in signals.items():
        if key != 't':
            flux_signals[key] = np.array(val, dtype=np.float64)

    return time_vector, flux_signals


def smooth_signal(signal, window_len=128):
    """Smooth a signal using Savitzky-Golay filter."""
    if len(signal) < window_len:
        window_len = max(5, len(signal) // 2 * 2 - 1)
    return savgol_filter(signal, window_length=window_len, polyorder=2)


def build_parser_function(coils_list):
    """Build AMReX parser implicit function string for union of trapezoidal coils."""
    box_exprs = []
    for c in coils_list:
        z0, z1 = c['z_lo'], c['z_hi']
        dz = z1 - z0
        r1c, r2c, drc = c['r1c'], c['r2c'], c['drc']
        slope = (r2c - r1c) / dz if dz > 0 else 0.0
        r_inner = f"({r1c:.10f}+{slope:.10f}*(y-{z0:.10f}))"
        expr = (f"min(min(x-{r_inner}, {r_inner}+{drc:.10f}-x), "
                f"min(y-{z0:.10f}, {z1:.10f}-y))")
        box_exprs.append(expr)

    result = box_exprs[0]
    for expr in box_exprs[1:]:
        result = f"max({result}, {expr})"
    return result


def compute_coil_psi(coils_list, time_vector, flux_signals,
                     t, override_dict, smoothed_cache):
    """Compute psi (flux / 2pi) for all coils at time t."""
    values = []
    for c in coils_list:
        name = c['name']
        psi = 0.0

        if "ED" not in name:
            sig_key = 'F_' + name
            if sig_key in flux_signals:
                if sig_key not in smoothed_cache:
                    smoothed_cache[sig_key] = smooth_signal(flux_signals[sig_key])
                sig = smoothed_cache[sig_key]
                psi = (1.0 / (2 * np.pi)) * np.interp(t, time_vector, sig)

        if override_dict is not None and name in override_dict:
            flux_entry = override_dict[name]['flux']
            flux_val = flux_entry(t) if callable(flux_entry) else flux_entry
            psi = (1.0 / (2 * np.pi)) * flux_val

        values.append(psi)

    return values


def extract_nodal_data(mf, geom):
    """Extract the full nodal array from a MultiFab."""
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
            data = marr[ng0:ng0 + nr, ng1:ng1 + nz, 0, 0]
        else:
            data = marr[0, 0, ng1:ng1 + nz, ng0:ng0 + nr].T
        full_arr[lo[0]:lo[0] + nr, lo[1]:lo[1] + nz] = data

    problo = geom.ProbLo()
    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]
    r_arr = problo[0] + xp.arange(nr_full, dtype=xp.float64) * dx[0]
    z_arr = problo[1] + xp.arange(nz_full, dtype=xp.float64) * dx[1]
    return full_arr, r_arr, z_arr


def setup_solver(coils_list, Nr, Nz, rmax, zmax):
    """Set up geometry, box array, EB factory, and solver."""
    dom = amr.Box(amr.IntVect(0, 0), amr.IntVect(Nr - 1, Nz - 1))
    geom = amr.Geometry(dom, amr.RealBox(0.0, 0.0, rmax, zmax), 1, [0, 0])

    ba = amr.BoxArray(dom)
    ba.max_size(max(Nr, Nz, 32))
    dm = amr.DistributionMapping(ba)

    amr.EB2_Build(geom, required_coarsening_level=0,
                  max_coarsening_level=0, ngrow=4)

    eb_factory = amr.makeEBFabFactory(
        geom, ba, dm, amr.Vector_int([1, 1, 1]), amr.EBSupport.full
    )

    ba_nd = amr.BoxArray(ba)
    ba_nd.convert(amr.IntVect(1, 1))

    A = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    J = [amr.MultiFab(ba_nd, dm, 1, 1) for _ in range(3)]
    for d in range(3):
        A[d].set_val(0.0)
        J[d].set_val(0.0)

    bc = amr.NodalBoundaryHandler(periodic_axial=False, axial_dirichlet=False)
    solver = amr.VectorPoissonSolverNodal(
        geom, ba, dm, bc,
        is_rz=True, eb_enabled=True, eb_factory=eb_factory,
    )

    dx = [geom.ProbLength(d) / geom.domain.length(d) for d in range(2)]

    return {
        'geom': geom, 'ba': ba, 'dm': dm,
        'eb_factory': eb_factory,
        'ba_nd': ba_nd, 'A': A, 'J': J,
        'bc': bc, 'solver': solver, 'dx': dx,
    }


def solve_at_time(ctx, coils_list, coil_psi):
    """Update coil psi values and solve."""
    solver = ctx['solver']
    A, J, geom = ctx['A'], ctx['J'], ctx['geom']

    coil_specs = []
    for ic, c in enumerate(coils_list):
        cs = amr.VectorPoissonSolverNodal.CoilSpec()
        cs.z_lo = c['z_lo']
        cs.z_hi = c['z_hi']
        cs.r1c = c['r1c']
        cs.r2c = c['r2c']
        cs.drc = c['drc']
        cs.psi = coil_psi[ic]
        coil_specs.append(cs)

    solver.setEBCoils(1, coil_specs)
    solver.solve(A, J, 1e-12, 0.0, 200, 2)

    A_data, r_arr, z_arr = extract_nodal_data(A[1], geom)
    iters = solver.getNumIters(1)
    resid = solver.getResidual(1)

    return A_data, r_arr, z_arr, iters, resid


def coil_contains_point(c, r, z):
    """Check if point (r, z) is inside a trapezoidal coil."""
    if z < c['z_lo'] or z > c['z_hi']:
        return False
    dz = c['z_hi'] - c['z_lo']
    tau = (z - c['z_lo']) / dz if dz > 0 else 0.0
    r_inner = c['r1c'] + (c['r2c'] - c['r1c']) * tau
    return r >= r_inner and r <= r_inner + c['drc']


def plot_time_sequence(results, coils_list,
                       plot_file="EB_coils_time_sequence.png"):
    """Plot A_theta at multiple time snapshots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        return

    n_times = len(results)

    vmax_global = 0.0
    for res in results:
        vmax_global = max(vmax_global, np.nanmax(np.abs(res['A_data'])))

    r_np = results[0]['r_arr']
    z_np = results[0]['z_arr']
    r_span = r_np[-1] - r_np[0]
    z_span = z_np[-1] - z_np[0]

    panel_width = 14
    panel_height = max(panel_width * (r_span / z_span), 1.2)

    fig, axes = plt.subplots(n_times, 1,
                             figsize=(panel_width + 1.5,
                                      n_times * (panel_height + 0.3) + 1.0),
                             squeeze=False)

    dr, dz = results[0]['dr'], results[0]['dz']
    margin = 0.5 * max(dr, dz)

    for i, res in enumerate(results):
        ax = axes[i, 0]
        rr, zz = np.meshgrid(res['r_arr'], res['z_arr'], indexing="ij")

        inside = np.zeros_like(rr, dtype=bool)
        for c in coils_list:
            for ir in range(rr.shape[0]):
                for iz in range(rr.shape[1]):
                    if coil_contains_point(c, rr[ir, iz], zz[ir, iz]):
                        inside[ir, iz] = True
        A_plot = np.where(inside, np.nan, res['A_data'])

        pcm = ax.pcolormesh(res['z_arr'], res['r_arr'], A_plot,
                            shading="auto", cmap="RdBu_r",
                            vmin=-vmax_global, vmax=vmax_global)

        for c in coils_list:
            verts = [
                (c['z_lo'], c['r1c']),
                (c['z_hi'], c['r2c']),
                (c['z_hi'], c['r2c'] + c['drc']),
                (c['z_lo'], c['r1c'] + c['drc']),
            ]
            ax.add_patch(Polygon(verts, closed=True,
                                 fill=False, edgecolor='k', linewidth=0.5))

        ax.set_ylabel("r (m)")
        ax.set_title(f"t = {res['t']*1e6:.1f} μs", fontsize=10)
        ax.set_aspect("equal")
        if i < n_times - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("z (m)")

    fig.subplots_adjust(right=0.88, hspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label=r"$A_\theta$")
    fig.suptitle(r"$A_\theta$ — Vacuum coil fluxes vs. time", fontsize=13, y=0.99)

    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import time as timer

    rmax = 0.6
    zmax = 5.8
    Nr = 256
    Nz = 512

    override_dict = {'ED06': {'flux': 1.0}}

    coils_list = load_coils("coils_params/coils_data.pkl",
                            override_dict=override_dict)
    time_vector, flux_signals = load_signals("coils_params/signals_vacuum.pkl")

    time_snapshots = np.arange(0.0, 40e-6, 10e-6)

    parser_func = build_parser_function(coils_list)
    smoothed_cache = {}

    amr.initialize([
        "",
        "eb2.geom_type=parser",
        'eb2.parser_function=' + '"' + parser_func + '"',
    ])

    try:
        ctx = setup_solver(coils_list, Nr, Nz, rmax, zmax)
        dr, dz = ctx['dx']

        results = []
        for t_snap in time_snapshots:
            t0 = timer.time()

            coil_psi = compute_coil_psi(
                coils_list, time_vector, flux_signals,
                t_snap, override_dict, smoothed_cache)

            A_data, r_arr, z_arr, iters, resid = solve_at_time(
                ctx, coils_list, coil_psi)

            elapsed = timer.time() - t0
            print(f"  t={t_snap*1e6:7.1f} μs  iters={iters}  "
                  f"resid={resid:.1e}  time={elapsed:.2f}s")

            results.append({
                't': t_snap,
                'A_data': to_numpy(A_data),
                'r_arr': to_numpy(r_arr),
                'z_arr': to_numpy(z_arr),
                'dr': dr, 'dz': dz,
                'iters': iters, 'resid': resid,
            })

        plot_time_sequence(results, coils_list)

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()