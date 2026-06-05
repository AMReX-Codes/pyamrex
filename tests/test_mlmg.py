# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr

SD = amr.Config.spacedim


def make_geom(box, periodic=False):
    real_box = amr.RealBox([0.0] * SD, [1.0] * SD)
    return amr.Geometry(box, real_box, amr.CoordSys.cartesian, [int(periodic)] * SD)


def sin_product(bx, geom, nodal=False):
    """u = prod_d sin(pi x_d) at the cell centers (or nodes) of bx"""
    dx = geom.data().CellSize()
    u = 1.0
    for d in range(SD):
        offset = 0.0 if nodal else 0.5
        x = (np.arange(bx.small_end[d], bx.big_end[d] + 1) + offset) * dx[d]
        u = u * np.sin(np.pi * x).reshape([-1 if i == d else 1 for i in range(4)])
    return u


def fill_mf(mf, geom, factor, nodal=False):
    """fill a (no-ghost) MultiFab with factor * prod_d sin(pi x_d)"""
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        arr[...] = factor * sin_product(mfi.validbox(), geom, nodal)


def max_error(mf, geom, factor, nodal=False):
    err = 0.0
    ng = mf.n_grow_vect
    valid = tuple(
        slice(ng[d], -ng[d]) if d < SD and ng[d] > 0 else slice(None) for d in range(3)
    ) + (slice(None),)
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        u = factor * sin_product(mfi.validbox(), geom, nodal)
        err = max(err, float(np.abs(arr[valid] - u).max()))
    return err


def dirichlet_bc():
    lo = [amr.LinOpBCType.Dirichlet] * SD
    hi = [amr.LinOpBCType.Dirichlet] * SD
    return lo, hi


def test_mlmg_poisson():
    """solve del^2 u = rhs with homogeneous Dirichlet boundaries and
    compare against the analytic solution"""
    n_cell = 64
    box = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    sol = amr.MultiFab(ba, dm, 1, 1)
    rhs = amr.MultiFab(ba, dm, 1, 0)
    sol.set_val(0.0)
    # u = prod sin(pi x_d) -> del^2 u = -SD pi^2 u
    fill_mf(rhs, geom, -SD * np.pi**2)

    info = amr.LPInfo()
    linop = amr.MLPoisson([geom], [ba], [dm], info)
    linop.set_max_order(3)
    linop.set_domain_bc(*dirichlet_bc())
    linop.set_level_bc(0, None)  # homogeneous

    mlmg = amr.MLMG(linop)
    mlmg.set_max_iter(100)
    mlmg.set_verbose(0)
    resid = mlmg.solve([sol], [rhs], 1.0e-10, 0.0)
    assert resid < 1.0e-8

    # second-order accuracy
    err = max_error(sol, geom, 1.0)
    assert err < 3.0 * (np.pi / n_cell) ** 2

    # fluxes (-grad u) on faces are finite and of the expected magnitude
    fluxes = []
    for d in range(SD):
        fba = amr.BoxArray(ba)
        fba.surroundingNodes(d)
        fluxes.append(amr.MultiFab(fba, dm, 1, 0))
    mlmg.get_fluxes([fluxes])
    fmax = max(f.norm0(0, 0, False, True) for f in fluxes)
    assert np.isclose(fmax, np.pi, rtol=0.05)


def test_mlmg_poisson_composite():
    """a two-level composite Poisson solve"""
    n_cell = 32
    boxes = [
        amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1)),
        amr.Box(amr.IntVect(n_cell // 2), amr.IntVect(3 * n_cell // 2 - 1)),
    ]
    domains = [
        amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1)),
        amr.Box(amr.IntVect(0), amr.IntVect(2 * n_cell - 1)),
    ]
    geoms = [make_geom(d) for d in domains]
    bas, dms, sols, rhss = [], [], [], []
    for lev in range(2):
        ba = amr.BoxArray(boxes[lev])
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)
        sol = amr.MultiFab(ba, dm, 1, 1)
        rhs = amr.MultiFab(ba, dm, 1, 0)
        sol.set_val(0.0)
        fill_mf(rhs, geoms[lev], -SD * np.pi**2)
        bas.append(ba)
        dms.append(dm)
        sols.append(sol)
        rhss.append(rhs)

    linop = amr.MLPoisson(geoms, bas, dms, amr.LPInfo())
    linop.set_max_order(3)
    linop.set_domain_bc(*dirichlet_bc())
    for lev in range(2):
        linop.set_level_bc(lev, None)

    mlmg = amr.MLMG(linop)
    mlmg.set_max_iter(100)
    resid = mlmg.solve(sols, rhss, 1.0e-10, 0.0)
    assert resid < 1.0e-8

    for lev in range(2):
        err = max_error(sols[lev], geoms[lev], 1.0)
        assert err < 5.0 * (np.pi / n_cell) ** 2


def test_mlmg_abeclaplacian():
    """solve (a alpha - b del . beta grad) u = rhs with constant
    coefficients alpha = beta = 1 and compare against the analytic
    solution"""
    n_cell = 64
    box = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    ascalar = 1.0
    bscalar = 2.0

    sol = amr.MultiFab(ba, dm, 1, 1)
    rhs = amr.MultiFab(ba, dm, 1, 0)
    sol.set_val(0.0)
    # u = prod sin(pi x_d):
    # (a - b del^2) u = (1 + 2 SD pi^2) u
    fill_mf(rhs, geom, 1.0 + bscalar * SD * np.pi**2)

    acoef = amr.MultiFab(ba, dm, 1, 0)
    acoef.set_val(1.0)
    bcoef = []
    for d in range(SD):
        fba = amr.BoxArray(ba)
        fba.surroundingNodes(d)
        bcoef.append(amr.MultiFab(fba, dm, 1, 0))
        bcoef[d].set_val(1.0)

    linop = amr.MLABecLaplacian([geom], [ba], [dm], amr.LPInfo())
    linop.set_max_order(3)
    linop.set_domain_bc(*dirichlet_bc())
    linop.set_level_bc(0, None)
    linop.set_scalars(ascalar, bscalar)
    linop.set_a_coeffs(0, acoef)
    linop.set_b_coeffs(0, bcoef)

    mlmg = amr.MLMG(linop)
    mlmg.set_max_iter(100)
    resid = mlmg.solve([sol], [rhs], 1.0e-10, 0.0)
    assert resid < 1.0e-8

    err = max_error(sol, geom, 1.0)
    assert err < 10.0 * (np.pi / n_cell) ** 2

    # the constant-coefficient overloads give the same operator
    sol2 = amr.MultiFab(ba, dm, 1, 1)
    sol2.set_val(0.0)
    linop2 = amr.MLABecLaplacian([geom], [ba], [dm], amr.LPInfo())
    linop2.set_max_order(3)
    linop2.set_domain_bc(*dirichlet_bc())
    linop2.set_level_bc(0, None)
    linop2.set_scalars(ascalar, bscalar)
    linop2.set_a_coeffs(0, 1.0)
    linop2.set_b_coeffs(0, 1.0)
    mlmg2 = amr.MLMG(linop2)
    mlmg2.solve([sol2], [rhs], 1.0e-10, 0.0)

    diff = 0.0
    for mfi in sol:
        a = sol.array(mfi).to_xp(copy=False, order="F")
        b = sol2.array(mfi).to_xp(copy=False, order="F")
        diff = max(diff, float(np.abs(a - b).max()))
    assert diff < 1.0e-8


def test_mlmg_nodal_poisson():
    """solve the nodal Poisson equation del . (sigma grad) u = rhs"""
    n_cell = 64
    box = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    ba_nd = amr.BoxArray(ba)
    ba_nd.surroundingNodes()

    sol = amr.MultiFab(ba_nd, dm, 1, 1)
    rhs = amr.MultiFab(ba_nd, dm, 1, 0)
    sol.set_val(0.0)
    # u = prod sin(pi x_d) at the nodes -> del^2 u = -SD pi^2 u
    fill_mf(rhs, geom, -SD * np.pi**2, nodal=True)

    sigma = amr.MultiFab(ba, dm, 1, 0)
    sigma.set_val(1.0)

    linop = amr.MLNodeLaplacian([geom], [ba], [dm], amr.LPInfo())
    linop.set_domain_bc(*dirichlet_bc())
    linop.set_sigma(0, sigma)

    mlmg = amr.MLMG(linop)
    mlmg.set_max_iter(100)
    resid = mlmg.solve([sol], [rhs], 1.0e-10, 0.0)
    assert resid < 1.0e-8

    err = max_error(sol, geom, 1.0, nodal=True)
    assert err < 5.0 * (np.pi / n_cell) ** 2


def test_mlmg_comp_residual():
    """the residual of the exact discrete solution is (near) zero"""
    n_cell = 32
    box = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    dm = amr.DistributionMapping(ba)

    sol = amr.MultiFab(ba, dm, 1, 1)
    rhs = amr.MultiFab(ba, dm, 1, 0)
    res = amr.MultiFab(ba, dm, 1, 0)
    sol.set_val(0.0)
    fill_mf(rhs, geom, -SD * np.pi**2)

    linop = amr.MLPoisson([geom], [ba], [dm], amr.LPInfo())
    linop.set_domain_bc(*dirichlet_bc())
    linop.set_level_bc(0, None)

    mlmg = amr.MLMG(linop)
    mlmg.solve([sol], [rhs], 1.0e-11, 0.0)
    mlmg.comp_residual([res], [sol], [rhs])
    assert res.norm0(0, 0, False, True) < 1.0e-8 * rhs.norm0(0, 0, False, True)
