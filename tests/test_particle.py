# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def test_particle_init():
    p1 = amr.Particle_8_0()
    print(p1)

    p2 = amr.Particle_8_0(1.0, 2.0, 3.0)
    assert p2.x == 1.0 and p2.y == 2.0 and p2.z == 3.0

    p3 = amr.Particle_8_0(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)
    assert p3.x == 1.0 and p3.get_rdata(0) == 4.0 and p3.get_rdata(6) == 10.0

    p4 = amr.Particle_8_0(1.0, 2.0, 3.0, rdata_0=4.0)
    assert (
        p4.x == 1.0
        and p4.z == 3.0
        and p4.get_rdata(0) == 4.0
        and p4.get_rdata(1) == 0 == p4.get_rdata(2) == p4.get_rdata(3)
    )

    p5 = amr.Particle_8_0(x=1.0, rdata_1=1.0, rdata_3=3.0)
    assert (
        p5.x == 1.0
        and p5.get_rdata(1) == 1.0
        and p5.get_rdata(3) == 3.0
        and p5.get_rdata(4) == 0
    )


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_particle_set():
    p1 = amr.Particle_8_0()
    p1.setPos(1, 1.5)
    assert (
        np.isclose(p1.pos(0), 0)
        and np.isclose(p1.pos(1), 1.5)
        and np.isclose(p1.pos(2), 0)
    )
    p1.setPos([1.0, 1, 2])
    assert (
        np.isclose(p1.pos(0), 1)
        and np.isclose(p1.pos(1), 1)
        and np.isclose(p1.pos(2), 2)
    )
    p1.setPos(amr.RealVect(2, 3.3, 4.2))
    assert (
        np.isclose(p1.pos(0), 2)
        and np.isclose(p1.pos(1), 3.3)
        and np.isclose(p1.pos(2), 4.2)
    )

    print(p1.x, p1.y, p1.z)
    p1.x = 2.1
    assert np.isclose(p1.x, 2.1)
    p1.y = 3.2
    assert np.isclose(p1.y, 3.2)
    p1.z = 5.1
    assert np.isclose(p1.z, 5.1)


def test_rdata():
    p1 = amr.Particle_2_1()
    rvec = [1.5, 2.0]
    p1.set_rdata(rvec)
    assert np.allclose(p1.get_rdata(), rvec)
    p1.set_rdata(1, 2.5)
    assert np.allclose(p1.get_rdata(1), 2.5)

    with pytest.raises(ValueError):
        p1.set_rdata(100, 5.2)

    with pytest.raises(ValueError):
        p1.get_rdata(100)


def test_idata():
    p1 = amr.Particle_2_1()
    ivec = [-1]
    p1.set_idata(ivec)
    assert p1.get_idata() == ivec
    p1.set_idata(0, 3)
    assert p1.get_idata(0) == 3
    with pytest.raises(ValueError):
        p1.set_idata(100, 5)

    with pytest.raises(ValueError):
        p1.get_idata(100)
