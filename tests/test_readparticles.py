# -*- coding: utf-8 -*-

import shutil

import numpy as np
import pytest

import amrex.space3d as amr


def write_test_plotfile(filename, n_part):
    """Write a single-level plotfile with a mesh and a pure-SoA particle
    container, so it can be read back via ``amr.read_particles``.

    Returns the names of the runtime real/int components that were added.
    """
    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])

    ba = amr.BoxArray(domain_box)
    dm = amr.DistributionMapping(ba, 1)

    # mesh data, so PlotFileData can recover the geometry on read
    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(np.pi)
    amr.write_single_level_plotfile(
        filename, mf, amr.Vector_string(["density"]), geom, 1.0, 200
    )

    # pure-SoA particle container with runtime components
    pc = amr.ParticleContainer_pureSoA_3_0_polymorphic(geom, dm, ba)
    pc.arena = amr.The_Arena()
    myt = amr.ParticleInitType_pureSoA_3_0()
    myt.real_array_data = [0.0, 0.0, 0.0]  # x, y, z (overwritten by init_random)
    myt.int_array_data = []
    pc.init_random(n_part, 1, myt, False, real_box)

    # add 1 runtime real + 2 runtime int components and fill them
    pc.add_real_comp("w", True)
    pc.add_int_comp("i1", True)
    pc.add_int_comp("i2", True)
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            soa.get_real_data(3).assign(1.2345)  # "w": after x,y,z (idx 0,1,2)
            soa.get_int_data(0).assign(42)  # "i1"
            soa.get_int_data(1).assign(33)  # "i2"
    pc.redistribute()

    real_names = ["w"]
    int_names = ["i1", "i2"]
    pc.write_plotfile(
        filename,
        "particles",
        amr.Vector_string(real_names),
        amr.Vector_string(int_names),
    )
    return real_names, int_names


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_header():
    """ParticleHeader discovers the on-disk layout without a matching container."""
    plt_file_name = "plt_read_hdr"
    n_part = 15
    real_names, int_names = write_test_plotfile(plt_file_name, n_part)

    header = amr.ParticleHeader.read(plt_file_name, "particles")
    assert header.dim == 3
    # pure SoA: the AMREX_SPACEDIM positions are implicit, so only the runtime
    # real component "w" is counted here
    assert header.num_real == len(real_names)
    assert list(header.real_comp_names) == real_names
    assert header.num_int == len(int_names)
    assert list(header.int_comp_names) == int_names
    assert header.num_particles == n_part

    shutil.rmtree(plt_file_name)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_roundtrip():
    """amr.read_particles reconstructs a container from a plotfile with no prior
    knowledge of its compile-time layout."""
    plt_file_name = "plt_read_rt"
    n_part = 15
    real_names, int_names = write_test_plotfile(plt_file_name, n_part)

    pc = amr.read_particles(plt_file_name, "particles")

    # the runtime layout was reconstructed from the file
    for name in real_names:
        assert pc.has_real_comp(name)
    for name in int_names:
        assert pc.has_int_comp(name)
    assert pc.total_number_of_particles() == n_part

    # and the runtime component values round-tripped
    w_idx = pc.get_real_comp_index("w")
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            np.testing.assert_allclose(
                soa.get_real_data(w_idx).to_numpy(copy=False), 1.2345
            )

    shutil.rmtree(plt_file_name)
