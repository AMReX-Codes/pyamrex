# -*- coding: utf-8 -*-

import importlib.util
import shutil

import numpy as np
import pytest

import amrex.space3d as amr

has_openpmd = importlib.util.find_spec("openpmd_api") is not None
if has_openpmd:
    import openpmd_api as io

    from amrex.tools.pltfile_to_openpmd import convert

pytestmark = [
    pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3"),
    pytest.mark.skipif(not has_openpmd, reason="Requires openpmd_api"),
]


def write_single_level_plotfile(filename):
    """32^3 cells in 16^3 boxes, one linear-ramp component."""
    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])

    ba = amr.BoxArray(domain_box)
    ba.max_size(16)
    dm = amr.DistributionMapping(ba, 1)
    mf = amr.MultiFab(ba, dm, 1, 0)
    # a position-dependent value, so chunk placement errors are caught
    for mfi in mf:
        bx = mfi.tilebox()
        marr = mf.array(mfi).to_xp()
        i_s, j_s, k_s = tuple(bx.small_end)
        nx, ny, nz, _ = marr.shape
        i = np.arange(i_s, i_s + nx)[:, None, None]
        j = np.arange(j_s, j_s + ny)[None, :, None]
        k = np.arange(k_s, k_s + nz)[None, None, :]
        marr[..., 0] = i + 100 * j + 10000 * k

    amr.write_single_level_plotfile(
        filename, mf, amr.Vector_string(["ramp"]), geom, 1.5, 300
    )
    return mf


def test_convert_single_level_mesh(tmp_path):
    plt_name = str(tmp_path / "plt00300")
    write_single_level_plotfile(plt_name)

    out = str(tmp_path / "series_%T.h5")
    convert([plt_name], out, verbose=False)

    series = io.Series(out, io.Access.read_only)
    it = series.iterations[300]
    assert np.isclose(it.time, 1.5)

    mesh = it.meshes["ramp"]
    assert mesh.geometry == io.Geometry.cartesian
    assert mesh.axis_labels == ["z", "y", "x"]
    np.testing.assert_allclose(mesh.grid_spacing, [1.0 / 32.0] * 3)
    np.testing.assert_allclose(mesh.grid_global_offset, [-0.5] * 3)

    mrc = mesh[io.Mesh_Record_Component.SCALAR]
    assert list(mrc.shape) == [32, 32, 32]
    data = mrc.load_chunk()
    series.flush()

    # data is (z, y, x); rebuild the expected ramp
    k, j, i = np.meshgrid(np.arange(32), np.arange(32), np.arange(32), indexing="ij")
    np.testing.assert_array_equal(data, i + 100 * j + 10000 * k)

    # lossless-reconstruction metadata (1-element arrays may read as scalars)
    assert list(np.atleast_1d(it.get_attribute("amrex_level_steps"))) == [300]
    assert it.get_attribute("amrex_coord_sys") == 0
    ba_flat = it.get_attribute("amrex_box_array_lvl0")
    assert len(ba_flat) == 8 * 6  # 8 boxes, small_end+big_end each

    series.close()


def test_convert_multi_level_mesh(tmp_path):
    """Two-level hierarchy: level 1 refines the upper-x half of the domain."""
    plt_name = str(tmp_path / "plt00007")

    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    domain0 = amr.Box([0, 0, 0], [15, 15, 15])
    geom0 = amr.Geometry(domain0, real_box, amr.CoordSys.cartesian, [0, 0, 0])
    domain1 = amr.Box([0, 0, 0], [31, 31, 31])
    geom1 = amr.Geometry(domain1, real_box, amr.CoordSys.cartesian, [0, 0, 0])

    ba0 = amr.BoxArray(domain0)
    ba0.max_size(8)
    ba1 = amr.BoxArray(amr.Box([16, 0, 0], [31, 31, 31]))  # refined upper-x half
    ba1.max_size(16)

    mfs = []
    for ba, val in ((ba0, 1.0), (ba1, 2.0)):
        dm = amr.DistributionMapping(ba, 1)
        mf = amr.MultiFab(ba, dm, 1, 0)
        mf.set_val(val)
        mfs.append(mf)

    amr.write_multi_level_plotfile(
        plt_name, mfs, ["density"], [geom0, geom1], 0.25, [7, 7], [amr.IntVect(2)]
    )

    out = str(tmp_path / "ml_%T.h5")
    convert([plt_name], out, verbose=False)

    series = io.Series(out, io.Access.read_only)
    it = series.iterations[7]

    # level 0: plain name, full domain
    m0 = it.meshes["density"][io.Mesh_Record_Component.SCALAR]
    assert list(m0.shape) == [16, 16, 16]
    d0 = m0.load_chunk()
    series.flush()
    np.testing.assert_array_equal(d0, 1.0)

    # level 1: _lvl1 suffix, refinementRatio, sized to the refined index space
    m1 = it.meshes["density_lvl1"]
    assert m1.get_attribute("refinementRatio") == [2, 2, 2]
    mrc1 = m1[io.Mesh_Record_Component.SCALAR]
    assert list(mrc1.shape) == [32, 32, 32]
    np.testing.assert_allclose(m1.grid_spacing, [1.0 / 32.0] * 3)  # refined spacing
    # only the covered half is defined: read it back chunk-wise
    d1 = mrc1.load_chunk([0, 0, 16], [32, 32, 16])
    series.flush()
    np.testing.assert_array_equal(d1, 2.0)

    series.close()


def test_convert_particles(tmp_path):
    plt_name = str(tmp_path / "plt00042")
    n_part = 21

    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])
    ba = amr.BoxArray(domain_box)
    dm = amr.DistributionMapping(ba, 1)

    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(0.0)
    amr.write_single_level_plotfile(
        plt_name, mf, amr.Vector_string(["dummy"]), geom, 0.0, 42
    )

    pc = amr.ParticleContainer_pureSoA_3_0_polymorphic(geom, dm, ba)
    pc.arena = amr.The_Arena()
    myt = amr.ParticleInitType_pureSoA_3_0()
    myt.real_array_data = [0.0, 0.0, 0.0]
    myt.int_array_data = []
    pc.init_random(n_part, 7, myt, False, real_box)
    pc.add_real_comp("w", True)
    pc.add_int_comp("tag", True)
    for pti in pc.iterator(level=0):
        soa = pti.soa()
        soa.get_real_data(3).assign(3.25)
        soa.get_int_data(0).assign(9)
    pc.redistribute()
    pc.write_plotfile(
        plt_name, "electrons", amr.Vector_string(["w"]), amr.Vector_string(["tag"])
    )

    out = str(tmp_path / "parts_%T.h5")
    convert([plt_name], out, verbose=False)

    series = io.Series(out, io.Access.read_only)
    sp = series.iterations[42].particles["electrons"]

    x = sp["position"]["x"].load_chunk()
    w = sp["w"][io.Record_Component.SCALAR].load_chunk()
    tag = sp["tag"][io.Record_Component.SCALAR].load_chunk()
    pid = sp["id"][io.Record_Component.SCALAR].load_chunk()
    series.flush()

    assert x.size == n_part
    assert np.all((x >= -0.5) & (x <= 0.5))
    np.testing.assert_allclose(w, 3.25)
    np.testing.assert_array_equal(tag, 9)
    assert np.unique(pid).size == n_part  # ids are unique particle identities

    assert list(np.atleast_1d(sp.get_attribute("amrex_num_particles_per_level"))) == [
        n_part
    ]

    series.close()


def test_convert_field_selection_errors(tmp_path):
    plt_name = str(tmp_path / "plt00300")
    write_single_level_plotfile(plt_name)

    with pytest.raises(ValueError, match="unknown field"):
        convert([plt_name], str(tmp_path / "e_%T.h5"), fields=["nope"], verbose=False)

    with pytest.raises(FileNotFoundError, match="plotfile Header"):
        convert([str(tmp_path / "not_a_plotfile")], str(tmp_path / "f_%T.h5"))

    shutil.rmtree(plt_name)
