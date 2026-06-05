# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def _make_core(record=None):
    """An AmrCore subclass that records calls to its overridden virtuals.

    Built as a single-level (``max_level = 0``), non-periodic, Cartesian core.
    """
    if record is None:
        record = {}

    class RecordingCore(amr.AmrCore):
        def make_new_level_from_scratch(self, lev, time, ba, dm):
            record.setdefault("scratch", []).append((lev, ba.size))

        def make_new_level_from_coarse(self, lev, time, ba, dm):
            record.setdefault("coarse", []).append(lev)

        def remake_level(self, lev, time, ba, dm):
            record.setdefault("remake", []).append(lev)

        def clear_level(self, lev):
            record.setdefault("clear", []).append(lev)

        def error_est(self, lev, tags, time, ngrow):
            record.setdefault("error_est", []).append(lev)

    rb = amr.RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    n_cell = amr.Vector_int([16, 16, 16])
    ref_ratios = amr.Vector_IntVect([])  # max_level=0 -> no refinement ratios
    core = RecordingCore(rb, 0, n_cell, 0, ref_ratios, [0, 0, 0])
    return core, record


# ---------------------------------------------------------------------------
# AmrCore binding & trampoline
# ---------------------------------------------------------------------------
def test_amrcore_classes_are_bound():
    assert hasattr(amr, "AmrCore")
    assert hasattr(amr, "ParGDBBase")
    assert hasattr(amr, "AmrParGDB")
    # AmrCore derives from AmrMesh in C++ and in the bindings
    assert issubclass(amr.AmrCore, amr.AmrMesh)
    assert issubclass(amr.AmrParGDB, amr.ParGDBBase)


def test_amrcore_construct():
    core, _ = _make_core()
    assert core.max_level == 0
    # finest_level is -1 until init_from_scratch defines level 0
    assert core.finest_level == -1


def test_amrcore_trampoline_make_new_level_from_scratch():
    core, record = _make_core()
    core.init_from_scratch(0.0)

    # the Python override must have been called exactly once, for level 0
    assert "scratch" in record
    assert len(record["scratch"]) == 1
    lev, n_boxes = record["scratch"][0]
    assert lev == 0
    assert n_boxes >= 1
    assert core.finest_level == 0


def test_amrcore_trampoline_clear_level():
    core, record = _make_core()
    core.init_from_scratch(0.0)
    core.clear_level(0)
    assert record.get("clear") == [0]


def test_amrcore_missing_pure_virtual_raises():
    # A subclass that does NOT override make_new_level_from_scratch must raise
    # when AMReX tries to call that pure virtual during init_from_scratch.
    class IncompleteCore(amr.AmrCore):
        def make_new_level_from_coarse(self, *args):
            pass

        def remake_level(self, *args):
            pass

        def clear_level(self, lev):
            pass

        def error_est(self, *args):
            pass

    rb = amr.RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    core = IncompleteCore(
        rb, 0, amr.Vector_int([16, 16, 16]), 0, amr.Vector_IntVect([]), [0, 0, 0]
    )
    with pytest.raises(RuntimeError):
        core.init_from_scratch(0.0)


# ---------------------------------------------------------------------------
# AmrMesh geom / set_geometry accessors (added alongside AmrCore)
# ---------------------------------------------------------------------------
def test_amrmesh_geom_accessor():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    geom = core.geom(0)
    assert np.allclose(geom.ProbLo(), [0.0, 0.0, 0.0])
    assert np.allclose(geom.ProbHi(), [1.0, 1.0, 1.0])


def test_amrmesh_set_geometry():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    new_geom = amr.Geometry(
        core.geom(0).domain,
        amr.RealBox(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        0,
        [0, 0, 0],
    )
    core.set_geometry(0, new_geom)
    assert np.allclose(core.geom(0).ProbHi(), [2.0, 2.0, 2.0])


# ---------------------------------------------------------------------------
# ParGDB / AmrParGDB
# ---------------------------------------------------------------------------
def test_get_par_gdb():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    gdb = core.get_par_gdb()
    assert isinstance(gdb, amr.AmrParGDB)
    assert isinstance(gdb, amr.ParGDBBase)
    assert gdb.max_level() == 0
    assert gdb.finest_level() == 0
    assert gdb.level_defined(0)
    # geometry queried through the GDB matches the AmrCore's level-0 geometry
    assert np.allclose(gdb.geom(0).ProbHi(), [1.0, 1.0, 1.0])


def test_amrpargdb_construct_from_core():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    gdb = amr.AmrParGDB(core)
    assert gdb.max_level() == 0
    assert gdb.finest_level() == 0


def test_get_par_gdb_keeps_core_alive(assert_keeps_python_alive):
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    gdb = assert_keeps_python_alive(core, lambda: core.get_par_gdb())
    assert gdb.finest_level() == 0


# ---------------------------------------------------------------------------
# ParticleContainer construction from a ParGDB + in-place tile definition
# ---------------------------------------------------------------------------
def test_particle_container_from_gdb():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    gdb = core.get_par_gdb()

    pc = amr.ParticleContainer_2_1_3_1_default(gdb)
    assert pc.finest_level == 0
    assert pc.total_number_of_particles() == 0


def test_define_and_return_particle_tile():
    core, _ = _make_core()
    core.init_from_scratch(0.0)
    gdb = core.get_par_gdb()
    pc = amr.ParticleContainer_2_1_3_1_default(gdb)

    tile = pc.define_and_return_particle_tile(0, 0, 0)
    assert tile.num_particles == 0
    tile.resize(5)
    assert tile.size == 5


@pytest.mark.skipif(
    not hasattr(amr, "ParticleContainer_pureSoA_11_0_polymorphic"),
    reason="ImpactX pure-SoA particle container not built",
)
def test_pure_soa_container_from_gdb_add_particles():
    """End-to-end: AmrCore -> ParGDB -> pure-SoA PC -> in-place particle add.

    This mirrors the ImpactX construction path
    ``ImpactXParticleContainer(amr_core->GetParGDB())`` followed by
    ``AddNParticles``.
    """
    core, _ = _make_core()
    core.init_from_scratch(0.0)

    pc = amr.ParticleContainer_pureSoA_11_0_polymorphic(core.get_par_gdb())
    pc.arena = amr.The_Arena()  # PolymorphicArenaAllocator needs an arena
    names = [f"r{i}" for i in range(11)]
    pc.set_soa_compile_time_names(names, [])

    npart = 100
    tile = pc.define_and_return_particle_tile(0, 0, 0)
    tile.resize(npart)
    soa = tile.get_struct_of_arrays()

    # write the first real component and the particle ids
    x = np.arange(npart, dtype=np.float64)
    np.array(soa.get_real_data(0), copy=False)[:] = x

    idcpu = np.array(soa.get_idcpu_data(), copy=False)
    amr.pack_ids(idcpu, np.arange(1, npart + 1, dtype=np.int64))
    amr.pack_cpus(idcpu, np.zeros(npart, dtype=np.int32))

    assert pc.number_of_particles_at_level(0) == npart
    assert pc.total_number_of_particles() == npart

    # read back via the iterator and confirm the written data round-trips
    read = []
    for pti in pc.iterator(level=0):
        read.append(np.array(pti.soa().get_real_data(0), copy=False).copy())
    read = np.concatenate(read)
    assert read.size == npart
    assert np.allclose(np.sort(read), x)
