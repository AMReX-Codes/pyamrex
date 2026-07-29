# -*- coding: utf-8 -*-

import os
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

    # Manual: Read Particle Header START
    # inspect the on-disk particle component layout, without reading the data
    header = amr.ParticleHeader.read(plt_file_name, "particles")

    print(header.real_comp_names)  # e.g., ["w"]
    print(header.int_comp_names)  # e.g., ["i1", "i2"]
    print(header.num_particles)  # e.g., 15
    print(header.is_checkpoint)  # False for plotfiles, True for checkpoints
    # Manual: Read Particle Header END

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

    # Manual: Read Plotfile Particles START
    # read all particles from <plotfile>/particles/ into a new container;
    # the component names and layout are discovered from the file
    pc = amr.read_particles(plt_file_name, "particles")

    # access the data per tile, e.g., as zero-copy numpy/cupy arrays ...
    w_idx = pc.get_real_comp_index("w")  # runtime component from the file
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            x = soa.get_real_data(0).to_xp()  # position x
            w = soa.get_real_data(w_idx).to_xp()  # runtime component "w"

    # ... or copy all (MPI rank-local) particles into a pandas DataFrame
    # df = pc.to_df()
    # Manual: Read Plotfile Particles END

    # the runtime layout was reconstructed from the file
    for name in real_names:
        assert pc.has_real_comp(name)
    for name in int_names:
        assert pc.has_int_comp(name)
    assert pc.total_number_of_particles() == n_part

    # and the runtime component values round-tripped
    assert x.size == n_part
    np.testing.assert_allclose(w, 1.2345)

    shutil.rmtree(plt_file_name)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_missing_header():
    """Wrong paths raise a catchable Python exception instead of aborting."""
    plt_file_name = "plt_read_missing"
    write_test_plotfile(plt_file_name, 5)

    # wrong particle sub-directory
    with pytest.raises(FileNotFoundError, match="particle Header"):
        amr.read_particles(plt_file_name, "no_such_dir")

    # non-existing plotfile directory
    with pytest.raises(FileNotFoundError, match="particle Header"):
        amr.read_particles("no_such_plotfile", "particles")

    shutil.rmtree(plt_file_name)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_particle_only_output():
    """A bare particle output without a top-level plotfile Header raises with
    container=None and reads fine into a user-provided container.

    Note: conforming AMReX particle plotfiles always include a top-level
    plotfile Header (applications write a dummy MultiFab for pure-particle
    outputs to ensure that), so this layout is off-spec - but it must fail
    with a clear Python exception, not an amrex::Abort.
    """
    plt_file_name = "plt_read_ponly"
    n_part = 10

    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])
    ba = amr.BoxArray(domain_box)
    dm = amr.DistributionMapping(ba, 1)

    pc = amr.ParticleContainer_pureSoA_3_0_polymorphic(geom, dm, ba)
    pc.arena = amr.The_Arena()
    myt = amr.ParticleInitType_pureSoA_3_0()
    myt.real_array_data = [0.0, 0.0, 0.0]  # x, y, z (overwritten by init_random)
    myt.int_array_data = []
    pc.init_random(n_part, 1, myt, False, real_box)
    pc.write_plotfile(
        plt_file_name, "particles", amr.Vector_string([]), amr.Vector_string([])
    )

    with pytest.raises(FileNotFoundError, match="plotfile Header"):
        amr.read_particles(plt_file_name, "particles")

    # reading into an explicitly geometry-defined container works
    # Manual: Read Particles Existing Container START
    # define the geometry in the application, e.g., for a checkpoint restart,
    # then fill the container from the file
    pc_read = amr.ParticleContainer_pureSoA_3_0_polymorphic(geom, dm, ba)
    pc_read.arena = amr.The_Arena()
    pc_read = amr.read_particles(plt_file_name, "particles", container=pc_read)
    # Manual: Read Particles Existing Container END
    assert pc_read.total_number_of_particles() == n_part

    shutil.rmtree(plt_file_name)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_checkpoint_needs_container():
    """An application checkpoint's top-level Header is not in plotfile format:
    geometry recovery must fail with a clear error pointing to 'container'."""
    plt_file_name = "plt_read_chk_src"
    chk_file_name = "chk_read_fake"
    write_test_plotfile(plt_file_name, 5)

    # fake an application checkpoint: valid particle data next to a
    # checkpoint-format top-level Header (as written by amrex::Amr::checkPoint)
    os.makedirs(chk_file_name)
    shutil.copytree(
        os.path.join(plt_file_name, "particles"),
        os.path.join(chk_file_name, "particles"),
    )
    with open(os.path.join(chk_file_name, "Header"), "w") as f:
        f.write("CheckPointVersion_1.0\n")

    with pytest.raises(ValueError, match="container"):
        amr.read_particles(chk_file_name, "particles")

    shutil.rmtree(plt_file_name)
    shutil.rmtree(chk_file_name)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_read_particles_multilevel():
    """Particles from a multi-level plotfile are read back completely: the
    auto-created container is single-level and gathers them all on level 0."""
    plt_file_name = "plt_read_ml"
    n_part = 24

    # two-level AmrCore hierarchy over [-0.5, 0.5]^3 with 16^3 coarse cells;
    # only the upper-x half of the domain is refined, so particles end up
    # distributed over both levels and the fine-level BoxArray covers neither
    # the whole domain nor the coarse index space
    class HalfRefinedCore(amr.AmrCore):
        def make_new_level_from_scratch(self, lev, time, ba, dm):
            pass

        def make_new_level_from_coarse(self, lev, time, ba, dm):
            pass

        def remake_level(self, lev, time, ba, dm):
            pass

        def clear_level(self, lev):
            pass

        def error_est(self, lev, tags, time, ngrow):
            tags.set_val(amr.TagBox.SET, amr.Box([8, 0, 0], [15, 15, 15]))

    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    core = HalfRefinedCore(
        real_box,
        1,
        amr.Vector_int([16, 16, 16]),
        0,  # cartesian
        amr.Vector_IntVect([amr.IntVect(2)]),
        [0, 0, 0],
    )
    core.init_from_scratch(0.0)
    assert core.finest_level == 1
    gdb = core.get_par_gdb()

    # matching two-level mesh plotfile, so the geometry can be recovered
    mfs = []
    for lev in range(2):
        mf = amr.MultiFab(gdb.box_array(lev), gdb.dist_map(lev), 1, 0)
        mf.set_val(np.pi)
        mfs.append(mf)
    amr.write_multi_level_plotfile(
        plt_file_name,
        mfs,
        ["density"],
        [core.geom(0), core.geom(1)],
        1.0,
        [200, 200],
        [amr.IntVect(2)],
    )

    # multi-level pure-SoA particle container with a runtime component
    pc = amr.ParticleContainer_pureSoA_3_0_polymorphic(gdb)
    pc.arena = amr.The_Arena()
    myt = amr.ParticleInitType_pureSoA_3_0()
    myt.real_array_data = [0.0, 0.0, 0.0]  # x, y, z (overwritten by init_random)
    myt.int_array_data = []
    pc.init_random(n_part, 1, myt, False, real_box)
    pc.add_real_comp("w", True)
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            pti.soa().get_real_data(3).assign(1.2345)  # "w": after x,y,z
    pc.redistribute()

    # the particle file must exercise both levels
    n_lev = [pc.number_of_particles_at_level(lvl) for lvl in range(2)]
    assert n_lev[0] > 0 and n_lev[1] > 0
    assert sum(n_lev) == n_part

    pc.write_plotfile(
        plt_file_name, "particles", amr.Vector_string(["w"]), amr.Vector_string([])
    )
    header = amr.ParticleHeader.read(plt_file_name, "particles")
    assert header.finest_level == 1

    pc_read = amr.read_particles(plt_file_name, "particles")
    assert pc_read.finest_level == 0
    assert pc_read.total_number_of_particles() == n_part

    w_idx = pc_read.get_real_comp_index("w")
    for pti in pc_read.iterator(level=0):
        np.testing.assert_allclose(
            pti.soa().get_real_data(w_idx).to_numpy(copy=False), 1.2345
        )

    shutil.rmtree(plt_file_name)
