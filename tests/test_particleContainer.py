# -*- coding: utf-8 -*-

import importlib

import numpy as np
import pytest

import amrex.space3d as amr


@pytest.fixture()
def Npart():
    return 21


@pytest.fixture(scope="function")
def empty_particle_container(std_geometry, distmap, boxarr):
    # This fixture includes the legacy AoS layout components, which for CuPy only run on CPU
    # or require managed memory, see https://github.com/cupy/cupy/issues/2031
    if amr.Config.have_gpu:
        return amr.ParticleContainer_2_1_3_1_managed(std_geometry, distmap, boxarr)
    else:
        return amr.ParticleContainer_2_1_3_1_default(std_geometry, distmap, boxarr)


@pytest.fixture(scope="function")
def empty_soa_particle_container(std_geometry, distmap, boxarr):
    pc = amr.ParticleContainer_pureSoA_8_0_default(std_geometry, distmap, boxarr)
    return pc


@pytest.fixture(scope="function")
def std_particle():
    myt = amr.ParticleInitType_2_1_3_1()
    myt.real_struct_data = [0.5, 0.6]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2, 0.3]
    myt.int_array_data = [1]
    return myt


@pytest.fixture(scope="function")
def particle_container(Npart, std_geometry, distmap, boxarr, std_real_box):
    # This fixture includes the legacy AoS layout components, which for CuPy only run on CPU
    # or require managed memory, see https://github.com/cupy/cupy/issues/2031
    if amr.Config.have_gpu:
        pc = amr.ParticleContainer_2_1_3_1_managed(std_geometry, distmap, boxarr)
    else:
        pc = amr.ParticleContainer_2_1_3_1_default(std_geometry, distmap, boxarr)
    myt = amr.ParticleInitType_2_1_3_1()
    myt.real_struct_data = [0.5, 0.6]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2, 0.3]
    myt.int_array_data = [1]

    iseed = 1
    pc.init_random(Npart, iseed, myt, False, std_real_box)

    # add runtime components: 1 real 2 int
    pc.add_real_comp("b", True)
    pc.add_int_comp("i1", True)
    pc.add_int_comp("i2", True)

    # assign some values to runtime components
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            soa.get_real_data(2).assign(1.2345)
            soa.get_int_data(1).assign(42)
            soa.get_int_data(2).assign(33)

    return pc


@pytest.fixture(scope="function")
def soa_particle_container(Npart, std_geometry, distmap, boxarr, std_real_box):
    pc = amr.ParticleContainer_pureSoA_8_0_default(std_geometry, distmap, boxarr)
    myt = amr.ParticleInitType_pureSoA_8_0()
    myt.real_array_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    myt.int_array_data = []

    with pytest.raises(Exception):
        pc.set_soa_compile_time_names(
            ["x", "y", "z", "z", "b", "c", "d", "e"], []
        )  # error: z added twice
    pc.set_soa_compile_time_names(["x", "y", "z", "a", "b", "c", "d", "e"], [])

    iseed = 1
    pc.init_random(Npart, iseed, myt, False, std_real_box)

    # add runtime components: 1 real 2 int
    with pytest.raises(Exception):
        pc.add_real_comp("a", True)  # already used as a compile-time component
    pc.add_real_comp("f", True)
    pc.add_int_comp("i1", True)
    pc.add_int_comp("i2", True)

    # assign some values to runtime components
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            soa.get_real_data(8).assign(1.2345)
            soa.get_int_data(0).assign(42)
            soa.get_int_data(1).assign(33)

    return pc


def test_particleInitType():
    myt = amr.ParticleInitType_2_1_3_1()
    print(myt.real_struct_data)
    print(myt.int_struct_data)
    print(myt.real_array_data)
    print(myt.int_array_data)

    myt.real_struct_data = [0.5, 0.7]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2, 0.4]
    myt.int_array_data = [1]

    assert np.allclose(myt.real_struct_data, [0.5, 0.7])
    assert np.allclose(myt.int_struct_data, [5])
    assert np.allclose(myt.real_array_data, [0.5, 0.2, 0.4])
    assert np.allclose(myt.int_array_data, [1])


def test_n_particles(particle_container, Npart):
    pc = particle_container
    assert pc.OK()
    assert (
        pc.num_struct_real == amr.ParticleContainer_2_1_3_1_default.num_struct_real == 2
    )
    assert (
        pc.num_struct_int == amr.ParticleContainer_2_1_3_1_default.num_struct_int == 1
    )
    assert (
        pc.num_array_real == amr.ParticleContainer_2_1_3_1_default.num_array_real == 3
    )
    assert pc.num_array_int == amr.ParticleContainer_2_1_3_1_default.num_array_int == 1
    assert (
        pc.number_of_particles_at_level(0)
        == np.sum(pc.number_of_particles_in_grid(0))
        == Npart
    )


def test_pc_init():
    # This test only runs on CPU or requires managed memory,
    # see https://github.com/cupy/cupy/issues/2031
    pc = (
        amr.ParticleContainer_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleContainer_2_1_3_1_default()
    )

    print("bytespread", pc.byte_spread)
    print("capacity", pc.print_capacity())
    print("number_of_particles_at_level(0)", pc.number_of_particles_at_level(0))
    assert pc.number_of_particles_at_level(0) == 0

    bx = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    rb = amr.RealBox(0, 0, 0, 1, 1, 1)
    coord_int = 1  # RZ
    periodicity = [0, 0, 1]
    gm = amr.Geometry(bx, rb, coord_int, periodicity)

    ba = amr.BoxArray(bx)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    print("-------------------------")
    print("define particle container")
    pc.Define(gm, dm, ba)
    assert pc.OK()
    assert (
        pc.num_struct_real == amr.ParticleContainer_2_1_3_1_default.num_struct_real == 2
    )
    assert (
        pc.num_struct_int == amr.ParticleContainer_2_1_3_1_default.num_struct_int == 1
    )
    assert (
        pc.num_array_real == amr.ParticleContainer_2_1_3_1_default.num_array_real == 3
    )
    assert pc.num_array_int == amr.ParticleContainer_2_1_3_1_default.num_array_int == 1

    print("bytespread", pc.byte_spread)
    print("capacity", pc.print_capacity())
    print("number_of_particles_at_level(0)", pc.number_of_particles_at_level(0))
    assert pc.total_number_of_particles() == pc.number_of_particles_at_level(0) == 0
    assert pc.OK()

    print("---------------------------")
    print("add a particle to each grid")
    Npart_grid = 1
    iseed = 1
    myt = amr.ParticleInitType_2_1_3_1()
    myt.real_struct_data = [0.5, 0.4]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2, 0.4]
    myt.int_array_data = [1]
    pc.init_random_per_box(Npart_grid, iseed, myt)
    ngrid = ba.size
    npart = Npart_grid * ngrid

    print("NumberOfParticles", pc.number_of_particles_at_level(0))
    assert pc.total_number_of_particles() == pc.number_of_particles_at_level(0) == npart
    assert pc.OK()

    print(f"Finest level = {pc.finest_level}")

    print("Iterate particle boxes & set values")
    # lvl = 0
    for lvl in range(pc.finest_level + 1):
        print(f"at level {lvl}:")
        for pti in pc.iterator(level=lvl):
            print("...")
            assert pti.num_particles == 1
            assert pti.num_real_particles == 1
            assert pti.num_neighbor_particles == 0
            assert pti.level == lvl
            print(pti.pair_index)
            print(pti.geom(level=lvl))

            # note: cupy does not yet support this
            # https://github.com/cupy/cupy/issues/2031
            aos = pti.aos()
            aos_arr = aos.to_numpy()
            aos_arr[0]["x"] = 0.30
            aos_arr[0]["y"] = 0.35
            aos_arr[0]["z"] = 0.40

            # TODO: this seems to write into a copy of the data
            soa = pti.soa()
            real_arrays = soa.get_real_data()
            int_arrays = soa.get_int_data()
            real_arrays[0] = [0.55]
            real_arrays[1] = [0.22]
            int_arrays[0] = [2]

            assert np.allclose(real_arrays[0], np.array([0.55]))
            assert np.allclose(real_arrays[1], np.array([0.22]))
            assert np.allclose(int_arrays[0], np.array([2]))

    # read-only
    for lvl in range(pc.finest_level + 1):
        for pti in pc.const_iterator(level=lvl):
            assert pti.num_particles == 1
            assert pti.num_real_particles == 1
            assert pti.num_neighbor_particles == 0
            assert pti.level == lvl

            aos = pti.aos()
            aos_arr = aos.to_numpy()
            assert np.isclose(aos[0].x, 0.30)
            assert np.isclose(aos[0].y, 0.35)
            assert np.isclose(aos[0].z, 0.40)
            assert np.isclose(aos_arr[0]["z"], 0.40)

            soa = pti.soa()
            real_arrays = soa.get_real_data()
            int_arrays = soa.get_int_data()
            print(real_arrays[0])
            print(int_arrays[0])
            # TODO: this does not work yet and is still the original data
            # assert np.allclose(real_arrays[0], np.array([0.55]))
            # assert np.allclose(real_arrays[1], np.array([0.22]))
            # assert np.allclose(int_arrays[0], np.array([2]))


def test_particle_init(Npart, particle_container):
    pc = particle_container
    assert (
        pc.number_of_particles_at_level(0)
        == np.sum(pc.number_of_particles_in_grid(0))
        == Npart
    )

    # pc.resizeData()
    print(pc.num_local_tiles_at_level(0))
    lev = pc.get_particles(0)
    print(len(lev.items()))
    assert pc.num_local_tiles_at_level(0) == len(lev.items())
    for tile_ind, pt in lev.items():
        print("tile", tile_ind)
        real_arrays = pt.get_struct_of_arrays().get_real_data()
        int_arrays = pt.get_struct_of_arrays().get_int_data()
        aos = pt.get_array_of_structs()
        aos_arr = aos.to_numpy()
        if len(real_arrays) > 0:
            assert np.isclose(real_arrays[0][0], 0.5) and np.isclose(
                real_arrays[1][0], 0.2
            )
            assert isinstance(int_arrays[0][0], int)
            assert int_arrays[0][0] == 1
            assert isinstance(aos_arr[0]["rdata_0"], np.floating)
            assert isinstance(aos_arr[0]["idata_0"], np.integer)
            assert (
                np.isclose(aos_arr[0]["rdata_0"], 0.5) and aos_arr[0]["idata_0"] == 5
            )  # fixme in SP: random value np.int32(-2147483648) == 5

            aos_arr[0]["idata_0"] = 2
            aos1 = pt.get_array_of_structs()
            print(aos1[0])
            print(aos[0])
            print(aos_arr[0])
            assert (
                aos_arr[0]["idata_0"]
                == aos[0].get_idata(0)
                == aos1[0].get_idata(0)
                == 2
            )

            print("soa test")
            real_arrays[1][0] = -1.2

            ra1 = pt.get_struct_of_arrays().get_real_data()
            print(real_arrays)
            print(ra1)
            for ii, arr in enumerate(real_arrays):
                assert np.allclose(arr.to_numpy(), ra1[ii].to_numpy())

            print("soa int test")
            iarr_np = int_arrays[0].to_numpy()
            iarr_np[0] = -3
            ia1 = pt.get_struct_of_arrays().get_int_data()
            ia1_np = ia1[0].to_numpy()
            print(iarr_np)
            print(ia1_np)
            assert np.allclose(iarr_np, ia1_np)

    print(
        "---- is the particle tile recording changes or passed by reference? --------"
    )
    lev1 = pc.get_particles(0)
    for tile_ind, pt in lev1.items():
        print("tile", tile_ind)
        real_arrays = pt.get_struct_of_arrays().get_real_data()
        int_arrays = pt.get_struct_of_arrays().get_int_data()
        aos = pt.get_array_of_structs()
        print(aos[0])
        assert aos[0].get_idata(0) == 2
        assert np.isclose(real_arrays[1][0], -1.2)
        assert int_arrays[0][0] == -3


def test_per_cell(empty_particle_container, std_geometry, std_particle):
    pc = empty_particle_container
    pc.init_one_per_cell(0.5, 0.5, 0.5, std_particle)
    assert pc.OK()

    lev = pc.get_particles(0)
    assert pc.num_local_tiles_at_level(0) == len(lev.items())

    sum_1 = 0
    for tile_ind, pt in lev.items():
        print("tile", tile_ind)
        real_arrays = pt.get_struct_of_arrays().get_real_data()
        sum_1 += np.sum(real_arrays[1])
    print(sum_1)
    ncells = std_geometry.domain.numPts()
    print("ncells from box", ncells)
    print("NumberOfParticles", pc.number_of_particles_at_level(0))
    assert (
        pc.total_number_of_particles() == pc.number_of_particles_at_level(0) == ncells
    )
    print("npts * real_1", ncells * std_particle.real_array_data[1])
    assert np.isclose(ncells * std_particle.real_array_data[1], sum_1)


def test_soa_pc_numpy(soa_particle_container, Npart):
    """Used in docs/source/usage/compute.rst"""
    pc = soa_particle_container
    assert pc.number_of_particles_at_level(0) == Npart

    class Config:
        have_gpu = False

    # Manual: Pure SoA Compute PC Detailed START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # iterate over mesh-refinement levels
    for lvl in range(pc.finest_level + 1):
        # loop local tiles of particles
        for pti in pc.iterator(level=lvl):
            # compile-time and runtime attributes
            soa = pti.soa().to_xp()

            # print all particle ids in the tile
            print("idcpu =", soa.idcpu)

            x = soa.real["x"]
            y = soa.real["y"]

            # write to all particles in the tile
            # note: careful, if you change particle positions, you might need to
            #       redistribute particles before continuing the simulation step
            soa.real["x"][:] = 0.30
            soa.real["y"][:] = 0.35
            soa.real["z"][:] = 0.40

            soa.real["a"][:] = x[:] ** 2
            soa.real["b"][:] = x[:] + y[:]
            soa.real["c"][:] = 0.50
            # ...

            # all int attributes
            for soa_int in soa.int.values():
                soa_int[:] = 12
    # Manual: Pure SoA Compute PC Detailed END

    # Manual: Pure SoA Compute PC Simple pti START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # iterate over particles on level 0
    for pti in pc.iterator(level=0):
        # print all particle ids in the tile
        print("idcpu =", pti["idcpu"])

        x = pti["x"]  # this is automatically a cupy or numpy
        y = pti["y"]  #   array, depending on Config.have_gpu

        # write to all particles in the chunk
        # note: careful, if you change particle positions, you might need to
        #       redistribute particles before continuing the simulation step
        pti["x"][:] = 0.30
        pti["y"][:] = 0.35
        pti["z"][:] = 0.40

        pti["a"][:] = x[:] ** 2
        pti["b"][:] = x[:] + y[:]
        pti["c"][:] = 0.50
        # ...

        # int attributes
        pti["i1"][:] = 12
        pti["i2"][:] = 13
    # Manual: Pure SoA Compute PC Simple pti END


def test_pc_numpy(particle_container, Npart):
    """Used in docs/source/usage/compute.rst"""
    pc = particle_container
    assert pc.number_of_particles_at_level(0) == Npart

    class Config:
        have_gpu = False

    # Manual: Legacy Compute PC Detailed START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # iterate over mesh-refinement levels
    for lvl in range(pc.finest_level + 1):
        # loop local tiles of particles
        for pti in pc.iterator(level=lvl):
            # default layout: AoS with positions and idcpu
            # note: not part of the new PureSoA particle container layout
            aos = (
                pti.aos().to_numpy(copy=True)
                if Config.have_gpu
                else pti.aos().to_numpy()
            )

            # additional compile-time and runtime attributes in SoA format
            soa = pti.soa().to_xp()

            # notes:
            # Only the next lines are the "HOT LOOP" of the computation.
            # For efficiency, use numpy array operation for speed on CPUs.
            # For GPUs use .to_cupy() above and compute with cupy or numba.

            # print all particle ids in the chunk
            print("idcpu =", aos[:]["idcpu"])

            # write to all particles in the chunk
            aos[:]["x"] = 0.30
            aos[:]["y"] = 0.35
            aos[:]["z"] = 0.40

            print(soa.real)
            for soa_real in soa.real.values():
                soa_real[:] = 42.0

            for soa_int in soa.int.values():
                soa_int[:] = 12
    # Manual: Legacy Compute PC Detailed END


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
@pytest.mark.skipif(
    amr.Config.precision_particles == "SINGLE",
    reason="Requires DOUBLE precision particles",
)
def test_pc_df(particle_container, Npart):
    pc = particle_container
    print(f"pc={pc}")
    df = pc.to_df()
    print(df.columns)
    print(df)

    assert len(df.columns) == 14


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_soa_pc_empty_df(empty_soa_particle_container, Npart):
    pc = empty_soa_particle_container
    print(f"pc={pc}")
    df = pc.to_df()
    assert df is None


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_df_mpi(soa_particle_container, Npart):
    pc = soa_particle_container
    print(f"pc={pc}")
    df = pc.to_df(local=False)
    if df is not None:
        # only rank 0
        print(df.columns)
        print(df)


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_soa_pc_df(soa_particle_container, Npart):
    """Used in docs/source/usage/compute.rst"""
    pc = soa_particle_container

    class Config:
        have_gpu = False

    # Manual: Pure SoA Compute PC Pandas START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # local particles on all levels
    df = pc.to_df()  # this is a copy!
    print(df)

    # read
    print(df["x"])

    # write (into copy!)
    df["x"] = 0.30
    df["y"] = 0.35
    df["z"] = 0.40

    df["a"] = df["x"] ** 2
    df["b"] = df["x"] + df["y"]
    df["c"] = 0.50

    # int attributes
    # df["i1"] = 12
    # df["i2"] = 12
    # ...

    print(df)

    # Manual: Pure SoA Compute PC Pandas END


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_pc_empty_df(empty_particle_container, Npart):
    pc = empty_particle_container
    print(f"pc={pc}")
    df = pc.to_df()
    assert df is None


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
@pytest.mark.skipif(
    amr.Config.precision_particles == "SINGLE",
    reason="Requires DOUBLE precision particles",
)
@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_pc_df_mpi(particle_container, Npart):
    pc = particle_container
    print(f"pc={pc}")
    df = pc.to_df(local=False)
    if df is not None:
        # only rank 0
        print(df.columns)
        print(df)

        assert len(df.columns) == 14
