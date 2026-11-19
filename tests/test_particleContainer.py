# -*- coding: utf-8 -*-

import importlib
import sys

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
    pc = amr.ParticleContainer_pureSoA_11_0_polymorphic(std_geometry, distmap, boxarr)
    pc.arena = amr.The_Arena()
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
    pc = amr.ParticleContainer_pureSoA_11_0_polymorphic(std_geometry, distmap, boxarr)
    pc.arena = amr.The_Arena()
    myt = amr.ParticleInitType_pureSoA_11_0()
    myt.real_array_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2]
    myt.int_array_data = []

    with pytest.raises(Exception):
        pc.set_soa_compile_time_names(
            ["x", "y", "z", "z", "b", "c", "d", "e", "f", "g", "h"], []
        )  # error: z added twice
    pc.set_soa_compile_time_names(
        ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h"], []
    )

    iseed = 1
    pc.init_random(Npart, iseed, myt, False, std_real_box)

    # add runtime components: 1 real 2 int
    with pytest.raises(Exception):
        pc.add_real_comp("a", True)  # already used as a compile-time component
    pc.add_real_comp("w", True)
    pc.add_int_comp("i1", True)
    pc.add_int_comp("i2", True)

    # assign some values to runtime components
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            soa = pti.soa()
            soa.get_real_data(11).assign(1.2345)
            soa.get_int_data(0).assign(42)
            soa.get_int_data(1).assign(33)

    yield pc

    pc.clear_particles()


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


def test_particle_iterators_keep_container_alive(
    particle_container, assert_keeps_python_alive
):
    pc = particle_container

    assert_keeps_python_alive(pc, lambda: pc.iterator(level=0))
    assert_keeps_python_alive(pc, lambda: pc.Iterator(pc, level=0))
    assert_keeps_python_alive(pc, lambda: pc.ConstIterator(pc, level=0))


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
    return

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


def _make_empty_soa_like(std_geometry, distmap, boxarr):
    """An empty container with the same layout as soa_particle_container"""
    pc = amr.ParticleContainer_pureSoA_11_0_polymorphic(std_geometry, distmap, boxarr)
    pc.arena = amr.The_Arena()
    pc.set_soa_compile_time_names(
        ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h"], []
    )
    pc.add_real_comp("w", True)
    pc.add_int_comp("i1", True)
    pc.add_int_comp("i2", True)
    return pc


def _random_columns(pc, npart, seed=42):
    """Particle columns of a pure SoA container, with positions in the domain"""
    rng = np.random.default_rng(seed)
    data = {name: rng.uniform(0.0, 1.0, npart) for name in pc.real_soa_names}
    for k, name in enumerate(pc.int_soa_names):
        data[name] = np.arange(npart, dtype=np.int32) + 100 * k
    return data


def _particle_columns(pc, gather=False):
    """Particle data as NumPy arrays by component name, without pandas

    With gather=True, the data of all MPI ranks is returned on every rank.
    """
    columns = {}
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            if pti.size == 0:
                continue
            soa = pti.soa().to_numpy(copy=True)
            items = [("idcpu", soa.idcpu), *soa.real.items(), *soa.int.items()]
            for name, array in items:
                columns.setdefault(name, []).append(np.asarray(array))
    parts = [columns]
    if gather and amr.Config.have_mpi:
        from mpi4py import MPI

        parts = MPI.COMM_WORLD.allgather(columns)
    names = {name for part in parts for name in part}
    return {
        name: np.concatenate([a for part in parts for a in part.get(name, [])])
        for name in names
    }


def _assert_same_values(stored, expected):
    """Compare stored particle values with input values, in any order

    The input is cast to the stored type, like the copy into the container.
    """
    np.testing.assert_array_equal(
        np.sort(stored), np.sort(np.asarray(expected).astype(stored.dtype))
    )


def _has_cuda_cupy():
    if importlib.util.find_spec("cupy") is None:
        return False
    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_soa_pc_add_df_roundtrip(
    soa_particle_container, std_geometry, distmap, boxarr, Npart
):
    """to_df(local=False) followed by add_df(local=False) restores all particles"""
    df = soa_particle_container.to_df(local=False)

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    # Manual: Pure SoA Add DF START
    # pc: a particle container, e.g., empty
    # df: a pandas.DataFrame with one column per particle component, as in
    #     pc.to_df(); only needed on the root rank (None on other ranks).
    #     Without an "idcpu" column, new particle ids are created.
    pc.add_df(df, local=False)  # add df of the root rank, then redistribute
    # Manual: Pure SoA Add DF END

    assert pc.total_number_of_particles() == Npart

    df2 = pc.to_df(local=False)
    if df is not None:
        # only the root rank
        df = df.sort_values("idcpu").reset_index(drop=True)
        df2 = df2.sort_values("idcpu").reset_index(drop=True)
        assert list(df2.columns) == list(df.columns)
        assert (df2.dtypes == df.dtypes).all()
        assert df2.equals(df)


@pytest.mark.parametrize("distribute", ["redistribute", "equally", "none"])
def test_soa_pc_add_arrays_new_ids(std_geometry, distmap, boxarr, distribute):
    """Particles without idcpu get new, unique and valid ids"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 17
    root = amr.ParallelDescriptor.IOProcessor()

    data = _random_columns(pc, npart) if root else None
    pc.add_arrays(data, local=False, distribute=distribute)
    assert pc.total_number_of_particles() == npart

    # a second batch continues the ids of this rank
    pc.add_arrays(data, local=False, distribute=distribute)
    assert pc.total_number_of_particles() == 2 * npart

    columns = _particle_columns(pc, gather=True)
    if root:
        idcpu = columns["idcpu"]
        assert all(amr.is_valid(int(v)) for v in idcpu)
        pairs = set(zip(amr.unpack_ids(idcpu), amr.unpack_cpus(idcpu)))
        assert len(pairs) == 2 * npart
        _assert_same_values(columns["x"], np.concatenate([data["x"], data["x"]]))


def test_soa_pc_add_arrays_local(std_geometry, distmap, boxarr):
    """With local=True, every rank adds its own particles"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 5
    rank = amr.ParallelDescriptor.MyProc()
    # only ranks that own a box can hold particles
    owners = set(distmap.ProcessorMap())

    data = _random_columns(pc, npart, seed=rank) if rank in owners else None
    pc.add_arrays(data, local=True)
    assert pc.total_number_of_particles() == npart * len(owners)

    # empty input on every rank adds nothing
    pc.add_arrays(None, local=True)
    pc.add_arrays({}, local=True)
    assert pc.total_number_of_particles() == npart * len(owners)


def test_soa_pc_add_arrays_podvector(std_geometry, distmap, boxarr):
    """pyAMReX PODVectors are accepted as columns"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    if amr.ParallelDescriptor.MyProc() not in set(distmap.ProcessorMap()):
        pytest.skip("This MPI rank owns no box")
    npart = 9
    data = _random_columns(pc, npart)
    data["x"] = amr.PODVector_real_std.from_numpy(data["x"])

    pc.add_arrays(data, local=True, distribute="none")
    assert pc.total_number_of_particles(True, True) == npart
    _assert_same_values(_particle_columns(pc)["x"], data["x"].to_numpy())


@pytest.mark.parametrize("source", ["xp", "podvector"])
def test_soa_pc_add_arrays_self_view(std_geometry, distmap, boxarr, source):
    """Columns that view the tile that receives the particles are copied first"""
    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 1000
    add = dict(local=True, distribute="none")
    pc.add_arrays(_random_columns(pc, npart), **add)
    before = _particle_columns(pc)

    # the first tile of the first box of this rank, which add_arrays appends to
    grid = list(distmap.ProcessorMap()).index(amr.ParallelDescriptor.MyProc())
    soa = pc.define_and_return_particle_tile(0, grid, 0).get_struct_of_arrays()
    assert soa.size == npart
    if source == "xp":
        soa = soa.to_xp()  # NumPy, CuPy or dpnp views, no copies
        columns = {"idcpu": soa.idcpu, **soa.real, **soa.int}
    else:
        columns = {"idcpu": soa.get_idcpu_data()}
        columns.update(
            (name, soa.get_real_data(i)) for i, name in enumerate(pc.real_soa_names)
        )
        columns.update(
            (name, soa.get_int_data(i)) for i, name in enumerate(pc.int_soa_names)
        )
    del soa

    pc.add_arrays(columns, **add)
    assert pc.total_number_of_particles(True, True) == 2 * npart
    after = _particle_columns(pc)
    for name, values in before.items():
        np.testing.assert_array_equal(after[name], np.concatenate([values, values]))


def test_pack_ids_views():
    """pack_ids/pack_cpus write strided views element by element"""
    ids = np.arange(1, 6, dtype=np.int64)
    cpus = np.arange(10, 15, dtype=np.int32)
    expected = np.zeros(5, dtype=np.uint64)
    amr.pack_ids(expected, ids)
    amr.pack_cpus(expected, cpus)

    idcpu = np.zeros(10, dtype=np.uint64)
    amr.pack_ids(idcpu[::2], np.repeat(ids, 2)[::2])
    amr.pack_cpus(idcpu[::2], np.repeat(cpus, 2)[::2])
    np.testing.assert_array_equal(idcpu[::2], expected)
    np.testing.assert_array_equal(idcpu[1::2], 0)


def test_soa_pc_add_arrays_errors(std_geometry, distmap, boxarr):
    """Invalid inputs raise on all ranks and leave the container unchanged"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 4
    data = _random_columns(pc, npart)

    missing = dict(data)
    del missing["w"]
    with pytest.raises((KeyError, RuntimeError)):
        pc.add_arrays(missing, local=True)

    unknown = dict(data, foo=np.zeros(npart))
    with pytest.raises((ValueError, RuntimeError)):
        pc.add_arrays(unknown, local=True)

    unequal = dict(data, y=np.zeros(npart + 1))
    with pytest.raises((ValueError, RuntimeError)):
        pc.add_arrays(unequal, local=True)

    two_d = dict(data, y=np.zeros((npart, 2)))
    with pytest.raises((ValueError, RuntimeError)):
        pc.add_arrays(two_d, local=True)

    if amr.Config.have_mpi:
        # errors on the root rank are raised on all ranks
        with pytest.raises((KeyError, RuntimeError)):
            root = amr.ParallelDescriptor.IOProcessor()
            pc.add_arrays(missing if root else None, local=False)

    assert pc.total_number_of_particles() == 0


def test_pc_add_arrays_aos(empty_particle_container):
    """Legacy AoS containers are not supported"""
    with pytest.raises(NotImplementedError):
        empty_particle_container.add_arrays({}, local=True)


def test_soa_pc_add_arrays_no_cupy_import():
    """Adding NumPy columns does not import CuPy"""
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        import amrex.space3d as amr

        # allocate GPU memory on demand, like conftest.py: the GPU may be
        # shared with other test processes
        amr.initialize(
            ["amrex.the_arena_is_managed=0", "amrex.the_arena_init_size=0"]
        )
        bx = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(7, 7, 7))
        rb = amr.RealBox(0, 0, 0, 1.0, 1.0, 1.0)
        gm = amr.Geometry(bx, rb, 0, [0, 0, 0])
        ba = amr.BoxArray(bx)
        dm = amr.DistributionMapping(ba)
        pc = amr.ParticleContainer_pureSoA_3_0_default(gm, dm, ba)
        n = 3
        data = {name: np.full(n, 0.5) for name in pc.real_soa_names}
        pc.add_arrays(data)
        assert pc.total_number_of_particles() == n
        del pc
        amr.finalize()
        for name in ("cupy", "dpnp", "dpctl"):
            assert name not in sys.modules, f"{name} was imported"
        """
    )
    # run as an MPI singleton: drop the process manager variables of mpiexec
    launcher_prefixes = ("PMI", "PMIX", "HYDRA", "HYDI", "MPIR", "OMPI")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    if amr.ParallelDescriptor.MyProc() != 0:
        pytest.skip("Runs on MPI rank 0 only")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_soa_pc_add_arrays_without_cupy_dpnp(
    std_geometry, distmap, boxarr, monkeypatch
):
    """Host data is added without CuPy or dpnp, also on GPU builds"""
    for name in ("cupy", "dpnp", "dpctl"):
        monkeypatch.setitem(sys.modules, name, None)

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    if amr.ParallelDescriptor.MyProc() not in set(distmap.ProcessorMap()):
        pytest.skip("This MPI rank owns no box")
    npart = 7
    data = _random_columns(pc, npart)
    data["y"] = amr.PODVector_real_std.from_array(data["y"])
    data["i1"] = data["i1"].astype(np.int64)  # cast to the int component type

    pc.add_arrays(data, local=True, distribute="none")
    assert pc.total_number_of_particles(True, True) == npart
    columns = _particle_columns(pc)
    for name in ("x", "i1"):
        _assert_same_values(columns[name], data[name])
    _assert_same_values(columns["y"], data["y"].to_numpy())


@pytest.mark.skipif(not _has_cuda_cupy(), reason="CuPy with a CUDA device required")
@pytest.mark.parametrize("local", [True, False])
def test_soa_pc_add_arrays_cupy(std_geometry, distmap, boxarr, local):
    """CuPy columns, also mixed with NumPy columns"""
    import cupy as cp

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 11
    root = amr.ParallelDescriptor.IOProcessor()

    data = _random_columns(pc, npart)
    # all Real columns on the device, int columns stay NumPy
    data_cp = {
        name: cp.asarray(col) if col.dtype.kind == "f" else col
        for name, col in data.items()
    }
    # only ranks that own a box can hold particles
    owners = set(distmap.ProcessorMap())
    rank = amr.ParallelDescriptor.MyProc()
    if (local and rank in owners) or root:
        pc.add_arrays(data_cp, local=local)
    else:
        pc.add_arrays(local=local)

    nranks = len(owners) if local else 1
    assert pc.total_number_of_particles() == npart * nranks

    columns = _particle_columns(pc, gather=True)
    if root:
        for name in data:
            _assert_same_values(columns[name], np.tile(data[name], nranks))


def _owns_box(distmap):
    return amr.ParallelDescriptor.MyProc() in set(distmap.ProcessorMap())


def test_soa_pc_add_arrays_hybrid(std_geometry, distmap, boxarr):
    """Columns as a mapping, keyword arguments, scalars and fill values"""
    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 6
    data = _random_columns(pc, npart)
    add = dict(local=True, distribute="none")

    # keyword arguments
    pc.add_arrays(**data, **add)
    # a mapping plus keyword arguments, with broadcast scalars
    mapping = {name: data[name] for name in ("x", "y", "z")}
    rest = {name: v for name, v in data.items() if name not in mapping}
    rest["w"] = 2.5
    rest["i1"] = 7
    pc.add_arrays(mapping, **rest, **add)
    # fill values for all missing components
    pc.add_arrays(x=data["x"], y=data["y"], z=data["z"], fill_missing=0, **add)
    assert pc.total_number_of_particles(True, True) == 3 * npart

    columns = _particle_columns(pc)
    _assert_same_values(columns["x"], np.tile(data["x"], 3))
    _assert_same_values(
        columns["w"], np.concatenate([data["w"], np.full(npart, 2.5), np.zeros(npart)])
    )
    _assert_same_values(
        columns["i1"], np.concatenate([data["i1"], np.full(npart, 7), np.zeros(npart)])
    )

    # errors, before any particle is added
    with pytest.raises(ValueError, match="both"):
        pc.add_arrays(mapping, x=data["x"], **rest, **add)
    with pytest.raises(ValueError, match="redistrbute"):
        pc.add_arrays(**data, redistrbute=True, **add)
    with pytest.raises(KeyError, match="fill_missing"):
        pc.add_arrays(x=data["x"], **add)
    with pytest.raises(ValueError, match="array"):
        pc.add_arrays({name: 1.0 for name in data}, **add)
    with pytest.raises(ValueError, match="idcpu"):
        pc.add_arrays(**data, idcpu=2**63, **add)
    with pytest.raises(TypeError):
        pc.add_arrays(x=data["x"], fill_missing=[0.0], **add)
    with pytest.raises(TypeError):
        pc.add_arrays([1.0, 2.0], **add)
    assert pc.total_number_of_particles(True, True) == 3 * npart


def test_soa_pc_add_arrays_option_named_component(std_geometry, distmap, boxarr):
    """Components named like options are given in the mapping"""
    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    pc.add_real_comp("local", True)
    npart = 3
    data = _random_columns(pc, npart)
    local_values = data.pop("local")

    pc.add_arrays({"local": local_values}, **data, local=True, distribute="none")
    _assert_same_values(_particle_columns(pc)["local"], local_values)


def test_soa_pc_add_arrays_casts(std_geometry, distmap, boxarr):
    """Values that cannot be stored without silent data loss raise"""
    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 4
    data = _random_columns(pc, npart)
    add = dict(local=True, distribute="none")

    bad = {
        TypeError: [
            dict(i1=np.array(["a"] * npart)),  # strings
            dict(x=np.ones(npart, dtype=np.complex128)),  # complex
            dict(idcpu=np.ones(npart)),  # floating point ids
        ],
        ValueError: [
            dict(i1=np.full(npart, 2**40, dtype=np.int64)),  # int32 overflow
            dict(i1=np.full(npart, 0.5)),  # not an integer
            dict(i1=np.full(npart, np.nan)),  # not a number
            dict(i1=2**40),  # scalar int32 overflow
            dict(idcpu=np.zeros(npart, dtype=np.uint64)),  # invalid particles
            dict(idcpu=np.ones(npart, dtype=np.int32)),  # cast, but invalid
            dict(i1=float("inf")),  # scalar not finite
            dict(i1=10**400),  # Python int beyond the float range
            dict(w=10**400),  # Python int beyond the float range
            dict(idcpu=np.full(npart, -1, dtype=np.int32)),  # negative ids
            dict(idcpu=np.ones(npart, dtype=np.int8)),  # cast, but invalid
        ],
    }
    for error, cases in bad.items():
        for case in cases:
            with pytest.raises(error):
                pc.add_arrays(**dict(data, **case), **add)
    assert pc.total_number_of_particles(True, True) == 0

    # lossless or intended casts
    ok = dict(
        data,
        i1=np.arange(npart, dtype=np.float64),  # integer values
        i2=np.arange(npart, dtype=np.uint64),  # small values
    )
    pc.add_arrays(**ok, **add)
    columns = _particle_columns(pc)
    for name in ("x", "i1", "i2"):
        _assert_same_values(columns[name], ok[name])


def test_soa_pc_add_arrays_error_on_one_rank(std_geometry, distmap, boxarr):
    """Invalid data on one rank raises on all ranks, instead of hanging"""
    nranks = amr.ParallelDescriptor.NProcs()
    if nranks < 2:
        pytest.skip("Requires at least 2 MPI ranks")
    rank = amr.ParallelDescriptor.MyProc()
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 3
    data = _random_columns(pc, npart) if _owns_box(distmap) else {}
    bad = dict(data, i1=np.array(["a"] * npart)) if data else {}

    # local=True, collective because of distribute="redistribute"
    with pytest.raises(TypeError if rank == 1 and bad else RuntimeError):
        pc.add_arrays(bad if rank == 1 else data, local=True)
    # local=False: invalid data on the root rank
    with pytest.raises(
        TypeError if amr.ParallelDescriptor.IOProcessor() else RuntimeError
    ):
        pc.add_arrays(
            bad if amr.ParallelDescriptor.IOProcessor() else None, local=False
        )
    assert pc.total_number_of_particles() == 0


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
@pytest.mark.parametrize("root", ["first", "last"])
def test_soa_pc_add_arrays_split(std_geometry, distmap, boxarr, root):
    """local=False splits the particles 1/N over the ranks that own boxes"""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    root_rank = 0 if root == "first" else nranks - 1
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 17

    data = _random_columns(pc, npart) if comm.Get_rank() == root_rank else None
    pc.add_arrays(data, local=False, root_rank=root_rank, distribute="equally")

    owners = sorted(set(distmap.ProcessorMap()))
    navg, nleft = divmod(npart, len(owners))
    expected = [0] * nranks
    for k, owner in enumerate(owners):
        expected[owner] = navg + (1 if k < nleft else 0)
    counts = comm.allgather(pc.total_number_of_particles(True, True))
    assert counts == expected

    columns = _particle_columns(pc, gather=True)
    if comm.Get_rank() == root_rank:
        _assert_same_values(columns["x"], data["x"])


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_add_arrays_root_redistribute(
    std_geometry, distmap, boxarr, monkeypatch
):
    """local=False, distribute="redistribute" sends the particles only once"""
    import amrex.extensions.ParticleContainer as ext

    def no_scatter(*args, **kwargs):
        raise AssertionError("the 1/N scatter is not needed with redistribute")

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 17
    io = amr.ParallelDescriptor.IOProcessor()
    data = _random_columns(pc, npart) if io else None
    with monkeypatch.context() as m:
        m.setattr(ext, "_scatter_columns", no_scatter)
        pc.add_arrays(data, local=False)
    assert pc.total_number_of_particles() == npart
    columns = _particle_columns(pc, gather=True)
    if io:
        _assert_same_values(columns["x"], data["x"])

    # a root that owns no box splits the particles 1/N first
    nranks = amr.ParallelDescriptor.NProcs()
    if nranks < 2:
        return
    root_rank = nranks - 1
    no_root = amr.DistributionMapping(amr.Vector_int([0] * boxarr.size))
    pc = _make_empty_soa_like(std_geometry, no_root, boxarr)
    data = (
        _random_columns(pc, npart)
        if root_rank == amr.ParallelDescriptor.MyProc()
        else None
    )
    pc.add_arrays(data, local=False, root_rank=root_rank)
    assert pc.total_number_of_particles() == npart


@pytest.mark.parametrize("distribute", ["redistribute", "equally", "none"])
def test_soa_pc_add_arrays_not_local_on_all_ranks(
    std_geometry, distmap, boxarr, distribute
):
    """local=False with particles on several ranks raises on all ranks"""
    nranks = amr.ParallelDescriptor.NProcs()
    if nranks < 2:
        pytest.skip("Requires at least 2 MPI ranks")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    data = _random_columns(pc, 5)
    root = amr.ParallelDescriptor.IOProcessor()

    with pytest.raises(RuntimeError if root else ValueError, match="local=True|failed"):
        pc.add_arrays(data, local=False, distribute=distribute)
    assert pc.total_number_of_particles() == 0

    # no particles on the other ranks: None, empty columns or only scalars
    for other in (None, {}, {"x": np.empty(0)}, {"w": 1.0}):
        pc.add_arrays(data if root else other, local=False, distribute=distribute)
    assert pc.total_number_of_particles() == 4 * 5

    # local=True adds the particles of every rank
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    pc.add_arrays(data if _owns_box(distmap) else None, local=True)
    assert pc.total_number_of_particles() == 5 * len(set(distmap.ProcessorMap()))


def test_soa_pc_add_arrays_distribute_errors(std_geometry, distmap, boxarr):
    """Invalid distribute values and combinations raise on all ranks"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    data = _random_columns(pc, 3) if _owns_box(distmap) else None
    with pytest.raises(ValueError, match="distribute must be one of"):
        pc.add_arrays(data, distribute="scatter")
    with pytest.raises(ValueError, match="needs local=False"):
        pc.add_arrays(data, local=True, distribute="equally")
    # the bool flag of earlier drafts is no option
    with pytest.raises(ValueError, match="distribute='none'"):
        pc.add_arrays(data, local=True, redistribute=False)
    assert pc.total_number_of_particles() == 0


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_add_arrays_root_keeps(std_geometry, distmap, boxarr):
    """local=False, distribute="none": the root keeps all particles"""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 17
    io = amr.ParallelDescriptor.IOProcessor()
    data = _random_columns(pc, npart) if io else None

    # two batches, one redistribute
    pc.add_arrays(data, local=False, distribute="none")
    pc.add_arrays(data, local=False, distribute="none")
    counts = comm.allgather(pc.total_number_of_particles(True, True))
    io_rank = amr.ParallelDescriptor.IOProcessorNumber()
    assert counts == [2 * npart if r == io_rank else 0 for r in range(len(counts))]
    pc.redistribute()
    assert pc.total_number_of_particles() == 2 * npart
    columns = _particle_columns(pc, gather=True)
    if io:
        _assert_same_values(columns["x"], np.concatenate([data["x"], data["x"]]))

    # a root without a box cannot keep them
    nranks = comm.Get_size()
    if nranks < 2:
        return
    root_rank = nranks - 1
    no_root = amr.DistributionMapping(amr.Vector_int([0] * boxarr.size))
    pc = _make_empty_soa_like(std_geometry, no_root, boxarr)
    data = _random_columns(pc, npart) if comm.Get_rank() == root_rank else None
    with pytest.raises(ValueError, match="owns no box"):
        pc.add_arrays(data, local=False, root_rank=root_rank, distribute="none")
    assert pc.total_number_of_particles() == 0


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_add_arrays_comm(std_geometry, distmap, boxarr):
    """A communicator whose ranks do not match AMReX's raises on all ranks"""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    data = _random_columns(pc, 5) if amr.ParallelDescriptor.IOProcessor() else None

    # a duplicate works
    pc.add_arrays(data, local=False, comm=comm.Dup())
    assert pc.total_number_of_particles() == 5

    if comm.Get_size() < 2:
        return
    reversed_comm = comm.Split(0, comm.Get_size() - comm.Get_rank())
    with pytest.raises(ValueError):
        pc.add_arrays(data, local=False, comm=reversed_comm)
    assert pc.total_number_of_particles() == 5

    # local=True with redistribute is collective, too: a rank-local
    # communicator would let ranks without errors enter redistribute alone
    local_data = _random_columns(pc, 3) if _owns_box(distmap) else None
    for bad_comm in (MPI.COMM_SELF, reversed_comm):
        with pytest.raises(ValueError, match="AMReX MPI ranks"):
            pc.add_arrays(local_data, local=True, comm=bad_comm)
    assert pc.total_number_of_particles() == 5


def test_soa_pc_particle_gdb(std_geometry, distmap, boxarr):
    """Particle BoxArray and DistributionMapping per level"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    assert pc.particle_box_array(0).size == boxarr.size
    assert list(pc.particle_distribution_map(0).ProcessorMap()) == list(
        distmap.ProcessorMap()
    )
    for level in (-1, 1):
        with pytest.raises(IndexError):
            pc.particle_box_array(level)
        with pytest.raises(IndexError):
            pc.particle_distribution_map(level)


def test_soa_pc_reserve_particle_ids():
    """Reserve ranges of particle ids"""
    PC = amr.ParticleContainer_pureSoA_11_0_polymorphic

    first = PC.reserve_particle_ids(0)
    assert PC.reserve_particle_ids(0) == first  # nothing reserved
    assert PC.reserve_particle_ids(5) == first
    assert PC.reserve_particle_ids(0) == first + 5

    with pytest.raises(ValueError):
        PC.reserve_particle_ids(-1)
    with pytest.raises(ValueError):
        PC.reserve_particle_ids(2**63 - 1)  # exhausts the id range
    assert PC.reserve_particle_ids(0) == first + 5  # unchanged


def test_soa_pc_add_arrays_manual(std_geometry, distmap, boxarr):
    """Used in docs/source/usage/compute.rst"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 10
    rng = np.random.default_rng(seed=1)
    x, y, z = (rng.uniform(0.0, 1.0, npart) for _ in range(3))

    # Manual: Pure SoA Add Arrays START
    # one array (NumPy, CuPy, dpnp, PODVector, ...) or scalar per component;
    # with local=False, the particles of the I/O rank are added and redistributed
    if amr.ParallelDescriptor.IOProcessor():
        pc.add_arrays(x=x, y=y, z=z, w=1.0, fill_missing=0.0, local=False)
    else:
        pc.add_arrays(local=False)
    # Manual: Pure SoA Add Arrays END

    assert pc.total_number_of_particles() == npart


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_soa_pc_add_df_idcpu(soa_particle_container, std_geometry, distmap, boxarr):
    """Particle ids from an idcpu column or index, cast to uint64"""
    df = soa_particle_container.to_df()
    if df is None:
        pytest.skip("No particles on this MPI rank")
    # to_df: idcpu is a column, the index counts rows
    assert "idcpu" in df.columns and df["idcpu"].dtype == np.uint64
    assert df.index.name is None and list(df.index) == list(range(len(df)))
    ids = df["idcpu"].to_numpy()
    add = dict(local=True, distribute="none")

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    pc.add_df(df.set_index("idcpu"), **add)  # uint64 index
    pc.add_df(df.assign(idcpu=ids.view(np.int64)), **add)  # int64 column
    pc.add_df(df.assign(idcpu=ids.view(np.int64)).set_index("idcpu"), **add)
    _assert_same_values(_particle_columns(pc)["idcpu"], np.tile(ids, 3))

    # an index named idcpu that holds no particle ids
    no_ids = df.drop(columns="idcpu")
    no_ids.index.name = "idcpu"
    with pytest.raises(ValueError, match="invalid particle ids"):
        pc.add_df(no_ids, **add)


def test_soa_pc_add_arrays_more_errors(std_geometry, distmap, boxarr):
    """Helpful errors for common mistakes"""
    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    data = _random_columns(pc, 4)
    add = dict(local=True, distribute="none")

    with pytest.raises(ValueError, match="positional"):
        pc.add_arrays(data=data, **add)
    with pytest.raises(TypeError, match="fill_missing"):
        pc.add_arrays(data, fill_missing="zero", **add)  # even if nothing is missing

    # 0-d arrays are scalars
    zero_d = dict(data, w=np.array(2.5))
    pc.add_arrays(zero_d, **add)
    _assert_same_values(_particle_columns(pc)["w"], np.full(4, 2.5))

    if amr.Config.precision_particles == "SINGLE":
        with pytest.raises(ValueError, match="overflow"):
            pc.add_arrays(dict(data, w=np.full(4, 1e300)), **add)
        with pytest.raises(ValueError, match="overflow"):
            pc.add_arrays(dict(data, w=1e300), **add)
    assert pc.total_number_of_particles(True, True) == 4


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_parallel_descriptor_communicator():
    """AMReX's communicator, for mpi4py"""
    from mpi4py import MPI

    comm = MPI.Comm.f2py(amr.ParallelDescriptor.Communicator())
    assert comm.Get_size() == amr.ParallelDescriptor.NProcs()
    assert comm.Get_rank() == amr.ParallelDescriptor.MyProc()
    io_rank = comm.bcast(
        amr.ParallelDescriptor.IOProcessorNumber(),
        root=amr.ParallelDescriptor.IOProcessorNumber(),
    )
    assert io_rank == amr.ParallelDescriptor.IOProcessorNumber()


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_add_arrays_root_rank(std_geometry, distmap, boxarr):
    """An invalid root rank raises on all ranks"""
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    for root_rank in (-1, amr.ParallelDescriptor.NProcs()):
        with pytest.raises(ValueError, match="root_rank"):
            pc.add_arrays(local=False, root_rank=root_rank)


@pytest.mark.skipif(
    importlib.util.find_spec("dpnp") is None, reason="dpnp is not available"
)
def test_soa_pc_add_arrays_dpnp(std_geometry, distmap, boxarr):
    """dpnp columns, also mixed with NumPy columns"""
    import dpnp as dp

    if not _owns_box(distmap):
        pytest.skip("This MPI rank owns no box")
    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    npart = 9
    data = _random_columns(pc, npart)
    # float32: SYCL devices may not support float64
    data_dp = {
        name: dp.asarray(col.astype(np.float32))
        for name, col in data.items()
        if col.dtype.kind == "f"
    }
    data_dp["i1"] = dp.asarray(data["i1"].astype(np.uint64))  # small values
    data_dp["i2"] = data["i2"]

    pc.add_arrays(data_dp, local=True, distribute="none")
    columns = _particle_columns(pc)
    for name, col in data.items():
        expected = col.astype(np.float32) if col.dtype.kind == "f" else col
        _assert_same_values(columns[name], expected)


def test_soa_pc_undefined():
    """A container without geometry raises instead of crashing"""
    pc = amr.ParticleContainer_pureSoA_3_0_default()
    with pytest.raises(RuntimeError, match="not defined"):
        pc.particle_box_array(0)
    with pytest.raises(RuntimeError, match="not defined"):
        pc.particle_distribution_map(0)
    with pytest.raises(RuntimeError, match="not defined"):
        pc.add_arrays(x=[0.5], y=[0.5], z=[0.5], local=True, distribute="none")


@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_soa_pc_add_arrays_large_count(std_geometry, distmap, boxarr, monkeypatch):
    """Scatters beyond the MPI count limit need MPI-4 large counts"""
    import amrex.extensions.ParticleContainer as ext

    pc = _make_empty_soa_like(std_geometry, distmap, boxarr)
    io = amr.ParallelDescriptor.IOProcessor()
    data = _random_columns(pc, 17) if io else None

    # pretend 17 particles exceed the count limit of an MPI-3 library
    monkeypatch.setattr(ext, "_MAX_MPI_COUNT", 10)
    monkeypatch.setattr(ext, "_mpi_large_count", lambda: False)
    with pytest.raises(ValueError if io else RuntimeError, match="large count|failed"):
        pc.add_arrays(data, local=False, distribute="equally")
    assert pc.total_number_of_particles() == 0

    monkeypatch.setattr(ext, "_mpi_large_count", lambda: True)
    pc.add_arrays(data, local=False, distribute="equally")
    assert pc.total_number_of_particles() == 17
