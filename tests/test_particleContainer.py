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
    pc = amr.ParticleContainer_1_1_2_1_default(std_geometry, distmap, boxarr)
    return pc


@pytest.fixture(scope="function")
def std_particle():
    myt = amr.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]
    return myt


@pytest.fixture(scope="function")
def particle_container(Npart, std_geometry, distmap, boxarr, std_real_box):
    pc = amr.ParticleContainer_1_1_2_1_default(std_geometry, distmap, boxarr)
    myt = amr.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]

    iseed = 1
    pc.InitRandom(Npart, iseed, myt, False, std_real_box)
    return pc


@pytest.fixture(scope="function")
def soa_particle_container(Npart, std_geometry, distmap, boxarr, std_real_box):
    pc = amr.ParticleContainer_pureSoA_8_0_default(std_geometry, distmap, boxarr)
    myt = amr.ParticleInitType_pureSoA_8_0()
    myt.real_array_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    myt.int_array_data = []

    iseed = 1
    pc.InitRandom(Npart, iseed, myt, False, std_real_box)
    return pc


def test_particleInitType():
    myt = amr.ParticleInitType_1_1_2_1()
    print(myt.real_struct_data)
    print(myt.int_struct_data)
    print(myt.real_array_data)
    print(myt.int_array_data)

    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]

    assert np.allclose(myt.real_struct_data, [0.5])
    assert np.allclose(myt.int_struct_data, [5])
    assert np.allclose(myt.real_array_data, [0.5, 0.2])
    assert np.allclose(myt.int_array_data, [1])


def test_n_particles(particle_container, Npart):
    pc = particle_container
    assert pc.OK()
    assert pc.NStructReal == amr.ParticleContainer_1_1_2_1_default.NStructReal == 1
    assert pc.NStructInt == amr.ParticleContainer_1_1_2_1_default.NStructInt == 1
    assert pc.NArrayReal == amr.ParticleContainer_1_1_2_1_default.NArrayReal == 2
    assert pc.NArrayInt == amr.ParticleContainer_1_1_2_1_default.NArrayInt == 1
    assert (
        pc.NumberOfParticlesAtLevel(0) == np.sum(pc.NumberOfParticlesInGrid(0)) == Npart
    )


def test_pc_init():
    pc = amr.ParticleContainer_1_1_2_1_default()

    print("bytespread", pc.ByteSpread())
    print("capacity", pc.PrintCapacity())
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.NumberOfParticlesAtLevel(0) == 0

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
    assert pc.NStructReal == amr.ParticleContainer_1_1_2_1_default.NStructReal == 1
    assert pc.NStructInt == amr.ParticleContainer_1_1_2_1_default.NStructInt == 1
    assert pc.NArrayReal == amr.ParticleContainer_1_1_2_1_default.NArrayReal == 2
    assert pc.NArrayInt == amr.ParticleContainer_1_1_2_1_default.NArrayInt == 1

    print("bytespread", pc.ByteSpread())
    print("capacity", pc.PrintCapacity())
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.TotalNumberOfParticles() == pc.NumberOfParticlesAtLevel(0) == 0
    assert pc.OK()

    print("---------------------------")
    print("add a particle to each grid")
    Npart_grid = 1
    iseed = 1
    myt = amr.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]
    pc.InitRandomPerBox(Npart_grid, iseed, myt)
    ngrid = ba.size
    npart = Npart_grid * ngrid

    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.TotalNumberOfParticles() == pc.NumberOfParticlesAtLevel(0) == npart
    assert pc.OK()

    print(f"Finest level = {pc.finest_level}")

    print("Iterate particle boxes & set values")
    # lvl = 0
    for lvl in range(pc.finest_level + 1):
        print(f"at level {lvl}:")
        for pti in pc.iterator(pc, level=lvl):
            print("...")
            assert pti.num_particles == 1
            assert pti.num_real_particles == 1
            assert pti.num_neighbor_particles == 0
            assert pti.level == lvl
            print(pti.pair_index)
            print(pti.geom(level=lvl))

            aos = pti.aos()
            aos_arr = aos.to_numpy()
            aos_arr[0]["x"] = 0.30
            aos_arr[0]["y"] = 0.35
            aos_arr[0]["z"] = 0.40

            # TODO: this seems to write into a copy of the data
            soa = pti.soa()
            real_arrays = soa.GetRealData()
            int_arrays = soa.GetIntData()
            real_arrays[0] = [0.55]
            real_arrays[1] = [0.22]
            int_arrays[0] = [2]

            assert np.allclose(real_arrays[0], np.array([0.55]))
            assert np.allclose(real_arrays[1], np.array([0.22]))
            assert np.allclose(int_arrays[0], np.array([2]))

    # read-only
    for lvl in range(pc.finest_level + 1):
        for pti in pc.const_iterator(pc, level=lvl):
            assert pti.num_particles == 1
            assert pti.num_real_particles == 1
            assert pti.num_neighbor_particles == 0
            assert pti.level == lvl

            aos = pti.aos()
            aos_arr = aos.to_numpy()
            assert aos[0].x == 0.30
            assert aos[0].y == 0.35
            assert aos[0].z == 0.40
            assert aos_arr[0]["z"] == 0.40

            soa = pti.soa()
            real_arrays = soa.GetRealData()
            int_arrays = soa.GetIntData()
            print(real_arrays[0])
            print(int_arrays[0])
            # TODO: this does not work yet and is still the original data
            # assert np.allclose(real_arrays[0], np.array([0.55]))
            # assert np.allclose(real_arrays[1], np.array([0.22]))
            # assert np.allclose(int_arrays[0], np.array([2]))


def test_particle_init(Npart, particle_container):
    pc = particle_container
    assert (
        pc.NumberOfParticlesAtLevel(0) == np.sum(pc.NumberOfParticlesInGrid(0)) == Npart
    )

    # pc.resizeData()
    print(pc.numLocalTilesAtLevel(0))
    lev = pc.GetParticles(0)
    print(len(lev.items()))
    assert pc.numLocalTilesAtLevel(0) == len(lev.items())
    for tile_ind, pt in lev.items():
        print("tile", tile_ind)
        real_arrays = pt.GetStructOfArrays().GetRealData()
        int_arrays = pt.GetStructOfArrays().GetIntData()
        aos = pt.GetArrayOfStructs()
        aos_arr = aos.to_numpy()
        if len(real_arrays) > 0:
            assert np.isclose(real_arrays[0][0], 0.5) and np.isclose(
                real_arrays[1][0], 0.2
            )
            assert isinstance(int_arrays[0][0], int)
            assert int_arrays[0][0] == 1
            assert isinstance(aos_arr[0]["rdata_0"], np.floating)
            assert isinstance(aos_arr[0]["idata_0"], np.integer)
            assert np.isclose(aos_arr[0]["rdata_0"], 0.5) and aos_arr[0]["idata_0"] == 5

            aos_arr[0]["idata_0"] = 2
            aos1 = pt.GetArrayOfStructs()
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

            ra1 = pt.GetStructOfArrays().GetRealData()
            print(real_arrays)
            print(ra1)
            for ii, arr in enumerate(real_arrays):
                assert np.allclose(arr.to_numpy(), ra1[ii].to_numpy())

            print("soa int test")
            iarr_np = int_arrays[0].to_numpy()
            iarr_np[0] = -3
            ia1 = pt.GetStructOfArrays().GetIntData()
            ia1_np = ia1[0].to_numpy()
            print(iarr_np)
            print(ia1_np)
            assert np.allclose(iarr_np, ia1_np)

    print(
        "---- is the particle tile recording changes or passed by reference? --------"
    )
    lev1 = pc.GetParticles(0)
    for tile_ind, pt in lev1.items():
        print("tile", tile_ind)
        real_arrays = pt.GetStructOfArrays().GetRealData()
        int_arrays = pt.GetStructOfArrays().GetIntData()
        aos = pt.GetArrayOfStructs()
        print(aos[0])
        assert aos[0].get_idata(0) == 2
        assert real_arrays[1][0] == -1.2
        assert int_arrays[0][0] == -3


def test_per_cell(empty_particle_container, std_geometry, std_particle):
    pc = empty_particle_container
    pc.InitOnePerCell(0.5, 0.5, 0.5, std_particle)
    assert pc.OK()

    lev = pc.GetParticles(0)
    assert pc.numLocalTilesAtLevel(0) == len(lev.items())

    sum_1 = 0
    for tile_ind, pt in lev.items():
        print("tile", tile_ind)
        real_arrays = pt.GetStructOfArrays().GetRealData()
        sum_1 += np.sum(real_arrays[1])
    print(sum_1)
    ncells = std_geometry.Domain().numPts()
    print("ncells from box", ncells)
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.TotalNumberOfParticles() == pc.NumberOfParticlesAtLevel(0) == ncells
    print("npts * real_1", ncells * std_particle.real_array_data[1])
    assert ncells * std_particle.real_array_data[1] == sum_1


def test_soa_pc_numpy(soa_particle_container, Npart):
    """Used in docs/source/usage/compute.rst"""
    pc = soa_particle_container

    class Config:
        have_gpu = False

    # Manual: Pure SoA Compute PC START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # iterate over every mesh-refinement levels (no MR: lev=0)
    for lvl in range(pc.finest_level + 1):
        # get every local chunk of particles
        for pti in pc.iterator(pc, level=lvl):
            # additional compile-time and runtime attributes in SoA format
            soa = pti.soa().to_cupy() if Config.have_gpu else pti.soa().to_numpy()

            # notes:
            # Only the next lines are the "HOT LOOP" of the computation.
            # For efficiency, use numpy array operation for speed.

            # write to all particles in the chunk
            # note: careful, if you change particle positions, you need to
            #       redistribute particles before continuing the simulation step
            print(soa.real)
            soa.real[0][()] = 0.30  # x
            soa.real[1][()] = 0.35  # y
            soa.real[2][()] = 0.40  # z

            # all other real attributes
            for soa_real in soa.real[3:]:
                soa_real[()] = 42.0

            # all int attributes
            for soa_int in soa.int:
                soa_int[()] = 12
    # Manual: Pure SoA Compute PC END


def test_pc_numpy(particle_container, Npart):
    """Used in docs/source/usage/compute.rst"""
    pc = particle_container

    class Config:
        have_gpu = False

    # Manual: Legacy Compute PC START
    # code-specific getter function, e.g.:
    # pc = sim.get_particles()
    # Config = sim.extension.Config

    # iterate over every mesh-refinement levels (no MR: lev=0)
    for lvl in range(pc.finest_level + 1):
        # get every local chunk of particles
        for pti in pc.iterator(pc, level=lvl):
            # default layout: AoS with positions and cpuid
            # note: not part of the new PureSoA particle container layout
            aos = (
                pti.aos().to_numpy(copy=True)
                if Config.have_gpu
                else pti.aos().to_numpy()
            )

            # additional compile-time and runtime attributes in SoA format
            soa = pti.soa().to_cupy() if Config.have_gpu else pti.soa().to_numpy()

            # notes:
            # Only the next lines are the "HOT LOOP" of the computation.
            # For efficiency, use numpy array operation for speed on CPUs.
            # For GPUs use .to_cupy() above and compute with cupy or numba.

            # print all particle ids in the chunk
            print(aos[()]["cpuid"])

            # write to all particles in the chunk
            aos[()]["x"] = 0.30
            aos[()]["y"] = 0.35
            aos[()]["z"] = 0.40

            for soa_real in soa.real:
                soa_real[()] = 42.0

            for soa_int in soa.int:
                soa_int[()] = 12
    # Manual: Legacy Compute PC END


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not available"
)
def test_pc_df(particle_container, Npart):
    pc = particle_container
    print(f"pc={pc}")
    df = pc.to_df()
    print(df.columns)
    print(df)


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
@pytest.mark.skipif(not amr.Config.have_mpi, reason="Requires AMReX_MPI=ON")
def test_pc_df_mpi(particle_container, Npart):
    pc = particle_container
    print(f"pc={pc}")
    df = pc.to_df(local=False)
    if df is not None:
        # only rank 0
        print(df.columns)
        print(df)
