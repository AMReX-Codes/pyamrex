# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex


@pytest.fixture()
def Npart():
    return 21


@pytest.fixture(scope="function")
def empty_particle_container(std_geometry, distmap, boxarr):
    pc = amrex.ParticleContainer_1_1_2_1_std(std_geometry, distmap, boxarr)
    return pc


@pytest.fixture(scope="function")
def std_particle():
    myt = amrex.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]
    return myt


@pytest.fixture(scope="function")
def particle_container(Npart, std_geometry, distmap, boxarr, std_real_box):
    pc = amrex.ParticleContainer_1_1_2_1_std(std_geometry, distmap, boxarr)
    myt = amrex.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]

    iseed = 1
    pc.InitRandom(Npart, iseed, myt, False, std_real_box)
    return pc


def test_particleInitType():
    myt = amrex.ParticleInitType_1_1_2_1()
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
    assert pc.NStructReal == amrex.ParticleContainer_1_1_2_1_std.NStructReal == 1
    assert pc.NStructInt == amrex.ParticleContainer_1_1_2_1_std.NStructInt == 1
    assert pc.NArrayReal == amrex.ParticleContainer_1_1_2_1_std.NArrayReal == 2
    assert pc.NArrayInt == amrex.ParticleContainer_1_1_2_1_std.NArrayInt == 1
    assert (
        pc.NumberOfParticlesAtLevel(0) == np.sum(pc.NumberOfParticlesInGrid(0)) == Npart
    )


def test_pc_init():
    pc = amrex.ParticleContainer_1_1_2_1_std()

    print("bytespread", pc.ByteSpread())
    print("capacity", pc.PrintCapacity())
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.NumberOfParticlesAtLevel(0) == 0

    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0, 0, 0, 1, 1, 1)
    coord_int = 1  # RZ
    periodicity = [0, 0, 1]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    print("-------------------------")
    print("define particle container")
    pc.Define(gm, dm, ba)
    assert pc.OK()
    assert pc.NStructReal == amrex.ParticleContainer_1_1_2_1_std.NStructReal == 1
    assert pc.NStructInt == amrex.ParticleContainer_1_1_2_1_std.NStructInt == 1
    assert pc.NArrayReal == amrex.ParticleContainer_1_1_2_1_std.NArrayReal == 2
    assert pc.NArrayInt == amrex.ParticleContainer_1_1_2_1_std.NArrayInt == 1

    print("bytespread", pc.ByteSpread())
    print("capacity", pc.PrintCapacity())
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.TotalNumberOfParticles() == pc.NumberOfParticlesAtLevel(0) == 0
    assert pc.OK()

    print("---------------------------")
    print("add a particle to each grid")
    Npart_grid = 1
    iseed = 1
    myt = amrex.ParticleInitType_1_1_2_1()
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

    print("Iterate particle boxes & set values")
    lvl = 0
    for pti in amrex.ParIter_1_1_2_1_std(pc, level=lvl):
        print("...")
        assert pti.num_particles == 1
        assert pti.num_real_particles == 1
        assert pti.num_neighbor_particles == 0
        assert pti.level == lvl
        print(pti.pair_index)
        print(pti.geom(level=lvl))

        aos = pti.aos()
        aos_arr = np.array(aos, copy=False)
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
    for pti in amrex.ParConstIter_1_1_2_1_std(pc, level=lvl):
        assert pti.num_particles == 1
        assert pti.num_real_particles == 1
        assert pti.num_neighbor_particles == 0
        assert pti.level == lvl

        aos = pti.aos()
        aos_arr = np.array(aos, copy=False)
        assert aos[0].x == 0.30
        assert aos[0].y == 0.35
        assert aos[0].z == 0.40
        assert aos_arr[0]["z"] == 0.40

        soa = pti.soa()
        real_arrays = soa.GetRealData()
        int_arrays = soa.GetIntData()
        print(real_arrays[0])
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
        aos_arr = np.array(aos, copy=False)
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
                assert np.allclose(np.array(arr), np.array(ra1[ii]))

            print("soa int test")
            iarr_np = np.array(int_arrays[0], copy=False)
            iarr_np[0] = -3
            ia1 = pt.GetStructOfArrays().GetIntData()
            ia1_np = np.array(ia1[0], copy=False)
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
        real_arrays = pt.GetStructOfArrays().GetRealData()
        sum_1 += np.sum(real_arrays[1])
    print(sum_1)
    ncells = std_geometry.Domain().numPts()
    print("ncells from box", ncells)
    print("NumberOfParticles", pc.NumberOfParticlesAtLevel(0))
    assert pc.TotalNumberOfParticles() == pc.NumberOfParticlesAtLevel(0) == ncells
    print("npts * real_1", ncells * std_particle.real_array_data[1])
    assert ncells * std_particle.real_array_data[1] == sum_1
