# -*- coding: utf-8 -*-

import random
import shutil
from dataclasses import dataclass

import numpy as np
import pytest

import amrex.space3d as amr

# Particle container constructor -- depending on if gpus are available use the
# GPU-enabled version. This uses the FHDeX Particle Container for the test
if amr.Config.have_gpu:
    PCC = amr.ParticleContainer_16_4_0_0_managed
else:
    PCC = amr.ParticleContainer_16_4_0_0_default


@dataclass
class Particle:
    """
    Helper class to work with particle data
    """

    x: float
    y: float
    z: float

    idx: int

    def to_soa(self, aos_numpy):
        """Fill amr particle SoA with x, y, z, idx data"""
        aos_numpy["x"] = self.x
        aos_numpy["y"] = self.y
        aos_numpy["z"] = self.z
        aos_numpy["idata_0"] = self.idx


def generate_test_particles(n_part):
    """
    Returns a list of test particles scattered throught the domain
    """
    particles = list()

    def generator():
        return 1 - 2 * random.random()

    for i in range(n_part):
        particles.append(Particle(x=generator(), y=generator(), z=generator(), idx=i))

    return particles


def particle_container(Rpart, std_geometry, distmap, boxarr, std_real_box):
    """
    Generate a fresh particle container, containing copies from Rpart
    """
    pc = PCC(std_geometry, distmap, boxarr)

    iseed = 1
    myt = amr.ParticleInitType_16_4_0_0()
    pc.init_random(len(Rpart), iseed, myt, False, std_real_box)

    particles_tile_ct = 0
    # assign some values to runtime components
    for lvl in range(pc.finest_level + 1):
        for pti in pc.iterator(level=lvl):
            aos = pti.aos()
            aos_numpy = aos.to_numpy(copy=False)
            for i, p in enumerate(aos_numpy):
                Rpart[i + particles_tile_ct].to_soa(p)
            particles_tile_ct += len(aos_numpy)

    pc.redistribute()
    return pc


def check_particles_container(pc, reference_particles):
    """
    Checks the contents of `pc` against `reference_particles`
    """
    for lvl in range(pc.finest_level + 1):
        for i, pti in enumerate(pc.iterator(level=lvl)):
            aos = pti.aos()
            for p in aos.to_numpy(copy=True):
                ref = reference_particles[p["idata_0"]]
                assert Particle(x=p["x"], y=p["y"], z=p["z"], idx=p["idata_0"]) == ref


def write_test_plotfile(filename, reference_part):
    """
    Write single-level plotfile (in order to read it back in).
    """
    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])

    ba = amr.BoxArray(domain_box)
    dm = amr.DistributionMapping(ba, 1)
    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(np.pi)

    time = 1.0
    level_step = 200
    var_names = amr.Vector_string(["density"])
    amr.write_single_level_plotfile(filename, mf, var_names, geom, time, level_step)

    pc = particle_container(reference_part, geom, dm, ba, real_box)
    pc.write_plotfile(filename, "particles")


def load_test_plotfile_particle_container(plot_file_name):
    """
    Load signle-level plotfile and return particle container
    """
    plt = amr.PlotFileData(plot_file_name)

    probDomain = plt.probDomain(0)
    probLo = plt.probLo()
    probHi = plt.probHi()
    domain_box = amr.Box(probDomain.small_end, probDomain.big_end)
    real_box = amr.RealBox(probLo, probHi)
    std_geometry = amr.Geometry(domain_box, real_box, plt.coordSys(), [0, 0, 0])

    pc = amr.ParticleContainer_16_4_0_0_default(
        std_geometry,
        plt.DistributionMap(plt.finestLevel()),
        plt.boxArray(plt.finestLevel()),
    )
    pc.restart(plot_file_name, "particles")

    return pc


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_plotfile_particle_data_read():
    """
    Generate and then read a plot file containing particle data. Checks that
    the particle data matches the original particle list.
    """
    # seed RNG to make test reproducible -- comment out this line to generate new
    # random particle positions every time.
    random.seed(1)

    plt_file_name = "plt_test"
    # Reference particle lists
    n_part = 15
    reference_part = generate_test_particles(n_part)
    # Write a test plotfile containing the reference particles in a paritcle
    # container
    write_test_plotfile(plt_file_name, reference_part)
    # Load the particle container from the test plot file
    pc = load_test_plotfile_particle_container(plt_file_name)
    # Check that the particles in the loaded particle container match the
    # original particle list
    check_particles_container(pc, reference_part)

    # clean up after yourself
    shutil.rmtree(plt_file_name)
