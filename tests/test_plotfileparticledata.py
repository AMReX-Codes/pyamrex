from dataclasses import dataclass
import amrex.space3d as amr

if not amr.initialized():
    amr.initialize([])

plt_file_name = "plt0000600"

@dataclass
class Particle:
    x: float
    y: float
    z: float

    ib_id: int
    idx: int


def load_data(plot_file_name):
    plt = amr.PlotFileData(plt_file_name)

    probDomain   = plt.probDomain(0)
    probLo       = plt.probLo()
    probHi       = plt.probHi()
    domain_box   = amr.Box(probDomain.small_end, probDomain.big_end)
    real_box     = amr.RealBox(probLo, probHi)
    std_geometry = amr.Geometry(domain_box, real_box, plt.coordSys(), [0, 0, 0])

    pc = amr.ParticleContainer_16_4_0_0_default(
        std_geometry,
        plt.DistributionMap(plt.finestLevel()),
        plt.boxArray(plt.finestLevel())
    )
    pc.restart(plt_file_name, "immbdy_markers")

    particles = list()
    for pti in pc.iterator(pc, level=plt.finestLevel()):
        aos = pti.aos()
        for p in aos.to_numpy(copy=True):
            particles.append(
                Particle(x=p[0], y=p[1], z=p[2], ib_id=p[-1], idx=p[-2])
            )

    return particles


particles = load_data(plt_file_name)
for p in particles:
    print(p)
    print(p.x)
