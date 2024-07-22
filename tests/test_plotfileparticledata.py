import amrex.space3d as amr

if not amr.initialized():
    amr.initialize([])

plt_file_name = "plt0001400"
plt = amr.PlotFileData(plt_file_name)

probDomain = plt.probDomain(0)
probSize = plt.probSize()
probLo = plt.probLo()
probHi = plt.probHi()
cellSize = plt.cellSize(0)
varNames = plt.varNames()
nComp = plt.nComp()
nGrowVect = plt.nGrowVect(0)

print(f"{probDomain=}")
print(f"{probSize=}")
print(f"{probLo=}")
print(f"{probHi=}")
print(f"{cellSize=}")
print(f"{varNames=}")
print(f"{nComp=}")
print(f"{nGrowVect=}")

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
    for p in aos.to_numpy():
        particles.append(p)


for p in particles:
    print(p)
