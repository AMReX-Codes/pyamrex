


def test_ptile_data():
    
    ptd = amrex.ParticleTileData_1_1_2_1()
    sp = amrex.Particle_3_2()
    # sp.setPos([1,2,3])
    sp.x = 1
    # ptd.setSuperParticle(sp,0)
    # ptd.getSuperParticle(0)
    print(ptd.m_size)
    print(ptd.m_num_runtime_real)
    print(ptd.m_num_runtime_int)
    pass


def test_init_ptile():
    pt = amrex.ParticleTile_1_1_2_1()

    print(pt.empty())
    print(pt.size())

    pt.define(4,3)

    print(pt.empty())
    print(pt.size())
    pass

def test_ptile_funs():

    pt = amrex.ParticleTile_1_1_2_1()
    print('num particles', pt.numParticles())
    print('num real particles', pt.numRealParticles())
    print('num neighbor particles', pt.numNeighborParticles())
    print('num totalparticles', pt.numTotalParticles())
    print('num Neighbors', pt.getNumNeighbors())
    pt.setNumNeighbors(3)
    print('num Neighbors', pt.getNumNeighbors())
    pt.resize(5)
    print('tile is empty?', pt.empty())
    print('tile size', pt.size())


def test_ptile_pushback_ptiledata():

    pt = amrex.ParticleTile_1_1_2_1()
    p = amrex.Particle_1_1(1.,2.,3,4.,5)
    # p.set_rdata([4.])
    # p.set_idata([5])
    sp = amrex.Particle_3_2(5.,6.,7.,8.,9.,10.,11,12)
    # sp.set_rdata([8.,9.,10.])
    # sp.set_idata([11,12])
    pt.push_back(p)
    pt.push_back(sp)

    print('num particles', pt.numParticles())
    print('num real particles', pt.numRealParticles())
    print('num neighbor particles', pt.numNeighborParticles())
    print('num totalparticles', pt.numTotalParticles())
    print('num Neighbors', pt.getNumNeighbors())
    print('num Neighbors', pt.getNumNeighbors())
    print('tile is empty?', pt.empty())
    print('tile size', pt.size())

    td = pt.getParticleTileData()
    print('particle tile data size', td.m_size)

    # # for ii in range(pt.size()):
    for ii in range(td.m_size):
        print('particle',ii)
        print(td[ii])
    