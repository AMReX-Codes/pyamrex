# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

@pytest.fixture()
def particle_container(std_geometry, distmap, boxarr):
    pc = amrex.ParticleContainer_1_1_2_1(std_geometry, distmap, boxarr)
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
    pass
