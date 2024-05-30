# -*- coding: utf-8 -*-

import numpy as np
import pytest
import os
from pathlib import Path

import amrex.space3d as amr

def test_plotfiledata_read():
    parent_path = Path(os.path.abspath(__file__)).parent
    plt = amr.PlotFileData(str(parent_path / "projz04000"))
    
    assert(plt.spaceDim() == 3)
    assert(plt.time() == 56649980960510.086)
    assert(plt.finestLevel() == 0)
    assert(plt.refRatio(0) == 0)
    assert(plt.coordSys() == amr.CoordSys.cartesian)
    
    probDomain = plt.probDomain(0)
    probSize = plt.probSize()
    probLo = plt.probLo()
    probHi = plt.probHi()
    cellSize = plt.cellSize(0)
    varNames = plt.varNames()
    nComp = plt.nComp()
    nGrowVect = plt.nGrowVect(0)

    #assert(probDomain == amr.Box([0,0,0],[1024-1,256-1,1-1])) # TypeError: unhashable type: 'instancemethod'.
    print(probDomain)

    assert(probSize == [2.4688e+21, 6.172e+20, 6.172e+20])
    assert(probLo == [0.0, 0.0, 0.0])
    assert(probHi == [2.4688e+21, 6.172e+20, 6.172e+20])
    assert(cellSize == [2.4109375e+18, 2.4109375e+18, 6.172e+20])

    #assert(varNames == amr.Vector_string(['nH', 'nH_cloud', 'nH_wind']))
    print(varNames)
    
    assert(nComp == 3)
    assert(nGrowVect == amr.IntVect(0,0,0))
    
    for compname in varNames:
        mfab_comp = plt.get(0, compname)
        nboxes = 0
        
        for mfi in mfab_comp:
            bx = mfi.tilebox()
            marr = mfab_comp.array(mfi)            
            # numpy/cupy representation: non-copying view, including the
            # guard/ghost region
            marr_xp = marr.to_xp()
            assert(marr_xp.shape == (1024, 256, 1, 1))
            nboxes += 1
            
        assert(nboxes == 1)
        #print(mfab_comp.min())
        #print(mfab_comp.max())
        
