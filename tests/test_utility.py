# -*- coding: utf-8 -*-

import amrex.space3d as amr


def test_concatenate():
    pltname = amr.concatenate("plt", 1000, 5)
    assert pltname == "plt01000"


def test_print():
    print("hello from everyone")
    amr.Print("byeee from IO processor")


def test_parallel_reductions():
    """ParallelDescriptor reductions (single rank: identity; the values
    are returned, not modified in place)"""
    pd = amr.ParallelDescriptor
    nprocs = pd.NProcs()

    assert pd.ReduceRealSum(1.5) == 1.5 * nprocs
    assert pd.ReduceRealMin(2.5) == 2.5
    assert pd.ReduceRealMax(3.5) == 3.5
    assert pd.ReduceRealMin([1.0, 2.0]) == [1.0, 2.0]
    assert pd.ReduceRealMax([1.0, 2.0]) == [1.0, 2.0]
    assert pd.ReduceRealSum([1.0, 2.0]) == [1.0 * nprocs, 2.0 * nprocs]

    assert pd.ReduceIntSum(2) == 2 * nprocs
    assert pd.ReduceIntMin(2) == 2
    assert pd.ReduceIntMax(2) == 2
    assert pd.ReduceLongSum(2) == 2 * nprocs
    assert pd.ReduceLongMin(2) == 2
    assert pd.ReduceLongMax(2) == 2
    assert pd.ReduceBoolAnd(True)
    assert pd.ReduceBoolOr(False) is False

    pd.Barrier()
