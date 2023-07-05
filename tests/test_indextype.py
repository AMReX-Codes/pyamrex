# -*- coding: utf-8 -*-

import pytest

import amrex.space3d as amr


@pytest.mark.skipif(amr.Config.spacedim != 1, reason="Requires AMREX_SPACEDIM = 1")
def test_indextype_1d():
    obj = amr.IndexType(amr.IndexType.CellIndex.NODE)
    assert obj.node_centered()
    assert not obj.cell_centered()
    with pytest.raises(IndexError):
        obj[-2]
    with pytest.raises(IndexError):
        obj[1]


@pytest.mark.skipif(amr.Config.spacedim != 2, reason="Requires AMREX_SPACEDIM = 2")
def test_indextype_2d():
    obj = amr.IndexType(amr.IndexType.CellIndex.NODE, amr.IndexType.CellIndex.CELL)
    assert obj.node_centered(0)
    assert obj.cell_centered(1)
    assert obj.node_centered(-2)
    assert obj.cell_centered(-1)

    with pytest.raises(IndexError):
        obj[-3]
    with pytest.raises(IndexError):
        obj[2]


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_indextype_3d():
    obj = amr.IndexType(
        amr.IndexType.CellIndex.NODE,
        amr.IndexType.CellIndex.CELL,
        amr.IndexType.CellIndex.NODE,
    )

    # Check indexing
    assert obj.node_centered(0)
    assert obj.cell_centered(1)
    assert obj.node_centered(2)
    assert obj.node_centered(-3)
    assert obj.cell_centered(-2)
    assert obj.node_centered(-1)
    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

    # Check methods
    obj.set(1)
    assert obj.node_centered()
    obj.unset(1)
    assert not obj.node_centered()


def test_indextype_static():
    cell = amr.IndexType.cell_type()
    for i in range(amr.Config.spacedim):
        assert not cell.test(i)

    node = amr.IndexType.node_type()
    for i in range(amr.Config.spacedim):
        assert node[i]

    assert cell == amr.IndexType.cell_type()
    assert node == amr.IndexType.node_type()
    assert cell < node


def test_indextype_conversions():
    node = amr.IndexType.node_type()
    assert node.ix_type() == amr.IntVect(1)
    assert node.to_IntVect() == amr.IntVect(1)
