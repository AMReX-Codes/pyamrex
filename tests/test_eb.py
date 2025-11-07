# -*- coding: utf-8 -*-

import pytest

import amrex.space3d as amr


@pytest.mark.skipif(not amr.Config.have_eb, reason="Requires -DAMReX_EB=ON")
def test_makeEBFab():
    pass

    # TODO:
    # EB2_Build(...)
    # makeEBFabFactory(...)
