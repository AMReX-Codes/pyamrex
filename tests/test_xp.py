# -*- coding: utf-8 -*-

import subprocess
import sys

import pytest

import amrex.space3d as amr
from amrex.extensions.dlpack_helpers import xp_module_name

# CuPy/dpnp are optional dependencies, so a GPU build without them installed
# must still import -- only touching amr.xp may then fail.
XP_NAME = xp_module_name(amr)


def test_xp_matches_build():
    """amr.xp is the array namespace this build was compiled for."""
    expected = {
        None: "numpy",
        "CUDA": "cupy",
        "HIP": "cupy",
        "SYCL": "dpnp",
    }[amr.Config.gpu_backend]
    assert XP_NAME == expected

    pytest.importorskip(XP_NAME, reason=f"optional dependency {XP_NAME} not installed")
    assert amr.xp.__name__ == expected


def test_xp_is_cached():
    """xp resolves once, then is a plain module global."""
    pytest.importorskip(XP_NAME, reason=f"optional dependency {XP_NAME} not installed")
    assert amr.xp is amr.xp
    assert vars(amr)["xp"] is amr.xp


def test_xp_listed_in_dir():
    """xp is discoverable (tab completion, stub generation) before first use.

    Run in a subprocess, so xp is not yet resolved by other tests.
    """
    code = (
        "import amrex.space3d as amr; "
        "assert 'xp' not in vars(amr), 'xp resolved at import'; "
        "assert 'xp' in dir(amr), 'xp not in dir()'; "
        "print('listed')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "listed" in out.stdout


def test_no_module_getattr():
    """No module-level PEP 562 __getattr__.

    In the generated .pyi stubs it would tell type checkers that any
    attribute exists, hiding typos, and break the Sphinx docs build, which
    executes the stubs.
    """
    assert "__getattr__" not in vars(amr)


def test_missing_attribute_raises():
    """Lazy xp must not swallow genuine attribute errors."""
    with pytest.raises(AttributeError, match="no attribute"):
        amr.this_does_not_exist


def test_import_does_not_pull_in_gpu_array_library():
    """Importing pyAMReX must not import CuPy or dpnp.

    They are optional dependencies. Run in a subprocess so this holds
    regardless of what the rest of the test session has already imported.
    """
    code = (
        "import sys; import amrex.space3d as amr; "
        "assert 'cupy' not in sys.modules, 'cupy imported by import amrex'; "
        "assert 'dpnp' not in sys.modules, 'dpnp imported by import amrex'; "
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
