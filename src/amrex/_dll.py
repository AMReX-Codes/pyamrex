# -*- coding: utf-8 -*-

import os
from pathlib import Path

# Keep the os.add_dll_directory() handles alive so the registered search paths
# stay active for the whole interpreter lifetime.
_DLL_DIRECTORY_HANDLES = {}


def _iter_windows_dll_directories(package_file):
    package_dir = Path(package_file).resolve().parent
    package_root = package_dir.parent
    seen = set()

    # Prefer package-local directories first so vendored wheel layouts can load
    # dependent DLLs even when they never appear on PATH.
    candidates = (
        package_dir,
        package_dir / ".libs",
        package_root,
        package_root / ".libs",
    )

    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in seen:
                seen.add(candidate_str)
                yield candidate_str

    # PATH is still needed for developer/source installs that link against DLLs
    # provided by an existing AMReX or application environment.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue

        path = Path(os.path.expandvars(os.path.expanduser(entry)))
        try:
            resolved = path.resolve()
        except OSError:
            continue

        if not resolved.exists():
            continue

        resolved_str = str(resolved)
        if resolved_str not in seen:
            seen.add(resolved_str)
            yield resolved_str


def add_windows_dll_directories(package_file):
    """Keep DLL directories registered for the lifetime of the process.

    Python 3.8+ requires dependent DLL search paths to be registered
    explicitly before importing the extension module.

    Refs.:
    - https://github.com/python/cpython/issues/80266
    - https://docs.python.org/3.8/library/os.html#os.add_dll_directory
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    package_dir = str(Path(package_file).resolve().parent)
    if package_dir in _DLL_DIRECTORY_HANDLES:
        return

    handles = []
    for dll_dir in _iter_windows_dll_directories(package_file):
        try:
            handles.append(os.add_dll_directory(dll_dir))
        except OSError:
            continue

    _DLL_DIRECTORY_HANDLES[package_dir] = handles
