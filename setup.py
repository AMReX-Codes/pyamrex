#!/usr/bin/env python3
#
# Copyright 2021-2023 The AMReX Community
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
import os
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext


class CopyPreBuild(build):
    def initialize_options(self):
        build.initialize_options(self)
        # We just overwrite this because the default "build" (and "build/lib")
        # clashes with directories many developers have in their source trees;
        # this can create confusing results with "pip install .", which clones
        # the whole source tree by default
        self.build_base = os.path.join("_tmppythonbuild", "amrex")

    def run(self):
        # remove existing build directory
        #   by default, this stays around. We want to make sure generated
        #   files like amrex_*d_pybind.*.(so|pyd) are always only the
        #   ones we want to package and not ones from an earlier wheel's stage
        if os.path.exists(self.build_base):
            shutil.rmtree(self.build_base)

        # call superclass
        build.run(self)

        # copy Python module artifacts and sources
        dst_path = os.path.join(self.build_lib, "amrex")
        shutil.copytree(PYAMREX_libdir, dst_path, dirs_exist_ok=True)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        from packaging.version import parse

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake 3.24.0+ must be installed to build the following "
                + "extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = parse(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
        if cmake_version < parse("3.24.0"):
            raise RuntimeError("CMake >= 3.24.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        r_dim = re.search(r"amrex_(1|2|3)d", ext.name)
        dims = r_dim.group(1).upper()

        pyv = sys.version_info
        cmake_args = [
            # Python: use the calling interpreter in CMake
            # https://cmake.org/cmake/help/latest/module/FindPython.html#hints
            # https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-version-selection
            f"-DPython_ROOT_DIR={sys.prefix}",
            f"-DPython_FIND_VERSION={pyv.major}.{pyv.minor}.{pyv.micro}",
            "-DPython_FIND_VERSION_EXACT=TRUE",
            "-DPython_FIND_STRATEGY=LOCATION",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.join(extdir, "amrex"),
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DCMAKE_PYTHON_OUTPUT_DIRECTORY=" + extdir,
            "-DAMReX_SPACEDIM=" + dims,
            ## variants
            "-DAMReX_OMP=" + AMReX_OMP,
            "-DAMReX_GPU_BACKEND=" + AMReX_GPU_BACKEND,
            "-DAMReX_MPI:BOOL=" + AMReX_MPI,
            "-DAMReX_PRECISION=" + AMReX_PRECISION,
            #'-DAMReX_PARTICLES_PRECISION=' + AMReX_PARTICLES_PRECISION,
            "-DpyAMReX_CCACHE=" + PYAMREX_CCACHE,
            "-DpyAMReX_IPO=" + PYAMREX_IPO,
            ## dependency control (developers & package managers)
            "-DpyAMReX_amrex_internal=" + AMReX_internal,
            "-DpyAMReX_pybind11_internal=" + pybind11_internal,
            # PEP-440 conformant version from package
            "-DpyAMReX_VERSION_INFO=" + self.distribution.get_version(),
            #        see PICSAR and openPMD below
            ## static/shared libs
            "-DBUILD_SHARED_LIBS:BOOL=" + BUILD_SHARED_LIBS,
            ## Unix: rpath to current dir when packaged
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF",
            # Windows: has no RPath concept, all `.dll`s must be in %PATH%
            #          or same dir as calling executable
        ]
        # further dependency control (developers & package managers)
        if AMReX_repo:
            cmake_args.append("-DpyAMReX_amrex_repo=" + AMReX_repo)
        if AMReX_branch:
            cmake_args.append("-DpyAMReX_amrex_branch=" + AMReX_branch)
        if AMReX_src:
            cmake_args.append("-DpyAMReX_amrex_src=" + AMReX_src)

        if sys.platform == "darwin":
            cmake_args.append("-DCMAKE_INSTALL_RPATH=@loader_path")
        else:
            # values: linux*, aix, freebsd, ...
            #   just as well win32 & cygwin (although Windows has no RPaths)
            cmake_args.append("-DCMAKE_INSTALL_RPATH=$ORIGIN")

        cfg = "Debug" if self.debug else "RelWithDebInfo"  # 'Release'
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), os.path.join(extdir, "amrex")
                ),
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        # this environment variable is standardized in CMake
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # optional -j parameter in the build_ext call, not supported by pip
            if hasattr(self, "parallel") and self.parallel:
                build_args += ["-j{}".format(self.parallel)]
            else:
                build_args += ["-j2"]

        build_dir = os.path.join(self.build_temp, dims)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)
        # note that this does not call install;
        # we pick up artifacts directly from the build output dirs


with open("./README.md", encoding="utf-8") as f:
    long_description = f.read()

# Allow to control options via environment vars.
#   Work-around for https://github.com/pypa/setuptools/issues/1712
# Pick up existing AMReX libraries or...
PYAMREX_libdir = os.environ.get("PYAMREX_LIBDIR")

PYAMREX_CCACHE = os.environ.get("PYAMREX_CCACHE", "ON")
PYAMREX_IPO = os.environ.get("PYAMREX_IPO", "ON")

# ... build AMReX libraries with CMake
#   note: changed default for SHARED, SPACEDIM, MPI, TESTING and EXAMPLES
AMReX_OMP = os.environ.get("AMREX_OMP", "OFF")
AMReX_GPU_BACKEND = os.environ.get("AMREX_GPU_BACKEND", "NONE")
AMReX_MPI = os.environ.get("AMREX_MPI", "OFF")
AMReX_PRECISION = os.environ.get("AMREX_PRECISION", "DOUBLE")
#   single value or as a list 1;2;3
AMReX_SPACEDIM = os.environ.get("AMREX_SPACEDIM", "1;2;3")
BUILD_SHARED_LIBS = os.environ.get("AMREX_BUILD_SHARED_LIBS", "OFF")
# CMake dependency control (developers & package managers)
AMReX_src = os.environ.get("AMREX_SRC")
AMReX_internal = os.environ.get("AMREX_INTERNAL", "ON")
AMReX_repo = os.environ.get("AMREX_REPO")
AMReX_branch = os.environ.get("AMREX_BRANCH")
pybind11_internal = os.environ.get("PYBIND11_INTERNAL", "ON")

# https://cmake.org/cmake/help/v3.0/command/if.html
if AMReX_MPI.upper() in ["1", "ON", "TRUE", "YES"]:
    AMReX_MPI = "ON"
else:
    AMReX_MPI = "OFF"

# for CMake
cxx_modules = []  # values: amrex_1d, amrex_2d, amrex_3d
cmdclass = {}  # build extensions

# externally pre-built: pick up pre-built pyAMReX libraries
if PYAMREX_libdir:
    cmdclass = dict(build=CopyPreBuild)
# CMake: build pyAMReX ourselves
else:
    # TODO: modernize into a single build
    cmdclass = dict(build_ext=CMakeBuild)
    for dim in [x.lower() for x in AMReX_SPACEDIM.split(";")]:
        name = dim + "d"
        cxx_modules.append(CMakeExtension("amrex_" + name))

# Get the package requirements from the requirements.txt file
install_requires = []
with open("./requirements.txt") as f:
    install_requires = [line.strip("\n") for line in f.readlines()]
    if AMReX_MPI == "ON":
        install_requires.append("mpi4py>=2.1.0")

# keyword reference:
#   https://packaging.python.org/guides/distributing-packages-using-setuptools
setup(
    name="amrex",
    # note PEP-440 syntax: x.y.zaN but x.y.z.devN
    version="24.11",
    packages=["amrex"],
    # Python sources:
    package_dir={"": "src"},
    # pyAMReX authors:
    author="Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang",
    author_email="axelhuebl@lbl.gov",
    # wheel/pypi packages:
    maintainer="Axel Huebl",
    maintainer_email="axelhuebl@lbl.gov",
    description="AMReX: Software Framework for Block Structured AMR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=("AMReX AMR openscience mpi hpc research modeling simulation"),
    url="https://amrex-codes.github.io/amrex",
    project_urls={
        "Documentation": "https://amrex-codes.github.io/amrex/docs_html",
        "Tutorials": "https://amrex-codes.github.io/amrex/tutorials_html/",
        "Doxygen": "https://amrex-codes.github.io/amrex/doxygen/",
        "Reference": "https://doi.org/10.5281/zenodo.2555438",
        "Source": "https://github.com/AMReX-Codes/amrex",
        "Tracker": "https://github.com/AMReX-Codes/amrex/issues",
    },
    # CMake: self-built as extension module
    ext_modules=cxx_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    python_requires=">=3.8",
    tests_require=["pytest"],
    install_requires=install_requires,
    # cmdclass={'test': PyTest},
    # platforms='any',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Environment :: Console",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        (
            "License :: OSI Approved :: BSD License"
        ),  # TODO: use real SPDX: BSD-3-Clause-LBNL
    ],
    # new PEP 639 format
    license="BSD-3-Clause-LBNL",
    license_files=["LICENSE"],
)
