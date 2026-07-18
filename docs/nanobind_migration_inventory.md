# Nanobind migration inventory

This is the checked-in disposition record for the binding migration. It was
generated from the converted tree with `rg` and must be updated whenever a new
binding source or interop mechanism is added.

## Binding source coverage

All 64 files containing native binding declarations are covered:

- `src/pyAMReX.cpp`
- `src/AmrCore/{AmrCore,AmrMesh,TagBox}.cpp`
- `src/Boundary/{BCRec,BCUtil,PhysBCFunct}.cpp`
- `src/EB/{EB,EBFabFactory}.cpp`
- `src/Base/{AMReX,Algorithm,Arena,Array4,BaseFab,Box,BoxArray,CoordSys,Dim3,DistributionMapping,FArrayBox,FabArray,Geometry,IndexType,IntVect,MFInfo,MPMD,MultiFab,PODVector,ParGDB,ParallelDescriptor,ParmParse,Periodicity,PlotFileUtil,RealBox,RealVect,SmallMatrix,Utility,Vector,Version,VisMF,iMultiFab}.cpp`
- `src/Base/{Array4,MultiFab,SmallMatrix,Vector}.H`
- `src/Base/Array4_{complex,complex_const,float,float_const,int,int_const,uint,uint_const}.cpp`
- `src/Particle/{ParticleContainer,ParticleContainer_FHDeX,ParticleContainer_ImpactX,ParticleContainer_SoA,ParticleContainer_WarpX,ParticleContainer_tests}.cpp`
- `src/Particle/{ArrayOfStructs,Particle,ParticleContainer,ParticleTile,StructOfArrays}.H`

`src/pyAMReX.H` is the common nanobind include and scalar-format facade used
by those sources.

## Semantic inventory and disposition

| Concern | Files | Disposition |
| --- | --- | --- |
| Lifetime and return policies | `AmrCore/{AmrCore,AmrMesh,TagBox}.cpp`; `Base/{AMReX,Arena,BaseFab,Box,FArrayBox,FabArray,IntVect,MPMD,MultiFab,PODVector,ParGDB,VisMF}.cpp`; `Base/Array4.H`; `Boundary/PhysBCFunct.cpp`; `EB/EBFabFactory.cpp`; `Particle/{ArrayOfStructs,ParticleContainer,ParticleTile,StructOfArrays}.H` | Policies were translated by owner/referent relationship. Existing owner-deletion and view-lifetime tests cover Array4, FabArray/MultiFab, AmrCore/ParGDB, and particle iterator/tile/container families. |
| NumPy and accelerator arrays | `Base/{Array4,Vector}.H`; `Base/{BaseFab,IntVect,PODVector}.cpp`; `Base/SmallMatrix.H`; `Particle/ArrayOfStructs.H`; `Particle/ParticleContainer.cpp` | Typed inputs use `nb::ndarray`; explicit array-interface dictionaries remain for AMReX-owned zero-copy memory. Array-like copy inputs are normalized through `numpy.asarray`; view-producing paths retain their owners. |
| Trampolines and callbacks | `AmrCore/AmrCore.cpp`; `Boundary/{PhysBCFunct.H,PhysBCFunct.cpp}` | `AmrCore` uses `NB_TRAMPOLINE`; mutable `TagBoxArray` is passed as a borrowed reference. `PhysBCFunctUser` stores a callable wrapper that acquires the GIL, preserves mutation-by-reference, and synchronizes the GPU stream before invocation. |
| Enums | `AmrCore/TagBox.cpp`; `Base/{Box,CoordSys,IndexType,PODVector}.cpp`; `Boundary/BCRec.cpp`; `EB/EBFabFactory.cpp` | Native `nb::enum_` bindings preserve names and arithmetic/IntEnum behavior. Direction is registered once as an enum with dimension-appropriate members. |
| Raw Python and MPI API | `Base/MPMD.cpp` | Raw CPython access is isolated to the mpi4py communicator bridge. Ordinary object operations use nanobind; the published mpi4py C-API path remains documented for follow-up when mpi4py is available in the build matrix. |
| Type reuse | `Base/SmallMatrix.H` | The old binder-internal type lookup was removed. `nb::type<T>()` and borrowed public class handles deterministically reuse already-registered vector specializations. |
| I/O and exceptions | `Base/ParmParse.cpp` and binding-wide `std::runtime_error` paths | `ParmParse` writes its captured stream through `sys.stdout.write`; dynamic exception strings have stable storage and existing exception tests remain authoritative. |
| Build and packaging | `dependencies.json`, `cmake/dependencies/nanobind.cmake`, `CMakeLists.txt`, `setup.py` | Pinned, local-source, parent-provided, and external `find_package` modes are supported. Private extension basenames and installed target aliases are unchanged. pyAMReX owns IPO and stripping policy explicitly. |
| Stubs | `CMakeLists.txt`, `.github/update_stub.sh`, `.github/workflows/stubs.yml` | `nanobind_add_stub` generates the three native extension stubs. Pull requests fail on stale output; trusted development pushes may auto-commit updates. |
| Cross-extension types | `tests/downstream_nanobind/`, `docs/source/developers/nanobind_extensions.rst` | The default nanobind domain is intentional. The smoke extension accepts and returns an `IntVect` reference; pybind11 extensions are documented as ABI-incompatible and must migrate/rebuild. |

## Residual references

The private module basenames `amrex_*d_pybind` remain intentionally for Python
package compatibility. Historical release notes may still mention pybind11.
There are no active pybind11 includes, dependencies, configuration variables,
or namespace uses in build and binding sources.

The one direct `nb::detail` use is nanobind's public trampoline `ticket` type in
`AmrCore.cpp`. It is required to pass a mutable, non-copyable callback argument
with explicit reference policy; the stock override macro otherwise applies the
automatic policy and aborts while attempting ownership. This use is confined
to the public `nanobind/trampoline.h` mechanism and is covered by the refined
level `ErrorEst` regression test.

## Validation record

The implementation was validated locally on macOS/AppleClang with Python
3.14, nanobind 2.12.0, and a Debug AMReX build for 1D, 2D, and 3D with EB
enabled where supported:

- all three native modules compile and import individually;
- the CPU suite reports 218 passing tests and 55 accelerator/optional-backend
  skips, including callback exception and lifetime regressions;
- all generated native stubs compile as Python syntax;
- the installed CMake package configures and builds
  `tests/downstream_nanobind`, whose identity call returns the original
  `IntVect` Python object;
- both source-built and `PYAMREX_LIBDIR` prebuilt-artifact wheels have native
  CPython/platform tags and import from isolated target directories;
- wheel inspection finds no nanobind build files, pybind11 files, bytecode, or
  `__pycache__` directories.

Compiler, MPI, Windows, Linux, Emscripten, and accelerator coverage is supplied
by the repository CI matrix rather than this single-host record.
