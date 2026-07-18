# pybind11-to-nanobind migration plan

## Status

Implemented on the nanobind migration branch. The disposition record is
`docs/nanobind_migration_inventory.md`; downstream ABI guidance is in
`docs/source/developers/nanobind_extensions.rst`.

The implementation pins nanobind 2.12.0, converts the complete native binding
tree, preserves the historical private extension basenames, generates native
stubs with `nanobind_add_stub`, and retains the default domain for downstream
type sharing. Platform and accelerator configurations remain enforced by the
existing CI matrix.

## Goals

- Replace pybind11 with nanobind in the C++ extension modules, dependency
  management, CMake build, packaging, stub generation, CI, and documentation.
- Preserve the public Python API and behavior of `amrex.space1d`,
  `amrex.space2d`, and `amrex.space3d`.
- Preserve supported build variants: 1D/2D/3D, MPI, EB, single precision,
  shared/static AMReX, CUDA, HIP, SYCL, Emscripten, macOS, Linux, and Windows.
- Preserve zero-copy host and accelerator array views and their ownership
  guarantees.
- Keep the installed CMake package usable by downstream projects and make the
  cross-extension type-sharing policy explicit.
- Establish evidence for the expected nanobind benefits by measuring clean
  build time, incremental build time, extension size, import time, and a small
  set of binding-heavy calls before and after the migration.

## Non-goals

- Renaming Python classes, methods, arguments, or the public `amrex.space*d`
  packages.
- Renaming the existing private extension filenames
  (`amrex_1d_pybind`, `amrex_2d_pybind`, and `amrex_3d_pybind`) in the migration
  release. Although the names become historical, retaining them avoids an
  unrelated import and wheel-layout break. A later release can deprecate them
  separately.
- Enabling Python's stable ABI or free-threaded Python in the first nanobind
  release. Both change the compatibility matrix and should be evaluated after
  behavioral parity is established.
- Replacing the setuptools/CMake packaging architecture with
  scikit-build-core. That can be considered independently after the binding
  migration.
- Redesigning AMReX C++ APIs or the pure-Python extension layer.

## Current-state inventory

The binding is one source family compiled three times, once per
`AMReX_SPACEDIM`. `src/pyAMReX.H` imports pybind11 centrally and all 66 C++
binding source/header files depend on it. `src/pyAMReX.cpp` defines the three
module entry points and registers types in dependency order.

The important migration surfaces are:

| Surface | Current implementation | Migration concern |
| --- | --- | --- |
| Dependency acquisition | `dependencies.json` and `cmake/dependencies/pybind11.cmake` support pinned FetchContent, a local source tree, or `find_package` | Reproduce all three modes for nanobind and preserve offline/package-manager builds |
| Module build | Handwritten `MODULE` targets link `pybind11::module`; pybind11 helpers set the suffix and strip binaries | Reconcile `nanobind_add_module` defaults with existing IPO, stripping, CUDA source properties, Emscripten handling, output paths, and install/export rules |
| Python packaging | `setup.py` drives one CMake build per dimension and copies prebuilt artifacts when `PYAMREX_LIBDIR` is set | Preserve both source-build and prebuilt-library wheel paths and platform-specific RPATH/DLL behavior |
| Binding API | Approximately 75 class declarations and 276 `py::overload_cast` uses | Most declarations translate directly, but the whole module must use one binding runtime |
| Ownership | 32 `keep_alive` and 47 explicit return-value policies | Nanobind's ownership model differs; every policy must be reviewed semantically, not translated blindly |
| Inheritance | AMReX/FabArray/particle hierarchies plus a Python-subclassable `AmrCore` trampoline | Verify base registration order, virtual dispatch, pure-virtual failures, and trampoline lifetime |
| Python callbacks | `PhysBCFunctUser` stores `std::function`; `AmrCore` calls Python overrides | Verify callable conversion, exception propagation, GIL acquisition, object lifetime, and GPU synchronization |
| NumPy/buffers | `Array4`, `SmallMatrix`, `PODVector`, `IntVect`, particle arrays, and `pack_ids`/`pack_cpus` use `array_t`, `buffer_info`, format descriptors, or array interface dictionaries | Port inputs to typed `nb::ndarray` or explicit Python buffer handling while preserving dtype, shape, stride, copy/view, and const behavior |
| Accelerator arrays | `__cuda_array_interface__` and host/device copy paths expose AMReX memory | Preserve pointer, stream, allocator, lifetime, and synchronization semantics on CUDA; revalidate HIP/SYCL behavior |
| Internal APIs | `SmallMatrix.H` calls `py::detail::get_type_info`; `MPMD.cpp` inspects mpi4py/CPython objects | Replace the type-registry dependency with a supported registry and isolate raw CPython/mpi4py handling behind tested helpers |
| Python conveniences | Dynamic attributes, operators, native enums, exceptions, STL conversion/binding, `gc`/`sys` imports, and ostream redirection | Select the corresponding nanobind headers/APIs and test observable behavior |
| Stubs | Committed `.pyi` files are generated with a patched `pybind11-stubgen` workflow | Move to `nanobind_add_stub`, preserve post-processing only where still necessary, and review the full generated API diff |
| Downstream extensions | Installed `pyAMReX` CMake targets and symbol-hiding comments anticipate other extension modules | Decide nanobind's type domain and document that pybind11 and nanobind types do not transparently interoperate across extension boundaries |

The inventory counts are navigation aids, not completion criteria. Before
implementation, generate a checked-in or CI artifact that lists every binding
source and every use of lifetime policies, arrays, callbacks, enums, raw Python
API calls, and nanobind-incompatible internals. The migration is complete only
when every inventory row has a recorded disposition.

## Target decisions

1. **Use one binding runtime per process-facing type graph.** Do not build a
   module containing a mixture of pybind11 and nanobind registrations. Convert
   the three pyAMReX extensions atomically on the migration branch.
2. **Preserve Python names and private extension basenames initially.** Change
   `PYBIND11_MODULE` to `NB_MODULE`, but keep `amrex_*d_pybind` as the module
   names and output basenames for compatibility.
3. **Use an explicit nanobind domain policy.** Start with the default/global
   domain so separately built nanobind extensions can exchange pyAMReX types.
   Add a two-extension integration test. Do not select `NB_DOMAIN` merely for
   symbol isolation: a private domain would prevent downstream modules from
   reusing registered AMReX types.
4. **Retain the current Python minimum until verified.** Select and pin a
   nanobind release that supports the project's Python and compiler matrix. If
   current nanobind requires a higher Python version, treat raising pyAMReX's
   minimum as an explicit release decision rather than an incidental build
   fix.
5. **Do not enable `STABLE_ABI` in this migration.** It applies only to a subset
   of supported Python versions and would obscure binding parity failures.
6. **Use supported APIs only.** Replace `py::detail::get_type_info` with a
   pyAMReX-owned type-registration cache keyed by the concrete C++ type (or an
   equivalent documented nanobind facility). Keep raw CPython use confined to
   the mpi4py bridge where the external C API requires it.
7. **Make array contracts explicit.** Use `nb::ndarray` for typed array
   arguments and results when it expresses the existing contract. Continue to
   implement `__array_interface__`/`__cuda_array_interface__` where pyAMReX
   intentionally exposes AMReX-owned non-owning memory. Do not turn a view into
   a copy, accept a previously rejected dtype/stride, or weaken constness as an
   accidental side effect.
8. **Keep optimization ownership clear.** Prefer `nanobind_add_module` for the
   supported extension suffix and nanobind runtime setup, but disable or
   override its size/strip choices where they conflict with `pyAMReX_IPO`,
   CMake build types, CUDA compilation, or debuggable builds. Record the final
   ownership of LTO, stripping, and visibility in CMake comments.

## Execution plan

### Phase 0: Freeze the compatibility contract

1. Build the unmodified pybind11 implementation in a representative CPU
   configuration (`AMReX_SPACEDIM=1;2;3`, MPI on, EB on) and record:
   - the complete pytest/CTest results;
   - `dir()` and `inspect.signature`-style API snapshots for all three public
     modules;
   - generated stubs;
   - exception classes/messages for tested failure paths;
   - wheel contents and extension filenames;
   - clean/incremental build time, binary size, import time, and selected call
     overhead.
2. Add focused tests where current behavior is under-specified:
   - each `keep_alive`/`reference_internal` ownership family, including owner
     deletion and garbage collection;
   - `AmrCore` Python overrides, missing pure overrides, Python exceptions, and
     callback object destruction;
   - `PhysBCFunctUser` callback lifetime, mutation-by-reference, exception
     propagation, and GIL-safe invocation from C++;
   - NumPy input dtype, rank, shape, C/F/strided layout, read-only inputs,
     copy-versus-view behavior, and owner deletion;
   - `__array_interface__` and `__cuda_array_interface__` pointer, dtype,
     shape, stride, stream, and lifetime behavior;
   - native enum identity, inheritance, integer conversion, repr, and stub
     shape;
   - ostream redirection and exception translation;
   - mpi4py communicator extraction and MPMD copier references.
3. Add a downstream smoke project that builds a second extension, imports a
   pyAMReX type, accepts it as an argument, and returns a reference. Run it
   against the installed CMake package. This becomes the type-domain contract.

**Exit criterion:** the baseline artifacts and missing semantic tests are in
place and pass with pybind11. No migration code is merged before this gate.

### Phase 1: Prove build and ecosystem feasibility

Create a short-lived spike before converting the source tree:

1. Select a current nanobind release and test its Python, CMake, compiler,
   CUDA/HIP/SYCL, Emscripten, and platform requirements against CI. Pin the
   chosen release in `dependencies.json`.
2. Implement `cmake/dependencies/nanobind.cmake`, mirroring the existing
   controls:
   - `pyAMReX_nanobind_src` for a local checkout;
   - `pyAMReX_nanobind_internal` for the pinned FetchContent dependency;
   - `pyAMReX_nanobind_repo` and `pyAMReX_nanobind_branch`;
   - `find_package(nanobind CONFIG REQUIRED)` for external/package-manager
     builds.
3. Build a minimal extension with `nanobind_add_module` through the same
   setuptools-driven and standalone-CMake paths used by pyAMReX. Validate
   suffixes, multi-config output directories, Windows exports, macOS/Linux
   RPATH, Emscripten output, stripping, IPO, CUDA compilation, install, CMake
   export, wheel assembly, and `PYAMREX_LIBDIR` reuse.
4. Build the downstream smoke extension in the same/default nanobind domain.
   Also document the expected failure mode for a downstream extension that is
   still built with pybind11, so dependent projects can coordinate their own
   migration.

**Exit criterion:** a minimal nanobind module works in every packaging mode,
the downstream type-sharing test passes between nanobind modules, and any
unsupported platform/version decision has maintainer approval.

### Phase 2: Establish the nanobind source foundation

1. Replace the shared includes in `src/pyAMReX.H` with the narrow nanobind
   headers required for core bindings, functions, strings, STL containers,
   callables, operators, enums, and ndarrays. Use `namespace nb = nanobind`
   consistently rather than retaining a misleading `py` alias.
2. Change all initializer signatures to `nb::module_ &`, convert the three
   entry points to `NB_MODULE`, and retain their existing names and registration
   order.
3. Perform the low-risk mechanical API pass:
   - `py::` to `nb::`;
   - module, class, constructor, argument, operator, property, and exception
     APIs;
   - documented name changes such as nanobind's property helpers;
   - `nb::overload_cast` and const overload selection;
   - imports, handles, casts, tuples, lists, dicts, strings, and `none`.
4. Add STL headers per converted type (`vector`, `string`, `function`, smart
   pointers, and so on) rather than relying on one broad transitive include.
5. Compile after each binding subsystem (`Base`, `Boundary`, `AmrCore`, `EB`,
   and `Particle`) even though the final switch must land atomically.

**Exit criterion:** all source files compile and all three modules import in a
minimal CPU build. Temporary compatibility shims are listed and have removal
issues or tasks.

### Phase 3: Port semantics-heavy bindings

Handle these as reviewed work packages with their focused tests running after
each change:

1. **Ownership and inheritance:** audit every `keep_alive` and return policy.
   Map each policy from its intended owner/referent relationship, then exercise
   it with weak references or CPython refcount checks. Verify FabArray,
   MultiFab, particle iterator/tile/container, ParGDB, TagBox, and arena
   hierarchies.
2. **AmrCore trampoline:** replace `PYBIND11_OVERRIDE_PURE_NAME` with
   nanobind's documented trampoline/override mechanism. Preserve snake-case
   override names, pure-virtual errors, mutable `TagBoxArray` reference
   semantics, exception propagation, and Python-subclass lifetime.
3. **Callbacks:** include nanobind's function caster for `PhysBCFunctUser` and
   explicitly verify GIL behavior. If callbacks can be reached from a native
   thread, acquire the GIL at that boundary and test it. Retain the existing GPU
   stream synchronization before the Python call.
4. **Enums and operators:** port all native enums and arithmetic/comparison
   operators. Compare identity, inheritance from `enum.Enum`/`enum.IntEnum`,
   conversions, repr, default values, and generated annotations to baseline.
5. **Dynamic attributes and static properties:** verify `MFIter`, `MultiFab`,
   particle iterators, and `StructOfArrays`, plus class-level configuration
   properties that currently receive a Python object parameter.
6. **SmallMatrix type reuse:** remove `py::detail::get_type_info` and the raw
   cast of its internal type object. Introduce a supported, deterministic
   registry so shared vector specializations are bound exactly once and the
   same Python type is reused.
7. **I/O and exceptions:** replace pybind11 ostream redirection in `ParmParse`
   with nanobind's supported mechanism or a small CPython stream adapter.
   Preserve exception classes and user-visible messages.
8. **MPI bridge:** port ordinary Python object operations to nanobind while
   keeping mpi4py communicator access behind a small checked helper. Prefer
   mpi4py's published C API if available in the supported versions; test
   `None`, wrong object types, valid intracommunicators, and Python errors.

**Exit criterion:** focused CPU tests pass under sanitizers where practical,
and no source depends on nanobind internals.

### Phase 4: Port array, buffer, and accelerator interop

1. Convert typed NumPy inputs in `Array4.H`, `SmallMatrix.H`, `PODVector.cpp`,
   `IntVect.cpp`, and `ParticleContainer.cpp` to appropriately constrained
   `nb::ndarray` signatures. State device, dtype, dimensionality, contiguity,
   read-only, and conversion requirements in the C++ type when possible.
2. Where the old code used `buffer_info` and `format_descriptor`, replace
   format-string comparisons with nanobind dtype checks and use explicit
   shape/stride/data access. Keep current behavior for non-contiguous arrays;
   resolve existing TODOs separately rather than silently tightening or
   broadening accepted inputs.
3. Preserve ownership for every zero-copy view. An `Array4` constructed over a
   Python array must keep that array alive. A NumPy/CuPy view over AMReX storage
   must retain the owning AMReX object for at least as long as the view.
4. Revalidate manually constructed `__array_interface__` dictionaries for
   Array4, PODVector, SmallMatrix, Vector, and particle AoS/SoA data. Confirm
   that nanobind scalar/dtype helpers produce the same NumPy `typestr`/`descr`
   values, including particle alignment and single precision.
5. Revalidate `__cuda_array_interface__`, host/device copies, pinned/managed/
   async allocators, empty arrays, and stream synchronization with CUDA. Run
   the available HIP and SYCL suites to catch host-accessibility assumptions.
6. Keep DLPack adoption out of this migration unless it is required to restore
   an existing behavior; add it later as a separate feature.

**Exit criterion:** the full CPU NumPy suite and accelerator array suites pass,
with explicit tests proving pointer equality for views and independence for
copies.

### Phase 5: Complete build, packaging, stubs, and documentation

1. Replace the pybind11 dependency include/link/helper calls in
   `CMakeLists.txt` with the validated nanobind integration. Preserve target
   aliases and installed CMake target names.
2. Rename configuration inputs throughout `setup.py`, README, and install docs:
   - `PYBIND11_INTERNAL` to `NANOBIND_INTERNAL`;
   - `pyAMReX_pybind11_*` to `pyAMReX_nanobind_*`.
   If a compatibility window is desired, accept the old names for one release
   with a CMake/setup warning, reject conflicting old/new values, and document
   the removal version.
3. Replace pybind11 dependency metadata, licenses, links, comments, and
   developer documentation. Do not rewrite historical release notes or
   comments that intentionally explain old compatibility workarounds without
   first deciding whether the workaround is obsolete.
4. Replace `pybind11-stubgen` and its patched dependency with
   `nanobind_add_stub`. Generate committed stubs for all dimensions, keep the
   current `py.typed` packaging, and port only necessary enum/default-value
   post-processing from `.github/update_stub.sh`.
5. Review the entire stub diff as an API diff. Expected formatting changes are
   acceptable; missing types, changed names, altered overloads, weakened
   return annotations, or enum regressions block the migration.
6. Update the stub workflow so pull requests fail on stale stubs, while only
   trusted development-branch pushes auto-commit regenerated files.
7. Inspect built wheels on Linux, macOS, and Windows. Verify no pybind11 or
   build-only nanobind files are accidentally packaged, imports work from an
   isolated environment, and prebuilt-artifact packaging still works.

**Exit criterion:** no active build or documentation path depends on pybind11,
all committed stubs match generated output, and source-built plus prebuilt
wheels import in isolation.

### Phase 6: Full validation and rollout

Run the existing CI matrix, with failures grouped by behavior rather than fixed
platform-by-platform:

- GNU and Clang release/debug builds, libc++, MPI on/off, single/double
  precision, and 1D/2D/3D;
- CUDA, HIP, and SYCL builds and their array interoperability tests;
- macOS/AppleClang;
- Windows MSVC, ClangCL, and WSL, including shared/static and installed tests;
- EB and configured particle-container variants;
- Emscripten/Pyodide if it remains a supported build;
- standalone CMake build/install/CTest, `pip install .`, wheel build/install,
  `PYAMREX_LIBDIR`, external nanobind discovery, and local nanobind source;
- the downstream second-extension type-sharing test.

Compare the Phase 0 metrics on the same machine and configuration. Treat
performance numbers as evidence, not hard gates, unless maintainers set an
explicit threshold. Any substantial regression in compile time, binary size,
import time, or binding call overhead must be explained before release.

Before merging:

1. Run a repository-wide search for active `pybind11`, `PYBIND11`, and
   `py::` references and classify every remaining match as historical,
   compatibility-only, generated, or erroneous.
2. Publish a migration note for downstream extension authors covering the
   required nanobind version/domain, CMake target usage, and the lack of direct
   pybind11/nanobind type interoperability.
3. Mark the first nanobind release as an ABI transition even if the Python API
   is unchanged; downstream native extensions must rebuild.
4. Keep a pybind11-based release branch available for critical fixes until one
   nanobind release has completed the normal validation cycle.

**Final acceptance criterion:** the public API snapshots, focused semantic
tests, full CI matrix, installed/downstream integration tests, wheel inspection,
and stub review all pass, and every inventory item has a documented result.

## Suggested change sequence

Because all bindings belong to one registered type graph, avoid merging a
half-pybind11/half-nanobind module. A reviewable sequence is:

1. Baseline tests, API inventory, metrics, and downstream smoke project.
2. Nanobind dependency/build spike, kept separate until its platform findings
   are incorporated into this plan.
3. Atomic source conversion with subsystem-organized commits, including
   semantics-heavy and array fixes.
4. Build/package/configuration switch and isolated wheel validation.
5. Stub generator, generated stubs, CI, documentation, and compatibility
   cleanup.
6. Full matrix validation and downstream migration note.

The source-conversion branch may contain temporarily non-building intermediate
commits for review, but the merge commit must build and test all selected
dimensions with nanobind only.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A translated lifetime policy compiles but creates a use-after-free or leak | Test owner deletion and garbage collection for each relationship family; use ASan in the CPU build |
| Downstream native extensions cannot consume pyAMReX types | Fix and test the nanobind domain policy; provide a second-extension sample and migration note |
| NumPy/CuPy behavior changes from view to copy or accepts a wrong layout | Snapshot pointer/shape/stride/dtype behavior and test every allocator/backend |
| Trampoline/callback exceptions terminate native code or run without the GIL | Add override/callback exception and native-thread tests; acquire the GIL at explicit native-to-Python boundaries |
| Nanobind optimization defaults conflict with AMReX/CMake/CUDA flags | Make IPO, size optimization, stripping, visibility, and CUDA setup ownership explicit and inspect compile/link commands in CI |
| Generated stubs change the apparent API | Review stubs as a first-class compatibility artifact and fail CI on stale output |
| Raw mpi4py layout assumptions break on a supported mpi4py version | Prefer its supported C API, isolate the bridge, and test the oldest/newest supported mpi4py versions |
| Emscripten or a less common accelerator is unsupported by the selected nanobind version | Resolve in the Phase 1 spike and make support changes explicit before source conversion |
| Private module names containing `_pybind` cause confusion | Preserve them for compatibility now; address naming later with a deprecation/alias plan |

## Current documentation references

The implementation should re-check these references when work begins, since
nanobind evolves independently of this repository:

- [nanobind porting guide](https://nanobind.readthedocs.io/en/latest/porting.html)
- [nanobind CMake API](https://nanobind.readthedocs.io/en/latest/api_cmake.html)
- [nanobind packaging guide](https://nanobind.readthedocs.io/en/latest/packaging.html)
- [nanobind ndarray guide](https://nanobind.readthedocs.io/en/latest/ndarray.html)
- [nanobind ownership guide](https://nanobind.readthedocs.io/en/latest/ownership.html)
- [nanobind typing and stub generation](https://nanobind.readthedocs.io/en/latest/typing.html)

The current nanobind documentation confirms the key migration primitives used
by this plan: `NB_MODULE`, `nanobind_add_module`, `nanobind_add_stub`,
`nb::overload_cast`, typed `nb::ndarray`, and nanobind's holder-free class
model. Exact version selection remains a Phase 1 compatibility decision.
