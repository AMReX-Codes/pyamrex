.. _usage-api:

Python API
==========

Imports
-------

pyAMReX provides the public imports ``amrex.space1d``, ``amrex.space2d`` and ``amrex.space3d``, mirroring the compile-time option ``AMReX_SPACEDIM``.

Due to limitations in AMReX, currently, only one of the imports can be used at a time in the same Python process.
For example:

.. code-block:: python

   import amrex.space3d as amr

A 1D or 2D AMReX run needs its own Python process.
Another dimensionality *cannot be imported into the same Python process* after choosing a specific dimensionality for import.

For brevity, below the 3D APIs are shown.
pyAMReX classes and functions follow the same structure as the `C++ AMReX APIs <https://amrex-codes.github.io/amrex/doxygen/>`__.


.. _usage-api-base:

Base
----

.. autoclass:: amrex.space3d.AMReX
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Config
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.initialize

.. autofunction:: amrex.space3d.finalize

.. autofunction:: amrex.space3d.initialized

.. autofunction:: amrex.space3d.size

.. autoclass:: amrex.space3d.Arena
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Direction
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.CoordSys
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.DistributionMapping
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.GeometryData
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Geometry
   :members:
   :undoc-members:

.. automodule:: amrex.space3d.ParallelDescriptor
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Periodicity
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.RealBox
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.AlmostEqual

Indexing: Box, IntVect and IndexType
""""""""""""""""""""""""""""""""""""

`Corresponding AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/Basics.html#box-intvect-and-indextype>`__.

.. autoclass:: amrex.space3d.IntVect
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Box
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.BoxArray
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Dim3
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.XDim3
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.IndexType
   :members:
   :undoc-members:

Vectors
"""""""

.. autoclass:: amrex.space3d.RealVect
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.min

.. autofunction:: amrex.space3d.max

``amrex::Vector<T>`` is implemented for many types, e.g.,

.. autoclass:: amrex.space3d.Vector_Real
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Vector_int
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Vector_string
   :members:
   :undoc-members:

Boundary Conditions
"""""""""""""""""""

Boundary records describe the mathematical boundary types used for each
MultiFab component and coordinate direction. A common cell-centered fill
sequence is to fill interior or periodic ghost cells first and then fill
physical-domain ghost cells:

.. code-block:: python

   sd = amr.Config.spacedim
   bc = amr.Vector_BCRec([
       amr.BCRec(
           lo=[amr.BCType.foextrap] * sd,
           hi=[amr.BCType.foextrap] * sd,
       )
   ])

   mf.fill_boundary()
   amr.fill_domain_boundary(mf, geom, bc)

``fill_domain_boundary`` handles extrapolation and reflection boundary
types. For external Dirichlet values, ``BCType.ext_dir`` or
``BCType.ext_dir_cc``, fill the relevant ghost cells from application
code. ``PhysBCFunctUser`` provides a Python callback hook with the same
component-range convention as AMReX FillPatch routines:

.. code-block:: python

   def fill_ext_dir(mf, dcomp, ncomp, nghost, time, bccomp):
       # Fill external Dirichlet ghost cells for mf components
       # [dcomp, dcomp + ncomp).
       pass

   physbc = amr.PhysBCFunctUser(fill_ext_dir)
   physbc(mf, 0, 1, mf.n_grow_vect, 0.0, 0)

In physical boundary functors, ``dcomp`` is the first destination
component in the ``MultiFab`` and ``bccomp`` is the first matching entry
in the ``Vector_BCRec``.

.. autoclass:: amrex.space3d.BCType
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.PhysBCType
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.BCRec
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Vector_BCRec
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.setBC

.. autofunction:: amrex.space3d.fill_domain_boundary

.. autoclass:: amrex.space3d.PhysBCFunctNoOp
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.CpuBndryFuncFab
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.PhysBCFunct_CpuBndryFuncFab
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.PhysBCFunctUser
   :members:
   :undoc-members:

Data Containers
"""""""""""""""

``amrex::Array4<T>`` is implemented for many floating point and integer types, e.g.,

.. autoclass:: amrex.space3d.Array4_double
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.BaseFab_Real
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.FArrayBox
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.MultiFab
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.MFInfo
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.MFIter
   :members:
   :undoc-members:

``amrex::PODVector<T, Allocator>`` is implemented for many allocators, e.g.,

.. autoclass:: amrex.space3d.PODVector_real_arena
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.PODVector_int_pinned
   :members:
   :undoc-members:

MultiFab Operations
"""""""""""""""""""

Averaging operations between AMR levels and between data centerings.
Functions that take one ``MultiFab`` per coordinate direction expect a list
of exactly ``AMREX_SPACEDIM`` entries and raise ``ValueError`` otherwise.

.. autofunction:: amrex.space3d.average_down

.. autofunction:: amrex.space3d.average_down_faces

.. autofunction:: amrex.space3d.average_down_edges

.. autofunction:: amrex.space3d.average_down_nodal

.. autofunction:: amrex.space3d.average_node_to_cellcenter

.. autofunction:: amrex.space3d.average_edge_to_cellcenter

.. autofunction:: amrex.space3d.average_face_to_cellcenter

.. autofunction:: amrex.space3d.average_cellcenter_to_face

.. autofunction:: amrex.space3d.sum_fine_to_coarse

Small Matrices and Vectors
""""""""""""""""""""""""""

.. autoclass:: amrex.space3d.SmallMatrix_6x6_F_SI1_double
   :members:
   :undoc-members:

Utility
"""""""

.. autoclass:: amrex.space3d.ParmParse
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.Print

.. autofunction:: amrex.space3d.d_decl

.. autofunction:: amrex.space3d.concatenate

.. autofunction:: amrex.space3d.write_single_level_plotfile

.. autofunction:: amrex.space3d.write_multi_level_plotfile

.. autoclass:: amrex.space3d.PlotFileData
   :members:
   :undoc-members:

Plotfile paths and directories, e.g. to write a multi-level plotfile
incrementally:

.. autofunction:: amrex.space3d.level_path

.. autofunction:: amrex.space3d.multifab_header_path

.. autofunction:: amrex.space3d.level_full_path

.. autofunction:: amrex.space3d.multifab_file_full_prefix

.. autofunction:: amrex.space3d.pre_build_director_hierarchy

.. autofunction:: amrex.space3d.util_create_directory

.. autofunction:: amrex.space3d.util_create_clean_directory

.. autofunction:: amrex.space3d.util_create_directory_destructive

.. autofunction:: amrex.space3d.file_exists


.. _usage-api-amrcore:

AmrCore
-------

Python subclasses implement the AMR callbacks with pyAMReX snake-case names:
``make_new_level_from_scratch``, ``make_new_level_from_coarse``,
``remake_level``, ``clear_level`` and ``error_est``.  The ``error_est``
callback receives a mutable ``TagBoxArray`` for the level being tagged.  The
tag array is a callback-scoped, non-owning view; mark cells for refinement with
``tags.set_val(amr.TagBox.SET, ...)`` and do not store it for later use.

A particle container can be connected to an ``AmrCore`` hierarchy through the
particle metadata broker returned by ``core.get_par_gdb()``.

.. code-block:: python

   class MyCore(amr.AmrCore):
       def make_new_level_from_scratch(self, lev, time, ba, dm):
           pass

       def make_new_level_from_coarse(self, lev, time, ba, dm):
           pass

       def remake_level(self, lev, time, ba, dm):
           pass

       def clear_level(self, lev):
           pass

       def error_est(self, lev, tags, time, ngrow):
           tags.set_val(amr.TagBox.SET)

   core = MyCore(rb, 1, n_cell, 0, ref_ratios, is_periodic)
   core.init_from_scratch(0.0)
   particles = amr.ParticleContainer_2_1_3_1_default(core.get_par_gdb())

.. autoclass:: amrex.space3d.AmrInfo
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.AmrMesh
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.AmrCore
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.TagBox
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.TagBoxArray
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.ParGDBBase
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.AmrParGDB
   :members:
   :undoc-members:

Interpolaters
"""""""""""""

Interpolaters map data from coarse to fine levels. They cannot (yet) be
implemented in Python, since their per-FAB ``interp`` methods are called in
performance-critical inner loops. Use the global instances below, which are
the same objects as their C++ counterparts:

``pc_interp``, ``node_bilinear_interp``, ``cell_bilinear_interp``,
``cell_cons_interp``, ``lincc_interp``, ``protected_interp``,
``quartic_interp``, ``quadratic_interp``, ``cell_quartic_interp``,
``face_linear_interp``, ``face_divfree_interp``, ``face_cons_linear_interp``
and the ``MFInterpolater`` variants ``mf_pc_interp``,
``mf_cell_cons_interp``, ``mf_lincc_interp``,
``mf_linear_slope_minmax_interp``, ``mf_cell_bilinear_interp`` and
``mf_node_bilinear_interp``.

.. autoclass:: amrex.space3d.InterpBase
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.Interpolater
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.MFInterpolater
   :members:
   :undoc-members:

FillPatch
"""""""""

Fill a ``MultiFab``, including its ghost cells, from same-level and/or
coarse-level data. ``mapper`` accepts any ``Interpolater`` or
``MFInterpolater``; ``physbcf`` accepts ``PhysBCFunctNoOp``,
``PhysBCFunct_CpuBndryFuncFab`` or a Python ``PhysBCFunctUser``.

.. autofunction:: amrex.space3d.fill_patch_single_level

.. autofunction:: amrex.space3d.fill_patch_two_levels

.. autofunction:: amrex.space3d.interp_from_coarse_level

Flux Registers
""""""""""""""

.. autoclass:: amrex.space3d.FluxRegister
   :members:
   :undoc-members:


.. _usage-api-particles:

Particles
---------

Particle support is implemented for both legacy (AoS+SoA) and pure SoA particle memory layouts in AMReX.
Additional runtime attributes (Real or Int) are always in SoA memory layout.

``amrex::StructOfArrays<NReal, NInt, Allocator>`` is implemented for many numbers of Real and Int arguments, and allocators, e.g.,

.. autoclass:: amrex.space3d.StructOfArrays_8_0_idcpu_default
   :members:
   :undoc-members:

``amrex::ParticleTile<T_ParticleType, NArrayReal, NArrayInt, Allocator>`` is implemented for both legacy (AoS+SoA) and pure SoA particle types, many number of Real and Int arguments, and allocators, e.g.,

.. autoclass:: amrex.space3d.ParticleTile_pureSoA_8_0_default
   :members:
   :undoc-members:

``amrex::ParticleTileData<T_ParticleType, NArrayReal>`` is implemented for both legacy (AoS+SoA) and pure SoA particle types, many number of Real and Int arguments, e.g.,

.. autoclass:: amrex.space3d.ParticleTileData_pureSoA_8_0
   :members:
   :undoc-members:

``amrex::ParticleContainer_impl<ParticleType, T_NArrayReal, T_NArrayInt, Allocator>`` is implemented for both legacy (AoS+SoA) and pure SoA particle types, many number of Real and Int arguments, and allocators, e.g.,

.. autoclass:: amrex.space3d.ParticleContainer_2_1_3_1_default
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.ParticleContainer_pureSoA_8_0_default
   :members:
   :undoc-members:

Likewise for other classes accessible and usable on particle containers:

.. autoclass:: amrex.space3d.ParIter_pureSoA_8_0_default
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.ParConstIter_pureSoA_8_0_default

.. autoclass:: amrex.space3d.ParticleInitType_2_1_3_1
   :members:
   :undoc-members:

.. TODO for pure SoA
.. .. autoclass:: amrex.space3d.ParticleInitType_pureSoA_8_0
..    :members:
..    :undoc-members:

AoS
"""

This is for the legacy, AoS + SoA particle containers only:

``amrex::ArrayOfStructs<T_ParticleType, Allocator>`` is implemented for many numbers of extra Real and Int arguments, and allocators, e.g.,

.. autoclass:: amrex.space3d.ArrayOfStructs_2_1_default
   :members:
   :undoc-members:

``amrex::Particle<T_NReal, T_NInt>`` is implemented for many numbers of extra Real and Int arguments, e.g.,

.. autoclass:: amrex.space3d.Particle_2_1
   :members:
   :undoc-members:

I/O
"""

Read back particle data from plotfiles and checkpoints, see :ref:`Read Back Plotfiles <usage-how-to-read-plotfiles>`:

.. autofunction:: amrex.space3d.read_particles

.. autoclass:: amrex.space3d.ParticleHeader
   :members:
   :undoc-members:

.. _usage-api-eb:

Embedded Boundaries
-------------------

Embedded boundary (EB) support in pyAMReX is still minimal. To build pyAMReX with
EB support, you need to add ``-DAMReX_EB=ON`` to CMake build options.

.. autofunction:: amrex.space3d.EB2_Build

.. autoclass:: amrex.space3d.EBFArrayBoxFactory
   :members:
   :undoc-members:

.. autofunction:: amrex.space3d.makeEBFabFactory
