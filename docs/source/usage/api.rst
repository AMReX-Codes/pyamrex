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


.. _usage-api-amrcore:

AmrCore
-------

.. autoclass:: amrex.space3d.AmrInfo
   :members:
   :undoc-members:

.. autoclass:: amrex.space3d.AmrMesh
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
