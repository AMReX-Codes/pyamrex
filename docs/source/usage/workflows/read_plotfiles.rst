.. _usage-how-to-read-plotfiles:

Read Back Plotfiles
===================

AMReX applications write field (mesh) and particle data as `plotfiles and checkpoint files <https://amrex-codes.github.io/amrex/docs_html/IO.html>`__.
This page shows how to read such files back into pyAMReX objects, using the same AMReX C++ reader logic that applications use for restarts.

The returned objects are regular pyAMReX containers:
field and particle data can be accessed with :ref:`zero-copy views <usage-zerocopy>` from NumPy, CuPy, and other libraries and processed as shown in :ref:`Compute <usage-compute>`.

.. note::

   For quick data analysis and visualization of AMReX plotfiles, `yt <https://yt-project.org>`__ is a feature-rich, dedicated package.
   Reading plotfiles directly with pyAMReX, as shown here, is ideal when the data shall be placed 1:1 into AMReX data structures again, e.g., to initialize or restart a simulation, to couple codes, or to post-process with the exact AMReX block-structure, zero-copy math and MPI-parallelism.


Field (Mesh) Data
-----------------

Use :py:class:`~amrex.space3d.PlotFileData` to open a plotfile, query its meta-data and read per-level field data as :py:class:`~amrex.space3d.MultiFab`:

.. literalinclude:: ../../../../tests/test_plotfiledata.py
   :language: python3
   :dedent: 4
   :start-after: # Manual: Read Plotfile Mesh START
   :end-before: # Manual: Read Plotfile Mesh END


Particle Data
-------------

Particle data in a plotfile or checkpoint is stored in a sub-directory (often called ``particles``, per species name, or similar).
Use :py:func:`~amrex.space3d.read_particles` to read it back into a particle container - no prior knowledge of the writing container's compile-time layout is needed, the number and names of the particle components are discovered from the file:

.. literalinclude:: ../../../../tests/test_readparticles.py
   :language: python3
   :dedent: 4
   :start-after: # Manual: Read Plotfile Particles START
   :end-before: # Manual: Read Plotfile Particles END

For multi-level files, particles from all levels are read.
The auto-created container is single-level: all particles are placed on level 0, preserving their positions and components (only the mesh-refinement level association is flattened).
To move particles on MR levels again, an explicit call to ``Redistribute`` needs to be made after reading.

To only inspect the on-disk layout, e.g., to check which components a file contains before reading it, use :py:class:`~amrex.space3d.ParticleHeader`:

.. literalinclude:: ../../../../tests/test_readparticles.py
   :language: python3
   :dedent: 4
   :start-after: # Manual: Read Particle Header START
   :end-before: # Manual: Read Particle Header END


Read Into an Existing Container
-------------------------------

:py:func:`~amrex.space3d.read_particles` can also fill an existing, geometry-defined container, via its ``container`` argument.
Use this to control the container type, MPI decomposition and mesh-refinement levels - or when the geometry cannot be recovered from the file itself:
application *checkpoints* store their top-level ``Header`` in an application-specific format, so their AMR geometry must be defined by the application before reading.

.. literalinclude:: ../../../../tests/test_readparticles.py
   :language: python3
   :dedent: 4
   :start-after: # Manual: Read Particles Existing Container START
   :end-before: # Manual: Read Particles Existing Container END

Related low-level APIs, mirroring the AMReX C++ workflows for restarts:

* ``pc.restart_checkpoint(dir, file, is_checkpoint)``: restore particles from a checkpoint/plotfile into a live container,
* ``VisMF.Read(name)`` / ``VisMF.Write(mf, name)``: read/write a single :py:class:`~amrex.space3d.MultiFab` at the raw multifab-file granularity,
* :py:func:`~amrex.space3d.write_single_level_plotfile` / :py:func:`~amrex.space3d.write_multi_level_plotfile` and ``pc.write_plotfile(dir, name, real_comp_names, int_comp_names)``: write fields and particles, e.g., to generate the files read back above.
