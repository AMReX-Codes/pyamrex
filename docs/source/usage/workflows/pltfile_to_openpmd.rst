.. _usage-how-to-pltfile-to-openpmd:

Convert Plotfiles to openPMD
============================

The ``pltfile-to-openpmd`` tool converts native AMReX plotfiles (mesh fields and particles) to `openPMD <https://www.openpmd.org>`__ series, readable by openPMD-viewer, ParaView, VisIt and the wider openPMD ecosystem.
The openPMD backend - HDF5 (``.h5``), ADIOS2 (``.bp``) or JSON (``.json``) - is selected by the output file extension.
It requires the `openpmd-api <https://openpmd-api.readthedocs.io>`__ Python package (``pip install openpmd-api``).

Each plotfile becomes one openPMD iteration, indexed by its level-0 step number:

.. code-block:: bash

   pltfile-to-openpmd -o sim_%T.h5 diags/plt00000 diags/plt00100

   # equivalent, e.g., if the entry point is not on PATH:
   python -m amrex.tools.pltfile_to_openpmd -o sim_%T.h5 diags/plt?????

or, from Python:

.. code-block:: python

   from amrex.tools.pltfile_to_openpmd import convert

   convert(["diags/plt00000", "diags/plt00100"], "sim_%T.h5")

Run ``pltfile-to-openpmd --help`` for all options (field/species selection, skipping particles, recording a time step, quiet mode).
The plotfile's dimensionality is detected automatically and the matching ``amrex.space{1,2,3}d`` module is used.

Data Mapping
------------

The conversion is information-preserving:

* Field data is copied at its on-disk precision, per AMR level, with every AMReX grid stored as one chunk of the level's dataset.
  AMReX's Fortran axis order is reversed into openPMD's C order (``axisLabels`` e.g. ``["z", "y", "x"]``).
* Mesh refinement levels follow the openPMD `PatchBasedMeshRefinement <https://github.com/openPMD/openPMD-standard/pull/252>`__ extension proposal: the coarsest level keeps the plain record name (readable by every openPMD tool), finer levels are suffixed ``_lvl<N>`` and carry a ``refinementRatio`` attribute.
* Particles are converted per species (discovered via :py:func:`~amrex.space3d.list_particle_species` and read via :py:func:`~amrex.space3d.read_particles`, see :ref:`Read Back Plotfiles <usage-how-to-read-plotfiles>`), with their component names verbatim, unpacked ``id`` and ``amrex_cpu`` records, and a constant ``positionOffset`` of zero.
* AMReX metadata without an openPMD equivalent is stored in ``amrex_``-prefixed attributes: per-level steps, box arrays, ghost-cell widths, the coordinate system, and per-species file metadata.
  Together with the chunk layout, this suffices to reconstruct the plotfile structure.

Limitations, by design of the source format and this tool:

* Plotfiles carry no unit metadata, so ``unitSI`` is 1 and ``unitDimension`` is dimensionless; record a time step with ``--dt`` if needed.
* The on-disk *ordering* of particles is not preserved (identities are, via ``id``/``amrex_cpu``); their mesh-refinement level assignment is recoverable from positions and the stored per-level box arrays - the same rule AMReX applies in ``Redistribute()``.
* Ghost cell *values* are not written (the valid region is); the ghost width is recorded in ``amrex_n_grow``.
