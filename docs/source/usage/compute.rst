.. _usage-compute:

Compute
=======

With zero-copy read and write access to data structures, this section presents how to compute in pyAMReX.

Since the pyAMReX data containers are wrappers to the C++ AMReX objects, it is worth reading:

* `AMReX Basics <https://amrex-codes.github.io/amrex/docs_html/Basics_Chapter.html>`__ and
* `AMReX parallelization strategy for MPI+X (e.g, GPUs, CPUs) <https://amrex-codes.github.io/amrex/docs_html/GPU.html>`__.

As a very short, simplified overview, this narrows down to:

* AMReX decomposes **index space** into **rectangular, block-structured, regular grids**,
* block are often intentionally slightly **over-decomposed**, there is >= one block per compute unit (CPU core or GPU),
* **particles** are chunked/tiled and usually decomposed like the field blocks,
* **refinement levels** are represented as (potentially sparse) levels of blocks.

Computations are thus performed (mostly) on whole blocks, which enables to use compute advanced acceleration techniques on CPUs or GPUs.


.. _usage-compute-fields:

Fields
------

The most common data structure to interact with is a `MultiFab <https://amrex-codes.github.io/amrex/docs_html/Basics.html#fabarray-multifab-and-imultifab>`__, which is a collection of boxes with associated field data.
The field data can have more than one component (in the slowest varying index), but all components have the same `staggering/centering <https://amrex-codes.github.io/amrex/docs_html/Basics.html#box>`__.

This is how to iterate and potentially compute for all blocks assigned to a local process in pyAMReX:

.. tab-set::

   .. tab-item:: Simple

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Simple START
         :end-before: # Manual: Compute Mfab Simple END

   .. tab-item:: Detailed

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Detailed START
         :end-before: # Manual: Compute Mfab Detailed END

   .. tab-item:: Global

      .. literalinclude:: ../../../tests/test_multifab.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Compute Mfab Global START
         :end-before: # Manual: Compute Mfab Global END


For a complete physics example that uses CPU/GPU agnostic Python code for computation on fields, see:

* `Heat Equation example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/HeatEquation/Source/main.py>`__

For many small CPU and GPU examples on how to compute on fields, see the following test cases:

* `MultiFab example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/MultiFab/main.py>`__

* .. dropdown:: Examples in ``test_multifab.py``

     .. literalinclude:: ../../../tests/test_multifab.py
        :language: python3
        :caption: This files is in ``tests/test_multifab.py``.


.. _usage-compute-particles:

Particles
---------

AMReX `Particles <https://amrex-codes.github.io/amrex/docs_html/Particle_Chapter.html>`__ are stored in the `ParticleContainer <https://amrex-codes.github.io/amrex/docs_html/Particle.html#the-particlecontainer>`__ class.

There are a few small differences to the `iteration over a ParticleContainer <https://amrex-codes.github.io/amrex/docs_html/Particle.html#iterating-over-particles>`__ compared to a ``MultiFab``:

* ``ParticleContainer`` is aware of mesh-refinement levels,
* AMReX supports a variety of data layouts for particles, the modern pure SoA + runtime attribute layout and the legacy AoS + SoA + runtime SoA attributes layout.

Here is the general structure for computing on particles:

.. tab-set::

   .. tab-item:: Modern (pure SoA) Layout

      .. tab-set::

         .. tab-item:: Simple

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Simple pti START
               :end-before: # Manual: Pure SoA Compute PC Simple pti END

         .. tab-item:: Detailed

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Detailed START
               :end-before: # Manual: Pure SoA Compute PC Detailed END

         .. tab-item:: Pandas (read-only)

            .. literalinclude:: ../../../tests/test_particleContainer.py
               :language: python3
               :dedent: 4
               :start-after: # Manual: Pure SoA Compute PC Pandas START
               :end-before: # Manual: Pure SoA Compute PC Pandas END


   .. tab-item:: Legacy (AoS + SoA) Layout

      .. literalinclude:: ../../../tests/test_particleContainer.py
         :language: python3
         :dedent: 4
         :start-after: # Manual: Legacy Compute PC Detailed START
         :end-before: # Manual: Legacy Compute PC Detailed END

For many small CPU and GPU examples on how to compute on particles, see the following test cases:

* .. dropdown:: Examples in ``test_particleContainer.py``

     .. literalinclude:: ../../../tests/test_particleContainer.py
        :language: python3
        :caption: This files is in ``tests/test_particleContainer.py``.

* .. dropdown:: Examples in ``test_aos.py``

     .. literalinclude:: ../../../tests/test_aos.py
        :language: python3
        :caption: This files is in ``tests/test_aos.py``.

* .. dropdown:: Examples in ``test_soa.py``

     .. literalinclude:: ../../../tests/test_soa.py
        :language: python3
        :caption: This files is in ``tests/test_soa.py``.


Other C++ Calls
---------------

pyAMReX exposes many more C++-written and GPU-accelerated AMReX functions for :py:class:`~amrex.space3d.MultiFab` and :ref:`particles <usage-api-particles>` to Python, which can be used to set, reduce, rescale, redistribute, etc. contained data.
Check out the detailed :ref:`API docs for more details <usage-api>` and use further third party libraries at will on the exposed data structures, replacing the hot loops described above.

You can also leave the Python world in pyAMReX and go back to C++ whenever needed.
For :ref:`some applications <usage_run>`, pyAMReX might act as *scriptable glue* that transports fields and particles from one (C++) function to another without recompilation, by exposing the functions and methods to Python.
