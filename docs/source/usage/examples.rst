.. _usage_examples:

Examples
========

.. _usage_run:

When to Use pyAMReX?
--------------------

pyAMReX is usually used for these kinds of workflows:

1. To enhance an existing AMReX application with Python, Data-Science and AI/ML capabilities,
2. To write a standalone application or test on AMReX, rapidly prototyped in Python.


Enhance an Application
----------------------

pyAMReX is used in large, production-quality high-performance applications.
See the following examples:

ImpactX
"""""""

`ImpactX <https://impactx.readthedocs.io>`__ is an s-based beam dynamics code including space charge effects.

* `Python examples <https://impactx.readthedocs.io/en/latest/usage/examples.html>`__
* `Python implementation <https://github.com/ECP-WarpX/impactx/tree/development/src/python>`__
* Highlight example: `Fully GPU-accelerated PyTorch+ImpactX simulation <https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html>`__


WarpX
"""""
`WarpX <https://warpx.readthedocs.io>`__ is an advanced, time-based electromagnetic & electrostatic Particle-In-Cell code.

* `Python (PICMI) examples <https://warpx.readthedocs.io/en/latest/usage/examples.html>`__
* `Python implementation <https://github.com/ECP-WarpX/WarpX/tree/development/Source/Python>`__
* Detailed workflow: `Extend a WarpX Simulation with Python <https://warpx.readthedocs.io/en/latest/usage/workflows/python_extend.html>`__


Standalone
----------

Please see the `AMReX Tutorials <https://amrex-codes.github.io/amrex/tutorials_html/Python_Tutorial.html>`__ for Python-written, GPU-accelerated AMReX examples:

* `MultiFab example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/MultiFab/main.py>`__
* `Heat Equation example <https://github.com/AMReX-Codes/amrex-tutorials/blob/main/GuidedTutorials/HeatEquation/Source/main.py>`__


Unit Tests
----------

We ensure the correctness of pyAMReX with `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`__.
Our tests are small, plentiful and can be found in the source code, see:

* `tests/ <https://github.com/AMReX-Codes/pyamrex/tree/development/tests>`__

The following sections on :ref:`compute workflows <usage-compute>` go in detail through selected unit tests, too.
