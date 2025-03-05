.. _developers-testing:

Testing
=======

Preparation
-----------

Prepare for running tests of pyAMReX by :ref:`building pyAMReX from source <install-developers>`.

In order to run our tests, you need to have a few :ref:`Python packages installed <install-dependencies>`:

.. code-block:: sh

   python3 -m pip install -U pip
   python3 -m pip install -U build packaging setuptools[core] wheel pytest
   python3 -m pip install -r requirements.txt


Run
---

You can run all our tests with:

.. code-block:: sh

   ctest --test-dir build --output-on-failure


Further Options
---------------

For faster compile-and-test iterations, build with ``-DpyAMReX_IPO=OFF``:

.. code-block:: sh

   ctest -S . -B build -DpyAMReX_IPO=OFF

After successful installation, with

.. code-block:: sh

   ctest --test-dir build --target pip_install

you can also run the unit tests individually.
For ``AMReX_MPI=ON``, please prepend the following commands with ``mpiexec -np <NUM_PROCS>``

.. code-block:: sh

   # Run all tests
   python3 -m pytest tests/

   # Run tests from a single file
   python3 -m pytest tests/test_intvect.py

   # Run a single test (useful during debugging)
   python3 -m pytest tests/test_intvect.py::test_iv_conversions

   # Run all tests, do not capture "print" output and be verbose
   python3 -m pytest -s -vvvv tests/
