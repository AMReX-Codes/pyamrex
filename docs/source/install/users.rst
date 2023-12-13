.. _install-users:

Users
=====

.. raw:: html

   <style>
   .rst-content section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Our community is here to help.
Please `report installation problems <https://github.com/AMReX-Codes/pyamrex/issues>`_ in case you should get stuck.

Choose **one** of the installation methods below to get started:


.. _install-conda:

.. only:: html

   .. image:: conda.svg

Using the Conda Package
-----------------------

A package for pyAMReX is available via the `Conda <https://conda.io>`_ package manager.

.. tip::

   We recommend to configure your conda to use the faster ``libmamba`` `dependency solver <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`__.

   .. code-block:: bash

      conda update -n base conda
      conda install -n base conda-libmamba-solver
      conda config --set solver libmamba

   We recommend to deactivate that conda self-activates its ``base`` environment.
   This `avoids interference with the system and other package managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__.

   .. code-block:: bash

      conda config --set auto_activate_base false

.. code-block:: bash

   conda create -n pyamrex -c conda-forge pyamrex
   conda activate pyamrex

.. note::

   The ``pyamrex`` `conda package <https://anaconda.org/conda-forge/pyamrex>`__ does not yet provide GPU support.


.. _install-spack:

.. only:: html

   .. image:: spack.svg

Using the Spack Package
-----------------------

.. note::

   Coming soon.


.. _install-pypi:

.. only:: html

   .. image:: pypi.svg

Using the PyPI Package
----------------------

.. note::

   Coming soon.


.. _install-brew:

.. only:: html

   .. image:: brew.svg

Using the Brew Package
----------------------

.. note::

   Coming soon.


.. _install-cmake:

.. only:: html

   .. image:: cmake.svg

From Source with CMake
----------------------

After installing the :ref:`pyAMReX dependencies <install-dependencies>`, you can also install pyAMReX from source with `CMake <https://cmake.org/>`_:

.. code-block:: bash

   # get the source code
   git clone https://github.com/AMReX-Codes/pyamrex.git $HOME/src/pyamrex
   cd $HOME/src/pyamrex

   # configure
   cmake -S . -B build

   # optional: change configuration
   ccmake build

   # compile & install
   #   on Windows:          --config Release
   cmake --build build -j 4 --target pip_install

We document the details in the :ref:`developer installation <install-developers>`.

Tips for macOS Users
--------------------

.. tip::

   Before getting started with package managers, please check what you manually installed in ``/usr/local``.
   If you find entries in ``bin/``, ``lib/`` et al. that look like you manually installed MPI, HDF5 or other software in the past, then remove those files first.

   If you find software such as MPI in the same directories that are shown as symbolic links then it is likely you `brew installed <https://brew.sh/>`__ software before.
   If you are trying annother package manager than ``brew``, run `brew unlink ... <https://docs.brew.sh/Tips-N%27-Tricks#quickly-remove-something-from-usrlocal>`__ on such packages first to avoid software incompatibilities.

See also: A. Huebl, `Working With Multiple Package Managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__, `Collegeville Workshop (CW20) <https://collegeville.github.io/CW20/>`_, 2020
