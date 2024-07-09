:orphan:

pyAMReX
-------

The Python binding pyAMReX bridges the compute in AMReX block-structured codes and data science:
it provides zero-copy application GPU data access for AI/ML, in situ analysis, application coupling and enables rapid, massively parallel prototyping.

pyAMReX is part of the `AMReX software ecosystem <https://amrex-codes.github.io>`__ and builds directly on the AMReX C++ library.


.. _contact:

Contact us
^^^^^^^^^^

If you are starting using pyAMReX, or if you have a user question, please pop in our `GitHub discussions page <https://github.com/AMReX-Codes/pyamrex/discussions>`__ and get in touch with the community.

The `pyAMReX GitHub repo <https://github.com/AMReX-Codes/pyamrex>`__ is the main communication platform.
Have a look at the action icons on the top right of the web page: feel free to watch the repo if you want to receive updates, or to star the repo to support the project.
For bug reports or to request new features, you can also open a new `issue <https://github.com/AMReX-Codes/pyamrex/issues>`__.

On our `discussion page <https://github.com/AMReX-Codes/pyamrex/discussions>`__, you can find already answered questions, add new questions, get help with installation procedures, discuss ideas or share comments.

.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#installation,
   section#usage,
   section#development,
   section#maintenance,
   section#epilogue {
       display:none;
   }
   </style>

.. toctree::
   :hidden:

   coc
   acknowledge_us

Installation
------------
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/users
   install/cmake
..   install/hpc
..   install/changelog
..   install/upgrade

Usage
-----
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/examples
   usage/api
   usage/zerocopy
   usage/compute
   usage/workflows
..   usage/tests

Development
-----------
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   developers/testing
   developers/documentation
   developers/repo_organization
   developers/implementation
   developers/doxygen
   developers/debugging
..   developers/contributing

Maintenance
-----------
.. toctree::
   :caption: MAINTENANCE
   :maxdepth: 1
   :hidden:

   maintenance/release
..   maintenance/performance_tests

Epilogue
--------
.. toctree::
   :caption: EPILOGUE
   :maxdepth: 1
   :hidden:

   glossary
   acknowledgements
