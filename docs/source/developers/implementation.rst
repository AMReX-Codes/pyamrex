.. _developers-implementation:

Implementation Details
======================

For now, please see these presentations:

- `A. Huebl et al., NREL virtual seminar, December 2022 <https://docs.google.com/presentation/d/1X3_OERbFcZEd-awNEEEoOd98VT-gmB1UDEfK7A4NgGc/edit?usp=sharing>`__
- `A. Huebl et al., "Exascale and ML Models for Accelerator Simulations", 6th European Advanced Accelerator Concepts workshop (EAAC23), Isola d'Elba, Italy, Sep 17 – 23, 2023 DOI:10.5281/zenodo.8362549 <https://doi.org/10.5281/zenodo.8362549>`__


Zero-Copy Data APIs
-------------------

pyAMReX implements the following `standardized data APIs <https://data-apis.org>`__:

- ``__array_interface__`` (CPU)
- ``__cuda_array_interface__`` (CUDA GPU)
- ``DLPack`` (`coming soon <https://github.com/AMReX-Codes/pyamrex/issues/9#issuecomment-1644288546>`__)

These APIs are automatically used when creating "views" (non-copy) numpy arrays, cupy arrays, PyTorch tensors, etc. from AMReX objects such as ``Array4`` and particle arrays.
