.. _developers-nanobind-extensions:

Downstream Nanobind Extensions
==============================

pyAMReX 26.07 changes its native binding ABI from pybind11 to nanobind 2.12.
The public Python package names and the historical private extension basenames
are unchanged, but downstream native extensions must rebuild and migrate their
bindings in the same release cycle.

Type sharing
------------

pyAMReX uses nanobind's default/global type domain. A separately built
nanobind extension can therefore accept registered AMReX types and return
borrowed references to them. Do not set ``NB_DOMAIN`` in an extension that
needs to exchange pyAMReX objects: a private domain deliberately creates an
independent type registry.

The minimum supported nanobind version is recorded in ``dependencies.json``.
Downstream projects should use a compatible nanobind release and link their
extension against the matching dimensional ``AMReX::amrex_*d`` target found
through the installed pyAMReX package. The executable contract is the project
in ``tests/downstream_nanobind``.

pybind11 interoperability
-------------------------

pybind11 and nanobind do not share registered C++ type metadata. A pybind11
extension cannot directly accept a pyAMReX object after this ABI transition,
even when both bindings name the same C++ type. Such extensions must migrate
to nanobind, use the default domain, and rebuild. Process-independent exchange
formats such as NumPy arrays remain an alternative boundary when coordinated
migration is not possible.

The first nanobind-based pyAMReX release is consequently a native ABI
transition, despite preserving its Python API.
