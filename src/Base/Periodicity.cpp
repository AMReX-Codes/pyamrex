/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Periodicity.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_SPACE.H>

#include <ios>
#include <sstream>


void init_Periodicity(py::module &m)
{
    using namespace amrex;

    py::class_< Periodicity >(m, "Periodicity")
        .def("__repr__",
            [](Periodicity const & p) {
                std::stringstream s;
                s << std::boolalpha
                  << AMREX_D_TERM(
                        p.isPeriodic(0),
                        << ", " << p.isPeriodic(1),
                        << ", " << p.isPeriodic(2));
                return "<amrex.Periodicity per direction '" + s.str() + "'>";
            }
        )

        .def(py::init<>())
        .def(py::init< IntVect const & >())

        .def_property_readonly("is_any_periodic", &Periodicity::isAnyPeriodic)
        .def_property_readonly("is_all_periodic", &Periodicity::isAllPeriodic)
        .def_property_readonly("domain", &Periodicity::Domain,
                               "Cell-centered domain Box \"infinitely\" long in non-periodic directions.")
        .def_property_readonly("shift_IntVect", &Periodicity::shiftIntVect)

        .def("is_periodic", &Periodicity::isPeriodic,
             py::arg("dir"))
        .def("__getitem__", &Periodicity::isPeriodic,
             py::arg("dir"))

        .def(pybind11::self == pybind11::self)
        //.def(pybind11::self != pybind11::self)

        .def_static("non_periodic", &Periodicity::NonPeriodic,
            "Return the Periodicity object that is not periodic in any direction")
    ;
}
