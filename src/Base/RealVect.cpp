/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_RealBox.H>
#include <AMReX_IntVect.H>

#include <sstream>
#include <string>
#include <optional>
#include <vector>


void init_RealVect(py::module &m) {
    using namespace amrex;

    auto py_realvect = py::class_< RealVect>(m, "RealVect")
          .def("__repr__",
               [](py::object& obj) {
                    py::str py_name = obj.attr("__class__").attr("__name__");
                    const std::string name = py_name;
                    const auto rv = obj.cast<RealVect>();
                    std::stringstream s;
                    s << rv;
                    return "<amrex." + name + " " + s.str() + ">";
               }
          )
          .def("__str",
               [](const RealVect& rv) {
                    std::stringstream s;
                    s << rv;
                    return s.str();
               })

          .def(py::init())
#if (AMREX_SPACEDIM > 1)
          .def(py::init<AMREX_D_DECL(Real, Real, Real)>())
#endif
          .def(py::init<const IntVect&>())
          .def(py::init<const std::vector<Real>&>())
          .def(py::init<Real>())
          .def("__getitem__",
               [](const RealVect& v, const int i) {
                    const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                    if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                         throw py::index_error(
                              "Index must be between 0 and " +
                              std::to_string(AMREX_SPACEDIM));
                    return v[ii];
               })
          .def("__setitem__",
               [](RealVect& v, const int i, const Real& val) {
                    const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                    if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                         throw py::index_error(
                              "Index must be between 0 and " +
                              std::to_string(AMREX_SPACEDIM));
                    return v[ii] = val;
               })
          .def("__eq__",&RealVect::operator==)
          .def("__ne__",&RealVect::operator!=)
          .def("__lt__",&RealVect::operator<)
          .def("__le__",&RealVect::operator<=)
          .def("__gt__",&RealVect::operator>)
          .def("__ge__",&RealVect::operator>=)

          .def("__iadd__",
               py::overload_cast<Real>(&RealVect::operator+=))
          .def("__iadd__",
               py::overload_cast<const RealVect&>(&RealVect::operator+=))
          .def("__add__",
               py::overload_cast<Real>(&RealVect::operator+, py::const_))
          .def(float() + py::self)
          .def(py::self + py::self)

          .def("__isub__",
               py::overload_cast<Real>(&RealVect::operator-=))
          .def("__isub__",
               py::overload_cast<const RealVect&>(&RealVect::operator-=))
          .def(float() - py::self)
          .def(py::self - py::self)
          .def("__sub__",
               py::overload_cast<Real>(&RealVect::operator-, py::const_))

          .def("__imul__",
               py::overload_cast<Real>(&RealVect::operator*=))
          .def("__imul__",
               py::overload_cast<const RealVect&>(&RealVect::operator*=))
          .def(float() * py::self)
          .def(py::self * py::self)
          .def("dotProduct", &RealVect::dotProduct, "Return dot product of this vector with another")
#if (AMREX_SPACEDIM == 3)
          .def("crossProduct", &RealVect::crossProduct<>, "Return cross product of this vector with another")
#endif
          .def("__mul__",
               py::overload_cast<Real>(&RealVect::operator*, py::const_))

          .def(py::self /= float())
          .def(py::self / float())
          .def(float() / py::self)
          .def(py::self / py::self)

          .def("scale", &RealVect::scale, "Multiplify each component of this vector by a scalar")
          .def("floor", &RealVect::floor, "Return an ``IntVect`` whose components are the std::floor of the vector components")
          .def("ceil", &RealVect::ceil, "Return an ``IntVect`` whose components are the std::ceil of the vector components")
          .def("round", &RealVect::round, "Return an ``IntVect`` whose components are the std::round of the vector components")

          .def("min", &RealVect::min, "Replace vector with the component-wise minima of this vector and another")
          .def("max", &RealVect::max, "Replace vector with the component-wise maxima of this vector and another")
          // ------ UNARY:----
          .def(+py::self)
          .def(-py::self)

          .def_property_readonly("sum", &RealVect::sum, "Sum of the components of this vector")
          .def_property_readonly("vectorLength", &RealVect::vectorLength, "Length or 2-Norm of this vector")
          .def_property_readonly("radSquared", &RealVect::radSquared, "Length squared of this vector")
          .def_property_readonly("product", &RealVect::product, "Product of entries of this vector")
          .def("minDir", &RealVect::minDir, "direction or index of minimum value of this vector")
          .def("maxDir", &RealVect::maxDir, "direction or index of maximum value of this vector")

          // Static
          .def_static("zero_vector", &RealVect::TheZeroVector)
          .def_static("unit_vector", &RealVect::TheUnitVector)

          .def("BASISREALV", [](int dir) -> RealVect {
              return amrex::BASISREALV(dir); },
              "return basis vector in given coordinate direction")
     ;

     m.def("min", [](const RealVect& a, const RealVect& b) {
         return min(a,b);
     });
     m.def("max", [](const RealVect& a, const RealVect& b) {
         return max(a,b);
     });
}
