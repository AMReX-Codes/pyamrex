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


void init_RealVect(nb::module_ &m) {
    using namespace amrex;

    auto py_realvect = nb::class_< RealVect>(m, "RealVect")
          .def("__repr__",
               [](nb::object& obj) {
                    nb::str py_name = obj.attr("__class__").attr("__name__");
                    const std::string name = nb::cast<std::string>(py_name);
                    const auto rv = nb::cast<RealVect>(obj);
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

          .def(nb::init())
#if (AMREX_SPACEDIM > 1)
          .def(nb::init<AMREX_D_DECL(Real, Real, Real)>())
#endif
          .def(nb::init<const IntVect&>())
          .def(nb::init<const std::vector<Real>&>())
          .def(nb::init<Real>())
          .def("__getitem__",
               [](const RealVect& v, const int i) {
                    const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                    if ((ii < 0) || (ii >= AMREX_SPACEDIM)) {
                         auto message = "Index must be between 0 and " +
                                        std::to_string(AMREX_SPACEDIM);
                         throw nb::index_error(message.c_str());
                    }
                    return v[ii];
               })
          .def("__setitem__",
               [](RealVect& v, const int i, const Real& val) {
                    const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                    if ((ii < 0) || (ii >= AMREX_SPACEDIM)) {
                         auto message = "Index must be between 0 and " +
                                        std::to_string(AMREX_SPACEDIM);
                         throw nb::index_error(message.c_str());
                    }
                    return v[ii] = val;
               })
          .def("__eq__",&RealVect::operator==)
          .def("__ne__",&RealVect::operator!=)
          .def("__lt__",&RealVect::operator<)
          .def("__le__",&RealVect::operator<=)
          .def("__gt__",&RealVect::operator>)
          .def("__ge__",&RealVect::operator>=)

          .def("__iadd__",
               nb::overload_cast<Real>(&RealVect::operator+=))
          .def("__iadd__",
               nb::overload_cast<const RealVect&>(&RealVect::operator+=))
          .def("__add__",
               nb::overload_cast<Real>(&RealVect::operator+, nb::const_))
          .def(float() + nb::self)
          .def(nb::self + nb::self)

          .def("__isub__",
               nb::overload_cast<Real>(&RealVect::operator-=))
          .def("__isub__",
               nb::overload_cast<const RealVect&>(&RealVect::operator-=))
          .def(float() - nb::self)
          .def(nb::self - nb::self)
          .def("__sub__",
               nb::overload_cast<Real>(&RealVect::operator-, nb::const_))

          .def("__imul__",
               nb::overload_cast<Real>(&RealVect::operator*=))
          .def("__imul__",
               nb::overload_cast<const RealVect&>(&RealVect::operator*=))
          .def(float() * nb::self)
          .def(nb::self * nb::self)
          .def("dotProduct", &RealVect::dotProduct, "Return dot product of this vector with another")
#if (AMREX_SPACEDIM == 3)
          .def("crossProduct",
               [](const RealVect& lhs, const RealVect& rhs) {
                   return lhs.crossProduct(rhs);
               },
               "Return cross product of this vector with another")
#endif
          .def("__mul__",
               nb::overload_cast<Real>(&RealVect::operator*, nb::const_))

          .def(nb::self /= float())
          .def(nb::self / float())
          .def(float() / nb::self)
          .def(nb::self / nb::self)

          .def("scale", &RealVect::scale, "Multiplify each component of this vector by a scalar")
          .def("floor", &RealVect::floor, "Return an ``IntVect`` whose components are the std::floor of the vector components")
          .def("ceil", &RealVect::ceil, "Return an ``IntVect`` whose components are the std::ceil of the vector components")
          .def("round", &RealVect::round, "Return an ``IntVect`` whose components are the std::round of the vector components")

          .def("min", &RealVect::min, "Replace vector with the component-wise minima of this vector and another")
          .def("max", &RealVect::max, "Replace vector with the component-wise maxima of this vector and another")
          // ------ UNARY:----
          .def(+nb::self)
          .def(-nb::self)

          .def_prop_ro("sum", &RealVect::sum, "Sum of the components of this vector")
          .def_prop_ro("vectorLength", &RealVect::vectorLength, "Length or 2-Norm of this vector")
          .def_prop_ro("radSquared", &RealVect::radSquared, "Length squared of this vector")
          .def_prop_ro("product", &RealVect::product, "Product of entries of this vector")
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
