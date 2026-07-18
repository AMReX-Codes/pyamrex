/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>

#include <array>
#include <sstream>
#include <string>
#include <optional>


void init_RealBox(nb::module_ &m) {
    using namespace amrex;

    nb::class_< RealBox >(m, "RealBox")
        .def("__repr__",
             [](nb::object& obj) {
                 nb::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = nb::cast<std::string>(py_name);
                 const auto rb = nb::cast<RealBox>(obj);
                 std::stringstream s;
                 s << rb;
                 return "<amrex." + name + " " + s.str() + ">";
            }
        )
        .def("__str",
             [](const RealBox& rb) {
                 std::stringstream s;
                 s << rb;
                 return s.str();
             })


        .def(nb::init())
        .def(nb::init<AMREX_D_DECL(Real, Real, Real),
                      AMREX_D_DECL(Real, Real, Real)>(),
             AMREX_D_DECL(nb::arg("x_lo"), nb::arg("y_lo"), nb::arg("z_lo")),
             AMREX_D_DECL(nb::arg("x_hi"), nb::arg("y_hi"), nb::arg("z_hi"))
        )
        .def(nb::init<const std::array<Real, AMREX_SPACEDIM>&,
                      const std::array<Real, AMREX_SPACEDIM>& >(),
             nb::arg("a_lo"), nb::arg("a_hi")
        )
        .def("__init__",
            [](RealBox *self, const Box bx, Array<Real, AMREX_SPACEDIM> dx,
               Array<Real, AMREX_SPACEDIM> base) {
                new (self) RealBox(bx, dx.data(), base.data());
            },
            nb::arg("bx"), nb::arg("dx"), nb::arg("base")
        )

        .def_prop_ro(
            "xlo",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xlo {AMREX_D_DECL(
                    rb.lo(0), rb.lo(1), rb.lo(2)
                )};
                return xlo;
            }
        )
        .def_prop_ro(
            "xhi",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xhi {AMREX_D_DECL(
                    rb.hi(0), rb.hi(1), rb.hi(2)
                )};
                return xhi;
            }
        )

        .def("lo", nb::overload_cast<int>(&RealBox::lo, nb::const_), "Get ith component of ``xlo``")
        .def("lo",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xlo {AMREX_D_DECL(
                    rb.lo(0), rb.lo(1), rb.lo(2)
                )};
                return xlo;
            },
            "Get all components of ``xlo``"
        )
        .def("hi", nb::overload_cast<int>(&RealBox::hi, nb::const_), "Get ith component of ``xhi``")
        .def("hi",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xhi {AMREX_D_DECL(
                    rb.hi(0), rb.hi(1), rb.hi(2)
                )};
                return xhi;
            },
            "Get all components of ``xhi``"
        )
        .def("setLo",
            [](RealBox & rb, const std::vector<Real>& a_lo){
                rb.setLo(a_lo.data() );
            },
            "Get ith component of ``xlo``"
        )
        .def("setLo", nb::overload_cast<int,Real>(&RealBox::setLo), "Get all components of ``xlo``")
        .def("setHi",
            [](RealBox & rb, const std::vector<Real>& a_hi){
                rb.setHi(a_hi.data() );
            },
            "Get all components of ``xlo``"
        )
        .def("setHi", nb::overload_cast<int,Real>(&RealBox::setHi), "Get ith component of ``xhi``")
        .def("length", &RealBox::length)
        .def("ok", &RealBox::ok, "Determine if RealBox satisfies ``xlo[i]<xhi[i]`` for ``i=0,1,...,AMREX_SPACEDIM``.")
        .def("volume", &RealBox::volume)
        .def("contains",
            [](RealBox& rb, XDim3 point, Real eps) {
                return rb.contains(point, eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            nb::arg("rb"),nb::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const RealVect& pt, Real eps) {
                return rb.contains(pt, eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            nb::arg("rb"),nb::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const RealBox& rb2, Real eps) {
                return rb.contains(rb2, eps );
            },
            "Determine if RealBox contains another RealBox, within tolerance ``eps``",
            nb::arg("rb"),nb::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const std::vector<Real>& pt, Real eps) {
                return rb.contains(pt.data(), eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            nb::arg("rb"),nb::arg("eps") = 0.0
        )
        .def("intersects", &RealBox::intersects, "determine if box intersects with a box")
    ;
    m.def("AlmostEqual",
            [](const RealBox& rb1, const RealBox& rb2, Real eps) {
                return AlmostEqual(rb1,rb2,eps);
            },
            "Determine if two boxes are equal to within a tolerance",
            nb::arg("rb1"), nb::arg("rb2"), nb::arg("eps") = 0.0);
}
