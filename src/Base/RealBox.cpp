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


void init_RealBox(py::module &m) {
    using namespace amrex;

    py::class_< RealBox >(m, "RealBox")
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto rb = obj.cast<RealBox>();
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


        .def(py::init())
        .def(py::init<AMREX_D_DECL(Real, Real, Real),
                      AMREX_D_DECL(Real, Real, Real)>(),
             AMREX_D_DECL(py::arg("x_lo"), py::arg("y_lo"), py::arg("z_lo")),
             AMREX_D_DECL(py::arg("x_hi"), py::arg("y_hi"), py::arg("z_hi"))
        )
        .def(py::init<const std::array<Real, AMREX_SPACEDIM>&,
                      const std::array<Real, AMREX_SPACEDIM>& >(),
             py::arg("a_lo"), py::arg("a_hi")
        )
        .def(py::init(
            [](const Box bx, Array<Real, AMREX_SPACEDIM> dx, Array<Real, AMREX_SPACEDIM> base) {
                return RealBox(bx, dx.data(), base.data());
            }),
            py::arg("bx"), py::arg("dx"), py::arg("base")
        )

        .def_property_readonly(
            "xlo",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xlo {AMREX_D_DECL(
                    rb.lo(0), rb.lo(1), rb.lo(2)
                )};
                return xlo;
            }
        )
        .def_property_readonly(
            "xhi",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xhi {AMREX_D_DECL(
                    rb.hi(0), rb.hi(1), rb.hi(2)
                )};
                return xhi;
            }
        )

        .def("lo", py::overload_cast<int>(&RealBox::lo, py::const_), "Get ith component of ``xlo``")
        .def("lo",
            [](RealBox const & rb){
                std::array<Real,AMREX_SPACEDIM> xlo {AMREX_D_DECL(
                    rb.lo(0), rb.lo(1), rb.lo(2)
                )};
                return xlo;
            },
            "Get all components of ``xlo``"
        )
        .def("hi", py::overload_cast<int>(&RealBox::hi, py::const_), "Get ith component of ``xhi``")
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
        .def("setLo", py::overload_cast<int,Real>(&RealBox::setLo), "Get all components of ``xlo``")
        .def("setHi",
            [](RealBox & rb, const std::vector<Real>& a_hi){
                rb.setHi(a_hi.data() );
            },
            "Get all components of ``xlo``"
        )
        .def("setHi", py::overload_cast<int,Real>(&RealBox::setHi), "Get ith component of ``xhi``")
        .def("length", &RealBox::length)
        .def("ok", &RealBox::ok, "Determine if RealBox satisfies ``xlo[i]<xhi[i]`` for ``i=0,1,...,AMREX_SPACEDIM``.")
        .def("volume", &RealBox::volume)
        .def("contains",
            [](RealBox& rb, XDim3 point, Real eps) {
                return rb.contains(point, eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            py::arg("rb"),py::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const RealVect& pt, Real eps) {
                return rb.contains(pt, eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            py::arg("rb"),py::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const RealBox& rb2, Real eps) {
                return rb.contains(rb2, eps );
            },
            "Determine if RealBox contains another RealBox, within tolerance ``eps``",
            py::arg("rb"),py::arg("eps") = 0.0
        )
        .def("contains",
            [](RealBox& rb, const std::vector<Real>& pt, Real eps) {
                return rb.contains(pt.data(), eps );
            },
            "Determine if RealBox contains ``pt``, within tolerance ``eps``",
            py::arg("rb"),py::arg("eps") = 0.0
        )
        .def("intersects", &RealBox::intersects, "determine if box intersects with a box")
    ;
    m.def("AlmostEqual",
            [](const RealBox& rb1, const RealBox& rb2, Real eps) {
                return AlmostEqual(rb1,rb2,eps);
            },
            "Determine if two boxes are equal to within a tolerance",
            py::arg("rb1"), py::arg("rb2"), py::arg("eps") = 0.0);
}
