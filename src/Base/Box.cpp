/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>

#include <sstream>
#include <optional>

namespace py = pybind11;
using namespace amrex;

namespace
{
    /** A little Wrapper class to iterate an amrex::Box via
     *  amrex::Box::next().
     */
    struct Box3DConstIter {
        Box m_box;
        std::optional<IntVect> m_it;

        Box3DConstIter(Box const & bx) : m_box(bx) {
            m_it = m_box.smallEnd();
        }

        Box3DConstIter& operator++() {
            // from FABio_ascii::write
            if (m_it < m_box.bigEnd()) {
                m_box.next(m_it.value());
                return *this;
            }
            else
            {
                m_it = std::nullopt;
                return *this;
            }
        }

        bool operator==(Box3DConstIter const & other) const {
            return other.m_it == m_it;
        }

        Box3DConstIter begin() const
        {
            return Box3DConstIter(m_box);
        }
        Box3DConstIter end() const
        {
            auto it = Box3DConstIter(m_box);
            it.m_it = std::nullopt;
            return it;
        }

        IntVect operator*() const
        {
            return m_it.value();
        }
    };
}

void init_Box(py::module &m) {
    py::class_< Direction >(m, "Direction");



    py::class_< Box >(m, "Box")
        .def("__repr__",
            [](Box const & b) {
                std::stringstream s;
                s << b.size();
                return "<amrex.Box of size '" + s.str() + "'>";
            }
        )

        .def(py::init< IntVect const &, IntVect const & >())
        .def(py::init< IntVect const &, IntVect const &, IntVect const & >())
        //.def(py::init< IntVect const &, IntVect const &, IndexType >())

        .def_property_readonly("small_end", [](Box const & bx){ return bx.smallEnd(); })
        .def_property_readonly("big_end", [](Box const & bx){ return bx.bigEnd(); })
        /*
        .def_property("small_end",
            &Box::smallEnd,
            py::overload_cast< IntVect const & >(&Box::setSmall))
        .def_property("big_end",
            &Box::bigEnd,
            &Box::setBig)

        .def_property("type",
            py::overload_cast<>(&Box::type, py::const_),
            &Box::setType)

        .def_property_readonly("ix_type", &Box::ixType)
        .def_property_readonly("size", &Box::size)
        */
        .def("length",
            py::overload_cast<>(&Box::length, py::const_),
            "Return IntVect of lengths of the Box")
        .def("length",
            py::overload_cast<int>(&Box::length, py::const_),
            "Return the length of the Box in given direction.")
            /*
        .def_property_readonly("is_empty", &Box::isEmpty)
        .def_property_readonly("ok", &Box::ok)
        .def_property_readonly("cell_centered", &Box::cellCentered)
        .def_property_readonly("num_pts", &Box::numPts)
        .def_property_readonly("d_num_pts", &Box::d_numPts)
        .def_property_readonly("volume", &Box::volume)
        .def_property_readonly("the_unit_box", &Box::TheUnitBox)
        .def_property_readonly("is_square", &Box::isSquare)

        // loVect3d
        // hiVect3d
        .def_property_readonly("lo_vect", &Box::loVect)
        .def_property_readonly("hi_vect", &Box::hiVect)

        .def("contains",
            py::overload_cast< IntVect const & >(&Box::contains, py::const_))
        .def("strictly_contains",
            py::overload_cast< IntVect const & >(&Box::strictly_contains, py::const_))
        //.def("intersects", &Box::intersects)
        //.def("same_size", &Box::sameSize)
        //.def("same_type", &Box::sameType)
        //.def("normalize", &Box::normalize)
        // longside
        // shortside
        // index
        // atOffset
        // atOffset3d
        // setRange
        // shiftHalf
        */
        .def("shift", [](Box & bx, IntVect const& iv) { return bx.shift(iv); })

        .def(py::self + IntVect())
        .def(py::self - IntVect())
        .def(py::self += IntVect())
        .def(py::self -= IntVect())

        .def("convert",
             py::overload_cast< IndexType >(&Box::convert))
        .def("convert",
             py::overload_cast< IntVect const & >(&Box::convert))

        .def("grow", [](Box & bx, IntVect const& iv) { return bx.grow(iv); })

        //.def("surrounding_nodes",
        //     py::overload_cast< >(&Box::surroundingNodes))
        //.def("surrounding_nodes",
        //     py::overload_cast< int >(&Box::surroundingNodes),
        //     py::arg("dir"))
        //.def("surrounding_nodes",
        //     py::overload_cast< Direction >(&Box::surroundingNodes),
        //     py::arg("d"))

        // enclosedCells
        // minBox
        // chop
        // grow
        // growLo
        // growHi
        // refine
        // coarsen
        // next
        // coarsenable

        // __getitem__

        /* iterate Box index space */
        .def("__iter__",
             [](Box const & bx) {
                 auto box_iter = Box3DConstIter(bx);
                 return py::make_iterator(box_iter.begin(), box_iter.end());
             },
            // Essential: keep object alive while iterator exists
             py::keep_alive<0, 1>()
        )

        .def("lbound", [](Box const &, Box const & other){ return lbound(other); })
        .def("ubound", [](Box const &, Box const & other){ return ubound(other); })
        .def("begin", [](Box const &, Box const & other){ return begin(other); })
        .def("end", [](Box const &, Box const & other){ return end(other); })
        // already an attribute
        //.def("length", [](Box const &, Box const & other){ return length(other); })
    ;

    // free standing C++ functions:
    m.def("lbound", [](Box const & other){ return lbound(other); });
    m.def("ubound", [](Box const & other){ return ubound(other); });
    m.def("begin", [](Box const & other){ return begin(other); });
    m.def("end", [](Box const & other){ return end(other); });
    m.def("length", [](Box const & other){ return length(other); });
}
