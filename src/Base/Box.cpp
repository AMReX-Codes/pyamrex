/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <optional>

namespace py = nanobind;
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

void init_Box(py::module_ &m) {
    py::class_< Direction >(m, "Direction");



    py::class_< Box >(m, "Box")
        .def("__repr__",
            [](Box const & b) {
                std::stringstream s;
                s << b.size();
                return "<amrex.Box of size '" + s.str() + "'>";
            }
        )

        .def(py::init< IntVect const &, IntVect const & >(),
             py::arg("small"), py::arg("big")
        )
        .def(py::init< IntVect const &, IntVect const &, IntVect const & >(),
             py::arg("small"), py::arg("big"), py::arg("typ")
        )
        .def(py::init< IntVect const &, IntVect const &, IndexType >(),
             py::arg("small"), py::arg("big"), py::arg("t")
        )
        .def(py::init(
                [](const std::array<int, AMREX_SPACEDIM>& small,
                   const std::array<int, AMREX_SPACEDIM>& big){
                    return Box(IntVect{small}, IntVect{big});
                }
             ),
             py::arg("small"), py::arg("big")
        )
        .def(py::init(
                [](const std::array<int, AMREX_SPACEDIM>& small,
                   const std::array<int, AMREX_SPACEDIM>& big,
                   IndexType t){
                    return Box(IntVect{small}, IntVect{big}, t);
                }
             ),
             py::arg("small"), py::arg("big"), py::arg("t")
        )

        .def_prop_rw_readonly("lo_vect", [](Box const & bx){ return bx.smallEnd(); })
        .def_prop_rw_readonly("hi_vect", [](Box const & bx){ return bx.bigEnd(); })
        .def_prop_rw_readonly("small_end", [](Box const & bx){ return bx.smallEnd(); })
        .def_prop_rw_readonly("big_end", [](Box const & bx){ return bx.bigEnd(); })
        /*
        .def_prop_rw("small_end",
            py::overload_cast<>(&Box::smallEnd, py::const_),
            py::overload_cast< IntVect const & >(&Box::setSmall))
        .def_prop_rw("big_end",
            &Box::bigEnd,
            &Box::setBig)
        */
        .def_prop_rw("type",
            py::overload_cast<>(&Box::type, py::const_),
            &Box::setType)

        .def_prop_rw_readonly("ix_type", &Box::ixType)
        .def_prop_rw_readonly("size", &Box::size)

        .def("length",
            py::overload_cast<>(&Box::length, py::const_),
            "Return IntVect of lengths of the Box")
        .def("length",
            py::overload_cast<int>(&Box::length, py::const_),
            "Return the length of the Box in given direction.")
        .def("numPts", &Box::numPts,
             "Return the number of points in the Box.")

        .def_prop_rw_readonly("is_empty", &Box::isEmpty)
        .def_prop_rw_readonly("ok", &Box::ok)
        .def_prop_rw_readonly("cell_centered", &Box::cellCentered)
        .def_prop_rw_readonly("num_pts", &Box::numPts)
        .def_prop_rw_readonly("d_num_pts", &Box::d_numPts)
        .def_prop_rw_readonly("volume", &Box::volume)
        .def_prop_rw_readonly("the_unit_box", &Box::TheUnitBox)
        .def_prop_rw_readonly("is_square", &Box::isSquare)

        .def("contains",
            py::overload_cast< IntVect const & >(&Box::contains, py::const_))
        .def("strictly_contains",
            py::overload_cast< IntVect const & >(&Box::strictly_contains, py::const_))
        .def("intersects", &Box::intersects)
        .def("same_size", &Box::sameSize)
        .def("same_type", &Box::sameType)
        .def("normalize", &Box::normalize)
        // longside
        // shortside
        // index
        // atOffset
        // atOffset3d
        // setRange
        // shiftHalf
        .def("shift", [](Box & bx, IntVect const& iv) { return bx.shift(iv); })

        .def(py::self + IntVect())
        .def(py::self - IntVect())
        .def(py::self += IntVect())
        .def(py::self -= IntVect())

        .def("convert",
             py::overload_cast< IndexType >(&Box::convert))
        .def("convert",
             py::overload_cast< IntVect const & >(&Box::convert))

        .def("grow",
             py::overload_cast< int >(&Box::grow),
             py::arg("n_cell")
        )
        .def("grow",
             py::overload_cast< IntVect const & >(&Box::grow),
             py::arg("n_cells")
        )
        .def("grow",
             py::overload_cast< int, int >(&Box::grow),
             py::arg("idir"), py::arg("n_cell")
        )
        .def("grow",
             py::overload_cast< Direction, int >(&Box::grow),
             py::arg("d"), py::arg("n_cell")
        )

        .def("surrounding_nodes",
             py::overload_cast< >(&Box::surroundingNodes))
        .def("surrounding_nodes",
             py::overload_cast< int >(&Box::surroundingNodes),
             py::arg("dir"))
        .def("surrounding_nodes",
             py::overload_cast< Direction >(&Box::surroundingNodes),
             py::arg("d"))

        .def("enclosed_cells",
             py::overload_cast< >(&Box::enclosedCells))
        .def("enclosed_cells",
             py::overload_cast< int >(&Box::enclosedCells),
             py::arg("dir"))
        .def("enclosed_cells",
             py::overload_cast< Direction >(&Box::enclosedCells),
             py::arg("d"))

        .def("make_slab",
             &Box::makeSlab,
             py::arg("direction"), py::arg("slab_index"))

        // minBox
        // chop
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
