/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Box.H>
#include <AMReX_IntVect.H>

#include <sstream>
#include <optional>


namespace
{
    using namespace amrex;

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
    using namespace amrex;

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

        .def_property("lo_vect",
            [](Box const & bx){ return bx.smallEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setSmall(bg); }
        )
        .def_property("hi_vect",
            [](Box const & bx){ return bx.bigEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setBig(bg); }
        )
        .def_property("small_end",
            [](Box const & bx){ return bx.smallEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setSmall(bg); }
        )
        .def_property("big_end",
            [](Box const & bx){ return bx.bigEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setBig(bg); }
        )

        .def_property("type",
            py::overload_cast<>(&Box::type, py::const_),
            &Box::setType)

        .def_property_readonly("ix_type", &Box::ixType)
        .def_property_readonly("size", &Box::size)

        .def("length",
            py::overload_cast<>(&Box::length, py::const_),
            "Return IntVect of lengths of the Box")
        .def("length",
            py::overload_cast<int>(&Box::length, py::const_),
            py::arg("dir"),
            "Return the length of the Box in given direction.")
        .def("numPts", &Box::numPts,
             "Return the number of points in the Box.")

        .def_property_readonly("is_empty", &Box::isEmpty)
        .def_property_readonly("ok", &Box::ok)
        .def_property_readonly("cell_centered", &Box::cellCentered,
            "Returns true if Box is cell-centered in all indexing directions.")
        .def_property_readonly("num_pts", &Box::numPts)
        .def_property_readonly("d_num_pts", &Box::d_numPts)
        .def_property_readonly("volume", &Box::volume)
        .def_property_readonly("the_unit_box", &Box::TheUnitBox)
        .def_property_readonly("is_square", &Box::isSquare)
        .def("contains",
            [](Box const & bx, IntVect const & p){ return bx.contains(p); },
            py::arg("p"),
            "Returns true if argument is contained within Box."
        )
        .def("strictly_contains",
            [](Box const & bx, IntVect const & p){ return bx.strictly_contains(p); },
            py::arg("p"),
            "Returns true if argument is strictly contained within Box."
        )
        .def("intersects", &Box::intersects, py::arg("b"),
            "Returns true if Boxes have non-null intersections.\n"
            "It is an error if the Boxes have different types."
        )
        .def("same_size", &Box::sameSize, py::arg("b"),
            "Returns true is Boxes same size, ie translates of each other,.\n"
            "It is an error if they have different types."
        )
        .def("same_type", &Box::sameType, py::arg("b"),
            "Returns true if Boxes have same type."
        )
        .def("normalize", &Box::normalize)
        // longside
        // shortside
        // index
        // atOffset
        // atOffset3d
        // setRange
        // shiftHalf
        .def("shift",
             py::overload_cast< int, int >(&Box::shift),
             py::arg("dir"), py::arg("nzones"),
             "Shift this Box nzones indexing positions in coordinate direction dir."
        )
        .def("shift",
            py::overload_cast< IntVect const & >(&Box::shift),
            py::arg("iv"),
            "Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ..."
        )

        .def(py::self + IntVect())
        .def(py::self - IntVect())
        .def(py::self += IntVect())
        .def(py::self -= IntVect())

        .def("convert",
             py::overload_cast< IndexType >(&Box::convert),
             py::arg("typ"),
             "Convert the Box from the current type into the\n"
             "argument type.  This may change the Box coordinates:\n"
             "type CELL -> NODE : increase coordinate by one on high end\n"
             "type NODE -> CELL : reduce coordinate by one on high end\n"
             "other type mappings make no change."
        )
        .def("convert",
             py::overload_cast< IntVect const & >(&Box::convert),
             py::arg("typ"),
             "Convert the Box from the current type into the\n"
             "argument type.  This may change the Box coordinates:\n"
             "type CELL -> NODE : increase coordinate by one on high end\n"
             "type NODE -> CELL : reduce coordinate by one on high end\n"
             "other type mappings make no change."
        )

        .def("grow",
             py::overload_cast< int >(&Box::grow),
             py::arg("n_cell"),
             "Grow Box in all directions by given amount.\n"
             "NOTE: n_cell negative shrinks the Box by that number of cells."
        )
        .def("grow",
             py::overload_cast< IntVect const & >(&Box::grow),
             py::arg("n_cells"),
             "Grow Box in each direction by specified amount."
        )
        .def("grow",
             py::overload_cast< int, int >(&Box::grow),
             py::arg("idir"), py::arg("n_cell"),
             "Grow the Box on the low and high end by n_cell cells\n"
             "in direction idir."
        )
        .def("grow",
             py::overload_cast< Direction, int >(&Box::grow),
             py::arg("d"), py::arg("n_cell")
        )
        /* TODO: Bind Orientation class first
        .def("grow",
             py::overload_cast< Orientation, int >(&Box::grow),
             py::arg("face"), py::arg("n_cell")=1,
             "Grow in the direction of the given face."
        )
        */
        .def("grow_low",
             py::overload_cast< int, int >(&Box::growLo),
             py::arg("idir"), py::arg("n_cell")=1,
             "Grow the Box on the low end by n_cell cells in direction idir.\n"
             "NOTE: n_cell negative shrinks the Box by that number of cells."
        )
        .def("grow_low",
             py::overload_cast< Direction, int >(&Box::growLo),
             py::arg("d"), py::arg("n_cell")=1
        )
        .def("grow_high",
             py::overload_cast< int, int >(&Box::growHi),
             py::arg("idir"), py::arg("n_cell")=1,
             "Grow the Box on the high end by n_cell cells in\n"
             "direction idir.  NOTE: n_cell negative shrinks the Box by that\n"
             "number of cells."
        )
        .def("grow_high",
             py::overload_cast< Direction, int >(&Box::growHi),
             py::arg("d"), py::arg("n_cell")=1
        )

        .def("surrounding_nodes",
             py::overload_cast< >(&Box::surroundingNodes),
             "Convert to NODE type in all directions.")
        .def("surrounding_nodes",
             py::overload_cast< int >(&Box::surroundingNodes),
             py::arg("dir"),
             "Convert to NODE type in given direction.")
        .def("surrounding_nodes",
             py::overload_cast< Direction >(&Box::surroundingNodes),
             py::arg("d"),
             "Convert to NODE type in given direction.")

        .def("enclosed_cells",
             py::overload_cast< >(&Box::enclosedCells),
             "Convert to CELL type in all directions.")
        .def("enclosed_cells",
             py::overload_cast< int >(&Box::enclosedCells),
             py::arg("dir"),
             "Convert to CELL type in given direction.")
        .def("enclosed_cells",
             py::overload_cast< Direction >(&Box::enclosedCells),
             py::arg("d"),
             "Convert to CELL type in given direction.")

        .def("make_slab",
             &Box::makeSlab,
             py::arg("direction"), py::arg("slab_index"),
             "Flatten the box in one direction.")

        // minBox
        // chop
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
        .def("begin",
            [](Box const &, Box const & other){ return begin(other); },
            py::arg("box")
        )
        .def("end",
            [](Box const &, Box const & other){ return end(other); },
            py::arg("box")
        )
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
