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

void init_Box(nb::module_ &m) {
    using namespace amrex;

    nb::enum_<Direction>(m, "Direction", nb::is_arithmetic())
        .value("x", Direction::x)
#if AMREX_SPACEDIM >= 2
        .value("y", Direction::y)
#endif
#if AMREX_SPACEDIM == 3
        .value("z", Direction::z)
#endif
    ;


    nb::class_< Box >(m, "Box")
        .def("__repr__",
            [](Box const & b) {
                std::stringstream s;
                s << b.size();
                return "<amrex.Box of size '" + s.str() + "'>";
            }
        )

        .def(nb::init< IntVect const &, IntVect const & >(),
             nb::arg("small"), nb::arg("big")
        )
        .def(nb::init< IntVect const &, IntVect const &, IntVect const & >(),
             nb::arg("small"), nb::arg("big"), nb::arg("typ")
        )
        .def(nb::init< IntVect const &, IntVect const &, IndexType >(),
             nb::arg("small"), nb::arg("big"), nb::arg("t")
        )
        .def("__init__",
                [](Box *self, const std::array<int, AMREX_SPACEDIM>& small,
                   const std::array<int, AMREX_SPACEDIM>& big){
                    new (self) Box(IntVect{small}, IntVect{big});
                },
             nb::arg("small"), nb::arg("big")
        )
        .def("__init__",
                [](Box *self, const std::array<int, AMREX_SPACEDIM>& small,
                   const std::array<int, AMREX_SPACEDIM>& big,
                   IndexType t){
                    new (self) Box(IntVect{small}, IntVect{big}, t);
                },
             nb::arg("small"), nb::arg("big"), nb::arg("t")
        )

        .def_prop_rw("lo_vect",
            [](Box const & bx){ return bx.smallEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setSmall(bg); }
        )
        .def_prop_rw("hi_vect",
            [](Box const & bx){ return bx.bigEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setBig(bg); }
        )
        .def_prop_rw("small_end",
            [](Box const & bx){ return bx.smallEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setSmall(bg); }
        )
        .def_prop_rw("big_end",
            [](Box const & bx){ return bx.bigEnd(); },
            [](Box & bx, IntVect const & bg){ bx.setBig(bg); }
        )

        .def_prop_rw("type",
            nb::overload_cast<>(&Box::type, nb::const_),
            &Box::setType)

        .def_prop_ro("ix_type", &Box::ixType)
        .def_prop_ro("size", &Box::size)

        .def("length",
            nb::overload_cast<>(&Box::length, nb::const_),
            "Return IntVect of lengths of the Box")
        .def("length",
            nb::overload_cast<int>(&Box::length, nb::const_),
            nb::arg("dir"),
            "Return the length of the Box in given direction.")
        .def("numPts", &Box::numPts,
             "Return the number of points in the Box.")

        .def_prop_ro("is_empty", &Box::isEmpty)
        .def_prop_ro("ok", &Box::ok)
        .def_prop_ro("cell_centered", &Box::cellCentered,
            "Returns true if Box is cell-centered in all indexing directions.")
        .def_prop_ro("num_pts", &Box::numPts)
        .def_prop_ro("d_num_pts", &Box::d_numPts)
        .def_prop_ro("volume", &Box::volume)
        .def_prop_ro("the_unit_box", [](Box const&) {
            return Box(Box::TheUnitBox());
        })
        .def_prop_ro("is_square", &Box::isSquare)
        .def("contains",
            [](Box const & bx, IntVect const & p){ return bx.contains(p); },
            nb::arg("p"),
            "Returns true if argument is contained within Box."
        )
        .def("strictly_contains",
            [](Box const & bx, IntVect const & p){ return bx.strictly_contains(p); },
            nb::arg("p"),
            "Returns true if argument is strictly contained within Box."
        )
        .def("intersects", &Box::intersects, nb::arg("b"),
            "Returns true if Boxes have non-null intersections.\n"
            "It is an error if the Boxes have different types."
        )
        .def("same_size", &Box::sameSize, nb::arg("b"),
            "Returns true is Boxes same size, ie translates of each other,.\n"
            "It is an error if they have different types."
        )
        .def("same_type", &Box::sameType, nb::arg("b"),
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
             nb::overload_cast< int, int >(&Box::shift),
             nb::arg("dir"), nb::arg("nzones"),
             "Shift this Box nzones indexing positions in coordinate direction dir."
        )
        .def("shift",
            nb::overload_cast< IntVect const & >(&Box::shift),
            nb::arg("iv"),
            "Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ..."
        )

        .def(nb::self + IntVect())
        .def(nb::self - IntVect())
        .def(nb::self += IntVect())
        .def(nb::self -= IntVect())

        .def("convert",
             nb::overload_cast< IndexType >(&Box::convert),
             nb::arg("typ"),
             "Convert the Box from the current type into the\n"
             "argument type.  This may change the Box coordinates:\n"
             "type CELL -> NODE : increase coordinate by one on high end\n"
             "type NODE -> CELL : reduce coordinate by one on high end\n"
             "other type mappings make no change."
        )
        .def("convert",
             nb::overload_cast< IntVect const & >(&Box::convert),
             nb::arg("typ"),
             "Convert the Box from the current type into the\n"
             "argument type.  This may change the Box coordinates:\n"
             "type CELL -> NODE : increase coordinate by one on high end\n"
             "type NODE -> CELL : reduce coordinate by one on high end\n"
             "other type mappings make no change."
        )

        .def("grow",
             nb::overload_cast< int >(&Box::grow),
             nb::arg("n_cell"),
             "Grow Box in all directions by given amount.\n"
             "NOTE: n_cell negative shrinks the Box by that number of cells."
        )
        .def("grow",
             nb::overload_cast< IntVect const & >(&Box::grow),
             nb::arg("n_cells"),
             "Grow Box in each direction by specified amount."
        )
        .def("grow",
             nb::overload_cast< int, int >(&Box::grow),
             nb::arg("idir"), nb::arg("n_cell"),
             "Grow the Box on the low and high end by n_cell cells\n"
             "in direction idir."
        )
        .def("grow",
             nb::overload_cast< Direction, int >(&Box::grow),
             nb::arg("d"), nb::arg("n_cell")
        )
        /* TODO: Bind Orientation class first
        .def("grow",
             nb::overload_cast< Orientation, int >(&Box::grow),
             nb::arg("face"), nb::arg("n_cell")=1,
             "Grow in the direction of the given face."
        )
        */
        .def("grow_low",
             nb::overload_cast< int, int >(&Box::growLo),
             nb::arg("idir"), nb::arg("n_cell")=1,
             "Grow the Box on the low end by n_cell cells in direction idir.\n"
             "NOTE: n_cell negative shrinks the Box by that number of cells."
        )
        .def("grow_low",
             nb::overload_cast< Direction, int >(&Box::growLo),
             nb::arg("d"), nb::arg("n_cell")=1
        )
        .def("grow_high",
             nb::overload_cast< int, int >(&Box::growHi),
             nb::arg("idir"), nb::arg("n_cell")=1,
             "Grow the Box on the high end by n_cell cells in\n"
             "direction idir.  NOTE: n_cell negative shrinks the Box by that\n"
             "number of cells."
        )
        .def("grow_high",
             nb::overload_cast< Direction, int >(&Box::growHi),
             nb::arg("d"), nb::arg("n_cell")=1
        )

        .def("surrounding_nodes",
             nb::overload_cast< >(&Box::surroundingNodes),
             "Convert to NODE type in all directions.")
        .def("surrounding_nodes",
             nb::overload_cast< int >(&Box::surroundingNodes),
             nb::arg("dir"),
             "Convert to NODE type in given direction.")
        .def("surrounding_nodes",
             nb::overload_cast< Direction >(&Box::surroundingNodes),
             nb::arg("d"),
             "Convert to NODE type in given direction.")

        .def("enclosed_cells",
             nb::overload_cast< >(&Box::enclosedCells),
             "Convert to CELL type in all directions.")
        .def("enclosed_cells",
             nb::overload_cast< int >(&Box::enclosedCells),
             nb::arg("dir"),
             "Convert to CELL type in given direction.")
        .def("enclosed_cells",
             nb::overload_cast< Direction >(&Box::enclosedCells),
             nb::arg("d"),
             "Convert to CELL type in given direction.")

        .def("make_slab",
             &Box::makeSlab,
             nb::arg("direction"), nb::arg("slab_index"),
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
                 return nb::make_iterator(
                     nb::type<Box>(), "BoxIterator",
                     box_iter.begin(), box_iter.end()
                 );
             },
             // Essential: keep object alive while iterator exists
             nb::keep_alive<0, 1>()
        )

        .def("lbound", [](Box const &, Box const & other){ return lbound(other); })
        .def("ubound", [](Box const &, Box const & other){ return ubound(other); })
        .def("begin",
            [](Box const &, Box const & other){ return begin(other); },
            nb::arg("box")
        )
        .def("end",
            [](Box const &, Box const & other){ return end(other); },
            nb::arg("box")
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
