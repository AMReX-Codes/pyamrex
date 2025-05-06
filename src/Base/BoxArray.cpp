/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>

#include <sstream>


void init_BoxArray(py::module &m) {
    using namespace amrex;

    /* A collection of Boxes stored in an Array.  It is a
     * reference-counted concrete class, not a polymorphic one; i.e. you
     * cannot use any of the List member functions with a BoxList
     */
    py::class_< BoxArray >(m, "BoxArray")
        .def("__repr__",
            [](BoxArray const & ba) {
                std::stringstream s;
                s << ba.size();
                return "<amrex.BoxArray of size '" + s.str() + "'>";
            }
        )

        // Construct an empty BoxArray
        .def(py::init<>())
        // Copy a BoxArray
        .def(py::init< BoxArray const & >())

        // Construct a BoxArray from a single Box.
        .def(py::init< Box const & >())
        // Construct a BoxArray from a list of Boxes.
        .def(py::init([](Vector<Box> bl) {
             return BoxArray(bl.dataPtr(), bl.size());
        }))

        // Construct a BoxArray from a BoxList.
        //.def(py::init< BoxList const& >())

        //BoxArray (const BoxArray& rhs, const BATransformer& trans);
        //BoxArray (BoxList&& bl, IntVect const& max_grid_size);

        .def_property_readonly("size", &BoxArray::size)
        .def_property_readonly("capacity", &BoxArray::capacity)
        .def_property_readonly("empty", &BoxArray::empty)
        .def_property_readonly("numPts", &BoxArray::numPts)
        .def_property_readonly("d_numPts", &BoxArray::d_numPts)
/*
        .def_property("type",
            py::overload_cast<>(&BoxArray::type, py::const_),
            &Box::setType)

        .def_property_readonly("length",
            py::overload_cast<>(&Box::length, py::const_))
        .def_property_readonly("is_empty", &Box::isEmpty)
*/

        .def("define",
            py::overload_cast< Box const & >(&BoxArray::define))
        //.def("define",
        //    py::overload_cast< BoxList const & >(&BoxArray::define))
        //.def("define",
        //    py::overload_cast< BoxList&& >(&BoxArray::define))

        .def("clear", &BoxArray::clear)
        .def("resize", &BoxArray::resize)

        .def("cell_equal", &BoxArray::CellEqual)

        .def("max_size",
            py::overload_cast< int >(&BoxArray::maxSize))
        .def("max_size",
            py::overload_cast< IntVect const& >(&BoxArray::maxSize))

        .def("refine",
            py::overload_cast< int >(&BoxArray::refine))
        .def("refine",
            py::overload_cast< IntVect const & >(&BoxArray::refine))

        //! Coarsen each Box in the BoxArray to the specified ratio.
        .def("coarsen",
            py::overload_cast< IntVect const & >(&BoxArray::coarsen))
        .def("coarsen",
            py::overload_cast< int >(&BoxArray::coarsen))

        //! Coarsen each Box in the BoxArray to the specified ratio.
        .def("coarsenable",
            py::overload_cast< int, int >(&BoxArray::coarsenable, py::const_))
        .def("coarsenable",
            py::overload_cast< IntVect const &, int >(&BoxArray::coarsenable, py::const_))
        .def("coarsenable",
            py::overload_cast< IntVect const &, IntVect const & >(&BoxArray::coarsenable, py::const_))

/*
    //! Grow and then coarsen each Box in the BoxArray.
    BoxArray& growcoarsen (int n, const IntVect& refinement_ratio);
    BoxArray& growcoarsen (IntVect const& ngrow, const IntVect& refinement_ratio);

    //! Grow each Box in the BoxArray by the specified amount.
    BoxArray& grow (int n);

    //! Grow each Box in the BoxArray by the specified amount.
    BoxArray& grow (const IntVect& iv);

    //! \brief Grow each Box in the BoxArray on the low and high ends
    //! by n_cell cells in the idir direction.
    BoxArray& grow (int idir, int n_cell);

    //! \brief Grow each Box in the BoxArray on the low end
    /?! by n_cell cells in the idir direction.
    BoxArray& growLo (int idir, int n_cell);

    //! \brief Grow each Box in the BoxArray on the high end
    //! by n_cell cells in the idir direction.
    BoxArray& growHi (int idir, int n_cell);
*/

    //! \brief Apply surroundingNodes(Box,int) to each Box in
    //! BoxArray.  See the documentation of Box for details.
    .def("surroundingNodes",
            py::overload_cast<>(&BoxArray::surroundingNodes))
    .def("surroundingNodes",
            py::overload_cast<int>(&BoxArray::surroundingNodes))

    //! Apply Box::enclosedCells() to each Box in the BoxArray.
    .def("enclosed_cells",
            py::overload_cast<>(&BoxArray::enclosedCells))
    .def("enclosed_cells",
            py::overload_cast<int>(&BoxArray::enclosedCells))

    //! Convert nodality of each box in the BoxArray
    .def("convert",
            py::overload_cast< IndexType >(&BoxArray::convert))
    .def("convert",
            py::overload_cast< IntVect const &>(&BoxArray::convert))
/*
    //! Apply Box::shift(int,int) to each Box in the BoxArray.
    BoxArray& shift (int dir, int nzones);

    //! Apply Box::shift(const IntVect &iv) to each Box in the BoxArray.
    BoxArray& shift (const IntVect &iv);

    //! Set element i in this BoxArray to Box ibox.
    void set (int i, const Box& ibox);

    //! Return element index of this BoxArray.
    Box operator[] (int index) const noexcept {
        return m_bat(m_ref->m_abox[index]);
    }

    //! Return element index of this BoxArray.
    Box operator[] (const MFIter& mfi) const noexcept;
*/
        .def("__getitem__",
             [](const BoxArray& ba, const int i) {
                 const int ii = (i >= 0) ? i : ba.size() + i;
                 if ((ii < 0) || (ii >= ba.size()))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(ba.size()));
                 return ba[ii];
             })


        //! Return element index of this BoxArray.
        .def("get", &BoxArray::get)

/*
    //! Return cell-centered box at element index of this BoxArray.
    Box getCellCenteredBox (int index) const noexcept {
        return m_bat.coarsen(m_ref->m_abox[index]);
    }

    //! \brief Return true if Box is valid and they all have the same
    //! IndexType.  Is true by default if the BoxArray is empty.
    bool ok () const;

    //! Return true if set of intersecting Boxes in BoxArray is null.
    bool isDisjoint () const;

    //! Create a BoxList from this BoxArray.
    BoxList boxList () const;

    //! True if the IntVect is within any of the Boxes in this BoxArray.
    bool contains (const IntVect& v) const;

    //! \brief True if the Box is contained in this BoxArray(+ng).
    //! The Box must also have the same IndexType as those in this BoxArray.
    bool contains (const Box& b, bool assume_disjoint_ba = false,
                   const IntVect& ng = IntVect(0)) const;

    //! \brief True if all Boxes in ba are contained in this BoxArray(+ng).
    bool contains (const BoxArray& ba, bool assume_disjoint_ba = false,
                   const IntVect& ng = IntVect(0)) const;

*/
        //! Return smallest Box that contains all Boxes in this BoxArray.
        .def("minimal_box",
            py::overload_cast<>(&BoxArray::minimalBox, py::const_))

/*
    Box minimalBox (Long& npts_avg_box) const;

    //! \brief True if the Box intersects with this BoxArray(+ghostcells).
    //! The Box must have the same IndexType as those in this BoxArray.
    bool intersects (const Box& b, int ng = 0) const;

    bool intersects (const Box& b, const IntVect& ng) const;

    //! Return intersections of Box and BoxArray
    std::vector< std::pair<int,Box> > intersections (const Box& bx) const;

    //! Return intersections of Box and BoxArray(+ghostcells).
    std::vector< std::pair<int,Box> > intersections (const Box& bx, bool first_only, int ng) const;

    std::vector< std::pair<int,Box> > intersections (const Box& bx, bool first_only, const IntVect& ng) const;

    //! intersect Box and BoxArray, then store the result in isects
    void intersections (const Box& bx, std::vector< std::pair<int,Box> >& isects) const;

    //! intersect Box and BoxArray(+ghostcells), then store the result in isects
    void intersections (const Box& bx, std::vector< std::pair<int,Box> >& isects,
                        bool first_only, int ng) const;

    void intersections (const Box& bx, std::vector< std::pair<int,Box> >& isects,
                        bool first_only, const IntVect& ng) const;

    //! Return box - boxarray
    BoxList complementIn (const Box& b) const;
    void complementIn (BoxList& bl, const Box& b) const;

    //! Clear out the internal hash table used by intersections.
    void clear_hash_bin () const;

    //! Change the BoxArray to one with no overlap and then simplify it (see the simplify function in BoxList).
    void removeOverlap (bool simplify=true);

    //! whether two BoxArrays share the same data
    static bool SameRefs (const BoxArray& lhs, const BoxArray& rhs) { return lhs.m_ref == rhs.m_ref; }

    struct RefID {
        RefID () noexcept : data(nullptr) {}
        explicit RefID (BARef* data_) noexcept : data(data_) {}
        bool operator<  (const RefID& rhs) const noexcept { return std::less<BARef*>()(data,rhs.data); }
        bool operator== (const RefID& rhs) const noexcept { return data == rhs.data; }
        bool operator!= (const RefID& rhs) const noexcept { return data != rhs.data; }
        friend std::ostream& operator<< (std::ostream& os, const RefID& id);
    private:
        BARef* data;
    };

    //! Return a unique ID of the reference
    RefID getRefID () const noexcept { return RefID { m_ref.get() }; }

*/
        //! Return index type of this BoxArray
        .def("ix_type", &BoxArray::ixType)

/*
    //! Return crse ratio of this BoxArray
    IntVect crseRatio () const noexcept { return m_bat.coarsen_ratio(); }

    static void Initialize ();
    static void Finalize ();
    static bool initialized;

    //! Make ourselves unique.
    void uniqify ();

    BoxList const& simplified_list () const; // For regular AMR grids only!!!
    BoxArray simplified () const; // For regular AMR grids only!!!

*/

    ;

    make_Vector<BoxArray> (m, "BoxArray");
}
