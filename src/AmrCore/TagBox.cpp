/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Array4.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_TagBox.H>

#include <sstream>


void init_TagBox(py::module& m)
{
    using namespace amrex;

    py::class_<TagBoxArray, FabArrayBase> py_tag_box_array(m, "TagBoxArray",
        "An array of TagBoxes: the data structure that marks cells for "
        "refinement, e.g., in AmrCore.error_est. The tag value of each "
        "cell is one of TagVal.CLEAR, TagVal.BUF or TagVal.SET.");

    // TagBox::TagVal as amrex.TagVal, with member aliases on TagBoxArray
    // mirroring the C++ TagBox::SET etc. vocabulary
    py::native_enum<TagBox::TagVal>(m, "TagVal", "enum.IntEnum",
            "The cell tag values: clear (do not refine), buffer (may be "
            "changed to set by buffering), set (refine)")
        .value("CLEAR", TagBox::CLEAR)
        .value("BUF", TagBox::BUF)
        .value("SET", TagBox::SET)
        .finalize()
    ;

    py_tag_box_array
        .def("__repr__",
             [](TagBoxArray const & tba) {
                 std::stringstream s;
                 s << tba.size();
                 return "<amrex.TagBoxArray of size '" + s.str() + "'>";
             }
        )

        .def(py::init<BoxArray const &, DistributionMapping const &, int>(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow") = 0)
        .def(py::init<BoxArray const &, DistributionMapping const &,
                      IntVect const &>(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow"))

        // the FabArray<TagBox> template base is not bound: bind the
        // commonly used members directly
        .def("box_array",
            [](TagBoxArray const & tba) { return tba.boxArray(); })
        .def("dist_map",
            [](TagBoxArray const & tba) { return tba.DistributionMap(); })

        .def("array",
            [](TagBoxArray & tba, MFIter const & mfi)
            { return tba.array(mfi); },
            py::arg("mfi"),
            // do not copy via brace init list
            py::return_value_policy::move,
            "Return the Array4 (of char) of the tags in the box of the "
            "MFIter")
        .def("const_array",
            [](TagBoxArray const & tba, MFIter const & mfi)
            { return tba.const_array(mfi); },
            py::arg("mfi"),
            // do not copy via brace init list
            py::return_value_policy::move)

        .def("set_val",
            [](TagBoxArray & tba, BoxArray const & ba, TagBox::TagVal val)
            { tba.setVal(ba, val); },
            py::arg("ba"), py::arg("val"),
            "Set all tags inside BoxArray ba to value val.")
        .def("set_val",
            [](TagBoxArray & tba, TagBox::TagVal val)
            { tba.setVal(static_cast<char>(val)); },
            py::arg("val"),
            "Set all tags to value val.")

        .def("buffer", &TagBoxArray::buffer, py::arg("nbuf"),
             "Grow the tagged region by the grow vector nbuf.")
        .def("map_periodic_remove_duplicates",
             &TagBoxArray::mapPeriodicRemoveDuplicates, py::arg("geom"),
             "Map tags across periodic boundaries and remove duplicates "
             "using geom.")
        .def("coarsen", &TagBoxArray::coarsen, py::arg("ratio"),
             "Coarsen the tag layout by ratio.")
        .def("has_tags", &TagBoxArray::hasTags, py::arg("bx"),
             "Return true if any cells inside bx are tagged (SET or BUF), "
             "false otherwise.")
    ;
    // note: iteration (`for mfi in tags`) is provided via the
    // FabArrayBase.__iter__ Python extension
}
