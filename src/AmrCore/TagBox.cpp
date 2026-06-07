/* Copyright 2024-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_TagBox.H>

#include <sstream>


namespace
{
    amrex::TagBox::TagType tag_value (amrex::TagBox::TagVal val)
    {
        return static_cast<amrex::TagBox::TagType>(val);
    }
}


void init_TagBox (py::module& m)
{
    using namespace amrex;

    py::class_< TagBox > py_TagBox(m, "TagBox");

    py::native_enum< TagBox::TagVal >(py_TagBox, "TagVal", "enum.IntEnum")
        .value("CLEAR", TagBox::TagVal::CLEAR)
        .value("BUF", TagBox::TagVal::BUF)
        .value("SET", TagBox::TagVal::SET)
        .export_values()
        .finalize()
    ;

    py_TagBox.def("__repr__",
        [](TagBox const&) {
            return "<amrex.TagBox>";
        }
    );

    py::class_< TagBoxArray, FabArrayBase >(m, "TagBoxArray")
        .def("__repr__",
            [](TagBoxArray const& tags) {
                std::stringstream s;
                s << tags.size();
                return "<amrex.TagBoxArray of size '" + s.str() + "'>";
            }
        )

        .def(py::init< BoxArray const&, DistributionMapping const&, int >(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow") = 0)
        .def(py::init< BoxArray const&, DistributionMapping const&, IntVect const& >(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow"))

        .def("clear", &TagBoxArray::clear)
        .def("ok", &TagBoxArray::ok)
        .def("__len__", &TagBoxArray::size)
        .def_property_readonly("size", &TagBoxArray::size)
        .def_property_readonly("local_size", &TagBoxArray::local_size)
        .def_property_readonly("n_grow_vect", &TagBoxArray::nGrowVect)
        .def_property_readonly("box_array",
             [](TagBoxArray const& tags) { return tags.boxArray(); },
             py::return_value_policy::reference_internal)
        .def_property_readonly("dist_map",
             [](TagBoxArray const& tags) { return tags.DistributionMap(); },
             py::return_value_policy::reference_internal)

        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val) {
                 tags.setVal(tag_value(val));
             },
             py::arg("val"))
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, int nghost) {
                 tags.setVal(tag_value(val), nghost);
             },
             py::arg("val"), py::arg("nghost"))
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, IntVect const& nghost) {
                 tags.setVal(tag_value(val), nghost);
             },
             py::arg("val"), py::arg("nghost"))
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, Box const& region, int nghost) {
                 tags.setVal(tag_value(val), region, nghost);
             },
             py::arg("val"), py::arg("region"), py::arg("nghost") = 0)
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, Box const& region, IntVect const& nghost) {
                 tags.setVal(tag_value(val), region, nghost);
             },
             py::arg("val"), py::arg("region"), py::arg("nghost"))
        .def("set_val",
             [](TagBoxArray& tags, BoxArray const& ba, TagBox::TagVal val) {
                 tags.setVal(ba, val);
             },
             py::arg("ba"), py::arg("val"))

        .def("buffer", &TagBoxArray::buffer, py::arg("nbuf"))
        .def("map_periodic_remove_duplicates",
             &TagBoxArray::mapPeriodicRemoveDuplicates, py::arg("geom"))
        .def("coarsen", &TagBoxArray::coarsen, py::arg("ratio"))
        .def("has_tags", &TagBoxArray::hasTags, py::arg("box"))
    ;
}
