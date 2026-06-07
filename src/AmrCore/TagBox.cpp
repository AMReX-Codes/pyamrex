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

    py::class_< TagBox > py_TagBox(
        m, "TagBox",
        R"pbdoc(
Cell-tag storage used by ``AmrCore.error_est``.

Use ``TagBox.SET`` to request refinement, ``TagBox.CLEAR`` to remove a tag and
``TagBox.BUF`` for AMReX-generated buffered tags.
)pbdoc");

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

    py::class_< TagBoxArray, FabArrayBase >(
        m, "TagBoxArray",
        R"pbdoc(
Distributed array of ``TagBox`` objects used during AMR error estimation.

Python ``AmrCore.error_est`` overrides receive a ``TagBoxArray`` and mark cells
with ``set_val(TagBox.SET, ...)``.
)pbdoc")
        .def("__repr__",
            [](TagBoxArray const& tags) {
                std::stringstream s;
                s << tags.size();
                return "<amrex.TagBoxArray of size '" + s.str() + "'>";
            }
        )

        .def(py::init< BoxArray const&, DistributionMapping const&, int >(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow") = 0,
             "Construct tag storage on ba/dm with an isotropic grow width.")
        .def(py::init< BoxArray const&, DistributionMapping const&, IntVect const& >(),
             py::arg("ba"), py::arg("dm"), py::arg("ngrow"),
             "Construct tag storage on ba/dm with per-direction grow widths.")

        .def("clear", &TagBoxArray::clear,
             "Release all tag data and metadata.")
        .def("ok", &TagBoxArray::ok,
             "Return True if the tag array is internally consistent.")
        .def("__len__", &TagBoxArray::size)
        .def_property_readonly("size", &TagBoxArray::size,
             "Number of boxes in the global tag layout.")
        .def_property_readonly("local_size", &TagBoxArray::local_size,
             "Number of tag boxes owned by this MPI rank.")
        .def_property_readonly("n_grow_vect", &TagBoxArray::nGrowVect,
             "Grow width of the tag storage in each coordinate direction.")
        .def_property_readonly("box_array",
             [](TagBoxArray const& tags) { return tags.boxArray(); },
             py::return_value_policy::reference_internal,
             "BoxArray defining the valid regions for this tag array.")
        .def_property_readonly("dist_map",
             [](TagBoxArray const& tags) { return tags.DistributionMap(); },
             py::return_value_policy::reference_internal,
             "DistributionMapping defining ownership of this tag array.")

        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val) {
                 tags.setVal(tag_value(val));
             },
             py::arg("val"),
             "Set all valid and grow cells in the tag array to val.")
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, int nghost) {
                 tags.setVal(tag_value(val), nghost);
             },
             py::arg("val"), py::arg("nghost"),
             "Set all valid cells plus nghost grow cells to val.")
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, IntVect const& nghost) {
                 tags.setVal(tag_value(val), nghost);
             },
             py::arg("val"), py::arg("nghost"),
             "Set all valid cells plus per-direction grow cells to val.")
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, Box const& region, int nghost) {
                 tags.setVal(tag_value(val), region, nghost);
             },
             py::arg("val"), py::arg("region"), py::arg("nghost") = 0,
             "Set cells intersecting region, optionally grown by nghost, to val.")
        .def("set_val",
             [](TagBoxArray& tags, TagBox::TagVal val, Box const& region, IntVect const& nghost) {
                 tags.setVal(tag_value(val), region, nghost);
             },
             py::arg("val"), py::arg("region"), py::arg("nghost"),
             "Set cells intersecting region with per-direction grow widths to val.")
        .def("set_val",
             [](TagBoxArray& tags, BoxArray const& ba, TagBox::TagVal val) {
                 tags.setVal(ba, val);
             },
             py::arg("ba"), py::arg("val"),
             "Set cells covered by ba to val.")

        .def("buffer", &TagBoxArray::buffer, py::arg("nbuf"),
             "Grow every SET tag by nbuf cells using AMReX tag-buffer rules.")
        .def("map_periodic_remove_duplicates",
             &TagBoxArray::mapPeriodicRemoveDuplicates, py::arg("geom"),
             "Map tags through periodic boundaries described by geom and remove duplicates.")
        .def("coarsen", &TagBoxArray::coarsen, py::arg("ratio"),
             "Coarsen tags in place by ratio.")
        .def("has_tags", &TagBoxArray::hasTags, py::arg("box"),
             "Return True if box contains any SET or BUF tags.")
    ;
}
