/* Copyright 2026 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_ParticleHeader.H>

#include <string>


void init_ParticleHeader(py::module& m)
{
    using namespace amrex;

    py::class_<ParticleHeader> particle_header(m, "ParticleHeader");

    py::class_<ParticleHeader::GridEntry>(particle_header, "GridEntry")
        .def(py::init<>())
        .def_readwrite("which", &ParticleHeader::GridEntry::which,
            "index of the binary data file (DATA_XXXXX) holding this grid's particles")
        .def_readwrite("count", &ParticleHeader::GridEntry::count,
            "number of particles stored for this grid")
        .def_readwrite("where", &ParticleHeader::GridEntry::where,
            "byte offset of this grid's particle data within the data file")
        .def("__repr__",
            [](ParticleHeader::GridEntry const & e) {
                return "<amrex.ParticleHeader.GridEntry: DATA_" +
                    std::to_string(e.which) + ", " +
                    std::to_string(e.count) + " particles at byte " +
                    std::to_string(e.where) + ">";
            })
        ;

    particle_header
        .def(py::init<>())

        .def_static("read", &ParticleHeader::read,
            py::arg("dir"), py::arg("file"),
            "Read and parse the ``Header`` of a particle plotfile/checkpoint.\n\n"
            "This discovers the on-disk layout (number and names of real/int\n"
            "components, precision, checkpoint flag, ...) without constructing a\n"
            "matching ParticleContainer first.\n\n"
            "Parameters\n"
            "----------\n"
            "dir : str\n"
            "    plotfile/checkpoint directory\n"
            "file : str\n"
            "    particle sub-directory name (e.g. ``\"particle0\"``)")

        .def_readwrite("version", &ParticleHeader::version,
            "raw version string of the file format")
        .def_readwrite("how", &ParticleHeader::how,
            "precision the data was written in: 'single' or 'double'")
        .def_readwrite("convert_ids", &ParticleHeader::convert_ids,
            "whether particle ids need conversion (Version_Two_Dot_One+)")
        .def_readwrite("dim", &ParticleHeader::dim,
            "AMREX_SPACEDIM the file was written with")
        .def_readwrite("num_real", &ParticleHeader::num_real,
            "number of real components (pure SoA: excludes the positions)")
        .def_readwrite("real_comp_names", &ParticleHeader::real_comp_names,
            "names of the real components (len == num_real)")
        .def_readwrite("num_int", &ParticleHeader::num_int,
            "number of integer components")
        .def_readwrite("int_comp_names", &ParticleHeader::int_comp_names,
            "names of the integer components (len == num_int)")
        .def_readwrite("is_checkpoint", &ParticleHeader::is_checkpoint,
            "True if the file is a checkpoint, False for a plotfile")
        .def_readwrite("num_particles", &ParticleHeader::num_particles,
            "total number of particles in the file")
        .def_readwrite("next_id", &ParticleHeader::next_id,
            "the next particle id to hand out")
        .def_readwrite("finest_level", &ParticleHeader::finest_level,
            "finest level present in the file")
        .def_property_readonly("grids",
            [](ParticleHeader const & h) {
                py::list levels;
                for (auto const & lev : h.grids) {
                    py::list entries;
                    for (auto const & entry : lev) {
                        entries.append(entry);
                    }
                    levels.append(entries);
                }
                return levels;
            },
            "per level and grid: where each grid's binary particle data is\n"
            "stored, as a list (levels) of lists of\n"
            ":py:class:`~amrex.ParticleHeader.GridEntry`. Grids without\n"
            "particles have a zero ``count``.")

        .def("__repr__",
            [](ParticleHeader const & h) {
                return "<amrex.ParticleHeader: " +
                    std::to_string(h.num_particles) + " particles, " +
                    std::to_string(h.num_real) + " real + " +
                    std::to_string(h.num_int) + " int comps, " +
                    (h.is_checkpoint ? "checkpoint" : "plotfile") + ">";
            })
        ;
}
