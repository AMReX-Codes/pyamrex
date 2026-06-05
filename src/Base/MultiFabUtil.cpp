/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Vector.H>

#include <cstddef>
#include <string>
#include <vector>


namespace
{
    /** Convert a (Python list grown) std::vector of pointers to an
     *  amrex::Vector of const pointers. */
    template< class T >
    amrex::Vector< T const* >
    to_const_vector (std::vector< T* > const & in)
    {
        amrex::Vector< T const* > out;
        out.reserve(in.size());
        for (auto const * ptr : in) { out.push_back(ptr); }
        return out;
    }

    template< class T >
    amrex::Vector< T* >
    to_vector (std::vector< T* > const & in)
    {
        return amrex::Vector< T* >(in.begin(), in.end());
    }

    /** Throw a ValueError if a per-direction MultiFab list has the wrong
     *  number of entries. */
    void check_num_dirs (std::string const & name, std::size_t size)
    {
        if (size != AMREX_SPACEDIM) {
            throw py::value_error(
                name + " must have exactly AMREX_SPACEDIM=" +
                std::to_string(AMREX_SPACEDIM) + " entries (one per "
                "direction), got " + std::to_string(size));
        }
    }

    /** Throw a ValueError if a MultiFab has too few components. */
    void check_num_comp (std::string const & name, int ncomp, int needed)
    {
        if (ncomp < needed) {
            throw py::value_error(
                name + " needs at least " + std::to_string(needed) +
                " components, got " + std::to_string(ncomp));
        }
    }
}


void init_MultiFabUtil (py::module& m)
{
    using namespace amrex;

    constexpr auto doc_average_down =
        "Average fine MultiFab onto crse MultiFab. Both MultiFabs are "
        "assumed to be cell-centered. This routine DOES NOT assume that "
        "the crse BoxArray is a coarsened version of the fine BoxArray. "
        "Includes volume weighting for curvilinear coordinates.";
    m.def("average_down",
          [](MultiFab const & S_fine, MultiFab & S_crse,
             Geometry const & fgeom, Geometry const & cgeom,
             int scomp, int ncomp, IntVect const & ratio)
          { average_down(S_fine, S_crse, fgeom, cgeom, scomp, ncomp, ratio); },
          py::arg("S_fine"), py::arg("S_crse"),
          py::arg("fgeom"), py::arg("cgeom"),
          py::arg("scomp"), py::arg("ncomp"), py::arg("ratio"),
          doc_average_down);
    m.def("average_down",
          [](MultiFab const & S_fine, MultiFab & S_crse,
             Geometry const & fgeom, Geometry const & cgeom,
             int scomp, int ncomp, int ratio)
          { average_down(S_fine, S_crse, fgeom, cgeom, scomp, ncomp, ratio); },
          py::arg("S_fine"), py::arg("S_crse"),
          py::arg("fgeom"), py::arg("cgeom"),
          py::arg("scomp"), py::arg("ncomp"), py::arg("ratio"),
          doc_average_down);

    constexpr auto doc_average_down_no_geom =
        "Average fine MultiFab onto crse MultiFab without volume "
        "weighting. This routine DOES NOT assume that the crse BoxArray "
        "is a coarsened version of the fine BoxArray. Works for both "
        "cell-centered and nodal MultiFabs.";
    m.def("average_down",
          [](FabArray<FArrayBox> const & S_fine, FabArray<FArrayBox> & S_crse,
             int scomp, int ncomp, IntVect const & ratio)
          { average_down(S_fine, S_crse, scomp, ncomp, ratio); },
          py::arg("S_fine"), py::arg("S_crse"),
          py::arg("scomp"), py::arg("ncomp"), py::arg("ratio"),
          doc_average_down_no_geom);
    m.def("average_down",
          [](FabArray<FArrayBox> const & S_fine, FabArray<FArrayBox> & S_crse,
             int scomp, int ncomp, int ratio)
          { average_down(S_fine, S_crse, scomp, ncomp, ratio); },
          py::arg("S_fine"), py::arg("S_crse"),
          py::arg("scomp"), py::arg("ncomp"), py::arg("ratio"),
          doc_average_down_no_geom);

    m.def("average_down_faces",
          [](std::vector<MultiFab*> const & fine,
             std::vector<MultiFab*> const & crse,
             IntVect const & ratio, int ngcrse)
          {
              check_num_dirs("fine", fine.size());
              check_num_dirs("crse", crse.size());
              auto const c_fine = to_const_vector(fine);
              auto v_crse = to_vector(crse);
              average_down_faces(c_fine, v_crse, ratio, ngcrse);
          },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("ngcrse") = 0,
          "Average fine face-based MultiFabs (one per direction) onto "
          "crse face-based MultiFabs.");
    m.def("average_down_faces",
          [](FabArray<FArrayBox> const & fine, FabArray<FArrayBox> & crse,
             IntVect const & ratio, int ngcrse)
          { average_down_faces(fine, crse, ratio, ngcrse); },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("ngcrse") = 0,
          "Average down for one face direction. It uses the IndexType of "
          "the MultiFabs to determine the direction. It is expected that "
          "one direction is nodal and the rest are cell-centered.");
    m.def("average_down_faces",
          [](FabArray<FArrayBox> const & fine, FabArray<FArrayBox> & crse,
             IntVect const & ratio, Geometry const & crse_geom)
          { average_down_faces(fine, crse, ratio, crse_geom); },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("crse_geom"),
          "Average down for one face direction, taking periodicity into "
          "account.");

    m.def("average_down_edges",
          [](std::vector<MultiFab*> const & fine,
             std::vector<MultiFab*> const & crse,
             IntVect const & ratio, int ngcrse)
          {
              check_num_dirs("fine", fine.size());
              check_num_dirs("crse", crse.size());
              auto const c_fine = to_const_vector(fine);
              auto v_crse = to_vector(crse);
              average_down_edges(c_fine, v_crse, ratio, ngcrse);
          },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("ngcrse") = 0,
          "Average fine edge-based MultiFabs (one per direction) onto "
          "crse edge-based MultiFabs.");
    m.def("average_down_edges",
          [](MultiFab const & fine, MultiFab & crse,
             IntVect const & ratio, int ngcrse)
          { average_down_edges(fine, crse, ratio, ngcrse); },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("ngcrse") = 0,
          "Average down for one edge direction. It uses the IndexType of "
          "the MultiFabs to determine the direction. It is expected that "
          "one direction is cell-centered and the rest are nodal.");

    m.def("average_down_nodal",
          [](FabArray<FArrayBox> const & fine, FabArray<FArrayBox> & crse,
             IntVect const & ratio, int ngcrse, bool mfiter_is_definitely_safe)
          { average_down_nodal(fine, crse, ratio, ngcrse,
                               mfiter_is_definitely_safe); },
          py::arg("fine"), py::arg("crse"),
          py::arg("ratio"), py::arg("ngcrse") = 0,
          py::arg("mfiter_is_definitely_safe") = false,
          "Average fine node-based MultiFab onto crse node-centered "
          "MultiFab.");

    m.def("average_node_to_cellcenter",
          [](MultiFab & cc, int dcomp, MultiFab const & nd,
             int scomp, int ncomp, int ngrow)
          { average_node_to_cellcenter(cc, dcomp, nd, scomp, ncomp, ngrow); },
          py::arg("cc"), py::arg("dcomp"),
          py::arg("nd"), py::arg("scomp"), py::arg("ncomp"),
          py::arg("ngrow") = 0,
          "Average nodal-based MultiFab onto cell-centered MultiFab.");

    m.def("average_edge_to_cellcenter",
          [](MultiFab & cc, int dcomp,
             std::vector<MultiFab*> const & edge, int ngrow)
          {
              check_num_dirs("edge", edge.size());
              check_num_comp("cc", cc.nComp(), dcomp + AMREX_SPACEDIM);
              auto const c_edge = to_const_vector(edge);
              average_edge_to_cellcenter(cc, dcomp, c_edge, ngrow);
          },
          py::arg("cc"), py::arg("dcomp"), py::arg("edge"),
          py::arg("ngrow") = 0,
          "Average edge-based MultiFabs (one per direction) onto a "
          "cell-centered MultiFab.");

    m.def("average_face_to_cellcenter",
          [](MultiFab & cc, int dcomp,
             std::vector<MultiFab*> const & fc, int ngrow)
          {
              check_num_dirs("fc", fc.size());
              check_num_comp("cc", cc.nComp(), dcomp + AMREX_SPACEDIM);
              auto const c_fc = to_const_vector(fc);
              average_face_to_cellcenter(cc, dcomp, c_fc, ngrow);
          },
          py::arg("cc"), py::arg("dcomp"), py::arg("fc"),
          py::arg("ngrow") = 0,
          "Average face-based MultiFabs (one per direction) onto a "
          "cell-centered MultiFab.");

    m.def("average_cellcenter_to_face",
          [](std::vector<MultiFab*> const & fc, MultiFab const & cc,
             Geometry const & geom, int ncomp,
             bool use_harmonic_averaging, int ngrow)
          {
              check_num_dirs("fc", fc.size());
              check_num_comp("cc", cc.nComp(), ncomp);
              if (cc.nGrowVect().min() < ngrow + 1) {
                  throw py::value_error(
                      "cc needs at least ngrow+1 ghost cells");
              }
              auto v_fc = to_vector(fc);
              average_cellcenter_to_face(v_fc, cc, geom, ncomp,
                                         use_harmonic_averaging, ngrow);
          },
          py::arg("fc"), py::arg("cc"), py::arg("geom"),
          py::arg("ncomp") = 1,
          py::arg("use_harmonic_averaging") = false,
          py::arg("ngrow") = 0,
          "Average a cell-centered MultiFab onto face-based MultiFabs "
          "(one per direction) with geometric weighting.");

    m.def("sum_fine_to_coarse",
          &sum_fine_to_coarse,
          py::arg("S_fine"), py::arg("S_crse"),
          py::arg("scomp"), py::arg("ncomp"), py::arg("ratio"),
          py::arg("cgeom"), py::arg("fgeom"),
          "Add a coarsened version of the data contained in the S_fine "
          "MultiFab to S_crse, including ghost cells.");
}
