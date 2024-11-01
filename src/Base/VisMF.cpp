/* Copyright 2024 The AMReX Community
 *
 * Authors: David Grote
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_VisMF.H>
#include <AMReX_MultiFab.H>

void init_VisMF(py::module &m)
{
    py::class_< amrex::VisMF > py_VisMF(m, "VisMF");

    py_VisMF
        .def_static("Write",
           [](const amrex::FabArray<amrex::FArrayBox> &mf, const std::string& name) {
               return amrex::VisMF::Write(mf, name);
           },
           py::arg("mf"), py::arg("name"),
           "Writes a Multifab to the specified file")
        .def_static("Read",
           [](const std::string &name) {
               amrex::MultiFab mf;
               if (amrex::VisMF::Exist(name)) {
                   amrex::VisMF::Read(mf, name);
               } else {
                   throw std::runtime_error("MultiFab file " + name + " couldn't be found!");
               }
               return mf;
               },
           py::return_value_policy::move,
           py::arg("name"),
           "Reads a MultiFab from the specified file")
        .def_static("Read",
           [](const std::string &name, amrex::MultiFab &mf) {
               if (amrex::VisMF::Exist(name)) {
                   amrex::VisMF::Read(mf, name);
               } else {
                   throw std::runtime_error("MultiFab file " + name + " couldn't be found!");
               }
               },
           py::arg("name"), py::arg("mf"),
           "Reads a MultiFab from the specified file into the given MultiFab. The BoxArray on the disk must match the BoxArray * in mf")
        ;
}
