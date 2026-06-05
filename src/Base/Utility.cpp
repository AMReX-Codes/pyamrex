/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Revathi Jambunathan, Axel Huebl
 */
#include "pyAMReX.H"

#include <AMReX_Utility.H>

#include <string>


void init_Utility(py::module& m)
{
    m.def("concatenate",
          &amrex::Concatenate,
          "Builds plotfile name",
          py::arg("root"), py::arg("num"), py::arg("mindigits")=5
    );

    m.def("second",
          [](){ return amrex::second(); },
          "Returns the current time as the number of seconds elapsed "
          "since some arbitrary point in the past (e.g., for wallclock "
          "timers)."
    );

    m.def("util_create_directory",
          [](std::string const & path, bool verbose) {
              return amrex::UtilCreateDirectory(path, 0755, verbose);
          },
          "Creates the specified directories. path may be either a full "
          "pathname or a relative pathname. It will create all the "
          "directories in the pathname, if they don't already exist.",
          py::arg("path"), py::arg("verbose") = false
    );
    m.def("util_create_clean_directory",
          &amrex::UtilCreateCleanDirectory,
          "Create a new directory, renaming the old one if it exists",
          py::arg("path"), py::arg("callbarrier") = true
    );
    m.def("util_create_directory_destructive",
          &amrex::UtilCreateDirectoryDestructive,
          "Create a new directory, removing the old one if it exists",
          py::arg("path"), py::arg("callbarrier") = true
    );
    m.def("file_exists",
          &amrex::FileExists,
          "Check if a file already exists. Return true if the filename "
          "is an existing file, directory, or link. For links, this "
          "operates on the link and not what the link points to.",
          py::arg("filename")
    );
}
