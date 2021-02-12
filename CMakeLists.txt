# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.18.0)
project(pyAMReX VERSION 21.02)

include(${pyAMReX_SOURCE_DIR}/cmake/pyAMReXFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()
# device link step not yet fully implemented in AMReX logic
# https://cmake.org/cmake/help/latest/policy/CMP0105.html
if(POLICY CMP0105)
    cmake_policy(SET CMP0105 OLD)
endif()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
set_ccache()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
# change the default build type to RelWithDebInfo (or Release) instead of Debug
set_default_build_type("RelWithDebInfo")

# this defined the variable BUILD_TESTING which is ON by default
#include(CTest)


# Dependencies ################################################################
#

# AMReX
#   builds AMReX from source (default) or finds an existing install
include(${pyAMReX_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11
#   builds pybind11 from git (default), form local source or
#   finds an existing install
include(${pyAMReX_SOURCE_DIR}/cmake/dependencies/pybind11.cmake)


# Targets #####################################################################
#
# collect all objects for compilation
add_library(pyAMReX MODULE src/pyAMReX.cpp)
add_library(pyAMReX::pyAMReX ALIAS pyAMReX)

# own headers
target_include_directories(pyAMReX PUBLIC
    $<BUILD_INTERFACE:${pyAMReX_SOURCE_DIR}/src>
)

# if we include <AMReX_buildInfo.H> we will need to call:
include(AMReXBuildInfo)
generate_buildinfo(pyAMReX "${pyAMReX_SOURCE_DIR}")
target_link_libraries(pyAMReX PRIVATE buildInfo::pyAMReX)

# add sources
add_subdirectory(src)

# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(pyAMReX PUBLIC cxx_std_14)
set_target_properties(pyAMReX PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link dependencies
target_link_libraries(pyAMReX PUBLIC AMReX::amrex)
target_link_libraries(pyAMReX PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

# binary name: we will encoded 1D, 2D and 3D in here so we can provide a
# wrapper library that pre-compiled all three geometry variants of AMReX
#set_pyAMReX_binary_name()

# set Python module properties
pybind11_extension(pyAMReX)
pybind11_strip(pyAMReX)
set_target_properties(pyAMReX PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")

# AMReX helper function: propagate CUDA specific target & source properties
if(AMReX_GPU_BACKEND STREQUAL CUDA)
    setup_target_for_cuda_compilation(pyAMReX)
    target_compile_features(pyAMReX PUBLIC cuda_std_14)
    set_target_properties(pyAMReX PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
endif()


# Defines #####################################################################
#
# TBD


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
# these files are used if pyAMReX is installed and picked up by a downstream
# project (not needed yet)

#include(CMakePackageConfigHelpers)
#write_basic_package_version_file("pyAMReXConfigVersion.cmake"
#    VERSION ${pyAMReX_VERSION}
#    COMPATIBILITY SameMajorVersion
#)


# Installs ####################################################################
#
# headers, libraries and executables
set(pyAMReX_INSTALL_TARGET_NAMES pyAMReX)

install(TARGETS ${pyAMReX_INSTALL_TARGET_NAMES}
    EXPORT pyAMReXTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# CMake package file for find_package(pyAMReX::pyAMReX) in depending projects
#install(EXPORT pyAMReXTargets
#    FILE pyAMReXTargets.cmake
#    NAMESPACE pyAMReX::
#    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
#)
#install(
#    FILES
#        ${pyAMReX_BINARY_DIR}/pyAMReXConfig.cmake
#        ${pyAMReX_BINARY_DIR}/pyAMReXConfigVersion.cmake
#    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
#)


# Tests #######################################################################
#
# TBD

#if(BUILD_TESTING)
#    enable_testing()
#
#    add_test(...)
#endif()


# Status Summary for Build Options ############################################
#
pyAMReX_print_summary()
