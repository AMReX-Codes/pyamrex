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


# C++ Standard in Superbuilds #################################################
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
set_cxx17_superbuild()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
set_ccache()


# Output (build) Directories ##################################################
#
set_default_build_dirs()


# Options and Variants ########################################################
#
set(pyAMReX_VERSION_INFO "" CACHE STRING
    "PEP-440 conformant version (set by distutils)")
# change the default build type to Release (or RelWithDebInfo) instead of Debug
set_default_build_type("Release")

# Testing logic with possibility to overwrite on a project basis in superbuilds
set_testing_option(pyAMReX_BUILD_TESTING)  # default: ON (BUILD_TESTING)


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

# C++ properties: at least a C++17 capable compiler is needed
target_compile_features(pyAMReX PUBLIC cxx_std_17)
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
set_target_properties(pyAMReX PROPERTIES
    # hide symbols for combining multiple pybind11 modules downstream & for
    # reduced binary size
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    # name of the pybind-generated python module, which is wrapped in another
    # fluffy front-end modules, so we can extend it with pure Python
    ARCHIVE_OUTPUT_NAME amrex_pybind
    LIBRARY_OUTPUT_NAME amrex_pybind
    # build output directories - mainly set to run tests from CMake & IDEs
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex
    PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex
    COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex
)
if(EMSCRIPTEN)
    set_target_properties(pyAMReX PROPERTIES
        PREFIX "")
else()
    pybind11_extension(pyAMReX)
endif()
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    pybind11_strip(pyAMReX)
endif()

# AMReX helper function: propagate CUDA specific target & source properties
if(AMReX_GPU_BACKEND STREQUAL CUDA)
    setup_target_for_cuda_compilation(pyAMReX)
    target_compile_features(pyAMReX PUBLIC cuda_std_17)
    set_target_properties(pyAMReX PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
endif()


# Defines #####################################################################
#
# for module __version__
target_compile_definitions(pyAMReX PRIVATE
    PYAMReX_VERSION_INFO=${pyAMReX_VERSION_INFO})


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
# default installation directories
set_default_install_dirs()

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


# pip helpers for the amrex package ###########################################
#
set(PYINSTALLOPTIONS "" CACHE STRING
    "Additional parameters to pass to `pip install`")

# build the wheel by re-using the shared library we build
add_custom_target(pip_wheel
    ${CMAKE_COMMAND} -E rm -f "amrex*.whl"
    COMMAND
        ${CMAKE_COMMAND} -E env PYAMREX_LIBDIR=$<TARGET_FILE_DIR:pyAMReX>
            ${Python_EXECUTABLE} -m pip wheel -v --no-build-isolation --use-feature=in-tree-build ${pyAMReX_SOURCE_DIR}
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
    DEPENDS
        pyAMReX
)

# this will also upgrade/downgrade dependencies, e.g., when the version of numpy changes
add_custom_target(pip_install_requirements
    ${Python_EXECUTABLE} -m pip install ${PYINSTALLOPTIONS} -r "${pyAMReX_SOURCE_DIR}/requirements.txt"
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
)

# We force-install because in development, it is likely that the version of
# the package does not change, but the code did change. We need --no-deps,
# because otherwise pip would also force reinstall all dependencies.
add_custom_target(pip_install
    ${CMAKE_COMMAND} -E env AMREX_MPI=${AMReX_MPI}
        ${Python_EXECUTABLE} -m pip install --force-reinstall --no-deps ${PYINSTALLOPTIONS} "amrex*.whl"
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
    DEPENDS
        pyAMReX pip_wheel pip_install_requirements
)


# Tests #######################################################################
#
if(pyAMReX_BUILD_TESTING)
    # copy Python wrapper library to build directory
    add_custom_command(TARGET pyAMReX POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${pyAMReX_SOURCE_DIR}/src/amrex
            $<TARGET_FILE_DIR:pyAMReX>
    )

    add_test(NAME pytest
        COMMAND ${Python_EXECUTABLE} -m pytest
            ${pyAMReX_SOURCE_DIR}/tests
        WORKING_DIRECTORY
            ${CMAKE_PYTHON_OUTPUT_DIRECTORY}
    )
endif()


# Status Summary for Build Options ############################################
#
pyAMReX_print_summary()
