macro(find_amrex)
    if(TARGET AMReX::amrex)
        message(STATUS "AMReX::amrex target already imported")
    elseif(pyAMReX_amrex_src)
        message(STATUS "Compiling local AMReX ...")
        message(STATUS "AMReX source path: ${pyAMReX_amrex_src}")
        if(NOT IS_DIRECTORY ${pyAMReX_amrex_src})
            message(FATAL_ERROR "Specified directory pyAMReX_amrex_src='${pyAMReX_amrex_src}' does not exist!")
        endif()
    elseif(pyAMReX_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        message(STATUS "AMReX repository: ${pyAMReX_amrex_repo} (${pyAMReX_amrex_branch})")
        include(FetchContent)
    endif()
    if(TARGET AMReX::amrex)
        # nothing to do, target already exists in the superbuild
    elseif(pyAMReX_amrex_internal OR pyAMReX_amrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(AMReX_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(AMReX_FPE ON CACHE BOOL "")
        else()
            set(AMReX_ASSERTIONS OFF CACHE BOOL "")
            set(AMReX_FPE OFF CACHE BOOL "")
        endif()

        set(AMReX_EB ON CACHE INTERNAL "")
        set(AMReX_PIC ON CACHE INTERNAL "")
        set(AMReX_ENABLE_TESTS OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
        set(AMReX_PARTICLES ON CACHE INTERNAL "")  # default: OFF

        if(pyAMReX_amrex_src)
            list(APPEND CMAKE_MODULE_PATH "${pyAMReX_amrex_src}/Tools/CMake")
            if(AMReX_GPU_BACKEND STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            add_subdirectory(${pyAMReX_amrex_src} _deps/localamrex-build/)
        else()
            if(AMReX_GPU_BACKEND STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            FetchContent_Declare(fetchedamrex
                GIT_REPOSITORY ${pyAMReX_amrex_repo}
                GIT_TAG        ${pyAMReX_amrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedamrex)
            list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)
        endif()

        message(STATUS "AMReX: Using version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
    elseif(NOT pyAMReX_amrex_internal)
        message(STATUS "Searching for pre-installed AMReX ...")
        # https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#importing-amrex-into-your-cmake-project
        # not strictly required yet to compile pyAMReX: EB
        find_package(AMReX 25.01 CONFIG REQUIRED COMPONENTS PARTICLES PIC)
        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")

        if(AMReX_GPU_BACKEND STREQUAL CUDA)
            enable_language(CUDA)
        endif()
    endif()
endmacro()

# local source-tree
set(pyAMReX_amrex_src ""
    CACHE PATH
    "Local path to AMReX source directory (preferred if set)")

# Git fetcher
option(pyAMReX_amrex_internal "Download & build AMReX" ON)
set(pyAMReX_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(pyAMReX_amrex_internal)")
set(pyAMReX_amrex_branch "25.01"
    CACHE STRING
    "Repository branch for pyAMReX_amrex_repo if(pyAMReX_amrex_internal)")

find_amrex()
