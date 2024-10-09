function(find_pybind11)
    if(TARGET pybind11::module)
        message(STATUS "pybind11::module target already imported")
    elseif(pyAMReX_pybind11_internal)
        if(pyAMReX_pybind11_src)
            message(STATUS "Compiling local pybind11 ...")
            message(STATUS "pybind11 source path: ${pyAMReX_pybind11_src}")
            if(NOT IS_DIRECTORY ${pyAMReX_pybind11_src})
                message(FATAL_ERROR "Specified directory pyAMReX_pybind11_src='${pyAMReX_pybind11_src}' does not exist!")
            endif()
        elseif(pyAMReX_pybind11_tar)
            message(STATUS "Downloading pybind11 ...")
            message(STATUS "pybind11 source: ${pyAMReX_pybind11_tar}")
        elseif(pyAMReX_pybind11_branch)
            message(STATUS "Downloading pybind11 ...")
            message(STATUS "pybind11 repository: ${pyAMReX_pybind11_repo} (${pyAMReX_pybind11_branch})")
            include(FetchContent)
        endif()
    endif()
    if(TARGET pybind11::module)
        # nothing to do, target already exists in the superbuild
    elseif(pyAMReX_pybind11_internal)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        if(pyAMReX_pybind11_src)
            add_subdirectory(${pyAMReX_pybind11_src} _deps/localpybind11-build/)
        else()
            include(FetchContent)
            if(pyAMReX_pybind11_tar)
                FetchContent_Declare(fetchedpybind11
                    URL             ${pyAMReX_pybind11_tar}
                    URL_HASH        ${pyAMReX_pybind11_tar_hash}
                    BUILD_IN_SOURCE OFF
                )
            else()
                FetchContent_Declare(fetchedpybind11
                    GIT_REPOSITORY ${pyAMReX_pybind11_repo}
                    GIT_TAG        ${pyAMReX_pybind11_branch}
                    BUILD_IN_SOURCE 0
                )
            endif()
            FetchContent_MakeAvailable(fetchedpybind11)

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDpybind11)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDpybind11)
        endif()
    elseif(NOT pyAMReX_pybind11_internal)
        find_package(pybind11 2.12.0 CONFIG REQUIRED)
        message(STATUS "pybind11: Found version '${pybind11_VERSION}'")
    endif()
endfunction()

option(pyAMReX_pybind11_internal "Download & build pybind11" ${pyAMReX_SUPERBUILD})

# local source-tree
set(pyAMReX_pybind11_src ""
    CACHE PATH
    "Local path to pybind11 source directory (preferred if set)")

# tarball fetcher
set(pyAMReX_pybind11_tar "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz"
    CACHE STRING
    "Remote tarball link to pull and build pybind11 from if(pyAMReX_pybind11_internal)")
set(pyAMReX_pybind11_tar_hash "SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20"
    CACHE STRING
    "Hash checksum of the tarball of pybind11 if(pyAMReX_pybind11_internal)")

# Git fetcher
set(pyAMReX_pybind11_repo "https://github.com/pybind/pybind11.git"
    CACHE STRING
    "Repository URI to pull and build pybind11 from if(pyAMReX_pybind11_internal)")
set(pyAMReX_pybind11_branch "v2.12.0"
    CACHE STRING
    "Repository branch for pyAMReX_pybind11_repo if(pyAMReX_pybind11_internal)")

find_pybind11()
