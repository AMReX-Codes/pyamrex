cmake_minimum_required(VERSION 3.24)
project(pyAMReXDownstreamNanobind LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(pyAMReX CONFIG REQUIRED COMPONENTS 3D)

nanobind_add_module(pyamrex_downstream NB_STATIC NOMINSIZE NOSTRIP module.cpp)
target_compile_features(pyamrex_downstream PRIVATE cxx_std_20)
target_link_libraries(pyamrex_downstream PRIVATE AMReX::amrex_3d)

enable_testing()
add_test(
    NAME downstream.nanobind-domain
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_domain.py
)
set_property(
    TEST downstream.nanobind-domain
    APPEND PROPERTY ENVIRONMENT
        "PYTHONPATH=$<TARGET_FILE_DIR:pyamrex_downstream>:$ENV{PYTHONPATH}"
)
