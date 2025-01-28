/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_FabArray.H>


void init_MFInfo(py::module &m)
{
    using namespace amrex;

    py::class_<MFInfo>(m, "MFInfo")
        .def_readwrite("alloc", &MFInfo::alloc)
        .def_readwrite("arena", &MFInfo::arena)
        .def_readwrite("tags", &MFInfo::tags)

        .def(py::init<>())

        .def("set_alloc", &MFInfo::SetAlloc)
        .def("set_arena", &MFInfo::SetArena)
                //.def("set_tag", py::overload_cast< std::string >(&MFInfo::SetTag))
        .def("set_tag", [](MFInfo &info, std::string tag) { info.SetTag(std::move(tag)); });

    py::class_<MFItInfo>(m, "MFItInfo")
        .def_readwrite("do_tiling", &MFItInfo::do_tiling)
        .def_readwrite("dynamic", &MFItInfo::dynamic)
        .def_readwrite("device_sync", &MFItInfo::device_sync)
        .def_readwrite("num_streams", &MFItInfo::num_streams)
        .def_readwrite("tilesize", &MFItInfo::tilesize)

        .def(py::init<>())

        .def("enable_tiling", &MFItInfo::EnableTiling,
             py::arg("ts") /*=FabArrayBase::mfiter_tile_size*/ )
        .def("set_dynamic", &MFItInfo::SetDynamic,
             py::arg("f"))
        .def("disable_device_sync", &MFItInfo::DisableDeviceSync)
        .def("set_device_sync", &MFItInfo::SetDeviceSync,
             py::arg("f"))
        .def("set_num_streams", &MFItInfo::SetNumStreams,
             py::arg("n"))
        .def("use_default_stream", &MFItInfo::UseDefaultStream);
}
