/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_FabArray.H>


void init_MFInfo(nb::module_ &m)
{
    using namespace amrex;

    nb::class_<MFInfo>(m, "MFInfo")
        .def_rw("alloc", &MFInfo::alloc)
        .def_rw("arena", &MFInfo::arena)
        .def_rw("tags", &MFInfo::tags)

        .def(nb::init<>())

        .def("set_alloc", &MFInfo::SetAlloc)
        .def("set_arena", &MFInfo::SetArena)
                //.def("set_tag", nb::overload_cast< std::string >(&MFInfo::SetTag))
        .def("set_tag", [](MFInfo &info, std::string tag) { info.SetTag(std::move(tag)); });

    nb::class_<MFItInfo>(m, "MFItInfo")
        .def_rw("do_tiling", &MFItInfo::do_tiling)
        .def_rw("dynamic", &MFItInfo::dynamic)
        .def_rw("device_sync", &MFItInfo::device_sync)
        .def_rw("num_streams", &MFItInfo::num_streams)
        .def_rw("tilesize", &MFItInfo::tilesize)

        .def(nb::init<>())

        .def("enable_tiling", &MFItInfo::EnableTiling,
             nb::arg("ts") /*=FabArrayBase::mfiter_tile_size*/ )
        .def("set_dynamic", &MFItInfo::SetDynamic,
             nb::arg("f"))
        .def("disable_device_sync", &MFItInfo::DisableDeviceSync)
        .def("set_device_sync", &MFItInfo::SetDeviceSync,
             nb::arg("f"))
        .def("set_num_streams", &MFItInfo::SetNumStreams,
             nb::arg("n"))
        .def("use_default_stream", &MFItInfo::UseDefaultStream);
}
