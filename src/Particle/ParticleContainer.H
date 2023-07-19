/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include "Base/Iterator.H"

#include <AMReX_BoxArray.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_ParticleContainer.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ArrayOfStructs.H>

#include <string>
#include <sstream>


template <typename T_ParticleType, int T_NArrayReal=0, int T_NArrayInt=0>
std::string particle_type_suffix ()
{
    std::string suffix;
    if constexpr (T_ParticleType::is_soa_particle)
        suffix = "pureSoA_" +
                 std::to_string(T_NArrayReal) + "_" +
                 std::to_string(T_NArrayInt);
    else
        suffix = std::to_string(T_ParticleType::NReal) + "_" +
                 std::to_string(T_ParticleType::NInt) + "_" +
                 std::to_string(T_NArrayReal) + "_" +
                 std::to_string(T_NArrayInt);

    return suffix;
}

template <bool is_const, typename T_ParIterBase>
void make_Base_Iterators (py::module &m, std::string allocstr)
{
    using namespace amrex;

    using iterator_base = T_ParIterBase;
    using container = typename iterator_base::ContainerType;
    using ParticleType = typename container::ParticleType;
    constexpr int NArrayReal = container::NArrayReal;
    constexpr int NArrayInt = container::NArrayInt;

    std::string const suffix = particle_type_suffix<ParticleType, NArrayReal, NArrayInt>();
    std::string particle_it_base_name = std::string("Par");
    if (is_const) particle_it_base_name += "Const";
    particle_it_base_name += "IterBase_" + suffix + "_" + allocstr;
    auto py_it_base = py::class_<iterator_base, MFIter>(m, particle_it_base_name.c_str())
        .def(py::init<container&, int>(),
            py::arg("particle_container"), py::arg("level"))
        .def(py::init<container&, int, MFItInfo&>(),
             py::arg("particle_container"), py::arg("level"), py::arg("info"))

        .def("particle_tile", &iterator_base::GetParticleTile,
                               py::return_value_policy::reference_internal)
        .def("soa", &iterator_base::GetStructOfArrays,
                               py::return_value_policy::reference_internal)

        .def_property_readonly_static("is_soa_particle", [](const py::object&){ return ParticleType::is_soa_particle;})
        .def_property_readonly("num_particles", &iterator_base::numParticles)
        .def_property_readonly("num_real_particles", &iterator_base::numRealParticles)
        .def_property_readonly("num_neighbor_particles", &iterator_base::numNeighborParticles)
        .def_property_readonly("level", &iterator_base::GetLevel)
        .def_property_readonly("pair_index", &iterator_base::GetPairIndex)
        .def("geom", &iterator_base::Geom, py::arg("level"))

        // eq. to void operator++()
        .def("__next__",
             &pyAMReX::iterator_next<iterator_base>,
             py::return_value_policy::reference_internal
        )
        .def("__iter__",
             [](iterator_base & it) -> iterator_base & {
                 return it;
             },
             py::return_value_policy::reference_internal
        )
    ;

    // only legacy particle has an AoS data structure for positions and id+cpu
    if constexpr (!ParticleType::is_soa_particle)
        py_it_base.def("aos",
                       &iterator_base::GetArrayOfStructs,
                       py::return_value_policy::reference_internal);
}

template <bool is_const, typename T_ParIter, template<class> class Allocator=amrex::DefaultAllocator>
void make_Iterators (py::module &m, std::string allocstr)
{
    using namespace amrex;

    using iterator = T_ParIter;
    using container = typename iterator::ContainerType;
    using ParticleType = typename container::ParticleType;
    constexpr int NArrayReal = container::NArrayReal;
    constexpr int NArrayInt = container::NArrayInt;

    using iterator_base = amrex::ParIterBase_impl<is_const, ParticleType, NArrayReal, NArrayInt, Allocator>;
    make_Base_Iterators< is_const, iterator_base >(m, allocstr);

    std::string const suffix = particle_type_suffix<ParticleType, NArrayReal, NArrayInt>();
    auto particle_it_name = std::string("Par");
    if (is_const) particle_it_name += "Const";
    particle_it_name += std::string("Iter_") + suffix + "_" + allocstr;
    py::class_<iterator, iterator_base>(m, particle_it_name.c_str())
        .def("__repr__",
             [particle_it_name](iterator const & pti) {
                 std::string r = "<amrex." + particle_it_name + " (";
                 if( !pti.isValid() ) { r.append("in"); }
                 r.append("valid)>");
                 return r;
             }
        )
        .def(py::init<container&, int>(),
             py::arg("particle_container"), py::arg("level"))
        .def(py::init<container&, int, MFItInfo&>(),
             py::arg("particle_container"), py::arg("level"), py::arg("info"))
        .def_property_readonly_static("is_soa_particle", [](const py::object&){ return ParticleType::is_soa_particle;})
    ;
}

template <typename T_ParticleType, int T_NArrayReal=0, int T_NArrayInt=0>
void make_ParticleInitData (py::module &m) {
    using namespace amrex;

    using ParticleType = T_ParticleType;
    using ParticleInitData = ParticleInitType<ParticleType::NReal, ParticleType::NInt, T_NArrayReal, T_NArrayInt>;
    // depends on https://github.com/AMReX-Codes/amrex/pull/3280
    // using ParticleInitData = ParticleInitType<ParticleType, T_NArrayReal, T_NArrayInt>;

    std::string const suffix = particle_type_suffix<T_ParticleType, T_NArrayReal, T_NArrayInt>();
    auto const particle_init_data_type =
        std::string("ParticleInitType_") + suffix;
    auto py_particle_init_data = py::class_<ParticleInitData>(m, particle_init_data_type.c_str())
        .def(py::init<>())
        .def_property_readonly_static("is_soa_particle", [](const py::object&){return ParticleType::is_soa_particle;})
        .def_readwrite("real_array_data", &ParticleInitData::real_array_data)
        .def_readwrite("int_array_data", &ParticleInitData::int_array_data)
    ;

    // only legacy particle has an AoS data structure for positions and id+cpu
    if constexpr (!ParticleType::is_soa_particle)
        py_particle_init_data
            .def_readwrite("real_struct_data", &ParticleInitData::real_struct_data)
            .def_readwrite("int_struct_data", &ParticleInitData::int_struct_data);
}

template <typename T_ParticleType, int T_NArrayReal=0, int T_NArrayInt=0,
          template<class> class Allocator=amrex::DefaultAllocator>
void make_ParticleContainer_and_Iterators (py::module &m, std::string allocstr)
{
    using namespace amrex;

    using ParticleType = T_ParticleType;
    using ParticleContainerType = ParticleContainer_impl<
        ParticleType, T_NArrayReal, T_NArrayInt,
        Allocator
    >;
    using ParticleInitData = typename ParticleContainerType::ParticleInitData;
    using ParticleTileType = typename ParticleContainerType::ParticleTileType;

    std::string const suffix = particle_type_suffix<T_ParticleType, T_NArrayReal, T_NArrayInt>();
    auto const particle_container_type = std::string("ParticleContainer_") + suffix + "_" + allocstr;
    auto py_pc = py::class_<ParticleContainerType>(m, particle_container_type.c_str())
        .def(py::init())
        .def(py::init<const Geometry&, const DistributionMapping&, const BoxArray&>())
        .def(py::init<const Vector<Geometry>&,
                      const Vector<DistributionMapping>&,
                      const Vector<BoxArray>&,
                      const Vector<int>&>())
        .def(py::init<const Vector<Geometry>&,
                      const Vector<DistributionMapping>&,
                      const Vector<BoxArray>&,
                      const Vector<IntVect>&>())

        .def_property_readonly_static("is_soa_particle", [](const py::object&){return ParticleType::is_soa_particle;})
        .def_property_readonly_static("NStructReal", [](const py::object&){return ParticleContainerType::NStructReal; })
        .def_property_readonly_static("NStructInt", [](const py::object&){return ParticleContainerType::NStructInt; })
        .def_property_readonly_static("NArrayReal", [](const py::object&){return ParticleContainerType::NArrayReal; })
        .def_property_readonly_static("NArrayInt", [](const py::object&){return ParticleContainerType::NArrayInt; })

        .def_property_readonly("finest_level", &ParticleContainerBase::finestLevel)

        // ParticleContainer ( const ParticleContainer &) = delete;
        // ParticleContainer& operator= ( const ParticleContainer & ) = delete;

        // ParticleContainer ( ParticleContainer && ) = default;
        // ParticleContainer& operator= ( ParticleContainer && ) = default;

        .def("Define",
                py::overload_cast<const Geometry&,
                                    const DistributionMapping&,
                                    const BoxArray&>
                (&ParticleContainerType::Define))

        .def("Define",
                py::overload_cast<const Vector<Geometry>&,
                                    const Vector<DistributionMapping>&,
                                    const Vector<BoxArray>&,
                                    const Vector<int>&>
                (&ParticleContainerType::Define))

        .def("Define",
                py::overload_cast<const Vector<Geometry>&,
                                    const Vector<DistributionMapping>&,
                                    const Vector<BoxArray>&,
                                    const Vector<IntVect>&>
                (&ParticleContainerType::Define))

        .def("numLocalTilesAtLevel", &ParticleContainerType::numLocalTilesAtLevel)

        .def("reserveData", &ParticleContainerType::reserveData)
        .def("resizeData", &ParticleContainerType::resizeData)

        // void InitFromAsciiFile (const std::string& file, int extradata,
        //                         const IntVect* Nrep = nullptr);

        // void InitFromBinaryFile (const std::string& file, int extradata);

        // void InitFromBinaryMetaFile
        //     void InitRandom (Long icount, ULong iseed,
        //                  const ParticleInitData& mass,
        //                  bool serialize = false, RealBox bx = RealBox());

        .def("Increment", &ParticleContainerType::Increment) // TODO pure SoA
        //.def("IncrementWithTotal", &ParticleContainerType::IncrementWithTotal, py::arg("mf"), py::arg("level"), py::arg("local")=false) // TODO pure SoA
        .def("Redistribute", &ParticleContainerType::Redistribute, py::arg("lev_min")=0, py::arg("lev_max")=-1,
                                            py::arg("nGrow")=0, py::arg("local")=0, py::arg("remove_negative")=true)
        .def("SortParticlesByCell", &ParticleContainerType::SortParticlesByCell)
        .def("SortParticlesByBin", &ParticleContainerType::SortParticlesByBin)
        .def("OK", &ParticleContainerType::OK, py::arg("lev_min") = 0, py::arg("lev_max") = -1, py::arg("nGrow")=0)
        .def("ByteSpread",&ParticleContainerType::ByteSpread)
        .def("PrintCapacity", &ParticleContainerType::PrintCapacity)
        .def("ShrinkToFit", &ParticleContainerType::ShrinkToFit)
        // Long NumberOfParticlesAtLevel (int level, bool only_valid = true, bool only_local = false) const;
        .def("NumberOfParticlesAtLevel", &ParticleContainerType::NumberOfParticlesAtLevel,
            py::arg("level"), py::arg("only_valid")=true, py::arg("only_local")=false)
        // Vector<Long> NumberOfParticlesInGrid  (int level, bool only_valid = true, bool only_local = false) const;
        .def("NumberOfParticlesInGrid", &ParticleContainerType::NumberOfParticlesInGrid,
            py::arg("level"), py::arg("only_valid")=true, py::arg("only_local")=false)
            // .def("DefineAndReturnParticleTile",
            //     py::overload_cast<int, int, int>
            //     (&ParticleContainerType::DefineAndReturnParticleTile))
                // Long TotalNumberOfParticles (bool only_valid=true, bool only_local=false) const;
        .def("TotalNumberOfParticles", &ParticleContainerType::TotalNumberOfParticles,
            py::arg("only_valid")=true, py::arg("only_local")=false)
        .def("RemoveParticlesAtLevel", &ParticleContainerType::RemoveParticlesAtLevel)
        .def("RemoveParticlesNotAtFinestLevel", &ParticleContainerType::RemoveParticlesNotAtFinestLevel)
        // void CreateVirtualParticles (int level, AoS& virts) const;
        //.def("CreateVirtualParticles", py::overload_cast<int, AoS&>(&ParticleContainerType::CreateVirtualParticles, py::const_),
        //    py::arg("level"), py::arg("virts"))
        //.def("CreateVirtualParticles", py::overload_cast<int, ParticleTileType&>(&ParticleContainerType::CreateVirtualParticles, py::const_),
        //     py::arg("level"), py::arg("virts")) // TODO pure SoA
        //.def("CreateGhostParticles", py::overload_cast<int, int, AoS&>(&ParticleContainerType::CreateGhostParticles, py::const_),
        //     py::arg("level"), py::arg("ngrow"), py::arg("ghosts"))
        //.def("CreateGhostParticles", py::overload_cast<int, int, ParticleTileType&>(&ParticleContainerType::CreateGhostParticles, py::const_),
        //     py::arg("level"), py::arg("ngrow"), py::arg("ghosts")) // TODO pure SoA
        //.def("AddParticlesAtLevel", py::overload_cast<AoS&, int, int>(&ParticleContainerType::AddParticlesAtLevel),
        //    py::arg("particles"), py::arg("level"), py::arg("ngrow")=0)
        .def("AddParticlesAtLevel", py::overload_cast<ParticleTileType&, int, int>(&ParticleContainerType::AddParticlesAtLevel),
            py::arg("particles"), py::arg("level"), py::arg("ngrow")=0)

        .def("clearParticles", &ParticleContainerType::clearParticles)
        // template <class PCType,
        //           std::enable_if_t<IsParticleContainer<PCType>::value, int> foo = 0>
        // void copyParticles (const PCType& other, bool local=false);
        // template <class PCType,
        //           std::enable_if_t<IsParticleContainer<PCType>::value, int> foo = 0>
        // void addParticles (const PCType& other, bool local=false);
        // template <class F, class PCType,
        //           std::enable_if_t<IsParticleContainer<PCType>::value, int> foo = 0,
        //           std::enable_if_t<! std::is_integral<F>::value, int> bar = 0>
        // void copyParticles (const PCType& other, F&&f, bool local=false);
        // template <class F, class PCType,
        //           std::enable_if_t<IsParticleContainer<PCType>::value, int> foo = 0,
        //           std::enable_if_t<! std::is_integral<F>::value, int> bar = 0>
        // void addParticles (const PCType& other, F&& f, bool local=false);
        // void WriteParticleRealData (void* data, size_t size, std::ostream& os) const;
        // // void ReadParticleRealData (void* data, size_t size, std::istream& is);
        // void Checkpoint (const std::string& dir, const std::string& name,
        //                  const Vector<std::string>& real_comp_names = Vector<std::string>(),
        //                  const Vector<std::string>& int_comp_names = Vector<std::string>()) const
        // void CheckpointPre ();
        // void CheckpointPost ();
        // void Restart (const std::string& dir, const std::string& file);
        // void Restart (const std::string& dir, const std::string& file, bool is_checkpoint);
        // void WritePlotFile (const std::string& dir, const std::string& name) const;
        // template <class F, typename std::enable_if<!std::is_same<F, Vector<std::string>&>::value>::type* = nullptr>
        // void WritePlotFile (const std::string& dir, const std::string& name, F&& f) const;
        // void WritePlotFile (const std::string& dir, const std::string& name,
        //                 const Vector<std::string>& real_comp_names,
        //                 const Vector<std::string>&  int_comp_names) const;
        // void WritePlotFile (const std::string& dir, const std::string& name,
        //                     const Vector<std::string>& real_comp_names) const;
        // void WritePlotFile (const std::string& dir,
        //                     const std::string& name,
        //                     const Vector<int>& write_real_comp,
        //                     const Vector<int>& write_int_comp) const;
        // void WritePlotFile (const std::string& dir,
        //                     const std::string& name,
        //                     const Vector<int>& write_real_comp,
        //                     const Vector<int>& write_int_comp,
        //                     const Vector<std::string>& real_comp_names,
        //                     const Vector<std::string>&  int_comp_names) const;
        // void WritePlotFilePre ();

        // void WritePlotFilePost ();
        .def("GetParticles", py::overload_cast<>(&ParticleContainerType::GetParticles), py::return_value_policy::reference_internal)
        .def("GetParticles", py::overload_cast<int>(&ParticleContainerType::GetParticles), py::return_value_policy::reference_internal)
        // .def("ParticlesAt", py::overload_cast<int,int,int>(&ParticleContainerType::ParticlesAt),
        //     py::return_value_policy::reference_internal)
        // .def("ParticlesAt", py::overload_cast<int,int,int>(&ParticleContainerType::ParticlesAt,py::const_))
        // .def("ParticlesAt", [](ParticleContainerType& pc, int lev, int grid, int tile) {
        //         return pc.ParticlesAt(lev, grid, tile);
        //     }, py::return_value_policy::reference_internal)
        // const ParticleTileType& ParticlesAt (int lev, int grid, int tile) const
        // { return m_particles[lev].at(std::make_pair(grid, tile)); }"Return the ParticleTile for level "lev", grid "grid" and tile "tile."
        //  *        Const version.
        //  *
        //  *        Here, grid and tile are integers that give the index and LocalTileIndex
        //  *        of the tile you want.
        //  *
        //  *        This is a runtime error if a ParticleTile at "grid" and "tile" has not been
        //  *        created yet.
        //  *
        //  *        The ParticleLevel must already exist, meaning that the "resizeData()"
        //  *        method of this ParticleContainer has been called."
        // ParticleTileType&       ParticlesAt (int lev, int grid, int tile)
        // { return m_particles[lev].at(std::make_pair(grid, tile)); }
        // template <class Iterator>
        // const ParticleTileType& ParticlesAt (int lev, const Iterator& iter) const
        //     { return ParticlesAt(lev, iter.index(), iter.LocalTileIndex()); }
        // template <class Iterator>
        // ParticleTileType&       ParticlesAt (int lev, const Iterator& iter)
        //     { return ParticlesAt(lev, iter.index(), iter.LocalTileIndex()); }
        // .def("DefineAndReturnParticleTile", py::overload_cast<int,int,int>(&ParticleContainerType::DefineAndReturnParticleTile))
        // .def("DefineAndReturnParticleTile", py::overload_cast<int,int,int>(&ParticleContainerType::DefineAndReturnParticleTile, py::const_))
        .def("DefineAndReturnParticleTile",
            [](ParticleContainerType& pc,
                int lev,
                int grid,
                int tile) {
                return pc.DefineAndReturnParticleTile(lev,grid,tile);
            })
        // ParticleTileType& DefineAndReturnParticleTile (int lev, int grid, int tile)
        // {
        //     m_particles[lev][std::make_pair(grid, tile)].define(NumRuntimeRealComps(), NumRuntimeIntComps());
        //     return ParticlesAt(lev, grid, tile);
        // }
        // template <class Iterator>
        // ParticleTileType& DefineAndReturnParticleTile (int lev, const Iterator& iter)
        // {
        //     auto index = std::make_pair(iter.index(), iter.LocalTileIndex());
        //     m_particles[lev][index].define(NumRuntimeRealComps(), NumRuntimeIntComps());
        //     return ParticlesAt(lev, iter);
        // }
    ;

    // TODO for pure SoA
    // depends on https://github.com/AMReX-Codes/amrex/pull/3280
    if constexpr (!T_ParticleType::is_soa_particle) {
        py_pc
            .def("InitRandom", py::overload_cast<Long, ULong, const ParticleInitData&, bool, RealBox>(&ParticleContainerType::InitRandom)) // TODO pure SoA
            .def("InitRandomPerBox", py::overload_cast<Long, ULong, const ParticleInitData&>(&ParticleContainerType::InitRandomPerBox)) // TODO pure SoA
            .def("InitOnePerCell", &ParticleContainerType::InitOnePerCell);
    }

    using iterator = amrex::ParIter_impl<ParticleType, T_NArrayReal, T_NArrayInt, Allocator>;
    make_Iterators< false, iterator, Allocator >(m, allocstr);
    using const_iterator = amrex::ParConstIter_impl<ParticleType, T_NArrayReal, T_NArrayInt, Allocator>;
    make_Iterators< true, const_iterator, Allocator >(m, allocstr);
}

/** Create ParticleContainers and Iterators
 */
template <typename T_ParticleType, int T_NArrayReal=0, int T_NArrayInt=0>
void make_ParticleContainer_and_Iterators (py::module &m)
{
    // TODO for pure SoA
    // depends on https://github.com/AMReX-Codes/amrex/pull/3280
    if constexpr (!T_ParticleType::is_soa_particle)
        make_ParticleInitData<T_ParticleType, T_NArrayReal, T_NArrayInt>(m);

    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/pybind11/pull/4581
    //make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
    //                                     std::allocator>(m, "std");            // CPU DefaultAllocator
    //make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
    //                                     amrex::ArenaAllocator>(m, "arena");   // GPU DefaultAllocator
#ifdef AMREX_USE_GPU
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
                                         std::allocator>(m, "std");
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
                                         amrex::DefaultAllocator>(m, "default");   // amrex::ArenaAllocator
#else
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
            amrex::DefaultAllocator>(m, "default");   // std::allocator
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
            amrex::ArenaAllocator>(m, "arena");
#endif
    //   end work-around
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
            amrex::PinnedArenaAllocator>(m, "pinned");
#ifdef AMREX_USE_GPU
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
                                         amrex::DeviceArenaAllocator>(m, "device");
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
                                         amrex::ManagedArenaAllocator>(m, "managed");
    make_ParticleContainer_and_Iterators<T_ParticleType, T_NArrayReal, T_NArrayInt,
                                         amrex::AsyncArenaAllocator>(m, "async");
#endif
}
