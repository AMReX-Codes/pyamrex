/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>
#include <AMReX_ParticleContainer.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ArrayOfStructs.H>

#include <string>
#include <sstream>

namespace py = pybind11;
using namespace amrex;

template <int T_NStructReal, int T_NStructInt=0, int T_NArrayReal=0, int T_NArrayInt=0,
          template<class> class Allocator=DefaultAllocator>
void make_ParticleContainer(py::module &m)
{
    using ParticleContainerType = ParticleContainer<
        T_NStructReal, T_NStructInt, T_NArrayReal, T_NArrayInt,
        Allocator
    >;

    using ParticleTileType = typename ParticleContainerType::ParticleTileType;
    using AoS = typename ParticleTileType::AoS;
    using ParticleInitData = typename ParticleContainerType::ParticleInitData;

    auto const particle_init_data_type = std::string("ParticleInitType_").append(std::to_string(T_NStructReal) + "_" + std::to_string(T_NStructInt) + "_" + std::to_string(T_NArrayReal) + "_" + std::to_string(T_NArrayInt));
    py::class_<ParticleInitData>(m, particle_init_data_type.c_str())
        .def(py::init<>())
        .def_readwrite("real_struct_data", &ParticleInitData::real_struct_data)
        .def_readwrite("int_struct_data", &ParticleInitData::int_struct_data)
        .def_readwrite("real_array_data", &ParticleInitData::real_array_data)
        .def_readwrite("int_array_data", &ParticleInitData::int_array_data)
    ;

    auto const particle_container_type = std::string("ParticleContainer_").append(std::to_string(T_NStructReal) + "_" + std::to_string(T_NStructInt) + "_" + std::to_string(T_NArrayReal) + "_" + std::to_string(T_NArrayInt));
    py::class_<ParticleContainerType>(m, particle_container_type.c_str())
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

        .def_property_readonly_static("NStructReal", [](const py::object&){return ParticleContainerType::NStructReal; })
        .def_property_readonly_static("NStructInt", [](const py::object&){return ParticleContainerType::NStructInt; })
        .def_property_readonly_static("NArrayReal", [](const py::object&){return ParticleContainerType::NArrayReal; })
        .def_property_readonly_static("NArrayInt", [](const py::object&){return ParticleContainerType::NArrayInt; })


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

        .def("InitRandom", py::overload_cast<Long, ULong, const ParticleInitData&, bool, RealBox>(&ParticleContainerType::InitRandom))

        .def("InitRandomPerBox", py::overload_cast<Long, ULong, const ParticleInitData&>(&ParticleContainerType::InitRandomPerBox))
        .def("InitOnePerCell", &ParticleContainerType::InitOnePerCell)
        
        .def("Increment", &ParticleContainerType::Increment)
        .def("IncrementWithTotal", &ParticleContainerType::IncrementWithTotal, py::arg("mf"), py::arg("level"), py::arg("local")=false)
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
            py::arg("onnly_valid")=true, py::arg("only_local")=false)
        .def("RemoveParticlesAtLevel", &ParticleContainerType::RemoveParticlesAtLevel)
        .def("RemoveParticlesNotAtFinestLevel", &ParticleContainerType::RemoveParticlesNotAtFinestLevel)
        // void CreateVirtualParticles (int level, AoS& virts) const;
        // .def("CreateVirtualParticles", py::overload_cast<int, AoS&>(&ParticleContainerType::CreateVirtualParticles), py::const_)
        .def("CreateVirtualParticles", [](ParticleContainerType& pc, int level, AoS& virts){ return pc.CreateVirtualParticles(level, virts);})
        // void CreateGhostParticles (int level, int ngrow, AoS& ghosts) const;
        // .def("CreateGhostParticles", &ParticleContainerType::CreateGhostParticles)
        .def("CreateGhostParticles", [](ParticleContainerType& pc, int level, int ngrow, AoS& ghosts) { 
            return pc.CreateGhostParticles(level, ngrow, ghosts); })
        .def("AddParticlesAtLevel", [](ParticleContainerType& pc, AoS& particles, int level, int nGrow=0) {
            pc.AddParticlesAtLevel(particles, level, nGrow);
        })

        // void CreateGhostParticles (int level, int ngrow, ParticleTileType& ghosts) const;
        // void AddParticlesAtLevel (ParticleTileType& particles, int level, int nGrow=0);
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
}


void init_ParticleContainer(py::module& m) {
    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleContainer< 1, 1, 2, 1> (m);
    make_ParticleContainer<0,0,7,0> (m);
}
