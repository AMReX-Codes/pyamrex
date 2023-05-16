foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        Particle.cpp
        StructOfArrays.cpp
        ArrayOfStructs.cpp
        ParticleTile.cpp
        ParticleContainer.cpp
        ParticleContainer_HiPACE.cpp
        ParticleContainer_ImpactX.cpp
        ParticleContainer_WarpX.cpp
    )
endforeach()
