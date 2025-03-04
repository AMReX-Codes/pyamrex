foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        ParticleContainer.cpp
        ParticleContainer_FHDeX.cpp
        ParticleContainer_ImpactX.cpp
        ParticleContainer_WarpX.cpp
    )
endforeach()
