foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        ParticleContainer.cpp
    )

    # this is primarily to save compile-time
    foreach(PC IN LISTS pyAMReX_CODES)
        target_sources(pyAMReX_${D}d
          PRIVATE
            ParticleContainer_${PC}.cpp
        )
    endforeach()
endforeach()
