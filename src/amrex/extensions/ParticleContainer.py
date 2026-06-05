"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

import warnings

from .Iterator import getitem, next


def iterator(self, *args, level=None):
    """Create an iterator over all particle tiles

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A ParticleContainer class in pyAMReX
    args : deprecated positional argument
    level : int | str, optional
        The MR level. Allowed values are [0:self.finest_level+1) and "all".
        If there is more than one MR level, the argument is required.

    Returns
    -------
    Iterator over all particle tiles at the specified level.

    Examples
    --------
    >>> pc.iterator(level="all")
    >>> pc.iterator(level=0)  # only particles on the the coarsest MR level
    """
    # Warn if a second positional argument is provided (ignored argument)
    if len(args) > 0:
        if len(args) == 1 and isinstance(args[0], int) and level is None:
            level = args[0]
        else:
            warnings.warn(
                "The second positional argument to iterator() is deprecated and ignored. "
                "Please update your code to use iterator(self, level=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    has_mr = self.finest_level > 0

    if level is None:
        if has_mr:
            raise ValueError(
                "level must be specified for multi-level ParticleContainers"
            )
        else:
            level = 0

    if level == "all":
        raise ValueError("level='all' is not yet supported for ParticleContainers")
        # TODO: This does not work
        # for lvl in range(self.finest_level + 1):
        #     yield self.Iterator(self, level=lvl)
    elif isinstance(level, int) and level >= 0:
        return self.Iterator(self, level=level)
    else:
        raise ValueError(
            f"level must be an integer in [0:{self.finest_level + 1}) or 'all', but got: {level}"
        )


def pc_to_df(self, local=True, comm=None, root_rank=0):
    """
    Copy all particles into a pandas.DataFrame

    Parameters
    ----------
    self : amrex.ParticleContainer_*
        A ParticleContainer class in pyAMReX
    local : bool
        MPI rank-local particles only
    comm : MPI Communicator
        if local is False, this defaults to mpi4py.MPI.COMM_WORLD
    root_rank : MPI root rank to gather to
        if local is False, this defaults to 0

    Returns
    -------
    A concatenated pandas.DataFrame with particles from all levels.

    Returns None if no particles were found.
    If local=False, then all ranks but the root_rank will return None.
    """
    import pandas as pd

    # silently ignore local=False for non-MPI runs
    if local is False:
        from inspect import getmodule

        amr = getmodule(self)
        if not amr.Config.have_mpi:
            local = True

    # create a DataFrame per particle box and append it to the list of
    # local DataFrame(s)
    dfs_local = []
    for lvl in range(self.finest_level + 1):
        for pti in self.const_iterator(level=lvl):
            if pti.size == 0:
                continue

            if self.is_soa_particle:
                soa_view = pti.soa().to_numpy(copy=True)

                next_df = pd.DataFrame()

                next_df["idcpu"] = soa_view.idcpu

                soa_np_real = soa_view.real
                for name, array in soa_np_real.items():
                    next_df[name] = array

                soa_np_int = soa_view.int
                for name, array in soa_np_int.items():
                    next_df[name] = array
            else:
                # AoS
                aos_np = pti.aos().to_numpy(copy=True)
                next_df = pd.DataFrame(aos_np)

                # SoA
                soa_view = pti.soa().to_numpy(copy=True)
                soa_np_real = soa_view.real
                soa_np_int = soa_view.int

                for name, array in soa_np_real.items():
                    next_df[f"SoA_{name}"] = array

                for name, array in soa_np_int.items():
                    next_df[f"SoA_{name}"] = array

            next_df.set_index("idcpu")
            next_df.index.name = "idcpu"

            dfs_local.append(next_df)

    # MPI Gather to root rank if requested
    if local:
        if len(dfs_local) == 0:
            df = None
        else:
            df = pd.concat(dfs_local)
    else:
        from mpi4py import MPI

        if comm is None:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # a list for each rank's list of DataFrame(s)
        df_list_list = comm.gather(dfs_local, root=root_rank)

        if rank == root_rank:
            flattened_list = [df for sublist in df_list_list for df in sublist]

            if len(flattened_list) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(flattened_list, ignore_index=True)
        else:
            df = None

    return df


def read_particles(
    amr, plotfile, particle_dir="particles", communicate=True, container=None
):
    """Read AMReX particle data from a plotfile or checkpoint.

    The on-disk layout (number and names of the real/int components, the
    precision and whether the file is a checkpoint) is discovered via
    :py:class:`amrex.ParticleHeader`, which uses the same AMReX C++ header reader
    as ``ParticleContainer::Restart``. A polymorphic, pure Struct-of-Arrays
    ParticleContainer is then configured with matching runtime components and
    restarted from the file - so no prior knowledge of the container's
    compile-time layout is required.

    Parameters
    ----------
    amr : module
        The dimension-specific pyAMReX module (e.g. ``amrex.space3d``).
    plotfile : str
        Path to the plotfile / checkpoint directory.
    particle_dir : str, optional
        Name of the particle sub-directory inside ``plotfile``
        (default: ``"particles"``).
    communicate : bool, optional
        Whether the added runtime components participate in redistribution
        (default: ``True``).
    container : ParticleContainer, optional
        An existing, geometry-defined container to restart into. If ``None``
        (default), the geometry is recovered from the plotfile's
        :py:class:`amrex.PlotFileData` and a fresh polymorphic pure-SoA
        container is created.

    Returns
    -------
    ParticleContainer
        The populated particle container. Iterate the data via, e.g.,
        ``for pti in pc.iterator(level=0): soa = pti.soa()``.
    """
    header = amr.ParticleHeader.read(plotfile, particle_dir)

    pc = container
    if pc is None:
        # recover the AMR geometry from the plotfile metadata
        plt = amr.PlotFileData(plotfile)
        finest = plt.finestLevel()
        prob_domain = plt.probDomain(0)
        domain_box = amr.Box(prob_domain.small_end, prob_domain.big_end)
        real_box = amr.RealBox(plt.probLo(), plt.probHi())
        geom = amr.Geometry(
            domain_box, real_box, plt.coordSys(), [0] * amr.Config.spacedim
        )

        pc_type_name = f"ParticleContainer_pureSoA_{amr.Config.spacedim}_0_polymorphic"
        try:
            pc_type = getattr(amr, pc_type_name)
        except AttributeError as e:
            raise AttributeError(
                f"pyAMReX was built without the pure-SoA container '{pc_type_name}'."
            ) from e
        pc = pc_type(geom, plt.DistributionMap(finest), plt.boxArray(finest))
        # the polymorphic allocator needs an arena before any tile allocation
        pc.arena = amr.The_Arena()

    # configure the runtime SoA components to match the on-disk layout. For pure
    # SoA particles the AMREX_SPACEDIM positions are compile-time components and
    # header.{real,int}_comp_names list exactly the runtime components to add.
    for name in header.real_comp_names:
        if not pc.has_real_comp(name):
            pc.add_real_comp(name, communicate)
    for name in header.int_comp_names:
        if not pc.has_int_comp(name):
            pc.add_int_comp(name, communicate)

    pc.restart_checkpoint(plotfile, particle_dir, header.is_checkpoint)
    return pc


def register_ParticleContainer_extension(amr):
    """ParticleContainer helper methods"""
    import inspect
    import sys

    # register member functions for every Par(Const)Iter* type
    for _, ParIter_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and (
                member.__name__.startswith("ParIter")
                or member.__name__.startswith("ParConstIter")
            )
        ),
    ):
        ParIter_type.__next__ = next
        ParIter_type.__iter__ = lambda self: self
        ParIter_type.__getitem__ = getitem

    # register member functions for every ParticleContainer_* type
    for _, ParticleContainer_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("ParticleContainer_")
        ),
    ):
        ParticleContainer_type.iterator = iterator
        ParticleContainer_type.const_iterator = (
            iterator  # TODO: simplified, code duplication
        )
        ParticleContainer_type.to_df = pc_to_df
