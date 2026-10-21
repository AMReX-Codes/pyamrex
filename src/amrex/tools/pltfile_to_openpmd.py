# -*- coding: utf-8 -*-
"""
Convert AMReX plotfiles to openPMD series.

This tool reads native AMReX plotfiles (mesh fields and particles) and writes
them as an openPMD series (HDF5, ADIOS2 or JSON, chosen by the output file
extension). It aims to be information-preserving:

* every field value is copied at its on-disk precision, per AMR level,
* mesh refinement levels are encoded following the openPMD
  ``PatchBasedMeshRefinement`` extension proposal
  (https://github.com/openPMD/openPMD-standard/pull/252): the coarsest level
  keeps the plain record name, finer levels are suffixed ``_lvl<N>`` and carry
  a ``refinementRatio`` attribute; every AMReX grid is stored as one chunk,
* particles are converted per species with their component names, ids and
  cpus,
* AMReX metadata that has no openPMD equivalent (level steps, box arrays,
  ghost cell widths, ...) is stored in ``amrex_``-prefixed attributes.

Not preserved: the on-disk *ordering* of particles (their identity is kept via
``id``/``cpu``) and their mesh-refinement level assignment, which is
recoverable from the positions and the stored per-level box arrays - the same
rule AMReX itself applies in ``Redistribute()``. Ghost cell *values* are not
written (the valid region is); the ghost width is recorded.

Usage (CLI)::

    pltfile-to-openpmd -o sim_%T.h5 plt00000 plt00100 ...
    python -m amrex.tools.pltfile_to_openpmd -o sim_%T.bp plt?????

Usage (module)::

    from amrex.tools.pltfile_to_openpmd import convert

    convert(["plt00000", "plt00100"], "sim_%T.h5")

Requires the ``openpmd_api`` Python package (``pip install openpmd-api``).
"""

import argparse
import glob
import os
import re

import numpy as np


def _peek_spacedim(plotfile):
    """Read the spatial dimension from a plotfile ``Header`` without pyAMReX.

    The header starts with: version string, number of components, that many
    component names, then the spatial dimension.
    """
    header = os.path.join(plotfile, "Header")
    if not os.path.isfile(header):
        raise FileNotFoundError(
            f"pltfile-to-openpmd: no plotfile Header found at '{header}'. "
            f"Is '{plotfile}' an AMReX plotfile directory?"
        )
    with open(header) as f:
        f.readline()  # version
        ncomp = int(f.readline())
        for _ in range(ncomp):
            f.readline()  # component names
        return int(f.readline())


def _import_amr(spacedim):
    """Import the pyAMReX module matching the plotfile's dimensionality."""
    import importlib

    try:
        return importlib.import_module(f"amrex.space{spacedim}d")
    except ImportError as e:
        raise ImportError(
            f"pltfile-to-openpmd: the plotfile is {spacedim}D, but pyAMReX was "
            f"built without amrex.space{spacedim}d."
        ) from e


def _fab_dtype(plotfile, lev):
    """Detect the on-disk floating point precision of a level's field data.

    Every FAB in the level's binary data files starts with an ASCII header
    ``FAB ((<bytes-per-real>, ...`` - 8 for double, 4 for single precision.
    Returns None if the layout is non-standard and detection fails.
    """
    for vismf_header in sorted(
        glob.glob(os.path.join(plotfile, f"Level_{lev}", "*_H"))
    ):
        with open(vismf_header) as f:
            m = re.search(r"^FabOnDisk: (\S+) (\d+)", f.read(), re.MULTILINE)
        if m is None:
            continue
        fab_file = os.path.join(os.path.dirname(vismf_header), m.group(1))
        try:
            with open(fab_file, "rb") as f:
                f.seek(int(m.group(2)))
                fab = f.read(16)
        except OSError:
            continue
        m = re.match(rb"FAB \(\((\d+),", fab)
        if m is not None:
            return {4: np.float32, 8: np.float64}.get(int(m.group(1)))
    return None


def _reversed_list(seq):
    """AMReX orders axes Fortran-style (x fastest); openPMD datasets are
    written in C order, so all per-axis vectors are reversed."""
    return list(seq)[::-1]


def _axis_labels(spacedim, coord_sys):
    """Axis labels in AMReX (Fortran) order, before reversal."""
    if int(coord_sys) == 1 and spacedim == 2:  # RZ / cylindrical
        return ["r", "z"]
    return ["x", "y", "z"][:spacedim]


def _mesh_name(varname, lev):
    return varname if lev == 0 else f"{varname}_lvl{lev}"


def _convert_mesh(amr, io, plt, iteration, plotfile, fields=None, verbose=False):
    """Write all (selected) field components of all levels of one plotfile."""
    spacedim = plt.spaceDim()
    varnames = [str(v) for v in plt.varNames()]
    if fields is not None:
        unknown = set(fields) - set(varnames)
        if unknown:
            raise ValueError(
                f"pltfile-to-openpmd: unknown field(s) {sorted(unknown)}; "
                f"the plotfile contains {varnames}"
            )
        varnames = [v for v in varnames if v in fields]

    axis_labels_c = _reversed_list(_axis_labels(spacedim, plt.coordSys()))
    geometry = {
        0: io.Geometry.cartesian,
        1: io.Geometry.cylindrical,
        2: io.Geometry.spherical,
    }.get(int(plt.coordSys()), io.Geometry.other)

    for lev in range(plt.finestLevel() + 1):
        domain = plt.probDomain(lev)
        extent_c = _reversed_list(
            [int(b) - int(s) + 1 for s, b in zip(domain.small_end, domain.big_end)]
        )
        dtype = _fab_dtype(plotfile, lev)
        n_grow = plt.nGrowVect(lev)
        if verbose and max(n_grow) > 0:
            print(
                f"  note: level {lev} stores {list(n_grow)} ghost cells; "
                "their values are not written (the valid region is)"
            )

        for varname in varnames:
            mf = plt.get(lev, varname)
            mesh = iteration.meshes[_mesh_name(varname, lev)]
            mrc = mesh[io.Mesh_Record_Component.SCALAR]

            mesh.geometry = geometry
            mesh.axis_labels = axis_labels_c
            mesh.data_order = "C"
            mesh.grid_spacing = _reversed_list([float(x) for x in plt.cellSize(lev)])
            mesh.grid_global_offset = _reversed_list([float(x) for x in plt.probLo()])
            if lev > 0:
                # refinement ratio towards the previous (coarser) level,
                # ordered like axis_labels (openPMD-standard PR #252)
                ratio = amr.IntVect(plt.refRatio(lev - 1))
                mesh.set_attribute(
                    "refinementRatio", _reversed_list([int(r) for r in ratio])
                )

            # in-cell position: 0.5 for cell centers, 0.0 on nodes
            ix_type = domain.ix_type
            mrc.position = _reversed_list(
                [0.0 if ix_type.node_centered(d) else 0.5 for d in range(spacedim)]
            )

            data_dtype = (
                dtype if dtype is not None else mf.array(next(iter(mf))).to_xp().dtype
            )
            mrc.reset_dataset(io.Dataset(np.dtype(data_dtype), extent_c))

            for mfi in mf:
                arr = mf.array(mfi).to_xp()  # (nx, ny, nz[, ...], ncomp) w/ ghosts
                box = mfi.validbox()
                lo = [int(s) for s in box.small_end]
                hi = [int(b) for b in box.big_end]
                # strip ghost cells: array indices start at the grown lower end
                sl = tuple(
                    slice(int(n_grow[d]), int(n_grow[d]) + (hi[d] - lo[d] + 1))
                    for d in range(spacedim)
                )
                valid = arr[sl + (0,)]
                # to C order: reverse the axes, match the dataset precision
                chunk = np.ascontiguousarray(valid.transpose()).astype(
                    data_dtype, copy=False
                )
                offset_c = _reversed_list(
                    [lo[d] - int(domain.small_end[d]) for d in range(spacedim)]
                )
                mrc.store_chunk(chunk, offset_c, list(chunk.shape))


def _soa_to_numpy(podvector):
    """Copy a PODVector view to a host numpy array."""
    return np.array(podvector.to_numpy(copy=True))


def _convert_particles(amr, io, iteration, plotfile, species=None, verbose=False):
    """Write all (selected) particle species of one plotfile."""
    found = amr.list_particle_species(plotfile)
    if species is not None:
        unknown = set(species) - set(found)
        if unknown:
            raise ValueError(
                f"pltfile-to-openpmd: unknown species {sorted(unknown)}; "
                f"the plotfile contains {found}"
            )
        found = [s for s in found if s in species]

    spacedim = amr.Config.spacedim
    axes = _axis_labels(spacedim, 0)

    for name in found:
        if verbose:
            print(f"  particles: {name}")
        header = amr.ParticleHeader.read(plotfile, name)
        pc = amr.read_particles(plotfile, name)
        np_total = header.num_particles

        sp = iteration.particles[name]

        # gather rank-local tiles (this tool is serial); positions are the
        # first AMREX_SPACEDIM SoA real components in a pure-SoA container
        idcpu_parts, real_parts, int_parts = [], [], []
        n_real = pc.num_real_comps
        n_int = pc.num_int_comps
        for lvl in range(pc.finest_level + 1):
            for pti in pc.iterator(level=lvl):
                soa = pti.soa()
                idcpu_parts.append(_soa_to_numpy(soa.get_idcpu_data()))
                real_parts.append(
                    [_soa_to_numpy(soa.get_real_data(j)) for j in range(n_real)]
                )
                int_parts.append(
                    [_soa_to_numpy(soa.get_int_data(j)) for j in range(n_int)]
                )

        def concat(parts, j=None):
            arrs = [p if j is None else p[j] for p in parts]
            return (
                np.concatenate(arrs)
                if arrs
                else np.array([], dtype=np.uint64 if j is None else np.float64)
            )

        idcpu = concat(idcpu_parts)
        assert idcpu.size == np_total, (
            f"read {idcpu.size} particles, header announces {np_total}"
        )

        def store(record_component, data):
            record_component.reset_dataset(io.Dataset(data.dtype, [np_total]))
            record_component.store_chunk(np.ascontiguousarray(data), [0], [data.size])

        # position + constant positionOffset (openPMD base records)
        for d, ax in enumerate(axes):
            store(sp["position"][ax], concat(real_parts, d))
            poff = sp["positionOffset"][ax]
            poff.reset_dataset(io.Dataset(np.dtype(np.float64), [np_total]))
            poff.make_constant(0.0)

        # identity: unpacked AMReX id and cpu
        store(sp["id"][io.Record_Component.SCALAR], amr.unpack_ids(idcpu))
        store(sp["amrex_cpu"][io.Record_Component.SCALAR], amr.unpack_cpus(idcpu))

        # named runtime components, verbatim as scalar records
        for j, comp in enumerate(header.real_comp_names):
            store(
                sp[str(comp)][io.Record_Component.SCALAR],
                concat(real_parts, spacedim + j),
            )
        for j, comp in enumerate(header.int_comp_names):
            store(sp[str(comp)][io.Record_Component.SCALAR], concat(int_parts, j))

        # AMReX metadata: file layout details with no openPMD equivalent
        sp.set_attribute("amrex_version", header.version)
        sp.set_attribute("amrex_is_checkpoint", int(header.is_checkpoint))
        sp.set_attribute("amrex_next_id", int(header.next_id))
        sp.set_attribute(
            "amrex_num_particles_per_level",
            [int(sum(e.count for e in entries)) for entries in header.grids],
        )


def convert(
    plotfiles,
    output,
    fields=None,
    species=None,
    no_particles=False,
    dt=0.0,
    author=None,
    verbose=True,
):
    """Convert AMReX plotfiles into one openPMD series.

    Parameters
    ----------
    plotfiles : list of str
        AMReX plotfile directories; each becomes one iteration, indexed by its
        level-0 step number.
    output : str
        openPMD series path; the extension selects the backend (``.h5``,
        ``.bp``, ``.json``) and a ``%T`` placeholder selects file-based
        iteration encoding.
    fields : list of str, optional
        Only convert these field components (default: all).
    species : list of str, optional
        Only convert these particle species (default: all).
    no_particles : bool, optional
        Skip particle data entirely.
    dt : float, optional
        Time step to record per iteration; plotfiles do not store one.
    author : str, optional
        openPMD author attribute, e.g. ``"Jane Doe <jane@example.com>"``.
    verbose : bool, optional
        Print progress.
    """
    try:
        import openpmd_api as io
    except ImportError as e:
        raise ImportError(
            "pltfile-to-openpmd requires the openpmd_api package: "
            "https://openpmd-api.readthedocs.io - e.g. 'pip install openpmd-api'"
        ) from e

    if not plotfiles:
        raise ValueError("pltfile-to-openpmd: no input plotfiles given")

    spacedim = _peek_spacedim(plotfiles[0])
    for p in plotfiles[1:]:
        if _peek_spacedim(p) != spacedim:
            raise ValueError(
                f"pltfile-to-openpmd: '{p}' is not {spacedim}D like "
                f"'{plotfiles[0]}'; convert equal-dimension files together"
            )
    amr = _import_amr(spacedim)

    initialized_here = False
    if not amr.initialized():
        amr.initialize([])
        initialized_here = True

    try:
        # open each plotfile, order iterations by ascending step
        plts = {}
        for p in plotfiles:
            plt = amr.PlotFileData(p.rstrip("/"))
            step = plt.levelStep(0)
            if step in plts:
                raise ValueError(
                    f"pltfile-to-openpmd: '{p}' and '{plts[step][0]}' both have "
                    f"step {step}; cannot write both into one series"
                )
            plts[step] = (p, plt)

        series = io.Series(output, io.Access.create)
        series.set_software("pyAMReX", amr.__version__)
        if author:
            series.author = author

        for step in sorted(plts):
            p, plt = plts[step]
            if verbose:
                print(f"converting {p} -> iteration {step}")
            it = series.write_iterations()[step]
            it.time = float(plt.time())
            it.dt = float(dt)
            it.time_unit_SI = 1.0

            it.set_attribute("amrex_plotfile_version", "unknown")
            it.set_attribute("amrex_finest_level", plt.finestLevel())
            it.set_attribute(
                "amrex_level_steps",
                [int(plt.levelStep(lev)) for lev in range(plt.finestLevel() + 1)],
            )
            it.set_attribute("amrex_coord_sys", int(plt.coordSys()))
            it.set_attribute(
                "amrex_n_grow",
                [
                    int(g)
                    for lev in range(plt.finestLevel() + 1)
                    for g in plt.nGrowVect(lev)
                ],
            )
            for lev in range(plt.finestLevel() + 1):
                ba = plt.boxArray(lev)
                flat = []
                for i in range(ba.size):
                    b = ba[i]
                    flat += [int(x) for x in b.small_end] + [int(x) for x in b.big_end]
                it.set_attribute(f"amrex_box_array_lvl{lev}", flat)

            _convert_mesh(amr, io, plt, it, p, fields=fields, verbose=verbose)
            if not no_particles:
                _convert_particles(amr, io, it, p, species=species, verbose=verbose)

            it.close()

        series.close()
    finally:
        if initialized_here:
            amr.finalize()

    if verbose:
        print(f"wrote {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pltfile-to-openpmd",
        description="Convert AMReX plotfiles to an openPMD series "
        "(HDF5/ADIOS2/JSON by output extension).",
    )
    parser.add_argument("plotfiles", nargs="+", help="AMReX plotfile directories")
    parser.add_argument(
        "-o",
        "--output",
        default="openpmd_%T.h5",
        help="output series (default: %(default)s); '%%T' selects file-based "
        "iteration encoding",
    )
    parser.add_argument("--fields", nargs="+", help="only convert these fields")
    parser.add_argument("--species", nargs="+", help="only convert these species")
    parser.add_argument(
        "--no-particles", action="store_true", help="skip particle data"
    )
    parser.add_argument(
        "--dt", type=float, default=0.0, help="time step to record (not in plotfiles)"
    )
    parser.add_argument("--author", help="openPMD author attribute")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)

    convert(
        args.plotfiles,
        args.output,
        fields=args.fields,
        species=args.species,
        no_particles=args.no_particles,
        dt=args.dt,
        author=args.author,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
