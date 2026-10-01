/* Copyright 2022 The AMReX Community
 *
 * Authors: Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_EBCellFlag.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>

#include <stdexcept>


void init_EBFabFactory (py::module& m)
{
    using namespace amrex;

    py::class_<EBFArrayBoxFactory, FabFactory<FArrayBox>>(m, "EBFArrayBoxFactory")
        .def("getVolFrac", &EBFArrayBoxFactory::getVolFrac,
            py::return_value_policy::reference_internal,
            "Return volume faction MultiFab");

    py::native_enum<EBSupport>(m, "EBSupport", "enum.Enum")
        .value("basic", EBSupport::basic)
        .value("volume", EBSupport::volume)
        .value("full", EBSupport::full)
        .export_values()
        .finalize()
    ;

    m.def(
        "makeEBFabFactory",
        [] (Geometry const& geom, BoxArray const& ba, DistributionMapping const& dm,
            Vector<int> const& ngrow, EBSupport support)
        {
            return makeEBFabFactory(geom, ba, dm, ngrow, support);
        },
        py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("ngrow"),
        py::arg("support"),
        "Make EBFArrayBoxFactory for given Geometry, BoxArray and DistributionMapping"
    );

    m.def(
        "makeStaircaseEBFabFactoryFromCupy",
        [] (Geometry const& geom, BoxArray const& ba, DistributionMapping const& dm,
            py::object mask_obj, Vector<int> const& ngrow, EBSupport support)
        {
            if (mask_obj.is_none()) {
                throw py::type_error(
                    "makeStaircaseEBFabFactoryFromCupy() requires a CuPy/NumPy mask object"
                );
            }

            py::object mask_host_obj;
            if (py::hasattr(mask_obj, "__cuda_array_interface__")) {
                mask_host_obj = py::module_::import("cupy").attr("asnumpy")(mask_obj);
            } else {
                mask_host_obj = py::module_::import("numpy").attr("asarray")(mask_obj);
            }

            py::array_t<unsigned char, py::array::c_style | py::array::forcecast> mask_host(
                mask_host_obj
            );

            if (mask_host.ndim() != 2) {
                throw py::value_error(
                    "makeStaircaseEBFabFactoryFromCupy() expects a 2D nodal mask"
                );
            }

            const int nr_nodal = geom.Domain().length(0) + 1;
            const int nz_nodal = geom.Domain().length(1) + 1;

            const int in_nr = static_cast<int>(mask_host.shape(0));
            const int in_nz = static_cast<int>(mask_host.shape(1));
            if (in_nr < nr_nodal || in_nz < nz_nodal) {
                throw py::value_error(
                    "makeStaircaseEBFabFactoryFromCupy() mask is smaller than domain nodal shape"
                );
            }

            // Accept masks with symmetric ghost padding and crop to the valid
            // nodal domain centered window.
            const int off_r = (in_nr - nr_nodal) / 2;
            const int off_z = (in_nz - nz_nodal) / 2;

            Vector<int> hmask(static_cast<std::size_t>(nr_nodal) * static_cast<std::size_t>(nz_nodal), 0);
            auto const mh = mask_host.unchecked<2>();
            for (int i = 0; i < nr_nodal; ++i) {
                for (int j = 0; j < nz_nodal; ++j) {
                    hmask[static_cast<std::size_t>(i) * static_cast<std::size_t>(nz_nodal)
                        + static_cast<std::size_t>(j)] = (mh(off_r + i, off_z + j) != 0) ? 1 : 0;
                }
            }

            auto eb_factory = makeEBFabFactory(geom, ba, dm, ngrow, support);

            auto& flags = const_cast<FabArray<EBCellFlagFab>&>(eb_factory->getMultiEBCellFlagFab());
            auto& vfrac = const_cast<MultiFab&>(eb_factory->getVolFrac());
            auto& levset = const_cast<MultiFab&>(eb_factory->getLevelSet());

            const IndexType levset_type = levset.boxArray().ixType();
            const bool levset_is_nodal = levset_type.nodeCentered(0)
#if (AMREX_SPACEDIM >= 2)
                && levset_type.nodeCentered(1)
#endif
                ;
            if (!levset_is_nodal) {
                throw std::runtime_error(
                    "makeStaircaseEBFabFactoryFromCupy() expected nodal EB levelset"
                );
            }

            Gpu::DeviceVector<int> dmask(hmask.size());
            Gpu::copy(Gpu::hostToDevice, hmask.begin(), hmask.end(), dmask.begin());
            int const* nodal_mask = dmask.data();

            const int nr_cc = geom.Domain().length(0);
            const int nz_cc = geom.Domain().length(1);

            for (MFIter mfi(vfrac, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.growntilebox(vfrac.nGrowVect());
                auto const& flag_arr = flags.array(mfi);
                auto const& vfrac_arr = vfrac.array(mfi);

                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (i >= 0 && i < nr_cc && j >= 0 && j < nz_cc) {
                        const bool n00 = (nodal_mask[i * nz_nodal + j] != 0);
                        const bool n10 = (nodal_mask[(i + 1) * nz_nodal + j] != 0);
                        const bool n01 = (nodal_mask[i * nz_nodal + (j + 1)] != 0);
                        const bool n11 = (nodal_mask[(i + 1) * nz_nodal + (j + 1)] != 0);
                        const bool cell_covered = n00 && n10 && n01 && n11;

                        auto cell = flag_arr(i, j, k);
                        if (cell_covered) {
                            cell.setCovered();
                            vfrac_arr(i, j, k) = Real(0.0);
                        } else {
                            cell.setRegular();
                            vfrac_arr(i, j, k) = Real(1.0);
                        }
                        flag_arr(i, j, k) = cell;
                    }
                });
            }

            for (MFIter mfi(levset, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& levset_arr = levset.array(mfi);

                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (i >= 0 && i < nr_nodal && j >= 0 && j < nz_nodal) {
                        const int idx = i * nz_nodal + j;
                        levset_arr(i, j, k) = (nodal_mask[idx] != 0) ? Real(1.0) : Real(-1.0);
                    }
                });
            }

            Gpu::streamSynchronize();

            const Periodicity period = eb_factory->Geom().periodicity();
            vfrac.FillBoundary(period);
            levset.FillBoundary(period);

            return eb_factory;
        },
        py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("mask"),
        py::arg("ngrow"), py::arg("support") = EBSupport::full,
        "Make staircase EBFArrayBoxFactory from a CuPy/NumPy nodal boolean mask "
        "(True/1 means embedded boundary / covered)."
    );

}
