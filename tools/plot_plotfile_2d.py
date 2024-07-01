import matplotlib.pyplot as plt
import numpy as np
import pytest

import amrex.space2d as amr


def plot_mf(arr, compname, plo, phi):
    plt.plot()
    im = plt.imshow(
        arr.T,
        origin="lower",
        interpolation="none",
        extent=[plo[0], phi[0], plo[1], phi[1]],
        aspect="equal",
    )
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{compname}.png", dpi=150)
    plt.close()


@pytest.mark.skipif(amr.Config.spacedim != 2, reason="Requires AMREX_SPACEDIM = 2")
def plot_plotfile_2d(filename, level=0):
    plt = amr.PlotFileData(filename)
    assert level <= plt.finestLevel()

    probDomain = plt.probDomain(level)
    probLo = plt.probLo()
    probHi = plt.probHi()
    varNames = plt.varNames()

    for compname in varNames:
        mfab_comp = plt.get(level, compname)
        arr = np.zeros((probDomain.big_end - probDomain.small_end + 1))
        for mfi in mfab_comp:
            bx = mfi.tilebox()
            marr = mfab_comp.array(mfi)
            marr_xp = marr.to_xp()
            i_s, j_s = tuple(bx.small_end)
            i_e, j_e = tuple(bx.big_end)
            arr[i_s : i_e + 1, j_s : j_e + 1] = marr_xp[:, :, 0, 0]
        plot_mf(arr, compname, probLo, probHi)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Plots each variable in a 2D plotfile using matplotlib."
    )
    parser.add_argument("filename", help="AMReX 2D plotfile to read")
    parser.add_argument("level", type=int, help="AMR level to plot (default: 0)")
    args = parser.parse_args()

    amr.initialize([])
    plot_plotfile_2d(args.filename, level=args.level)
