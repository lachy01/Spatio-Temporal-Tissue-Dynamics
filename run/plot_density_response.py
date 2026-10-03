"""Per-cell-type density response of the mixing weight, from the production fit.

    python run/plot_density_response.py

Small multiples, one panel per cell type on a shared axis: each panel is that
cell type's own-state centroid pushed through the fitted `h()` and `K`, as a
change in log-odds relative to the median density. The neighbourhood state is
held at its mean, which shifts every curve by a constant and so does not change
its shape (only `h` and `K` involve density).

The shaded band is the 5th-95th percentile of density among cells of that type:
the curve is supported by data there and extrapolated outside it.

No uncertainty band: the bootstrap replicates store `h` but not `K`, so the
per-type curves cannot be resampled from them.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import PortraitModel, build_archetypes  # noqa: E402

BLUE, INK, INK2, GRID, BAND = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#cde2fb"


def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    model = PortraitModel.from_fit(pe["params"], basis)
    arch, _W = build_archetypes(df, basis)
    dens = df["density"].to_numpy(float)
    arche = df["archetype"].astype(str).to_numpy()
    pbar = basis.nbr(df).mean(0)
    ox = float(np.median(df["oxygen"]))
    grid = np.geomspace(np.percentile(dens, 1), np.percentile(dens, 99), 300)
    med = float(np.median(dens))

    curves = {}
    for k, nm in enumerate(arch.names):
        Ak = np.repeat(arch.A[k][None], len(grid), 0)
        e = model.eta(Ak, np.repeat(pbar[None], len(grid), 0), grid, ox)
        e0 = model.eta(arch.A[k][None], pbar[None], np.array([med]), ox)[0]
        curves[nm] = e - e0

    # order panels by the response at the 95th percentile, so the reader sees the
    # ones that keep rising together
    q95 = float(np.percentile(dens, 95))
    i95 = int(np.argmin(np.abs(grid - q95)))
    order = sorted(arch.names, key=lambda nm: -curves[nm][i95])
    lo = min(c.min() for c in curves.values())
    hi = max(c.max() for c in curves.values())
    pad = 0.08 * (hi - lo)

    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharex=True, sharey=True)
    rows = []
    for ax, nm in zip(axes.ravel(), order):
        g = arche == nm
        b5, b95 = np.percentile(dens[g], [5, 95])
        ax.axvspan(b5, b95, color=BAND, alpha=0.55, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(med, color=INK2, lw=0.8, ls=":")
        ax.plot(grid, curves[nm], color=BLUE, lw=2)
        c = curves[nm]
        ipk = int(np.argmax(c))
        ax.set_xscale("log")
        ax.set_ylim(lo - pad, hi + pad)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(nm, fontsize=10.5, color=INK, loc="left")
        ax.text(0.98, 0.05, f"at q95: {c[i95]:+.2f}", transform=ax.transAxes,
                ha="right", fontsize=8.5, color=INK2)
        rows.append(dict(archetype=nm, at_q95=float(c[i95]),
                         at_p1=float(c[0]), at_p99=float(c[-1]),
                         peak_density=float(grid[ipk]),
                         peak_pct=float(100 * np.mean(dens < grid[ipk])),
                         type_density_p5=float(b5), type_density_p95=float(b95)))
    for ax in axes[-1]:
        ax.set_xlabel("density, cells / (100 µm)²")
    for ax in axes[:, 0]:
        ax.set_ylabel("change in log-odds of cycling\nvs median density")
    fig.suptitle("Density response of P(cycling), per cell type  -  production fit.  "
                 "Shaded: 5-95% of that type's own densities.  Dotted: median density.",
                 fontsize=10, color=INK2, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/density_response_production.png", dpi=160)
    out = pd.DataFrame(rows)
    out.to_csv("outputs/generated/density_response_production.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
