"""Local Moran's I (LISA) maps of the Pearson residuals of the production models.

    python run/spatial_lisa_production.py

`run/spatial_analysis.py` §3 mapped the earlier NB, an unpenalised matched-NB refit
and the Poisson mixture. This maps:
  NB-c                the production NB (outputs/division_classifier/nb_params_nbc_full.json)
  Poisson + Poisson   the production mixture (outputs/hpc/point_estimate.json)
  Poisson + NB        the mixture with an NB cycling component (outputs/poisson_nb/params.json)
with `spatial_analysis.lisa` unchanged: mean Pearson residual per 100 um bin (bins of
>= 20 cells), queen contiguity, row-standardised, 9,999 conditional permutations,
Benjamini-Hochberg at 5%; global Moran's I on the same bins with a 199-permutation null.

The observed Tirosh count's own LISA (bin-mean count) is the reference column of
lisa_maps_production_with_counts.png.

Writes outputs/spatial/lisa_maps_production{,_with_counts}.png, lisa_summary_production.csv and
lisa_cluster_composition_production.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import ListedColormap                          # noqa: E402
from matplotlib.patches import Patch                                  # noqa: E402

from spatial_analysis import BIN_UM, grid_index, lisa                 # noqa: E402

from tissue_dynamics.count_diagnostics import mixture_components      # noqa: E402
from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data                  # noqa: E402

OUT = "outputs/spatial"
MODELS = {"nbc": "NB-c (production NB)", "pois_pois": "Poisson + Poisson (production mixture)",
          "pois_nb": "Poisson + NB mixture"}


def pearson(y, comps):
    """Pearson residual of a mixture whose components are (weight, family, mean, alpha)."""
    mean = sum(w * m for w, _f, m, _a in comps)
    second = sum(w * (m + (a * m ** 2 if f == "nb" else 0.0) + m ** 2) for w, f, m, a in comps)
    return (y - mean) / np.sqrt(np.maximum(second - mean ** 2, 1e-12))


def plot_maps(L, keys, path):
    """Top: the binned quantity; bottom: its LISA clusters. "raw" = the observed count."""
    fig, axes = plt.subplots(2, len(keys), figsize=(5 * len(keys), 10.5))
    res_keys = [k for k in keys if k != "raw"]
    lim = max(np.nanpercentile(np.abs(L[m]["mean"]), 98) for m in res_keys)
    codes = {"ns": 1, "HH": 2, "LL": 3, "HL": 4, "LH": 5, "island": 1}
    cmap = ListedColormap(["#ffffff", "#e4e3df", "#e34948", "#2a78d6", "#f4b4b3", "#a9cbf2"])
    im_res = None
    for j, m in enumerate(keys):
        Lm = L[m]
        g = np.full((Lm["ny"], Lm["nx"]), np.nan)
        g[Lm["by"], Lm["bx"]] = Lm["mean"]
        ax = axes[0, j]
        if m == "raw":
            im = ax.imshow(g, origin="lower", cmap="Oranges", vmin=0,
                           vmax=np.nanpercentile(Lm["mean"], 98), interpolation="nearest")
            ax.set_title(f"Tirosh count (observed)\nmean count per {BIN_UM:.0f} µm bin", fontsize=10.5, loc="left")
            fig.colorbar(im, ax=ax, shrink=0.6, label="mean count")
        else:
            im_res = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title(f"{MODELS[m]}\nmean Pearson residual per {BIN_UM:.0f} µm bin", fontsize=10.5, loc="left")
        c = np.zeros((Lm["ny"], Lm["nx"]))
        c[Lm["by"], Lm["bx"]] = [codes[v] for v in Lm["cls"]]
        ax2 = axes[1, j]
        ax2.imshow(c, origin="lower", cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
        ax2.set_title(f"local Moran's I clusters (FDR 5%)\nHH {int(np.sum(Lm['cls'] == 'HH'))}, "
                      f"LL {int(np.sum(Lm['cls'] == 'LL'))} bins; global I = {Lm['I_global']:.3f}",
                      fontsize=10.5, loc="left")
        for a in (ax, ax2):
            a.invert_xaxis()                 # rotated 180 degrees, to match the other section maps
            a.invert_yaxis()
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        if m == "raw":
            fig.colorbar(im, ax=ax2, shrink=0.6).ax.set_visible(False)   # keep the columns aligned
    fig.colorbar(im_res, ax=axes[:, len(keys) - 1].tolist() if "raw" in keys else axes.ravel().tolist(),
                 shrink=0.35, anchor=(0, 0.92), label="mean residual (top row)")
    fig.legend(handles=[Patch(color="#e34948", label="HH: high among high"),
                        Patch(color="#2a78d6", label="LL: low among low"),
                        Patch(color="#f4b4b3", label="HL: high among low"),
                        Patch(color="#a9cbf2", label="LH: low among high"),
                        Patch(color="#e4e3df", label="not significant")],
               loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    names = json.load(open(cfg["data"]["table_parquet"].replace(".parquet", ".phi_names.json")))
    df.attrs["phi_names"] = names
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    y = np.asarray(D["y"], float)

    nbc = {k: jnp.asarray(np.asarray(v)) for k, v in
           json.load(open("outputs/division_classifier/nb_params_nbc_full.json")).items()}
    mu = np.exp(np.asarray(arms.nb_log_mu(nbc, D)))
    alpha = float(np.exp(np.asarray(arms.nb_log_alpha(nbc, D))))
    comps = {"nbc": [(np.ones_like(y), "nb", mu, alpha)],
             "pois_pois": mixture_components(pe["params"], D, spec)[0],
             "pois_nb": mixture_components(json.load(open("outputs/poisson_nb/params.json")), D, spec,
                                           cycling_nb=True)[0]}
    res = {m: pearson(y, c) for m, c in comps.items()}

    xy = df[["x", "y"]].to_numpy(float)
    gx, gy = grid_index(xy, BIN_UM)
    ctype = df["archetype"].astype(str).to_numpy()
    dens = df["density"].to_numpy(float)
    L, rows, chars = {}, [], []
    for m in MODELS:
        L[m] = lisa(res[m], gx, gy, seed=1)
        cls = L[m]["cls"]
        rows.append(dict(model=MODELS[m], bins=int(np.sum(cls != "island")), I_global=L[m]["I_global"],
                         I_global_null_sd=L[m]["I_global_null_sd"], I_global_p=L[m]["I_global_p"],
                         **{f"n_{c}": int(np.sum(cls == c)) for c in ("HH", "LL", "HL", "LH", "ns")}))
        print(rows[-1], flush=True)
        cell_bin = L[m]["idx"][L[m]["flat"]]
        cell_cls = np.where(cell_bin >= 0, cls[np.maximum(cell_bin, 0)], "dropped")
        for c in ("HH", "LL", "ns"):
            sel = cell_cls == c
            if sel.sum():
                comp = pd.Series(ctype[sel]).value_counts(normalize=True)
                chars.append(dict(model=MODELS[m], cluster=c, n_cells=int(sel.sum()),
                                  mean_count=float(y[sel].mean()), median_density=float(np.median(dens[sel])),
                                  mean_residual=float(res[m][sel].mean()),
                                  **{f"share:{t}": float(comp.get(t, 0.0)) for t in names}))
    pd.DataFrame(chars).to_csv(f"{OUT}/lisa_cluster_composition_production.csv", index=False)

    L["raw"] = lisa(y, gx, gy, seed=1)                  # the observed count, for reference
    cls = L["raw"]["cls"]
    rows.append(dict(model="raw Tirosh count", bins=int(np.sum(cls != "island")), I_global=L["raw"]["I_global"],
                     I_global_null_sd=L["raw"]["I_global_null_sd"], I_global_p=L["raw"]["I_global_p"],
                     **{f"n_{c}": int(np.sum(cls == c)) for c in ("HH", "LL", "HL", "LH", "ns")}))
    pd.DataFrame(rows).to_csv(f"{OUT}/lisa_summary_production.csv", index=False)
    plot_maps(L, list(MODELS), f"{OUT}/lisa_maps_production.png")
    plot_maps(L, ["raw"] + list(MODELS), f"{OUT}/lisa_maps_production_with_counts.png")
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
