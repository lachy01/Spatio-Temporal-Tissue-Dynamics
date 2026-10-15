"""Tissue maps: cell-type compartments, one panel per type, and the cycling fit.

    python run/plot_tissue_maps.py                         # breast
    MIX_DATASET=ovarian python run/plot_tissue_maps.py     # ovarian

From the section's mixture table (analysis cells) and, for the cycling maps,
its point estimate (outputs/hpc[_<dataset>]/point_estimate.json):

  tissue_compartments_<ds>.png  every cell coloured by compartment (three hues
                                plus grey: more hues cannot stay distinguishable
                                when every pair of types can sit side by side)
  cell_types_<ds>.png           one panel per cell type: that type in blue over
                                all cells in grey, so each type reads exactly
  cycling_<ds>.png              the prior pi(x) and the posterior w, each the
                                mean over cells in 25 um squares, one scale

Colours are the documented reference palette (dataviz skill, palette.md): the
first three categorical slots, which validate all-pairs; the blue sequential
ramp; muted ink for grey. Writes to outputs/figures/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                 # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402

from tissue_dynamics.mixture.data import load_config, phi_cell_types  # noqa: E402

OUT = "outputs/figures"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
GREY = "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BIN_UM = 25.0
COMPARTMENTS = {
    "breast": [("Epithelial", ["Cancer Epithelial", "Normal Epithelial"]),
               ("Stroma", ["CAFs", "PVL", "Endothelial"]),
               ("Immune", ["T-cells", "B-cells", "Myeloid", "Plasmablasts"])],
    "ovarian": [("Tumour", ["Tumour"]),
                ("Stroma", ["Fibroblasts", "Smooth Muscle Cells", "Pericytes", "Endothelial Cells"]),
                ("Immune", ["Macrophages", "T & NK Cells"])],
}
TITLE = {"breast": "Breast cancer section", "ovarian": "Ovarian cancer section"}


def style(ax):
    ax.set_facecolor(SURF)
    ax.set_aspect("equal")
    ax.invert_yaxis()                       # image orientation, as Xenium Explorer
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, x, y, um=1000):
    x0, x1 = np.nanmin(x), np.nanmax(x)
    y1 = np.nanmax(y)
    ax.plot([x0, x0 + um], [y1 + 0.03 * (y1 - np.nanmin(y))] * 2, color=INK2, lw=2,
            solid_capstyle="butt")
    ax.text(x0 + um / 2, y1 + 0.06 * (y1 - np.nanmin(y)), f"{um / 1000:g} mm", color=INK2,
            ha="center", va="top", fontsize=9)


def main():
    cfg = load_config()
    ds = cfg.get("dataset", "breast")
    os.makedirs(OUT, exist_ok=True)
    path = cfg["data"]["table_parquet"]
    df = pd.read_parquet(path)
    names = json.load(open(os.path.splitext(path)[0] + ".phi_names.json"))
    cols = sorted(c for c in df.columns if c.startswith("phi_") and c[4:].isdigit())
    ct = phi_cell_types(df[cols].to_numpy(float), names, cfg["data"]).astype(str)
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    n = len(df)
    # marker area from cell spacing: a dot about as wide as the mean distance
    # between cells at the figure's scale (points^2), so dense sections read solid
    area_um2 = (x.max() - x.min()) * (y.max() - y.min())
    spacing_pts = np.sqrt(area_um2 / n) / (x.max() - x.min()) * 10.0 * 72
    pt_c = float(np.clip(spacing_pts ** 2 * 0.7, 0.05, 2.0))
    pt = pt_c * 0.45
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "figure.facecolor": SURF, "savefig.facecolor": SURF})
    w_in = 10.0
    asp = (y.max() - y.min()) / (x.max() - x.min())

    # ---- compartments --------------------------------------------------------
    comp = np.full(n, "Other", dtype=object)
    for (cname, members) in COMPARTMENTS[ds]:
        comp[np.isin(ct, members)] = cname
    colour = {"Other": GREY, **{c: SLOT[i] for i, (c, _) in enumerate(COMPARTMENTS[ds])}}
    fig, ax = plt.subplots(figsize=(w_in, w_in * asp), constrained_layout=True)
    # one pass in random order: no compartment is systematically painted over another
    o = np.random.default_rng(0).permutation(n)
    ax.scatter(x[o], y[o], s=pt_c, c=[colour[c] for c in comp[o]], linewidths=0, rasterized=True)
    style(ax)
    scalebar(ax, x, y)
    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=colour[c], mec="none",
                      label=f"{c} ({np.mean(comp == c):.0%})")
               for c in [c for c, _ in COMPARTMENTS[ds]] + ["Other"] if (comp == c).any()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
              labelcolor=INK)
    members = "\n".join(f"{c}: {', '.join(mm)}" for c, mm in COMPARTMENTS[ds])
    if ds == "ovarian":
        members += "\nOther: Granulosa, Fallopian Tube Epithelium"
    ax.text(1.02, 0.0, members, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom")
    ax.set_title(f"{TITLE[ds]}: {n:,} cells by compartment", loc="left", color=INK, fontsize=12)
    fig.savefig(f"{OUT}/tissue_compartments_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- one panel per type ---------------------------------------------------------
    types = pd.Series(ct).value_counts().index.tolist()
    nc = 3 if len(types) == 9 else 4
    nr = int(np.ceil(len(types) / nc))
    pw = 4.2
    fig, axes = plt.subplots(nr, nc, figsize=(pw * nc, pw * asp * nr + 0.6 * nr))
    for ax, t in zip(axes.ravel(), types):
        m = ct == t
        ax.scatter(x[~m], y[~m], s=pt * 0.6, c=GRID, linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=pt * 1.5, c=SLOT[0], linewidths=0, rasterized=True)
        style(ax)
        ax.set_title(f"{t}  {m.sum():,} ({m.mean():.1%})", loc="left", color=INK, fontsize=10)
    for ax in axes.ravel()[len(types):]:
        ax.set_visible(False)
    fig.suptitle(f"{TITLE[ds]}: each cell type (blue) against all cells (grey)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/cell_types_{ds}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- cycling ---------------------------------------------------------------------
    hpc = os.environ.get("MIX_OUT", f"outputs/hpc_{ds}" if ds != "breast" else "outputs/hpc")
    pe_path = f"{hpc}/point_estimate.json"
    if not os.path.exists(pe_path):
        print(f"no {pe_path}: cycling maps skipped")
        return
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
    from tissue_dynamics.mixture.model import build_data, pi_hat, responsibilities
    pe = json.load(open(pe_path))
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    pi = np.asarray(pi_hat(P, D, spec))
    w = np.asarray(responsibilities(P, D, spec))
    xb = ((x - x.min()) // BIN_UM).astype(int)
    yb = ((y - y.min()) // BIN_UM).astype(int)
    shape = (yb.max() + 1, xb.max() + 1)
    flat = yb * shape[1] + xb
    cnt = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    cmap.set_bad(SURF)
    fig, axes = plt.subplots(1, 2, figsize=(2 * w_in * 0.6, w_in * 0.6 * asp + 1.2),
                             constrained_layout=True)
    for ax, v, ttl in ((axes[0], pi, f"Prior \u03c0(x), from state and neighbourhood (mean {pi.mean():.1%})"),
                       (axes[1], w, f"Posterior w, given the cell's counts (mean {w.mean():.1%})")):
        img = np.bincount(flat, weights=v, minlength=shape[0] * shape[1]).reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(cnt > 0, img / cnt, np.nan)
        im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=[x.min(), x.min() + shape[1] * BIN_UM, y.min() + shape[0] * BIN_UM, y.min()])
        style(ax)
        ax.invert_yaxis()
        ax.set_title(ttl, loc="left", color=INK, fontsize=10)
    scalebar(axes[0], x, y)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, aspect=40)
    cb.set_label(f"mean cycling probability per {BIN_UM:g} \u00b5m square", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2)
    fig.suptitle(f"{TITLE[ds]}: the mixture's cycling probability", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(f"{OUT}/cycling_{ds}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{{tissue_compartments,cell_types,cycling}}_{ds}.png")


if __name__ == "__main__":
    main()
