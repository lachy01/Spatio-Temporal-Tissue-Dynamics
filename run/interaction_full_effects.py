"""The full neighbourhood effect on cycling, own type x neighbour type, as matrices.

    python run/interaction_full_effects.py

`run/interaction_support.py` plots only the interaction B (double-centred, per SD).
This plots the whole neighbour effect a cell of own type a experiences: the main
effect c PLUS the interaction, evaluated at the type's centroid phi_a,

    eff_a = c + B' phi_a                       (neighbour coordinates, 16)
    log OR_a,l = (Q_nbr eff_a)_l * IQR_l       (cell-type scores, per IQR)

the spec 8.2 estimand of `run/estimands_report.py` (outputs/estimands_8_2.md),
in matrix form: rows = own type a, columns = neighbour type l, one matrix for the
contact block (0-15 um) and one for the regional block (15-80 um). IQR_l is the
interquartile range of the raw neighbour score l over cells. phi_a is the mean
own state of the cells labelled a; phi is centred, so the main effect c alone is
the effect for a cell at the average own state.

The scores are compositional (they sum to a constant), so a column is a contrast:
more of type l's signature in the neighbourhood at the expense of the others.

95% intervals: percentile, over the 300 spatial block bootstrap replicates
(outputs/hpc/bootstrap, each with its own c and B).

Writes outputs/interaction/full_effect.{png,csv}, full_effect_decomposition.png.
"""
import glob
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

from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/interaction"
ORDER = ["Cancer Epithelial", "Normal Epithelial", "T-cells", "B-cells", "Plasmablasts",
         "Myeloid", "CAFs", "Endothelial", "PVL"]
INK, MUTED, SURF = "#1f1f1e", "#8a8985", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])


def effects(c, B, phi_c, Q, iqr):
    """(own types, 18) log OR per IQR: main, interaction and total."""
    main = (Q @ np.asarray(c)) * iqr
    inter = np.stack([(Q @ (np.asarray(B).T @ p)) * iqr for p in phi_c])
    return main, inter, main[None, :] + inter


def heat(ax, M, sig, title, lim, rows, cols, annotate=True):
    im = ax.imshow(M, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                strong = sig is not None and sig[i, j]
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color=INK if strong else MUTED, fontweight="bold" if strong else "normal")
    ax.set_title(title, loc="left", fontsize=10)
    return im


def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    names = json.load(open(os.path.splitext(cfg["data"]["table_parquet"])[0] + ".phi_names.json"))
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    P = pe["params"]
    Q = basis.Q_nbr                                                   # (18, 16)
    nbr_names = list(basis.nbr_names)                                 # contact:..., regional:...
    iqr = np.subtract(*np.percentile(df[basis.nbr_cols].to_numpy(float), [75, 25], axis=0))
    own = basis.own(df)
    lab = df["archetype"].astype(str).to_numpy()
    phi_c = np.stack([own[lab == a].mean(0) for a in ORDER])

    B0 = np.asarray(P["U"]) @ np.asarray(P["V"]).T
    main0, int0, tot0 = effects(P["c"], B0, phi_c, Q, iqr)
    boot = [json.load(open(f)) for f in sorted(glob.glob("outputs/hpc/bootstrap/*.json"))]
    reps = [effects(r["c"], r["B"], phi_c, Q, iqr) for r in boot if r.get("B") is not None]
    print(f"{len(reps)} bootstrap replicates")
    tot_b = np.stack([t for _, _, t in reps])
    main_b = np.stack([m for m, _, _ in reps])
    int_b = np.stack([x for _, x, _ in reps])
    ci = lambda a: np.percentile(a, [2.5, 97.5], axis=0)              # noqa: E731
    (lo, hi), (mlo, mhi), (ilo, ihi) = ci(tot_b), ci(main_b), ci(int_b)
    sig, msig, isig = (lo > 0) | (hi < 0), (mlo > 0) | (mhi < 0), (ilo > 0) | (ihi < 0)

    # column order: the same type order within each block
    blk = {b: [nbr_names.index(f"{b}:{t}") for t in ORDER] for b in ("contact", "regional")}

    rows = []
    for i, a in enumerate(ORDER):
        for b, idx in blk.items():
            for t, j in zip(ORDER, idx):
                rows.append({"own type": a, "block": b, "neighbour type": t,
                             "log OR per IQR": tot0[i, j], "lo": lo[i, j], "hi": hi[i, j],
                             "OR per IQR": np.exp(tot0[i, j]), "CI excludes 0": bool(sig[i, j]),
                             "main effect (c)": main0[j], "interaction (B' phi_a)": int0[i, j]})
    pd.DataFrame(rows).to_csv(f"{OUT}/full_effect.csv", index=False)

    lim = float(np.ceil(np.abs(tot0).max() * 10) / 10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), layout="constrained")
    for ax, (b, idx) in zip(axes, blk.items()):
        im = heat(ax, tot0[:, idx], sig[:, idx],
                  f"{b} neighbours ({'0-15' if b == 'contact' else '15-80'} µm)", lim, ORDER, ORDER)
        ax.set_xlabel("neighbour type (more of its score in the neighbourhood, by one IQR)", fontsize=9)
    axes[0].set_ylabel("own type (cell at its type's centroid)", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("log odds ratio of cycling per IQR  (0 = no effect)", fontsize=9)
    cb.outline.set_visible(False)
    fig.suptitle("Full neighbourhood effect on cycling, c + B'φ_a: log odds ratio per IQR of each "
                 "neighbour score\nbold: 95% spatial block bootstrap interval excludes 0 "
                 f"({int(sig.sum())} of {sig.size}); grey: it does not", x=0.01, ha="left", fontsize=10.5)
    fig.savefig(f"{OUT}/full_effect.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    # decomposition: main effect (one row, every own type) + interaction = total
    lim2 = float(np.ceil(max(np.abs(tot0).max(), np.abs(int0).max()) * 10) / 10)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    for r, (b, idx) in enumerate(blk.items()):
        mrow = np.repeat(main0[idx][None], len(ORDER), axis=0)
        msg = np.repeat(msig[idx][None], len(ORDER), axis=0)
        heat(axes[r, 0], mrow, msg, f"{b}: main effect c (the same for every own type)", lim2, ORDER, ORDER)
        heat(axes[r, 1], int0[:, idx], isig[:, idx], f"{b}: interaction B'φ_a", lim2, ORDER, ORDER)
        im = heat(axes[r, 2], tot0[:, idx], sig[:, idx], f"{b}: total = main + interaction", lim2,
                  ORDER, ORDER)
        axes[r, 0].set_ylabel("own type", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("log odds ratio of cycling per IQR", fontsize=9)
    cb.outline.set_visible(False)
    fig.suptitle("Where the full effect comes from: neighbour main effect plus own x neighbour "
                 "interaction (log odds ratio per IQR; bold = 95% bootstrap interval excludes 0)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/full_effect_decomposition.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    for b, idx in blk.items():
        t = pd.DataFrame(np.exp(tot0[:, idx]), index=ORDER, columns=ORDER).round(2)
        print(f"\n{b}: odds ratio per IQR (rows own, columns neighbour)\n{t.to_string()}")
    print(f"\nCI excludes 0: total {int(sig.sum())}/{sig.size}, main {int(msig.sum())}/{msig.size}, "
          f"interaction {int(isig.sum())}/{isig.size}")


if __name__ == "__main__":
    main()
