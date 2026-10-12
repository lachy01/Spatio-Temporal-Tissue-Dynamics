"""How much does the data support each entry of the own x neighbour interaction?

    python run/interaction_support.py [--cv outputs/hpc/cv_rank]
                                      [--boot outputs/hpc/bootstrap]

The weight model's bilinear term is phi' B phibar, with phibar the split
neighbourhood [contact 0-15 um | regional 15-80 um]. In cell-type coordinates
B is a 9 x 18 matrix: own type k against contact type l and regional type l.
A full matrix would need every combination to be seen often enough to estimate
its own coefficient; the low-rank factorisation exists because many are not.
This measures that support, and sets the fitted interaction against it.

Support, two ways:

* By argmax type (intuitive). Every typed cell is typed by standardised argmax,
  as the table's `archetype` is. Contact: the number of (own k, contact
  neighbour l) pairs and the number of type-k cells with at least one type-l
  contact neighbour; enrichment = observed pairs / pairs expected if contact
  neighbours were typed at random. Regional: the mean fraction of type l among
  a type-k cell's regional neighbours.
* By the covariate the coefficient multiplies (what estimation sees). Entry
  (k, l) multiplies z = (phi_k - m_k)(phibar_l - mbar_l). Its effective number
  of cells is Kish's ESS of the second moment, (sum z^2)^2 / sum z^4: the
  number of cells that carry the information, small when a few unusual cells
  dominate it.

Fit: the full-data rank-3 point estimate (outputs/hpc/point_estimate.json),
mapped to cell-type coordinates B* = Q_own B Q_nbr' and put per SD of the own
score and per SD of the neighbour score. Its uncertainty is the spatial block
bootstrap (job `bootstrap`, 250 blocks resampled, each replicate refitted from
the point estimate): the bootstrap SE, and the 95% percentile interval. An
entry is reportable when that interval excludes zero; the stricter column asks
for |estimate / bootstrap SE| above the Bonferroni bound for all 162 entries.
B = UV' is invariant to rotations of the factors, so replicates need no
alignment.

The cv_rank fold fits (one per spatial fold, each trained on the other four)
add a delete-one-fold jackknife SE as a cross-check, and ranks 2 and 5 from the
same run show where the factorisation's rank matters. B* is double-centred
(each row and column sums to zero within a block), so an entry is an effect
relative to the average type, not an absolute one.

Writes outputs/interaction/{support.csv, support.png, fit_vs_support.png}.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config, split_neighbourhood  # noqa: E402
from tissue_dynamics.nb_baseline import all_cells                     # noqa: E402

OUT = "outputs/interaction"
BLOCKS = ("contact", "regional")


def b_star(r, basis, sd_own, sd_nbr):
    """A fit's B in cell-type coordinates, per SD of each score: (9, 18)."""
    B = np.asarray(r["B"], dtype=float)                       # model coordinates
    Bs = basis.Q_own @ B @ basis.Q_nbr.T                      # raw-score coordinates
    return Bs * sd_own[:, None] * sd_nbr[None, :]


def heat(ax, M, names, title, fmt, cmap, vmin=None, vmax=None, center=None, label=""):
    if center is not None:
        lim = np.nanmax(np.abs(M - center))
        vmin, vmax = center - lim, center + lim
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("neighbour type")
    ax.set_ylabel("own type")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=6.2)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="outputs/hpc/cv_rank")
    ap.add_argument("--boot", default="outputs/hpc/bootstrap")
    ap.add_argument("--point", default="outputs/hpc/point_estimate.json")
    a = ap.parse_args()

    cfg = load_config()
    dc = cfg["data"]
    path = dc["table_parquet"]
    df = pd.read_parquet(path)
    names = json.load(open(os.path.splitext(path)[0] + ".phi_names.json"))
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    d = len(names)

    # ---- argmax types for every typed cell, as build_table types them ----
    cells = all_cells(cfg, genes=[])
    P = cells[[f"phi_{nm}" for nm in names]].to_numpy(float)
    Z = (P - P.mean(0)) / P.std(0)
    ctype = Z.argmax(1)
    pos = pd.Series(np.arange(len(cells)), index=cells["cell_id"].astype(str))
    idx = pos.reindex(df["cell_id"].astype(str)).to_numpy().astype(int)
    agree = np.mean(np.asarray(names)[ctype[idx]] == df["archetype"].astype(str).to_numpy())
    print(f"argmax types reproduce the table's archetype for {100 * agree:.2f}% of cells")
    onehot = np.eye(d)[ctype]
    fc, n_c, fr, n_r = split_neighbourhood(cells[["x", "y"]].to_numpy(float), onehot,
                                           cells["typed"].to_numpy(),
                                           float(dc.get("contact_radius_um", 15.0)),
                                           float(dc["neighbour_radius_um"]))
    own = ctype[idx]
    fc, n_c, fr, n_r = fc[idx], n_c[idx], fr[idx], n_r[idx]

    rows = []
    pairs = np.zeros((d, d))
    exposed = np.zeros((d, d))
    reg_frac = np.zeros((d, d))
    for k in range(d):
        m = own == k
        cnt = n_c[m, None] * fc[m]                            # type-l contact neighbours per cell
        pairs[k] = cnt.sum(0)
        exposed[k] = (cnt >= 0.5).sum(0)
        reg_frac[k] = fr[m].mean(0)
    expect = pairs.sum(1, keepdims=True) * (pairs.sum(0, keepdims=True) / pairs.sum())
    enrich = pairs / expect

    # ---- covariate support: Kish ESS of z = (phi_k - m_k)(phibar_l - mbar_l) ----
    own_raw = df[[f"phi_{j:02d}" for j in range(d)]].to_numpy(float)
    own_c = own_raw - own_raw.mean(0)
    nbr_raw = {b: df[[f"{pre}{j:02d}" for j in range(d)]].to_numpy(float)
               for b, pre in zip(BLOCKS, ("nbrc_phi_", "nbrr_phi_"))}
    ess = {}
    for b in BLOCKS:
        nc = nbr_raw[b] - nbr_raw[b].mean(0)
        E = np.zeros((d, d))
        for k in range(d):
            z2 = (own_c[:, [k]] * nc) ** 2
            E[k] = z2.sum(0) ** 2 / (z2 ** 2).sum(0)
        ess[b] = E

    # ---- fitted interaction from the cv_rank fold fits ----
    sd_own = own_raw.std(0)
    sd_nbr = np.concatenate([nbr_raw[b].std(0) for b in BLOCKS])
    fits = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.cv, "*.json")))]
    # the fold fits are a cross-check on the same model only: a CV run fitted with
    # other technical covariates (the area-era runs, since the MisTIC switch) is
    # not used, and its columns are left empty
    fits = [r for r in fits if r.get("technical", "area") == basis.technical]
    Bfit = {}
    for rank in sorted({int(r["rank"]) for r in fits}):
        Bfit[rank] = np.stack([b_star(r, basis, sd_own, sd_nbr) for r in fits
                               if int(r["rank"]) == rank])
    if 3 in Bfit:
        g = Bfit[3].shape[0]
        B3cv = Bfit[3].mean(0)
        se3 = np.sqrt((g - 1) / g * ((Bfit[3] - B3cv) ** 2).sum(0))
    else:
        print(f"no rank-3 cv_rank fold fits for technical: {basis.technical} in {a.cv}; "
              "the jackknife cross-check columns are left empty")
        B3cv = se3 = None

    # ---- the full-data point estimate and its spatial block bootstrap ----
    pe = json.load(open(a.point))
    B3 = b_star({"B": np.asarray(pe["params"]["U"]) @ np.asarray(pe["params"]["V"]).T},
                basis, sd_own, sd_nbr)
    boot = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.boot, "0*.json")))]
    boot = [r for r in boot if int(r["rank"]) == int(pe["rank"])]
    Bb = np.stack([b_star(r, basis, sd_own, sd_nbr) for r in boot])
    se_b = Bb.std(0, ddof=1)
    lo_b, hi_b = np.percentile(Bb, [2.5, 97.5], axis=0)
    z_bonf = norm.ppf(1 - 0.025 / B3.size)
    print(f"bootstrap: {len(Bb)} replicates; Bonferroni bound for {B3.size} entries |z| > {z_bonf:.2f}")

    for bi, b in enumerate(BLOCKS):
        cols = slice(bi * d, (bi + 1) * d)
        for k in range(d):
            for l in range(d):
                rows.append({"block": b, "own": names[k], "neighbour": names[l],
                             "contact_pairs": pairs[k, l] if b == "contact" else np.nan,
                             "cells_exposed": exposed[k, l] if b == "contact" else np.nan,
                             "enrichment": enrich[k, l] if b == "contact" else np.nan,
                             "regional_fraction": reg_frac[k, l] if b == "regional" else np.nan,
                             "ess_cells": ess[b][k, l],
                             "B_rank3": B3[k, cols][l], "se_boot": se_b[k, cols][l],
                             "ci_lo": lo_b[k, cols][l], "ci_hi": hi_b[k, cols][l],
                             "z": B3[k, cols][l] / se_b[k, cols][l],
                             "B_rank3_cv": B3cv[k, cols][l] if B3cv is not None else np.nan,
                             "se_jackknife": se3[k, cols][l] if se3 is not None else np.nan,
                             **{f"B_rank{r}": Bfit[r].mean(0)[k, cols][l] for r in Bfit if r != 3}})
    tab = pd.DataFrame(rows)
    tab["reportable"] = (tab["ci_lo"] > 0) | (tab["ci_hi"] < 0)
    tab["reportable_bonferroni"] = tab["z"].abs() > z_bonf
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(f"{OUT}/support.csv", index=False)

    # ---- summaries ----
    print(f"\n{len(df):,} cells; {int(pairs.sum()):,} contact pairs")
    for b in BLOCKS:
        t = tab[tab["block"] == b]
        print(f"\n{b}: effective cells per interaction entry, median {t['ess_cells'].median():,.0f}, "
              f"range {t['ess_cells'].min():,.0f}-{t['ess_cells'].max():,.0f}; "
              f"entries below 1,000: {int((t['ess_cells'] < 1000).sum())}/81, "
              f"below 5,000: {int((t['ess_cells'] < 5000).sum())}/81")
        print(f"   rank-3 |B| per SD: median {t['B_rank3'].abs().median():.4f}, max {t['B_rank3'].abs().max():.4f}; "
              f"95% CI excludes 0: {int(t['reportable'].sum())}/81; "
              f"Bonferroni: {int(t['reportable_bonferroni'].sum())}/81")
        rs = tab[tab["block"] == b]
        print(f"   corr(log ESS, |B rank 3|) = {np.corrcoef(np.log(rs['ess_cells']), rs['B_rank3'].abs())[0, 1]:+.3f}; "
              f"corr(log ESS, bootstrap SE) = {np.corrcoef(np.log(rs['ess_cells']), rs['se_boot'])[0, 1]:+.3f}")
        if rs["se_jackknife"].notna().all():
            print(f"   bootstrap SE / jackknife SE: median {np.median(rs['se_boot'] / rs['se_jackknife']):.2f}; "
                  f"corr(point estimate, fold mean) {np.corrcoef(rs['B_rank3'], rs['B_rank3_cv'])[0, 1]:+.3f}")
        if "B_rank5" in rs:
            dlt = (rs["B_rank5"] - rs["B_rank3"]).abs()
            print(f"   |B rank 5 - B rank 3|: median {dlt.median():.4f}; "
                  f"corr with log ESS {np.corrcoef(np.log(rs['ess_cells']), dlt)[0, 1]:+.3f}")
    print("\ncontact pairs, rarest 8 (own -> neighbour):")
    t = tab[tab["block"] == "contact"].sort_values("contact_pairs").head(8)
    print(t[["own", "neighbour", "contact_pairs", "cells_exposed", "enrichment", "ess_cells",
             "B_rank3", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    print("\nreportable entries (95% bootstrap CI excludes 0), largest first (per SD):")
    t = tab[tab["reportable"]]
    t = t.reindex(t["B_rank3"].abs().sort_values(ascending=False).index)
    print(t[["block", "own", "neighbour", "B_rank3", "ci_lo", "ci_hi", "z", "reportable_bonferroni",
             "ess_cells"]].round(4).to_string(index=False))

    # ---- figures ----
    short = [n.replace(" Epithelial", " Epi.").replace("Plasmablasts", "Plasmabl.") for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    heat(axes[0, 0], np.log10(np.maximum(pairs, 1)), short,
         "Contact pairs (0-15 um), log10 count", lambda v: f"{v:.1f}", "Blues", label="log10 pairs")
    heat(axes[0, 1], reg_frac, short, "Regional neighbourhood (15-80 um): mean fraction of each type",
         lambda v: f"{v:.2f}", "Blues", vmin=0, label="fraction")
    heat(axes[1, 0], np.log10(ess["contact"]), short,
         "Contact: effective cells behind each interaction entry, log10", lambda v: f"{v:.1f}",
         "Greens", label="log10 ESS")
    heat(axes[1, 1], np.log10(ess["regional"]), short,
         "Regional: effective cells behind each interaction entry, log10", lambda v: f"{v:.1f}",
         "Greens", label="log10 ESS")
    fig.tight_layout()
    fig.savefig(f"{OUT}/support.png", dpi=140, bbox_inches="tight")

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for bi, b in enumerate(BLOCKS):
        cols = slice(bi * d, (bi + 1) * d)
        heat(axes[0, bi], B3[:, cols], short,
             f"Rank-3 interaction, {b}: effect per SD x SD (outlined: 95% CI excludes 0)",
             lambda v: f"{v:+.2f}", "RdBu_r", center=0.0, label="logit pi per SD x SD")
        for k, l in zip(*np.nonzero((lo_b[:, cols] > 0) | (hi_b[:, cols] < 0))):
            axes[0, bi].add_patch(plt.Rectangle((l - 0.5, k - 0.5), 1, 1, fill=False, ec="k", lw=1.2))
        t = tab[tab["block"] == b]
        ax = axes[1, bi]
        sc = ax.scatter(t["ess_cells"], t["B_rank3"].abs(), c=np.minimum(t["z"].abs(), 6),
                        cmap="viridis", s=34, edgecolor="none")
        ax.set_xscale("log")
        ax.set_xlabel("effective cells behind the entry (Kish ESS)")
        ax.set_ylabel("|rank-3 effect| per SD x SD")
        ax.set_title(f"{b}: fitted size against support (colour = |z|, bootstrap, capped at 6)",
                     fontsize=10)
        for _, r in t.reindex(t["B_rank3"].abs().sort_values(ascending=False).index).head(5).iterrows():
            ax.annotate(f"{r['own'][:10]} x {r['neighbour'][:10]}", (r["ess_cells"], abs(r["B_rank3"])),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="|z|")
        ax.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fit_vs_support.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}/support.csv, support.png, fit_vs_support.png")


if __name__ == "__main__":
    main()
