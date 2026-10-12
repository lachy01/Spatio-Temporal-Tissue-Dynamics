"""MisTIC T4 (report): does theta m absorb spillover from epithelium?

    python run/mistic_t4_spillover.py

G_epi: epithelial markers on the panel and outside the blocklist B. Of the
spec's candidates only EPCAM and CDH1 qualify (KRT8, KRT18 and KRT19 are not
on this panel); CLDN7 joins them if it is as epithelium-specific in this
section (at least 5x the mean count in epithelial cells as in the rest).

In non-epithelial analysis cells (argmax archetype neither Cancer nor Normal
Epithelial), G_epi's summed count goes through BASE-T and BASE-O twice, with
the technical null fitted to those cells: once with theta m^G_epi in the mean
(mu = Nt e^b0 + theta m, the new null) and once without (mu = Nt e^b0). Call
rates are plotted against the distance from each cell to the nearest Cancer
Epithelial analysis cell, in 10 um bins out to 100 um.

Expected: without the term, call rates rise toward epithelium -- epithelial
transcripts spilled into neighbouring cells look like expression; with it, the
curve flattens. Reported with the fitted theta.

Writes outputs/mistic/t4.md, t4_spillover.csv and t4_spillover.png.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from scipy.spatial import cKDTree                                     # noqa: E402

from tissue_dynamics import mistic as M                               # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402

CANDIDATES = ("EPCAM", "KRT8", "KRT18", "KRT19", "CDH1")
EXTRA = ("CLDN7",)
EPI = ("Cancer Epithelial", "Normal Epithelial")
EDGES = list(range(0, 101, 10)) + [np.inf]
MIN_ENRICH = 5.0


def main():
    t0 = time.time()
    cfg, df, out = M.test_setup()
    basis = build_basis(df, cfg, verbose=False)
    cov = M.load_covariates(cfg)
    ids = df["cell_id"].astype(str).to_numpy()
    cix = pd.Index(cov.counts.cell_ids).get_indexer(ids)
    isB = M.blocklist(cfg, cov.counts.genes)
    names = df.attrs["phi_names"]
    ctype = L.cell_type(df, names)
    epi = np.isin(ctype, EPI)
    panel = {g: i for i, g in enumerate(cov.counts.genes)}

    rows_g, genes = [], []
    for g in CANDIDATES + EXTRA:
        on = g in panel
        inB = bool(isB[panel[g]]) if on else None
        enr = np.nan
        if on and not inB:
            v = np.asarray(cov.counts.X[cix][:, panel[g]].todense()).ravel()
            enr = float(v[epi].mean() / max(v[~epi].mean(), 1e-9))
        use = bool(on and not inB and (g in CANDIDATES or enr >= MIN_ENRICH))
        rows_g.append({"gene": g, "on panel": on, "in B": inB, "epithelial / other mean count": enr,
                       "used": use})
        if use:
            genes.append(g)
    G = pd.DataFrame(rows_g)
    print(G.to_string(index=False), flush=True)

    sel = ~epi
    sub = df.loc[sel].reset_index(drop=True)
    X, off, _m = L.null_design(sub, basis)
    y = cov.counts.gene_sum(genes)[cix][sel]
    m_epi = cov.m_gene_set(genes)[0][cix][sel]
    ct = ctype[sel]
    xy = df[["x", "y"]].to_numpy(float)
    ce = ctype == "Cancer Epithelial"
    dist, _ = cKDTree(xy[ce]).query(xy[sel], k=1)
    band = pd.cut(dist, EDGES, right=False, labels=False)

    res, fits = {}, {}
    for name, m in (("with theta m", m_epi), ("without", None)):
        null = L.robust_null(y, X, off, m=m)
        mu0, a0 = null["mu"], null["alpha"]
        sd0 = np.sqrt(mu0 + a0 * mu0 ** 2)
        LT, _c, _ai, _p = L.base_t(y, mu0, a0, L.ALPHA_B_PRIMARY)
        LO, _cO, _a = L.base_o(y, mu0, sd0, L.base_o_scale((y - mu0) / sd0, ct), L.TD_PRIMARY)
        res[name] = (LT, LO)
        fits[name] = null
        print(f"  {name}: alpha0 {a0:.4f}" + (f", theta {null['theta_m']:.4f}" if m is not None
                                              else "") + f"; BASE-T {LT.mean():.4f}, BASE-O "
              f"{LO.mean():.4f}", flush=True)

    rows = []
    for k in range(len(EDGES) - 1):
        mm = band == k
        lab = f"{EDGES[k]}-{EDGES[k + 1]}" if np.isfinite(EDGES[k + 1]) else f">= {EDGES[k]}"
        r = {"distance to nearest Cancer Epithelial (um)": lab, "cells": int(mm.sum()),
             "mean G_epi count": float(y[mm].mean()) if mm.any() else np.nan,
             "mean m": float(m_epi[mm].mean()) if mm.any() else np.nan}
        for name, (LT, LO) in res.items():
            r[f"BASE-T, {name}"] = float(LT[mm].mean()) if mm.any() else np.nan
            r[f"BASE-O, {name}"] = float(LO[mm].mean()) if mm.any() else np.nan
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(f"{out}/t4_spillover.csv", index=False)

    def near_far(col):
        v = T[col].to_numpy()
        return v[0] / v[9] if v[9] > 0 else np.inf     # 0-10 um over 90-100 um

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(T))
    for ax, lab in zip(axes, ("BASE-T", "BASE-O")):
        for name, col in (("without", "#c0504d"), ("with theta m", "#1f4e79")):
            ax.plot(xs, T[f"{lab}, {name}"], "o-", color=col, label=f"null {name}")
        ax.set_xticks(xs)
        ax.set_xticklabels(T.iloc[:, 0], rotation=40, ha="right", fontsize=8)
        ax.set_xlabel("distance to the nearest Cancer Epithelial cell (um)")
        ax.set_ylabel(f"{lab} call rate on G_epi")
        ax.set_title(f"{lab}, non-epithelial cells", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"Spillover control: G_epi = {', '.join(genes)}; theta = "
                 f"{fits['with theta m']['theta_m']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/t4_spillover.png", dpi=150)
    plt.close(fig)

    def md(frame):
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in frame.iterrows():
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                                           for v in r.to_numpy()) + " |")
        return "\n".join(lines)

    f1 = fits["with theta m"]
    text = ["## T4. Spillover control (report)", "",
            f"G_epi = **{', '.join(genes)}**, in {int(sel.sum()):,} non-epithelial analysis cells. "
            f"Fitted theta = **{f1['theta_m']:.4f}** (SE of eta {f1['se'].get('eta', np.nan):.4f}); "
            f"alpha0 {f1['alpha']:.4f} with the term, {fits['without']['alpha']:.4f} without.", "",
            "Call rate at 0-10 um over the rate at 90-100 um:", "",
            *[f"- {lab}, {name}: **{near_far(f'{lab}, {name}'):.2f}**"
              for lab in ("BASE-T", "BASE-O") for name in ("without", "with theta m")], "",
            md(T), "", "Gene choice:", "", md(G), "", "![T4](t4_spillover.png)"]
    open(f"{out}/t4.md", "w").write("\n".join(text) + "\n")
    print("\n".join(text[4:11]))
    print(f"T4 done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
