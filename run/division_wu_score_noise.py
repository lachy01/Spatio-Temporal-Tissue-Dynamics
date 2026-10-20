"""How noisy is the whole-transcriptome cell-cycle score on Wu, and why can't it
calibrate the thresholds?

    python run/division_wu_score_noise.py

The Wu calibrations (run/division_wu_calibration*.py) took Wu's S/G2M call as the
truth: max(S score, G2M score) > 0, Seurat's CellCycleScoring, which scanpy's
`score_genes_cell_cycle` reimplements. This measures that truth's own noise, with
scanpy on Wu's raw counts (normalize_total 1e4, log1p, the Tirosh lists of
`gene_sets.py` with the aliases Wu uses):

1. Agreement with Wu's published Seurat scores (sanity).
2. Control-gene randomness: the score subtracts a random control set; ten
   `random_state`s give each cell's spread and how often its call flips.
3. Split-half repeatability: each cell's counts are split in two by binomial
   thinning (p = 0.5), so the halves are independent draws of the same cell at
   half depth. Each half is scored on its own. Reported: score correlation, call
   agreement (Cohen's kappa, MCC), by depth decile and type; the Spearman-Brown
   full-depth reliability of the continuous score; and how well a half's call
   recovers the full-depth call (the best any label could do against a truth
   measured at half depth).
4. Depth dependence: the call rate by depth quintile within each type.

Scope as the calibrations: ER+ patients for the headline numbers, all patients also
reported. Writes outputs/division_classifier/wu_score_noise/ (summary.json,
by_depth.csv, by_type.csv, figures, report.md).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from tissue_dynamics.gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES  # noqa: E402

OUT = "outputs/division_classifier/wu_score_noise"
WU_DIR = "data/wu2021/Wu_etal_2021_BRCA_scRNASeq"
WU_CACHE = "data/wu2021/wu_counts_csr.npz"
ALIASES = {"MLF1IP": "CENPU", "FAM64A": "PIMREG", "HN1": "JPT1"}
SEEDS = range(10)
SHOW = ("Cancer Epithelial", "T-cells", "B-cells", "Myeloid", "CAFs")
COL = dict(zip(SHOW, ("#c2410c", "#2563eb", "#7c3aed", "#0f766e", "#b45309")))
INK, MUTED = "#1f1f1e", "#6b6a66"


def say(*a):
    print(*a, flush=True)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def mcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def score(X, genes, seed=0):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X=X.copy(), var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    s = [g for g in (ALIASES.get(g, g) for g in TIROSH_S_GENES) if g in a.var_names]
    g2m = [g for g in (ALIASES.get(g, g) for g in TIROSH_G2M_GENES) if g in a.var_names]
    sc.tl.score_genes_cell_cycle(a, s_genes=s, g2m_genes=g2m, random_state=seed)
    S, G = a.obs["S_score"].to_numpy(float), a.obs["G2M_score"].to_numpy(float)
    return S, G, np.maximum(S, G)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    genes = pd.read_csv(f"{WU_DIR}/count_matrix_genes.tsv", header=None)[0].astype(str).to_numpy()
    bar = pd.read_csv(f"{WU_DIR}/count_matrix_barcodes.tsv", header=None)[0].astype(str).to_numpy()
    X = sp.load_npz(WU_CACHE).tocsr().astype(np.float32)
    meta = pd.read_parquet("data/wu2021/wu_cells.parquet").set_index("barcode").loc[bar].reset_index()
    umi = np.asarray(X.sum(1)).ravel()
    say(f"Wu: {X.shape[0]:,} cells x {X.shape[1]:,} genes; UMI matches metadata: "
        f"{np.allclose(umi, meta['nCount_RNA'])} ({time.time() - t0:.0f}s)")
    er = (meta["subtype"] == "ER+").to_numpy()
    typ = meta["celltype_major"].astype(str).to_numpy()
    out = {}

    # 1-2. full depth, ten control-gene seeds
    full = [score(X, genes, s) for s in SEEDS]
    S0, G0, M0 = full[0]
    call0 = M0 > 0
    seurat = np.maximum(meta["S_score"], meta["G2M_score"]).to_numpy(float)
    out["vs Wu's Seurat scores"] = {
        "r S": float(np.corrcoef(S0, meta["S_score"])[0, 1]),
        "r G2M": float(np.corrcoef(G0, meta["G2M_score"])[0, 1]),
        "r max": float(np.corrcoef(M0, seurat)[0, 1]),
        "call kappa": kappa(call0, seurat > 0),
        "call rate scanpy / Seurat (ER+)": [float(call0[er].mean()), float((seurat > 0)[er].mean())]}
    Ms = np.array([f[2] for f in full])
    calls = Ms > 0
    flip = calls.any(0) & ~calls.all(0)
    out["control-gene seeds"] = {
        "seeds": len(SEEDS),
        "median per-cell SD of max score": float(np.median(Ms.std(0))),
        "median |max score|": float(np.median(np.abs(M0))),
        "cells whose call flips across seeds (ER+)": float(flip[er].mean()),
        "of cells called at seed 0, share that flip (ER+)": float(flip[er & call0].mean())}
    say(f"scored at full depth x {len(SEEDS)} seeds ({time.time() - t0:.0f}s)")

    # 3. split halves
    rng = np.random.default_rng(1)
    A = X.copy()
    A.data = rng.binomial(X.data.astype(np.int64), 0.5).astype(np.float32)
    B = X.copy()
    B.data = X.data - A.data
    A.eliminate_zeros()
    B.eliminate_zeros()
    _, _, MA = score(A, genes, 0)
    _, _, MB = score(B, genes, 0)
    del A, B
    ca, cb = MA > 0, MB > 0
    say(f"split halves scored ({time.time() - t0:.0f}s)")

    def rel(sel):
        r = float(np.corrcoef(MA[sel], MB[sel])[0, 1])
        return {"cells": int(sel.sum()), "median UMI": float(np.median(umi[sel])),
                "split-half r": r, "Spearman-Brown full-depth r": 2 * r / (1 + r),
                "split-half call kappa": kappa(ca[sel], cb[sel]), "split-half call MCC": mcc(ca[sel], cb[sel]),
                "half call vs full call MCC": mcc(ca[sel], call0[sel]),
                "call rate full": float(call0[sel].mean()), "call rate half": float(ca[sel].mean())}

    out["split half, ER+"] = rel(er)
    out["split half, all patients"] = rel(np.ones(len(er), bool))
    dec = pd.qcut(umi[er], 10, labels=False)
    by_depth = pd.DataFrame([{"depth decile": d + 1, **rel(_mask(er, dec, d))} for d in range(10)])
    by_type = pd.DataFrame([{"type": t, **rel(er & (typ == t))} for t in sorted(set(typ[er]))
                            if (er & (typ == t)).sum() >= 200])
    by_depth.to_csv(f"{OUT}/by_depth.csv", index=False)
    by_type.to_csv(f"{OUT}/by_type.csv", index=False)

    # 4. call rate by depth quintile within type (full depth, seed 0)
    rate_rows = []
    for t in SHOW:
        sel = er & (typ == t)
        q = pd.qcut(umi[sel], 5, labels=False)
        for k in range(5):
            m = q == k
            rate_rows.append({"type": t, "quintile": k + 1, "median UMI": float(np.median(umi[sel][m])),
                              "call rate": float(call0[sel][m].mean()),
                              "flip share": float(flip[sel][m].mean())})
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(f"{OUT}/call_rate_by_depth.csv", index=False)
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)

    figures(MA, MB, er, umi, typ, by_depth, rates, Ms, call0)
    write_report(out, by_depth, by_type, rates)
    say(json.dumps(out, indent=1))
    say(f"done ({(time.time() - t0) / 60:.1f} min)")


def _mask(er, dec, d):
    m = np.zeros(len(er), bool)
    m[np.where(er)[0][dec == d]] = True
    return m


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.25, lw=0.6)


def figures(MA, MB, er, umi, typ, by_depth, rates, Ms, call0):
    # split-half scatter
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = np.percentile(np.r_[MA[er], MB[er]], [0.5, 99.8])
    hb = ax.hexbin(MA[er], MB[er], gridsize=70, bins="log", cmap="Blues", extent=(*lim, *lim), mincnt=1)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    r = np.corrcoef(MA[er], MB[er])[0, 1]
    ax.set_title(f"Split-half repeatability of the S/G2M score (ER+ Wu)\nr = {r:.2f}, "
                 f"call kappa = {kappa(MA[er] > 0, MB[er] > 0):.2f}", fontsize=10.5)
    ax.set_xlabel("max(S, G2M) score, half A of each cell's reads")
    ax.set_ylabel("max(S, G2M) score, half B")
    fig.colorbar(hb, ax=ax, label="cells (log scale)")
    _style(ax)
    fig.savefig(f"{OUT}/split_half_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # repeatability by depth
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = by_depth["median UMI"]
    ax.plot(x, by_depth["split-half r"], "o-", color="#2563eb", lw=2, ms=6, label="score correlation, half vs half")
    ax.plot(x, by_depth["Spearman-Brown full-depth r"], "o--", color="#2563eb", lw=1.5, ms=5, alpha=0.6,
            label="score reliability at full depth (Spearman-Brown)")
    ax.plot(x, by_depth["split-half call kappa"], "s-", color="#c2410c", lw=2, ms=6, label="call agreement (kappa)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("median UMI per cell (depth decile, ER+)")
    ax.set_ylabel("agreement")
    ax.set_title("Repeatability rises with depth; the call stays unreliable", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    fig.savefig(f"{OUT}/repeatability_by_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # call rate by depth within type
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for t in SHOW:
        d = rates[rates["type"] == t]
        ax.plot(d["median UMI"], 100 * d["call rate"], "o-", color=COL[t], lw=2, ms=5)
        ax.annotate(t, (d["median UMI"].iloc[-1], 100 * d["call rate"].iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("median UMI per cell (depth quintile within type, ER+)")
    ax.set_ylabel("called S/G2M (%)")
    ax.set_title("Shallow lymphocytes are called cycling most often", fontsize=10.5)
    _style(ax)
    fig.savefig(f"{OUT}/call_rate_by_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # control-gene seed noise against the score
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    m0 = Ms[0][er]
    flips = ((Ms > 0).any(0) & ~(Ms > 0).all(0))[er]
    bins = np.linspace(np.percentile(m0, 1), np.percentile(m0, 99), 40)
    idx = np.digitize(m0, bins)
    xs = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if (idx == i).sum() > 50]
    ys = [100 * flips[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 50]
    ax.plot(xs, ys, "o-", color="#0f766e", lw=2, ms=4)
    ax2 = ax.twinx()
    ax2.hist(m0, bins=bins, color="0.8", alpha=0.6)
    ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("max(S, G2M) score at full depth (seed 0); grey: distribution")
    ax.set_ylabel(f"call flips across {Ms.shape[0]} control-gene seeds (%)")
    ax.set_title("No gap at the cut: cells near 0 flip with the control-gene draw", fontsize=10.5)
    _style(ax)
    fig.savefig(f"{OUT}/control_seed_flips.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(out, by_depth, by_type, rates):
    def tab(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)
    keep = ["median UMI", "split-half r", "Spearman-Brown full-depth r", "split-half call kappa",
            "half call vs full call MCC", "call rate full"]
    txt = ["# Noise in Wu's whole-transcriptome S/G2M call (scanpy)\n",
           "Generated by `run/division_wu_score_noise.py`.\n",
           "```json\n" + json.dumps(out, indent=1) + "\n```\n",
           "## Split-half repeatability by depth decile (ER+)\n", tab(by_depth[["depth decile"] + keep]), "",
           "## By cell type (ER+)\n", tab(by_type[["type", "cells"] + keep]), "",
           "## Call rate by depth quintile within type (ER+)\n", tab(rates), "",
           "![split half](split_half_scatter.png)\n![by depth](repeatability_by_depth.png)\n"
           "![call rate](call_rate_by_depth.png)\n![seeds](control_seed_flips.png)\n"]
    open(f"{OUT}/report.md", "w").write("\n".join(txt))


if __name__ == "__main__":
    main()
