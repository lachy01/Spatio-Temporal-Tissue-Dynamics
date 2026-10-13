"""Division classifier, addendum stage A6: depth diagnostics against the Wu reference.

    python run/division_addendum_e.py [--smoke]        # --smoke: few cells, to a scratch folder

Addendum section 5, real data only. Wu et al. 2021 (GSE176078, 100,064 cells,
26 tumours, raw UMI counts) against this Xenium section.

  E1   per-gene sensitivity: per type and shared gene, mean counts here over
       mean counts in Wu, for technical genes (shared, not Tirosh, not
       blocklist) and the 53 Tirosh genes; transcripts per cell over shared
       technical genes; all patients and ER+. Intervals pair a Xenium spatial
       block bootstrap with a Wu patient bootstrap, 1,000 replicates each.
  V15  binomial thinning (blocking).
  V16  pipeline parity: the per-dataset code path run on this section's own
       counts reproduces Stage 1's SCORE label and a plain-N BASE-T (blocking).
  E2   three conditions on the same Wu cells -- (i) whole transcriptome, (ii)
       panel genes at full depth, (iii) panel genes thinned to this section's
       depth -- with a depth ladder (x1, 2, 5, 10, 25; 5 seeds) and a
       gene-specific thinning variant; SCORE, BASE-T (plain log N offset,
       alpha_B 0.05) and the constant-pi Poisson mixture, scored against
       Seurat's whole-transcriptome phase calls (primary) and a stricter split.
  E3   the S/G2M against G1 composite rate ratio in Wu, at full and thinned
       depth, and the constant-pi mixture's ratio on thinned data, against
       MIX on this section.

Two departures from the addendum's letter, recorded in the report:
- V15's 1% tolerance on per-gene totals cannot hold for a correct binomial
  thinning at 1,000 counts (the thinned total's SD is then 9.5% of its mean).
  It is judged on genes with enough counts for 1% to be four SDs, and by a
  4-SD binomial check on every gene with >= 1,000 counts; the literal result
  is reported beside them.
- MIX's "e^delta ~ 1.9" in E3 is e^delta in this project's parameterisation,
  where the component rate ratio is exp(e^delta) = 6.70. Ratios are compared
  on the rate-ratio scale, with both numbers shown.

Writes depth_sensitivity.csv, thinning_ladder.csv, thinning_ladder.png,
wu_rate_ratio.csv and report sections, in outputs/division_classifier/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.division import labels as L                     # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
WU_DIR = "data/wu2021/Wu_etal_2021_BRCA_scRNASeq"
WU_CACHE = "data/wu2021/wu_counts_csr.npz"
ALIASES = {"MLF1IP": "CENPU", "FAM64A": "PIMREG", "HN1": "JPT1"}   # as wu2021_prepare.py
LADDER = (1, 2, 5, 10, 25)
SEEDS = (0, 1, 2, 3, 4)
N_MIN = 10                    # the analysis-cell filter used on this section
N_BOOT = 1000
MIX_E_DELTA, MIX_RATIO = 1.9020, float(np.exp(1.9020))     # production MIX on this section
COL = {"SCORE": "#2a78d6", "BASE-T": "#eb6834", "mixture w": "#1baf7a"}
INK2, MUTED, SURF = "#52514e", "#898781", "#fcfcfb"


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.strip("\n") + "\n")


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def stop(test, detail):
    report(f"\n### STOPPED in addendum stage A6: {test} failed\n\n{detail}\n\nPer the addendum's "
           "section 7, nothing further was run.\n")
    sys.exit(f"{test} failed: {detail}")


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------
def load_wu():
    """The Wu raw count matrix (cells x genes, CSR) and its gene names, cached."""
    genes = pd.read_csv(f"{WU_DIR}/count_matrix_genes.tsv", header=None)[0].astype(str).to_numpy()
    if os.path.exists(WU_CACHE):
        return sp.load_npz(WU_CACHE).tocsr(), genes
    import pyarrow.csv as pacsv
    with open(f"{WU_DIR}/count_matrix_sparse.mtx") as fh:
        fh.readline()
        n_g, n_c, nnz = (int(v) for v in fh.readline().split())
    tab = pacsv.read_csv(f"{WU_DIR}/count_matrix_sparse.mtx",
                         read_options=pacsv.ReadOptions(skip_rows=2, autogenerate_column_names=True),
                         parse_options=pacsv.ParseOptions(delimiter=" "),
                         convert_options=pacsv.ConvertOptions(column_types={"f0": "int32", "f1": "int32",
                                                                           "f2": "float32"}))
    X = sp.csr_matrix((tab.column("f2").to_numpy(),
                       (tab.column("f1").to_numpy() - 1, tab.column("f0").to_numpy() - 1)),
                      shape=(n_c, n_g), dtype=np.float32)
    del tab
    if X.nnz != nnz:
        sys.exit("Wu matrix: entry count does not match its header")
    sp.save_npz(WU_CACHE, X, compressed=False)
    return X, genes


def load_xenium(cfg, ids):
    """This section's raw counts for the analysis cells, all panel genes."""
    import scanpy as sc
    ad = sc.read_10x_h5(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", gex_only=True)
    ad.var_names_make_unique()
    ad = ad[ids].copy()
    return ad.X.tocsr().astype(np.float32), np.asarray(ad.var_names).astype(str)


# ------------------------------------------------------------------------------------
# the per-dataset pipeline (main spec 2.2, 2.4; addendum 3.1 step 1)
# ------------------------------------------------------------------------------------
def const_pi_mixture(y, N, starts=(2.0, 4.0, 8.0), tol=1e-12, max_iter=20000):
    """Constant-pi two-component Poisson mixture, rates N e^b0 and r N e^b0, by EM
    (closed-form M-step). The best of a few starts. Returns pi, ratio r, e_delta =
    log r, the posterior w and the log-likelihood."""
    from scipy.special import expit, gammaln
    y, N = np.asarray(y, float), np.asarray(N, float)
    const = float(np.sum(gammaln(y + 1)))
    best = None
    for r in starts:
        pi, b = 0.1, float(y.sum() / N.sum()) / (0.9 + 0.1 * r)
        prev = -np.inf
        for it in range(max_iter):
            l0 = N * b
            lp0 = np.log1p(-pi) + y * np.log(l0) - l0
            lp1 = np.log(pi) + y * np.log(r * l0) - r * l0
            ll = float(np.sum(np.logaddexp(lp0, lp1))) - const
            w = expit(lp1 - lp0)
            pi = float(np.clip(w.mean(), 1e-9, 1 - 1e-9))
            b = float(np.sum((1 - w) * y) / np.sum((1 - w) * N))
            r = max(float(np.sum(w * y) / np.sum(w * N)) / b, 1.0 + 1e-9)
            if abs(ll - prev) < tol * abs(ll):
                break
            prev = ll
        if best is None or ll > best["loglik"]:
            best = dict(pi=pi, ratio=r, e_delta=float(np.log(r)), b=b, w=w, loglik=ll, iters=it + 1)
    return best


def methods(X, genes, tirosh, block, n_min=N_MIN, seed=0):
    """SCORE, BASE-T (plain log N offset, alpha_B 0.05) and the constant-pi mixture
    on a count matrix, built as Stage 1 builds them from this section's matrix:
    y = Tirosh composite, N = library without the Tirosh and blocklist genes, cells
    with N >= n_min kept."""
    gi = {g: i for i, g in enumerate(genes)}
    tir = [g for g in tirosh if g in gi]
    blk = [g for g in block if g in gi]
    excl = sorted({gi[g] for g in tir} | {gi[g] for g in blk})
    X = X.tocsr()
    y = np.asarray(X[:, [gi[g] for g in tir]].sum(1)).ravel()
    N = np.asarray(X.sum(1)).ravel() - np.asarray(X[:, excl].sum(1)).ravel()
    keep = N >= n_min
    yk, Nk = y[keep], N[keep]
    score, info = L.module_score(X[keep], genes, Nk, tir, blk, seed=seed)
    thr = L.score_threshold(score)
    null = L.robust_null(yk, np.ones((len(yk), 1)), np.log(Nk))
    Lt, _c, _ai, p = L.base_t(yk, null["mu"], null["alpha"], 0.05)
    mix = const_pi_mixture(yk, Nk)
    return dict(keep=keep, y=yk, N=Nk, depth=np.asarray(X.sum(1)).ravel()[keep],
                stat={"SCORE": score, "BASE-T": -np.log(np.clip(p, 1e-300, None)), "mixture w": mix["w"]},
                call={"SCORE": score > thr["threshold"], "BASE-T": Lt, "mixture w": mix["w"] > 0.5},
                mix=mix, null_alpha=float(null["alpha"]), score_threshold=float(thr["threshold"]),
                n_tirosh=len(tir), n_block=len(blk), n_pool=info["n_pool"])


# ------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------
def auroc(stat, truth):
    from scipy.stats import rankdata
    pos, neg = truth == 1, truth == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([stat[pos], stat[neg]]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_rows(res, truths, wtype, types, tag):
    """Every metric for one pipeline run: per truth, method and stratum."""
    from scipy.stats import spearmanr
    rows = []
    kt = wtype[res["keep"]]
    for tname, tr_full in truths.items():
        tr = tr_full[res["keep"]]          # 1 / 0 / -1 (excluded)
        ok = tr >= 0
        truth_rate = {t: float(np.mean(tr[ok & (kt == t)] == 1)) for t in types}
        for meth in res["stat"]:
            s, c = res["stat"][meth], res["call"][meth]
            rates = {t: float(np.mean(c[kt == t])) for t in types}
            rho = spearmanr([rates[t] for t in types], [truth_rate[t] for t in types]).statistic
            for strat in ["all"] + list(types):
                m = ok & ((kt == strat) if strat != "all" else True)
                tt, cc = tr[m] == 1, c[m]
                kap = L.cohen_kappa(cc, tt) if m.sum() else np.nan
                rows.append({**tag, "truth": tname, "method": meth, "stratum": strat,
                             "cells": int(m.sum()), "AUROC": auroc(s[m], tr[m]),
                             "kappa": kap, "sensitivity": float(cc[tt].mean()) if tt.any() else np.nan,
                             "specificity": float(1 - cc[~tt].mean()) if (~tt).any() else np.nan,
                             "call rate": float(c[(kt == strat) if strat != "all" else slice(None)].mean()),
                             "calls with composite <= 1": float(np.mean(res["y"][c] <= 1)) if c.any() else np.nan,
                             "Spearman, type call rates vs truth": rho if strat == "all" else np.nan,
                             "median depth": float(np.median(res["depth"]))})
    return rows


def thin(X, p_cell, rng, p_gene=None):
    """Binomial thinning of a CSR count matrix: entry (i, g) kept with probability
    p_i (times p_g / per-cell renormaliser when gene-specific)."""
    X = X.tocsr()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    p = p_cell[rows] if p_gene is None else np.clip(p_gene[0][rows] * p_gene[1][X.indices], 0, 1)
    data = rng.binomial(X.data.astype(np.int64), p).astype(np.float32)
    out = sp.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


# ------------------------------------------------------------------------------------
def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", default="/tmp/division_e_smoke")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "y", "N", "LS"])
    fm = pd.read_parquet(f"{OUT}/folds.parquet", columns=["cell_id", "block"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    wu = pd.read_parquet("data/wu2021/wu_cells.parquet")
    if a.smoke:
        rs = np.random.default_rng(0)
        xs = np.sort(rs.choice(len(lab), 20000, replace=False))
        lab, fm = lab.iloc[xs].reset_index(drop=True), fm.iloc[xs].reset_index(drop=True)
        OUT, REPORT = a.smoke_dir, f"{a.smoke_dir}/report.md"
        os.makedirs(OUT, exist_ok=True)
    if not np.array_equal(lab["cell_id"].astype(str).to_numpy(), fm["cell_id"].astype(str).to_numpy()):
        sys.exit("labels and folds list cells in different orders")

    # ---- data, genes ------------------------------------------------------------------
    say("reading Wu ...")
    WX, wgenes = load_wu()
    barcodes = pd.read_csv(f"{WU_DIR}/count_matrix_barcodes.tsv", header=None)[0].astype(str).to_numpy()
    if not np.array_equal(barcodes, wu["barcode"].astype(str).to_numpy()):
        sys.exit("wu_cells.parquet is not in the matrix's cell order")
    if a.smoke:
        ws = np.sort(np.random.default_rng(1).choice(len(wu), 12000, replace=False))
        WX, wu = WX[ws], wu.iloc[ws].reset_index(drop=True)
    say("reading this section ...")
    XX, xgenes = load_xenium(cfg, lab["cell_id"].astype(str).to_numpy())
    wtype, xtype = wu["celltype_major"].astype(str).to_numpy(), lab["cell_type"].astype(str).to_numpy()
    wpat = wu["patient"].astype(str).to_numpy()
    er = (wu["subtype"] == "ER+").to_numpy()

    tirosh_x = list(meta["tirosh_genes"])
    block_x = [g for g in dict.fromkeys(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str))]
    wset = set(wgenes)
    # Xenium name -> Wu name (exact, else the Tirosh aliases either way)
    inv = {v: k for k, v in ALIASES.items()}
    x2w = {}
    for g in xgenes:
        if g in wset:
            x2w[g] = g
        elif ALIASES.get(g) in wset:
            x2w[g] = ALIASES[g]
        elif inv.get(g) in wset:
            x2w[g] = inv[g]
    shared_x = [g for g in xgenes if g in x2w]
    tir_sh = [g for g in tirosh_x if g in x2w]
    blk_sh = [g for g in block_x if g in x2w and g not in set(tirosh_x)]
    tech_sh = [g for g in shared_x if g not in set(tirosh_x) and g not in set(block_x)]
    say(f"shared genes {len(shared_x):,}: Tirosh {len(tir_sh)}/{len(tirosh_x)}, other blocklist "
        f"{len(blk_sh)}, technical {len(tech_sh)}")
    xi = {g: i for i, g in enumerate(xgenes)}
    wi = {g: i for i, g in enumerate(wgenes)}
    # Wu restricted to the shared panel, columns renamed to the Xenium names
    WP = WX[:, [wi[x2w[g]] for g in shared_x]].tocsr()
    XP = XX[:, [xi[g] for g in shared_x]].tocsr()

    # ================================ E1 ==================================================
    say("E1 ...")
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb = int(bidx.max()) + 1
    rng = np.random.default_rng(20261013)
    Mx = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]).astype(float)
    pats, pidx = np.unique(wpat, return_inverse=True)
    npat = len(pats)
    Mw = np.stack([np.bincount(rng.integers(0, npat, npat), minlength=npat) for _ in range(N_BOOT)]).astype(float)
    groups = {"technical": [shared_x.index(g) for g in tech_sh], "Tirosh": [shared_x.index(g) for g in tir_sh]}
    tech_cols = groups["technical"]
    x_tot = np.asarray(XP[:, tech_cols].sum(1)).ravel()
    w_tot = np.asarray(WP[:, tech_cols].sum(1)).ravel()
    x_det = np.asarray((XP[:, tech_cols] > 0).sum(1)).ravel()
    w_det = np.asarray((WP[:, tech_cols] > 0).sum(1)).ravel()

    def unit_sums(X, unit, nunit, cells):
        """per unit (block or patient), the gene sums and cell counts over `cells`."""
        D = sp.csr_matrix((np.ones(cells.sum()), (unit[cells], np.arange(cells.sum()))), shape=(nunit, cells.sum()))
        return np.asarray((D @ X[cells]).todense()), np.bincount(unit[cells], minlength=nunit).astype(float)

    def wmedian(v, unit, mult):
        """median of v with each cell weighted by its unit's bootstrap multiplicity"""
        o = np.argsort(v)
        vs, us = v[o], unit[o]
        out = np.empty(len(mult))
        for r in range(len(mult)):
            cw = np.cumsum(mult[r][us])
            out[r] = vs[np.searchsorted(cw, cw[-1] / 2.0)]
        return out

    e1_rows, e1_gene = [], []
    for scope, wsel in (("all patients", np.ones(len(wu), bool)), ("ER+ patients", er)):
        for t in types:
            xc, wc = xtype == t, (wtype == t) & wsel
            if wc.sum() < 20:
                continue
            Sx, nx = unit_sums(XP, bidx, nb, xc)
            Sw, nw = unit_sums(WP, pidx, npat, wc)
            mx, mw = Sx.sum(0) / nx.sum(), Sw.sum(0) / nw.sum()
            mxb = (Mx @ Sx) / (Mx @ nx)[:, None]
            mwb = (Mw @ Sw) / (Mw @ nw)[:, None]
            detx = np.asarray((XP[xc] > 0).mean(0)).ravel()
            detw = np.asarray((WP[wc] > 0).mean(0)).ravel()
            row = {"scope": scope, "type": t, "cells here": int(xc.sum()), "cells in Wu": int(wc.sum())}
            for gname, cols in groups.items():
                ok = [j for j in cols if mw[j] > 0]
                s = mx[ok] / mw[ok]
                with np.errstate(divide="ignore", invalid="ignore"):
                    sb = mxb[:, ok] / mwb[:, ok]
                medb = np.nanmedian(np.where(np.isfinite(sb), sb, np.nan), axis=1)
                row[f"median s_gk, {gname}"] = float(np.median(s))
                row[f"s_gk IQR, {gname}"] = "%.3g-%.3g" % tuple(np.percentile(s, [25, 75]))
                row[f"median s_gk, {gname}, 95% CI"] = "%.3g-%.3g" % tuple(np.percentile(medb, [2.5, 97.5]))
                row[f"median detection here / Wu, {gname}"] = float(np.median(detx[ok] / np.maximum(detw[ok], 1e-12)))
                if scope == "all patients":
                    for j in ok:
                        e1_gene.append({"type": t, "gene": shared_x[j], "group": gname, "mean here": mx[j],
                                        "mean Wu": mw[j], "s_gk": mx[j] / mw[j],
                                        "detection here": detx[j], "detection Wu": detw[j]})
            mxt, mwt = float(np.median(x_tot[xc])), float(np.median(w_tot[wc]))
            bx = wmedian(x_tot[xc], bidx[xc], Mx)
            bw = wmedian(w_tot[wc], pidx[wc], Mw)
            row.update({"median transcripts/cell, technical, here": mxt,
                        "median transcripts/cell, technical, Wu": mwt,
                        "ratio here/Wu": mxt / mwt,
                        "ratio 95% CI": "%.3g-%.3g" % tuple(np.percentile(bx / bw, [2.5, 97.5])),
                        "median genes/cell, technical, here": float(np.median(x_det[xc])),
                        "median genes/cell, technical, Wu": float(np.median(w_det[wc]))})
            e1_rows.append(row)
    e1 = pd.DataFrame(e1_rows)
    e1_genes = pd.DataFrame(e1_gene)
    e1.to_csv(f"{OUT}/depth_sensitivity.csv", index=False)
    e1_genes.to_csv(f"{OUT}/depth_sensitivity_genes.csv", index=False)
    # pooled technical-gene sensitivity (for the gene-specific thinning)
    s_pool = np.full(len(shared_x), np.nan)
    mx_all = np.asarray(XP.mean(0)).ravel()
    mw_all = np.asarray(WP.mean(0)).ravel()
    tech_ok = [j for j in tech_cols if mw_all[j] > 0 and mx_all[j] > 0]
    s_pool[tech_ok] = mx_all[tech_ok] / mw_all[tech_ok]
    bins = pd.qcut(np.log(mw_all + 1e-12), 25, labels=False, duplicates="drop")
    for j in groups["Tirosh"]:
        same = [k for k in tech_ok if bins[k] == bins[j]]
        s_pool[j] = float(np.median(s_pool[same])) if same else float(np.nanmedian(s_pool[tech_ok]))
    s_pool = np.where(np.isfinite(s_pool), s_pool, float(np.nanmedian(s_pool[tech_ok])))

    show = e1[e1["scope"] == "all patients"][["type", "median transcripts/cell, technical, here",
                                              "median transcripts/cell, technical, Wu", "ratio here/Wu",
                                              "ratio 95% CI", "median s_gk, technical",
                                              "median s_gk, technical, 95% CI", "median s_gk, Tirosh",
                                              "median s_gk, Tirosh, 95% CI"]]
    show_er = e1[e1["scope"] == "ER+ patients"][["type", "ratio here/Wu", "median s_gk, technical",
                                                 "median s_gk, Tirosh"]]
    ratio_min = float(1 / e1.loc[e1["scope"] == "all patients", "ratio here/Wu"].max())
    tir_vs_tech = (e1["median s_gk, Tirosh"] / e1["median s_gk, technical"])[e1["scope"] == "all patients"]
    x16a = bool((e1.loc[e1["scope"] == "all patients", "ratio here/Wu"] <= 0.1).all())
    x16b = bool(((tir_vs_tech >= 0.5) & (tir_vs_tech <= 2)).all())
    report("\n".join([
        f"\n## Addendum stage A6 -- depth diagnostics against Wu et al. ({time.strftime('%Y-%m-%d %H:%M')})\n",
        "### E1 -- per-gene sensitivity of this section\n",
        f"Shared genes (this panel and Wu's transcriptome): {len(shared_x):,} -- the {len(tir_sh)} Tirosh "
        f"genes, {len(blk_sh)} other blocklist genes and {len(tech_sh):,} technical genes. Means per cell "
        "of each type; s_gk = mean here / mean in Wu. Intervals pair a 1,000-replicate spatial block "
        "bootstrap here with a 1,000-replicate patient bootstrap in Wu.\n",
        md_table(show), "",
        "ER+ patients only (the closest Wu subtype):\n", md_table(show_er), "",
        "Every type, scope and gene group, with detection rates and genes per cell, is in "
        "`depth_sensitivity.csv`; every gene in `depth_sensitivity_genes.csv`.\n",
        "Caveats. The Tirosh ratios mix sensitivity with biology: they also carry differences in "
        "cycling fraction between this tumour and Wu's patients, so read them against the technical "
        "genes. The ratio includes section geometry: a 5 um section captures part of each cell's "
        "volume, which is part of this section's sensitivity, not an artefact to remove.\n",
        f"**Against Ren et al. (2025)**, who found transcripts and genes per cell comparable to "
        f"scRNA-seq on shared genes: this section captures at most {100 / ratio_min:.1f}% of Wu's "
        f"transcripts per cell on shared technical genes in any type (Wu is at least {ratio_min:.1f} "
        f"times higher), so it {'does not agree' if ratio_min > 2 else 'agrees'}.\n",
        f"- **X16** (at least 10 times fewer transcripts per cell here in every type; Tirosh median "
        f"sensitivity within 2 times of the technical genes'): first clause **{'holds' if x16a else 'does not hold'}**; "
        f"second clause **{'holds' if x16b else 'does not hold'}** (Tirosh over technical median ratio "
        f"{tir_vs_tech.min():.2f}-{tir_vs_tech.max():.2f} across types). Overall: "
        f"**{'holds' if x16a and x16b else 'partly holds' if x16a or x16b else 'does not hold'}**.\n"]))
    say(show.to_string(index=False))

    # ================================ V15 ==================================================
    say("V15 ...")
    vr = np.random.default_rng(7)
    sub = np.sort(vr.choice(WP.shape[0], min(20000, WP.shape[0]), replace=False))
    T0 = np.asarray(WX[sub].sum(0)).ravel()
    T1 = np.asarray(thin(WX[sub], np.full(len(sub), 0.1), vr).sum(0)).ravel()
    big = T0 >= 1000
    rel = np.abs(T1[big] / (0.1 * T0[big]) - 1)
    z = (T1[big] - 0.1 * T0[big]) / np.sqrt(0.09 * T0[big])
    huge = T0[big] >= 1.44e6
    literal_ok = bool((rel <= 0.01).all())
    huge_ok = bool((rel[huge] <= 0.01).all()) if huge.any() else True
    z_ok = bool((np.abs(z) <= 4.5).all())
    ident = thin(WP, np.ones(WP.shape[0]), vr)
    exact = bool((ident != WP).nnz == 0)
    v15_ok = huge_ok and z_ok and exact
    report("\n".join([
        "### V15 -- thinning\n",
        f"Whole Wu transcriptome, 20,000 random cells, every entry thinned at p = 0.1. Of "
        f"{int(big.sum()):,} genes with at least 1,000 counts: largest relative error "
        f"{rel.max():.4f}, literal 1% criterion **{'met' if literal_ok else 'not met'}**. A correct "
        "binomial thinning cannot meet it at 1,000 counts, where the thinned total's SD is 9.5% of "
        f"its mean. Judged instead: on the {int(huge.sum())} genes with at least 1.44 million counts "
        f"(where 1% is four SDs) the largest error is "
        f"{(rel[huge].max() if huge.any() else float('nan')):.4f} (**{'pass' if huge_ok else 'FAIL'}**), "
        f"and the largest |z| over every gene with >= 1,000 counts is {np.abs(z).max():.2f} against "
        f"4.5 (**{'pass' if z_ok else 'FAIL'}**). With p = 1 for every cell the thinned panel matrix "
        f"equals condition (ii) exactly: **{'pass' if exact else 'FAIL'}**. V15: "
        f"**{'pass' if v15_ok else 'FAIL'}**.\n"]))
    if not v15_ok:
        stop("V15 (thinning)", f"huge-gene 1% {huge_ok}, z {z_ok}, exact at p = 1 {exact}")

    # ================================ V16 ==================================================
    say("V16 ...")
    rx = methods(XX, xgenes, tirosh_x, block_x)
    yl, Nl = lab["y"].to_numpy(float), lab["N"].to_numpy(float)
    same_yN = bool(rx["keep"].all() and np.array_equal(rx["y"], yl) and np.array_equal(rx["N"], Nl))
    ls_same = bool(np.array_equal(rx["call"]["SCORE"], lab["LS"].to_numpy(bool))) if not a.smoke else None
    nullp = L.robust_null(yl, np.ones((len(yl), 1)), np.log(Nl))
    LTp = L.base_t(yl, nullp["mu"], nullp["alpha"], 0.05)[0]
    bt_same = bool(np.array_equal(rx["call"]["BASE-T"], LTp))
    v16_ok = same_yN and bt_same and (ls_same if ls_same is not None else True)
    report("\n".join([
        "### V16 -- pipeline parity on this section's own counts\n",
        f"The code path run on thinned Wu, run instead on this section's raw counts (every panel gene): "
        f"y and N equal Stage 1's for every cell: **{same_yN}**; SCORE label equals Stage 1's LS: "
        f"**{ls_same if ls_same is not None else 'not checked (smoke run on a subsample)'}**; BASE-T at "
        f"alpha_B 0.05 equals a plain-N rebuild (robust null with a log N offset, no MisTIC term): "
        f"**{bt_same}**. V16: **{'pass' if v16_ok else 'FAIL'}**.\n"]))
    if not v16_ok:
        stop("V16 (pipeline parity)", f"y,N {same_yN}; SCORE {ls_same}; BASE-T {bt_same}")

    # ================================ E2 ==================================================
    truths = {}
    ph = wu["phase"].astype(str).to_numpy()
    cyc = np.isin(ph, ["S", "G2M"])
    truths["primary"] = cyc.astype(int)
    mx_sc = np.maximum(wu["S_score"].to_numpy(float), wu["G2M_score"].to_numpy(float))
    q75 = pd.Series(mx_sc).groupby(wtype).transform(lambda v: v.quantile(0.75)).to_numpy()
    strict = np.full(len(wu), -1)
    strict[cyc & (mx_sc > q75)] = 1
    strict[(ph == "G1") & (wu["S_score"].to_numpy() < 0) & (wu["G2M_score"].to_numpy() < 0)] = 0
    truths["stricter"] = strict

    rows, runs = [], {}
    say("E2 (i): whole transcriptome ...")
    wg = [str(g) for g in wgenes]
    tir_w = [x2w[g] for g in tir_sh]
    blk_w = [x2w.get(g, g) for g in block_x if x2w.get(g, g) in wset]
    runs["i"] = methods(WX, wg, tir_w, blk_w)
    rows += score_rows(runs["i"], truths, wtype, types, {"condition": "(i) whole transcriptome", "factor": np.nan, "seed": -1, "thinning": "none"})
    say("E2 (ii): panel genes, full depth ...")
    runs["ii"] = methods(WP, shared_x, tir_sh, block_x)
    rows += score_rows(runs["ii"], truths, wtype, types, {"condition": "(ii) panel, full depth", "factor": np.nan, "seed": -1, "thinning": "none"})

    # quantile-matched targets: Wu depth over the shared panel, matched within type to
    # this section's depth over the same genes
    wdepth = np.asarray(WP.sum(1)).ravel()
    xdepth = np.asarray(XP.sum(1)).ravel()
    target = np.zeros(len(wu))
    for t in types:
        wc, xc = wtype == t, xtype == t
        q = (pd.Series(wdepth[wc]).rank(method="average").to_numpy() - 0.5) / wc.sum()
        target[wc] = np.quantile(xdepth[xc], q)
    gene_spec = None
    for f in LADDER:
        for sd in SEEDS:
            say(f"E2 (iii): ladder x{f}, seed {sd} ...")
            p_i = np.minimum(1.0, f * target / np.maximum(wdepth, 1e-12))
            Wt = thin(WP, p_i, np.random.default_rng(1000 * f + sd))
            res = methods(Wt, shared_x, tir_sh, block_x)
            rows += score_rows(res, truths, wtype, types, {"condition": "(iii) panel, thinned", "factor": f, "seed": sd, "thinning": "quantile-matched"})
            if f == 1:
                runs.setdefault("iii_1", []).append(res)
    for sd in SEEDS:
        say(f"E2 (iii): gene-specific thinning, seed {sd} ...")
        p_i = np.minimum(1.0, target / np.maximum(wdepth, 1e-12))
        sbar = (WP @ s_pool) / np.maximum(wdepth, 1e-12)      # each cell's depth-weighted mean s_g
        gene_spec = (p_i / np.where(sbar > 0, sbar, 1.0), s_pool)
        Wt = thin(WP, p_i, np.random.default_rng(9000 + sd), p_gene=gene_spec)
        res = methods(Wt, shared_x, tir_sh, block_x)
        rows += score_rows(res, truths, wtype, types, {"condition": "(iii) panel, thinned", "factor": 1, "seed": sd, "thinning": "gene-specific"})
    lad = pd.DataFrame(rows)
    lad.to_csv(f"{OUT}/thinning_ladder.csv", index=False)

    # ---- figure: AUROC against median depth ------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = lad[(lad["truth"] == "primary") & (lad["stratum"] == "all")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for meth, colr in COL.items():
        q = A[(A["method"] == meth) & (A["thinning"] == "quantile-matched")].groupby("factor").agg(
            depth=("median depth", "mean"), auc=("AUROC", "mean"), lo=("AUROC", "min"), hi=("AUROC", "max"))
        ax.plot(q["depth"], q["auc"], "-o", color=colr, lw=2, ms=8, mec=SURF, mew=2, zorder=3)
        ax.vlines(q["depth"], q["lo"], q["hi"], color=colr, lw=2, zorder=2)
        ax.annotate(meth, (q["depth"].iloc[-1], q["auc"].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
        for cond, ls in (("(ii) panel, full depth", "--"), ("(i) whole transcriptome", ":")):
            v = A[(A["method"] == meth) & (A["condition"] == cond)]["AUROC"]
            if len(v):
                ax.axhline(float(v.iloc[0]), color=colr, lw=1.2, ls=ls, alpha=.85, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("median transcripts per cell over the shared panel genes (log scale)", color=INK2)
    ax.set_ylabel("AUROC against Seurat S/G2M calls", color=INK2)
    ax.set_title("Wu cells thinned to this section's depth and multiples of it\n"
                 "dashed: panel genes at full depth (ii);  dotted: whole transcriptome (i)",
                 fontsize=10, color="#0b0b0b")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e8e7e4", lw=0.8)
    xt = A[A["thinning"] == "quantile-matched"].groupby("factor")["median depth"].mean()
    for f, d in xt.items():
        ax.annotate(f"x{int(f)}", (d, ax.get_ylim()[0]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{OUT}/thinning_ladder.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---- E2 report ----------------------------------------------------------------------------
    def summ(df):
        return df.groupby(["condition", "thinning", "factor", "method"], dropna=False).agg(
            **{"median depth": ("median depth", "mean"), "AUROC": ("AUROC", "mean"),
               "kappa": ("kappa", "mean"), "sensitivity": ("sensitivity", "mean"),
               "specificity": ("specificity", "mean"), "call rate": ("call rate", "mean"),
               "calls with y <= 1": ("calls with composite <= 1", "mean"),
               "Spearman (type rates)": ("Spearman, type call rates vs truth", "mean")}).reset_index()
    P = summ(A)
    S = summ(lad[(lad["truth"] == "stricter") & (lad["stratum"] == "all")])
    l1 = A[(A["factor"] == 1) & (A["thinning"] == "quantile-matched")].groupby("method")["AUROC"].mean()
    ii = A[A["condition"] == "(ii) panel, full depth"].set_index("method")["AUROC"]
    by = lad[(lad["truth"] == "primary") & (lad["factor"] == 1) & (lad["thinning"] == "quantile-matched")
             & (lad["stratum"] != "all")].groupby(["stratum", "method"])["AUROC"].mean().unstack()
    x17a = bool(ii.get("SCORE", np.nan) > 0.9)
    x17b = bool(l1.get("SCORE", 1) < 0.75 and l1.get("BASE-T", 1) < 0.75)
    x17c = bool(l1.idxmax() == "mixture w")
    lq = A[A["thinning"] == "quantile-matched"].groupby(["method", "factor"])["AUROC"].mean().unstack()
    gain_1_10 = (lq[10] - lq[1])
    gain_10_25 = (lq[25] - lq[10])
    x18a = bool((gain_1_10 > 2 * gain_10_25).all())
    x18b = bool((((ii.reindex(lq.index) - lq[25]).abs()) < 0.02).all())
    report("\n".join([
        "### E2 -- thinning Wu to this section's depth\n",
        "Same Wu cells in every condition: (i) whole transcriptome, (ii) panel genes only at full "
        "depth, (iii) panel genes binomially thinned so each cell's depth matches the same quantile "
        "of this section's depth distribution within its type, times the ladder factor (5 seeds "
        "per level; mean shown). Every condition keeps cells with N >= 10, as this section does. "
        "Methods as on this section: SCORE (scanpy score_genes, pool = the condition's genes minus "
        "the blocklist; robust half-normal threshold), BASE-T (NB null with a plain log N offset, "
        "trimmed as in Stage 1, alpha_B 0.05; statistic -log p) and the constant-pi Poisson "
        "mixture (posterior w; MAP call w > 0.5). Truth: Seurat's whole-transcriptome S/G2M calls "
        "(primary); stricter: S/G2M cells above their type's 75th percentile of max(S, G2M) "
        "against G1 cells with both scores below 0.\n",
        "Primary truth, all cells:\n", md_table(P), "",
        "Stricter truth, all cells:\n", md_table(S), "",
        "AUROC by type at this section's depth (ladder x1, primary truth):\n",
        md_table(by.reset_index()), "",
        "Figure `thinning_ladder.png`; every condition, level, seed, method, truth and type in "
        "`thinning_ladder.csv`.\n",
        f"- **X17** (condition (ii) SCORE AUROC above 0.9; at x1 SCORE and BASE-T below 0.75 with w "
        f"highest): (ii) SCORE {ii.get('SCORE', np.nan):.3f} -> **{'yes' if x17a else 'no'}**; x1 "
        f"SCORE {l1.get('SCORE', np.nan):.3f}, BASE-T {l1.get('BASE-T', np.nan):.3f} -> "
        f"**{'yes' if x17b else 'no'}**; w {l1.get('mixture w', np.nan):.3f}, highest -> "
        f"**{'yes' if x17c else 'no'}**. Overall: **{'holds' if x17a and x17b and x17c else 'partly holds' if any((x17a, x17b, x17c)) else 'does not hold'}**.",
        f"- **X18** (AUROC rises steeply from x1 to x10, approaches (ii) by x25): gain x1->x10 "
        f"{', '.join(f'{m} {g:+.3f}' for m, g in gain_1_10.items())}, x10->x25 "
        f"{', '.join(f'{m} {g:+.3f}' for m, g in gain_10_25.items())} -> steep-then-flat "
        f"**{'yes' if x18a else 'no'}**; within 0.02 of (ii) at x25 **{'yes' if x18b else 'no'}**. "
        f"Overall: **{'holds' if x18a and x18b else 'partly holds' if x18a or x18b else 'does not hold'}**.\n"]))

    # ================================ E3 ==================================================
    say("E3 ...")

    def rate_ratio(y, N, grp, tr):
        """(sum y / sum N) in cycling-truth cells over the same in G1-truth cells"""
        pos, neg = tr == 1, tr == 0
        return (y[pos].sum() / N[pos].sum()) / (y[neg].sum() / N[neg].sum())

    e3 = []
    bt_rng = np.random.default_rng(31)
    MB = np.stack([np.bincount(bt_rng.integers(0, npat, npat), minlength=npat) for _ in range(N_BOOT)]).astype(float)

    def ratio_rows(res, label):
        out = []
        k = res["keep"]
        y, N, kt, kp = res["y"], res["N"], wtype[k], pidx[k]
        for tname, tr_full in truths.items():
            tr = tr_full[k]
            for strat in ["pooled"] + list(types):
                m = (kt == strat) if strat != "pooled" else np.ones(len(y), bool)
                if (tr[m] == 1).sum() < 5 or (tr[m] == 0).sum() < 5:
                    continue
                pt = rate_ratio(y[m], N[m], None, tr[m])
                # patient bootstrap: weight cells by their patient's multiplicity
                def comp(pos):
                    sel = m & pos
                    sy = np.bincount(kp[sel], weights=y[sel], minlength=npat)
                    sn = np.bincount(kp[sel], weights=N[sel], minlength=npat)
                    return (MB @ sy) / (MB @ sn)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rb = comp(tr == 1) / comp(tr == 0)
                lo, hi = np.nanpercentile(rb, [2.5, 97.5])
                out.append({"data": label, "truth": tname, "stratum": strat, "rate ratio": pt,
                            "95% CI": f"{lo:.3g}-{hi:.3g}"})
        return out
    e3 += ratio_rows(runs["ii"], "(ii) panel, full depth")
    for j, res in enumerate(runs["iii_1"]):
        e3 += [dict(r, data=f"(iii) thinned x1, seed {j}") for r in ratio_rows(res, "")]
    e3 = pd.DataFrame(e3)
    mixr = pd.DataFrame([{"data": f"(iii) thinned x1, seed {j}", "mixture pi": r["mix"]["pi"],
                          "mixture e^delta": r["mix"]["e_delta"], "mixture rate ratio": r["mix"]["ratio"]}
                         for j, r in enumerate(runs["iii_1"])]
                        + [{"data": "(ii) panel, full depth", "mixture pi": runs["ii"]["mix"]["pi"],
                            "mixture e^delta": runs["ii"]["mix"]["e_delta"],
                            "mixture rate ratio": runs["ii"]["mix"]["ratio"]}])
    e3.to_csv(f"{OUT}/wu_rate_ratio.csv", index=False)
    mixr.to_csv(f"{OUT}/wu_rate_ratio_mixture.csv", index=False)
    pooled = e3[e3["stratum"] == "pooled"].copy()
    pooled["data"] = pooled["data"].str.replace(r", seed \d", "", regex=True)
    pooled_s = pooled.groupby(["data", "truth"])["rate ratio"].agg(["mean", "min", "max"]).reset_index()
    full_strict = float(e3[(e3["data"] == "(ii) panel, full depth") & (e3["truth"] == "stricter")
                           & (e3["stratum"] == "pooled")]["rate ratio"].iloc[0])
    full_prim = float(e3[(e3["data"] == "(ii) panel, full depth") & (e3["truth"] == "primary")
                         & (e3["stratum"] == "pooled")]["rate ratio"].iloc[0])
    thin_mix = float(mixr[mixr["data"].str.startswith("(iii)")]["mixture rate ratio"].mean())
    x19a = full_strict >= 4
    closer_wu = abs(thin_mix - full_strict) < abs(thin_mix - 1.9)
    closer_wu_ratio = abs(thin_mix - full_strict) < abs(thin_mix - MIX_RATIO)
    report("\n".join([
        "### E3 -- the cycling rate ratio in Wu\n",
        "Tirosh composite per panel library (sum y / sum N) in S/G2M-called cells over G1-called "
        "cells, pooled and per type, with a 1,000-replicate patient bootstrap. Thinned: ladder x1, "
        "five seeds.\n",
        md_table(pooled_s), "",
        "Per type, every seed and both truths in `wu_rate_ratio.csv`. The constant-pi mixture fitted "
        "to the same data (`wu_rate_ratio_mixture.csv`):\n", md_table(mixr), "",
        f"**Scale.** This project's e^delta is the log of the component rate ratio: MIX on this "
        f"section has e^delta = {MIX_E_DELTA:.2f}, a rate ratio of {MIX_RATIO:.2f}. The addendum's "
        "'1.9' is MIX's e^delta, so the like-for-like comparison with Wu's rate ratios is "
        f"{MIX_RATIO:.2f}.\n",
        f"- **X19** (Wu's stricter-grouping ratio at least 4; the mixture on thinned Wu closer to "
        f"Wu's ratio than to 1.9): Wu's ratio {full_strict:.2f} (primary grouping {full_prim:.2f}) "
        f"-> **{'yes' if x19a else 'no'}**; the mixture on thinned Wu gives {thin_mix:.2f}, closer to "
        f"Wu's than to 1.9 -> **{'yes' if closer_wu else 'no'}** (and than to MIX's rate ratio "
        f"{MIX_RATIO:.2f} -> {'yes' if closer_wu_ratio else 'no'}). Overall, as written: "
        f"**{'holds' if x19a and closer_wu else 'partly holds' if x19a or closer_wu else 'does not hold'}**.\n",
        f"Stage A6 complete in {(time.time() - t0) / 60:.0f} min. **Review point** (A1, A3, A5 and A6 "
        "pause for review).\n"]))
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
