"""Division classifier comparison, Stage 5: biology (spec 5) and deviation (spec 6).

    python run/division_stage5_biology.py      # after run/wu2021_prepare.py

Reads Stage 3's out-of-fold predictions; nothing is refitted.

Spec 5, biological expectation:
  5.1 per cell type: the BASE-T, BASE-O and SCORE label rates; every arm's mean
      p; MIX's mean pi; and the Rogan-Gladen corrected BASE-T fraction
      (rate - mean alpha_i) / (sens - mean alpha_i), sens = mean P1(Y >= c^T_i)
      from MIX ("borrows from MIX").
  5.2 anchors. In situ: the fraction of Cancer Epithelial cells whose ESR1, PGR
      and ERBB2 counts exceed a per-gene technical null (the labels' null
      design, robustly trimmed: log N offset + area spline, or under MisTIC the
      weighted log N offset + theta m for the gene's own m), descriptive only.
      Wu et al. 2021: per-type S/G2M fractions (phase from
      run/wu2021_prepare.py), overall and within the patients of the clinical
      subtype whose receptor profile is closest to the section's. Closeness is
      measured on log2 enrichment of each receptor gene in Cancer Epithelial
      over Normal Epithelial (pooled transcripts per transcript captured): a
      ratio within one platform, so the platforms' depths cancel.
  5.3 qualitative checks, each pass / fail / unclear by a rule fixed here before
      any result was seen (stated in the report). Nothing is tuned to pass.

Spec 6, where the arms deviate:
  6.1 out-of-fold differences: MIX - NB-c (both count labels), MIX - LOG-T,
      MIX - LOG-O, NB-c - NB-d, and LOG-O - LOG-S as a difference of percentile
      ranks (their prevalences differ).
  6.2 each stratified by cell type, neighbourhood tumour fraction (deciles of
      the 80 um disc's Cancer Epithelial score), density decile, vascular
      proximity decile (the oxygen field u, L0 = 100 um) and signed distance
      to the tumour-region edge (100 um bins with Cancer Epithelial share >= 0.5;
      inside negative). Means with 95% block-bootstrap intervals.
  6.3 100 um binned label residual L - p for every arm and label, with local
      Moran's I as run/spatial_analysis.py does it (queen weights, 9,999
      conditional permutations, BH at 5%), set against the persistent clusters
      of the mixture's count residuals (outputs/spatial/lisa_bins_mixture.csv).
  6.4 every Stage 4 score for Cancer Epithelial and Plasmablasts.

Pre-stated expectations X4 and X5 are evaluated here.

Writes outputs/division_classifier/{biology.csv, receptor_context.csv,
deviation.csv, residual_lisa.csv, fraction_ladder.png, deviation_strata.png,
residual_maps.png} and the Stage 5 note in report.md.
"""
import importlib.util
import json
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
from scipy.stats import spearmanr                                     # noqa: E402

from tissue_dynamics import nb_baseline as nbb                        # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge, pois_p_ge      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

_s = importlib.util.spec_from_file_location("spatial", os.path.join(HERE, "spatial_analysis.py"))
spatial = importlib.util.module_from_spec(_s)
_s.loader.exec_module(spatial)

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_BOOT, SEED = 300, 20261008
RECEPTORS = ("ESR1", "PGR", "ERBB2")
RECEPTOR_P = (0.05, 0.001)
LOW = ("CAFs", "PVL", "Endothelial", "Normal Epithelial")
MID = ("Plasmablasts", "T-cells")
LT, LO = f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}", f"LO_{L.td_tag(L.TD_PRIMARY)}"
CT = LT.replace("LT_", "cT_")
# the tumour region is patchy (13% of cells), so no cell lies 300 um inside it
EDGE_BANDS = [(-np.inf, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 300), (300, np.inf)]
COL = {"NB-c": "#52514e", "NB-d": "#8a8984", "MIX": "#2a78d6", "LOG-T": "#eb6834",
       "LOG-O": "#eb6834", "LOG-S": "#1baf7a"}


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


class Groups:
    """Group means with a spatial block bootstrap (the Stage 4 replicates)."""

    def __init__(self, bidx, nb, M):
        self.b, self.nb, self.M = bidx, nb, M

    def sums(self, v, g, G):
        cell = self.b * G + g
        return np.bincount(cell, weights=v, minlength=self.nb * G).reshape(self.nb, G)

    def means(self, v, g, G, w=None):
        """Point and bootstrap means of v within each of G groups (g = -1 skips)."""
        ok = (g >= 0) & np.isfinite(v)
        v, gg = np.where(ok, v, 0.0), np.where(ok, g, 0)
        num = self.sums(v * ok, gg, G)
        den = self.sums(ok.astype(float), gg, G)
        with np.errstate(divide="ignore", invalid="ignore"):
            point = num.sum(0) / den.sum(0)
            boot = (self.M @ num) / (self.M @ den)
        return point, boot

    def ratio(self, num_v, den_v, g, G):
        num, den = self.sums(num_v, g, G), self.sums(den_v, g, G)
        with np.errstate(divide="ignore", invalid="ignore"):
            return num.sum(0) / den.sum(0), (self.M @ num) / (self.M @ den)


def receptor_null(y, Xa, off, m=None):
    """Per-gene technical null, as the labels' (spec 2.1): P(Y >= y). Under
    MisTIC `m` is this gene's own expected misassigned count."""
    fit = L.robust_null(y, Xa, off, m=m)
    return nb_p_ge(y, fit["mu"], fit["alpha"]), fit


def main():
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = types
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    oof = pd.read_parquet(f"{OUT}/oof_predictions.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    wu = pd.read_parquet("data/wu2021/wu_cells.parquet")
    wu_meta = json.load(open("data/wu2021/wu_cells.json"))
    ref = pd.read_csv("outputs/spatial/lisa_bins_mixture.csv")
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("oof_predictions", oof), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    n = len(df)
    ctype = lab["cell_type"].astype(str).to_numpy()
    T = len(types)
    tix = {t: i for i, t in enumerate(types)}
    g_type = np.array([tix[t] for t in ctype])
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb = int(bidx.max()) + 1
    rng = np.random.default_rng(SEED)
    M = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]
                 ).astype(float)
    G = Groups(bidx, nb, M)
    say(f"{n:,} cells, {nb} blocks; Wu atlas {len(wu):,} cells")

    # =================== 5.1 fractions per cell type ===========================
    srcs = {"BASE-T rate": lab[LT].to_numpy(float), "BASE-O rate": lab[LO].to_numpy(float),
            "SCORE rate": lab["LS"].to_numpy(float), "MIX mean pi": oof["pi_mix"].to_numpy()}
    for arm, c in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix")):
        srcs[f"{arm} p, BASE-T"] = oof[f"p_{c}_{LT}"].to_numpy()
        srcs[f"{arm} p, BASE-O"] = oof[f"p_{c}_{LO}"].to_numpy()
    srcs["LOG-T p, BASE-T"] = oof[f"p_logt_{LT}"].to_numpy()
    srcs["LOG-O p, BASE-O"] = oof[f"p_logo_{LO}"].to_numpy()
    srcs["LOG-S p, SCORE"] = oof["p_logs_LS"].to_numpy()
    g_all = np.zeros(n, int)
    bio_rows = []
    for name, v in srcs.items():
        for g, labels_, GG in ((g_type, types, T), (g_all, ["all"], 1)):
            pt, bt = G.means(v, g, GG)
            for k, t in enumerate(labels_):
                lo, hi = ci(bt[:, k])
                bio_rows.append({"source": name, "stratum": t, "estimate": pt[k], "ci_lo": lo, "ci_hi": hi})
    # Rogan-Gladen, borrowing the sensitivity from MIX
    alpha_i = lab[f"alphai_{L.ab_tag(L.ALPHA_B_PRIMARY)}"].to_numpy(float)
    sens_i = pois_p_ge(lab[CT].to_numpy(float), oof["lam1_mix"].to_numpy())
    yT = lab[LT].to_numpy(float)
    for g, labels_, GG in ((g_type, types, T), (g_all, ["all"], 1)):
        s_r, s_a, s_s, s_n = (G.sums(v, g, GG) for v in (yT, alpha_i, sens_i, np.ones(n)))
        rate, abar, sens = (s.sum(0) / s_n.sum(0) for s in (s_r, s_a, s_s))
        Mn = M @ s_n
        rb, ab, sb = ((M @ s) / Mn for s in (s_r, s_a, s_s))
        rg, rgb = (rate - abar) / (sens - abar), (rb - ab) / (sb - ab)
        for k, t in enumerate(labels_):
            lo, hi = ci(rgb[:, k])
            bio_rows.append({"source": "Rogan-Gladen BASE-T (borrows from MIX)", "stratum": t,
                             "estimate": rg[k], "ci_lo": lo, "ci_hi": hi,
                             "note": f"alpha {abar[k]:.4f}, sens {sens[k]:.3f}"})
    say(f"  5.1 fractions ({time.time() - t0:.0f}s)")

    # =================== 5.2 anchors ===========================================
    # in situ receptor context
    cfg_cells = nbb.all_cells(cfg, genes=list(RECEPTORS))
    pos = pd.Series(np.arange(len(cfg_cells)), index=cfg_cells["cell_id"].astype(str))
    ix = pos.reindex(ids).to_numpy()
    if np.isnan(ix).any():
        sys.exit("table cells missing from the Xenium cell list")
    ix = ix.astype(int)
    basis = build_basis(df, cfg, verbose=False)
    Xa, off, _m = L.null_design(df, basis)
    if basis.technical == "mistic":                # each gene's own m (MisTIC)
        from tissue_dynamics import mistic as MI

        cov = MI.load_covariates(cfg)
        cix = pd.Index(cov.counts.cell_ids).get_indexer(ids)
        m_gene = {g: cov.m_gene_set([g])[0][cix] for g in RECEPTORS}
    rec_rows, rec_counts = [], {}
    for gname in RECEPTORS:
        y = cfg_cells[f"g_{gname}"].to_numpy(float)[ix]
        rec_counts[gname] = y
        p, fit = receptor_null(y, Xa, off, m=m_gene[gname] if basis.technical == "mistic" else None)
        for stratum, m in (("Cancer Epithelial", ctype == "Cancer Epithelial"),
                           ("Normal Epithelial", ctype == "Normal Epithelial"),
                           ("all other types", ~np.isin(ctype, ["Cancer Epithelial", "Normal Epithelial"]))):
            row = {"gene": gname, "stratum": stratum, "cells": int(m.sum()),
                   "detected (y >= 1)": float(np.mean(y[m] >= 1)),
                   "transcripts per 1,000 captured": float(1000 * y[m].sum() / df["N"].to_numpy()[m].sum())}
            for a in RECEPTOR_P:
                row[f"above null, p < {a:g}"] = float(np.mean(p[m] < a))
            rec_rows.append(row)
        say(f"  {gname}: null alpha {fit['alpha']:.3f}")
    receptor = pd.DataFrame(rec_rows)
    receptor.to_csv(f"{OUT}/receptor_context.csv", index=False)

    def enrichment(counts, depth, types_):
        """log2 of pooled rate in Cancer Epithelial over Normal Epithelial, per gene."""
        out = {}
        ce, ne = types_ == "Cancer Epithelial", types_ == "Normal Epithelial"
        for gname in RECEPTORS:
            c = counts[gname]
            r_ce = (c[ce].sum() + 0.5) / depth[ce].sum()
            r_ne = (c[ne].sum() + 0.5) / depth[ne].sum()
            out[gname] = float(np.log2(r_ce / r_ne))
        return out

    situ = enrichment(rec_counts, df["N"].to_numpy(float), ctype)
    wu_types = wu["celltype_major"].astype(str).to_numpy()
    wu_counts = {gname: wu[gname].to_numpy(float) for gname in RECEPTORS}
    wu_depth = wu["nCount_RNA"].to_numpy(float)
    sub_rows = [{"profile": "this section (Xenium)", **situ, "distance": 0.0}]
    for sub in ("ER+", "HER2+", "TNBC"):
        m = wu["subtype"].to_numpy() == sub
        e = enrichment({g: v[m] for g, v in wu_counts.items()}, wu_depth[m], wu_types[m])
        sub_rows.append({"profile": f"Wu {sub}", **e,
                         "distance": float(np.sqrt(sum((e[g] - situ[g]) ** 2 for g in RECEPTORS)))})
    subtypes = pd.DataFrame(sub_rows)
    closest = subtypes.iloc[1:].sort_values("distance")["profile"].iloc[0].replace("Wu ", "")
    say(f"  closest Wu subtype by receptor enrichment: {closest}")

    wu = wu.assign(cyc=wu["phase"].isin(["S", "G2M"]).astype(float))
    wu_rows = []
    for scope, m in (("all patients", np.ones(len(wu), bool)),
                     (f"{closest} patients", wu["subtype"].to_numpy() == closest)):
        sub = wu[m]
        pooled = sub.groupby("celltype_major")["cyc"].mean()
        per_pt = sub.groupby(["celltype_major", "patient"])["cyc"].agg(["mean", "size"])
        per_pt = per_pt[per_pt["size"] >= 20]["mean"].groupby(level=0)
        for t in types:
            q = per_pt.get_group(t) if t in per_pt.groups else pd.Series(dtype=float)
            wu_rows.append({"scope": scope, "stratum": t, "cells": int((sub["celltype_major"] == t).sum()),
                            "patients": int(sub.loc[sub["celltype_major"] == t, "patient"].nunique()),
                            "S/G2M fraction (pooled)": float(pooled.get(t, np.nan)),
                            "patient median": float(q.median()) if len(q) else np.nan,
                            "patient IQR": (f"{q.quantile(.25):.3f}-{q.quantile(.75):.3f}" if len(q) else "")})
    wu_tab = pd.DataFrame(wu_rows)
    for r in wu_rows:
        bio_rows.append({"source": f"Wu S/G2M, {r['scope']}", "stratum": r["stratum"],
                         "estimate": r["S/G2M fraction (pooled)"], "ci_lo": np.nan, "ci_hi": np.nan,
                         "note": f"{r['patients']} patients, {r['cells']} cells"})
    bio = pd.DataFrame(bio_rows)
    bio.to_csv(f"{OUT}/biology.csv", index=False)
    say(f"  5.2 anchors ({time.time() - t0:.0f}s)")

    # =================== 5.3 qualitative checks ================================
    def per_type(source):
        b = bio[(bio["source"] == source) & (bio["stratum"].isin(types))].set_index("stratum")["estimate"]
        return b.reindex(types)

    wu_close = per_type(f"Wu S/G2M, {closest} patients")
    check_sources = ["MIX mean pi", "Rogan-Gladen BASE-T (borrows from MIX)", "BASE-T rate",
                     "BASE-O rate", "SCORE rate", "Wu S/G2M, all patients", f"Wu S/G2M, {closest} patients"]
    chk_rows = []
    for src in check_sources:
        v = per_type(src)
        rank = v.rank(ascending=False)
        ce_rank = int(rank["Cancer Epithelial"])
        c1 = "pass" if ce_rank == 1 else ("unclear" if ce_rank == 2 else "fail")
        above = {m: sum(v[m] > v[l] for l in LOW) for m in MID}
        below_ce = all(v[m] < v["Cancer Epithelial"] for m in MID)
        c2 = ("pass" if below_ce and all(a == len(LOW) for a in above.values())
              else "unclear" if all(a >= 2 for a in above.values()) else "fail")
        n_low = sum(rank[l] >= T - 4 for l in LOW)                 # bottom five of nine
        c3 = "pass" if n_low == len(LOW) else ("unclear" if n_low == len(LOW) - 1 else "fail")
        rho = float(spearmanr(v, wu_close).statistic) if not src.startswith("Wu") else np.nan
        c4 = ("" if np.isnan(rho) else "pass" if rho >= 0.6 else "unclear" if rho >= 0.3 else "fail")
        chk_rows.append({"source": src, "Cancer Epithelial highest": f"{c1} (rank {ce_rank})",
                         "Plasmablasts, T-cells intermediate": c2,
                         "CAFs, PVL, Endothelial, Normal Epi. low": f"{c3} ({n_low}/4 in bottom 5)",
                         f"Spearman with Wu {closest}": "" if np.isnan(rho) else f"{rho:+.2f} ({c4})"})
    checks = pd.DataFrame(chk_rows)

    # =================== 6.1-6.2 deviation =====================================
    def prank(v):
        r = np.empty(len(v))
        r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
        return r / (len(v) - 1)

    diffs = {"MIX - NB-c, BASE-T": oof[f"p_mix_{LT}"].to_numpy() - oof[f"p_nbc_{LT}"].to_numpy(),
             "MIX - NB-c, BASE-O": oof[f"p_mix_{LO}"].to_numpy() - oof[f"p_nbc_{LO}"].to_numpy(),
             "MIX - LOG-T, BASE-T": oof[f"p_mix_{LT}"].to_numpy() - oof[f"p_logt_{LT}"].to_numpy(),
             "MIX - LOG-O, BASE-O": oof[f"p_mix_{LO}"].to_numpy() - oof[f"p_logo_{LO}"].to_numpy(),
             "NB-c - NB-d, BASE-T": oof[f"p_nbc_{LT}"].to_numpy() - oof[f"p_nbd_{LT}"].to_numpy(),
             "NB-c - NB-d, BASE-O": oof[f"p_nbc_{LO}"].to_numpy() - oof[f"p_nbd_{LO}"].to_numpy(),
             "LOG-O - LOG-S (percentile rank)": prank(oof[f"p_logo_{LO}"].to_numpy())
             - prank(oof["p_logs_LS"].to_numpy())}
    ce_j = types.index("Cancer Epithelial")
    tumour_nbr = df[f"nbr_phi_{ce_j:02d}"].to_numpy(float)
    decile = lambda v: np.minimum((pd.Series(v).rank(pct=True, method="first").to_numpy() * 10).astype(int), 9)  # noqa: E731
    # signed distance to the tumour-region edge
    xy = df[["x", "y"]].to_numpy(float)
    gx, gy = spatial.grid_index(xy, 100.0)
    ny = gy.max() + 1
    flat = gx * ny + gy
    cnt = np.bincount(flat)
    ce_share = np.bincount(flat, weights=(ctype == "Cancer Epithelial").astype(float)) / np.maximum(cnt, 1)
    in_region = (ce_share[flat] >= 0.5) & (cnt[flat] >= 5)
    d_in = cKDTree(xy[~in_region]).query(xy[in_region], k=1)[0]
    d_out = cKDTree(xy[in_region]).query(xy[~in_region], k=1)[0]
    sdist = np.empty(n)
    sdist[in_region], sdist[~in_region] = -d_in, d_out
    band = np.full(n, -1)
    for k, (lo_, hi_) in enumerate(EDGE_BANDS):
        band[(sdist > lo_) & (sdist <= hi_)] = k
    band_names = ["<= -100", "-100 to -50", "-50 to 0", "0 to 50", "50 to 100", "100 to 300", "> 300"]
    stratifiers = {"cell type": (g_type, types),
                   "neighbourhood tumour fraction decile": (decile(tumour_nbr), [f"D{k + 1}" for k in range(10)]),
                   "density decile": (decile(df["density"].to_numpy()), [f"D{k + 1}" for k in range(10)]),
                   "vascular proximity decile (u)": (decile(df["oxygen"].to_numpy()), [f"D{k + 1}" for k in range(10)]),
                   "distance to tumour edge (um; inside < 0)": (band, band_names)}
    dev_rows = []
    for dname, dv in diffs.items():
        for sname, (g, levels) in stratifiers.items():
            pt, bt = G.means(dv, g, len(levels))
            for k, lev in enumerate(levels):
                lo, hi = ci(bt[:, k])
                dev_rows.append({"difference": dname, "stratifier": sname, "level": lev,
                                 "mean": pt[k], "ci_lo": lo, "ci_hi": hi,
                                 "mean |d|": float(np.mean(np.abs(dv[g == k]))) if np.any(g == k) else np.nan,
                                 "cells": int(np.sum(g == k))})
        pt, bt = G.means(dv, g_all, 1)
        lo, hi = ci(bt[:, 0])
        dev_rows.append({"difference": dname, "stratifier": "all", "level": "all", "mean": pt[0],
                         "ci_lo": lo, "ci_hi": hi, "mean |d|": float(np.mean(np.abs(dv))), "cells": n})
    dev = pd.DataFrame(dev_rows)
    dev.to_csv(f"{OUT}/deviation.csv", index=False)
    say(f"  6.1-6.2 deviation ({time.time() - t0:.0f}s); tumour region "
        f"{100 * in_region.mean():.1f}% of cells")

    # X4, X5: where the absolute disagreement concentrates
    def abs_by_type(dname):
        t = dev[(dev["difference"] == dname) & (dev["stratifier"] == "cell type")]
        return t.set_index("level")["mean |d|"].reindex(types)
    x4_t, x4_o = abs_by_type("NB-c - NB-d, BASE-T"), abs_by_type("NB-c - NB-d, BASE-O")
    x5_t, x5_o = abs_by_type("MIX - NB-c, BASE-T"), abs_by_type("MIX - NB-c, BASE-O")
    x4 = x4_t.idxmax() == "Cancer Epithelial" and x4_o.idxmax() == "Cancer Epithelial"
    top2 = lambda s: set(s.sort_values(ascending=False).index[:2])  # noqa: E731
    x5 = top2(x5_t) == {"Cancer Epithelial", "Plasmablasts"} and top2(x5_o) == {"Cancer Epithelial", "Plasmablasts"}

    # =================== 6.3 residual maps =====================================
    resid = {("BASE-T", a): lab[LT].to_numpy(float) - oof[f"p_{c}_{LT}"].to_numpy()
             for a, c in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix"), ("LOG-T", "logt"))}
    resid.update({("BASE-O", a): lab[LO].to_numpy(float) - oof[f"p_{c}_{LO}"].to_numpy()
                  for a, c in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix"), ("LOG-O", "logo"))})
    resid[("SCORE", "LOG-S")] = lab["LS"].to_numpy(float) - oof["p_logs_LS"].to_numpy()
    ref_key = {(int(a), int(b)): c for a, b, c in ref[["bx", "by", "cls"]].itertuples(index=False)}
    ref_hh = {k for k, c in ref_key.items() if c == "HH"}
    ref_ll = {k for k, c in ref_key.items() if c == "LL"}
    xmid, ymid = (gx.max() + 1) / 2, ny / 2
    quad = lambda bx_, by_: ("lower" if by_ < ymid else "upper") + "-" + ("left" if bx_ < xmid else "right")  # noqa: E731
    lisa_rows, lisa_maps = [], {}
    for key, rv in resid.items():
        Ls = spatial.lisa(rv, gx, gy, seed=1)
        cls = Ls["cls"]
        hh = {(int(a), int(b)) for a, b, c in zip(Ls["bx"], Ls["by"], cls) if c == "HH"}
        ll = {(int(a), int(b)) for a, b, c in zip(Ls["bx"], Ls["by"], cls) if c == "LL"}
        jac = lambda a, b: len(a & b) / max(len(a | b), 1)  # noqa: E731
        qs = {}
        for nm, s in (("HH", hh), ("LL", ll)):
            qq = pd.Series([quad(a, b) for a, b in s]).value_counts(normalize=True) if s else pd.Series(dtype=float)
            qs[nm] = ", ".join(f"{k} {v:.0%}" for k, v in qq.items())
        lisa_rows.append({"label": key[0], "arm": key[1], "global I": Ls["I_global"],
                          "HH bins (under-prediction)": len(hh), "LL bins (over-prediction)": len(ll),
                          "HH overlap with count-residual HH (Jaccard)": jac(hh, ref_hh),
                          "LL overlap with count-residual LL (Jaccard)": jac(ll, ref_ll),
                          "HH by quadrant": qs["HH"], "LL by quadrant": qs["LL"]})
        lisa_maps[key] = Ls
        say(f"  LISA {key}: I {Ls['I_global']:.3f}, HH {len(hh)}, LL {len(ll)}")
    ref_q = {nm: ", ".join(f"{k} {v:.0%}" for k, v in pd.Series([quad(a, b) for a, b in s])
                           .value_counts(normalize=True).items()) for nm, s in (("HH", ref_hh), ("LL", ref_ll))}
    lisa_tab = pd.DataFrame(lisa_rows)
    lisa_tab.to_csv(f"{OUT}/residual_lisa.csv", index=False)

    # =================== 6.4 over-dispersed types ==============================
    sc = pd.read_csv(f"{OUT}/scores.csv")
    sc = sc[(sc["threshold"] == "primary") & sc["stratum"].isin(["Cancer Epithelial", "Plasmablasts"])
            & sc["metric"].isin(["log loss", "Brier", "AUROC", "AUPRC", "mean p / rate"])
            & ~sc["arm"].str.contains(" - ")]
    od_rows = []
    for (lname, stratum, arm), g in sc.groupby(["label", "stratum", "arm"], sort=False):
        g = g.set_index("metric")
        od_rows.append({"label": lname, "type": stratum, "arm": arm,
                        **{m: f"{g.loc[m, 'estimate']:.4f} [{g.loc[m, 'ci_lo']:.4f}, {g.loc[m, 'ci_hi']:.4f}]"
                           for m in ("log loss", "AUROC", "AUPRC", "mean p / rate")}})
    od = pd.DataFrame(od_rows)

    # =================== figures ===============================================
    order = per_type("MIX mean pi").sort_values().index.tolist()
    yk = np.arange(T)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    panels = [("BASE-T (label scale)", ["BASE-T rate", "NB-c p, BASE-T", "NB-d p, BASE-T",
                                        "MIX p, BASE-T", "LOG-T p, BASE-T"]),
              ("BASE-O (label scale)", ["BASE-O rate", "NB-c p, BASE-O", "NB-d p, BASE-O",
                                        "MIX p, BASE-O", "LOG-O p, BASE-O"]),
              ("state scale and references", ["MIX mean pi", "Rogan-Gladen BASE-T (borrows from MIX)",
                                              "SCORE rate", "Wu S/G2M, all patients",
                                              f"Wu S/G2M, {closest} patients"])]
    mk = ["o", "s", "^", "D", "v"]
    cols5 = ["#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#eb6834"]
    cols_state = ["#2a78d6", "#7fb0e8", "#1baf7a", "#c2185b", "#f48fb1"]
    for ax, (title, srcs_) in zip(axes, panels):
        cs = cols_state if title.startswith("state") else cols5
        for j, src in enumerate(srcs_):
            v = per_type(src).reindex(order).to_numpy()
            ax.plot(v, yk + (j - 2) * 0.1, mk[j], color=cs[j], ms=6,
                    mfc="none" if src.endswith("rate") and not title.startswith("state") else cs[j],
                    label=src.replace(" (borrows from MIX)", "").replace(", BASE-T", "").replace(", BASE-O", ""))
        ax.axvline(0, color="#0b0b0b", lw=0.6)
        ax.set_yticks(yk)
        ax.set_yticklabels(order)
        ax.set_xlabel("fraction of cells")
        ax.set_title(title)
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Per-type cycling fractions: labels, arms, the mixture's state scale and the Wu reference",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fraction_ladder.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    show = ["MIX - NB-c, BASE-T", "MIX - NB-c, BASE-O", "MIX - LOG-T, BASE-T", "MIX - LOG-O, BASE-O"]
    dcol = {"MIX - NB-c, BASE-T": "#2a78d6", "MIX - NB-c, BASE-O": "#7fb0e8",
            "MIX - LOG-T, BASE-T": "#eb6834", "MIX - LOG-O, BASE-O": "#f5a87e"}
    for ax, sname in zip(axes, ("neighbourhood tumour fraction decile", "density decile",
                                "distance to tumour edge (um; inside < 0)")):
        for dname in show:
            t = dev[(dev["difference"] == dname) & (dev["stratifier"] == sname)]
            x = np.arange(len(t))
            ax.plot(x, t["mean"], "o-", ms=3.5, color=dcol[dname], label=dname)
            ax.fill_between(x, t["ci_lo"], t["ci_hi"], color=dcol[dname], alpha=0.15, lw=0)
            ax.set_xticks(x)
            ax.set_xticklabels(t["level"], rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="#0b0b0b", lw=0.8)
        ax.set_title(sname)
        ax.set_ylabel("mean out-of-fold difference in p")
        ax.grid(True, color="#e4e3df", lw=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deviation_strata.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    keys = [("BASE-T", "NB-c"), ("BASE-T", "MIX"), ("BASE-T", "LOG-T"), ("BASE-O", "NB-c"),
            ("BASE-O", "MIX"), ("BASE-O", "LOG-O"), ("SCORE", "LOG-S")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    cmap = {"HH": "#c62828", "LL": "#1565c0", "HL": "#f4a3a3", "LH": "#9cc3ef", "ns": "#e4e3df",
            "island": "#ffffff"}
    rb = np.array(sorted(ref_hh)) if ref_hh else np.zeros((0, 2))
    rl = np.array(sorted(ref_ll)) if ref_ll else np.zeros((0, 2))
    for ax, key in zip(axes.ravel(), keys + [None]):
        if key is None:
            Ls = None
            bins = ref[ref["cls"] != "island"]
            ax.scatter(bins["bx"] * 100, bins["by"] * 100, s=2.2, marker="s",
                       c=[cmap[c] for c in bins["cls"]], lw=0)
            ax.set_title("reference: mixture count residuals (spatial analysis)", fontsize=10)
        else:
            Ls = lisa_maps[key]
            ax.scatter(Ls["bx"] * 100, Ls["by"] * 100, s=2.2, marker="s",
                       c=[cmap[c] for c in Ls["cls"]], lw=0)
            ax.scatter(rb[:, 0] * 100, rb[:, 1] * 100, s=2.5, marker="s", facecolors="none",
                       edgecolors="#000000", lw=0.25)
            ax.scatter(rl[:, 0] * 100, rl[:, 1] * 100, s=2.5, marker="s", facecolors="none",
                       edgecolors="#555555", lw=0.25)
            ax.set_title(f"{key[0]}, {key[1]}: L - p (global I {Ls['I_global']:.3f})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Local Moran clusters of the out-of-fold label residual, 100 um bins "
                 "(red HH = under-prediction, blue LL = over-prediction; outlines: the mixture's "
                 "count-residual clusters)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/residual_maps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # =================== report =================================================
    def frac_table():
        cols = ["BASE-T rate", "BASE-O rate", "SCORE rate", "MIX mean pi",
                "Rogan-Gladen BASE-T (borrows from MIX)", "Wu S/G2M, all patients",
                f"Wu S/G2M, {closest} patients"]
        rows = []
        for t in order[::-1] + ["all"]:
            r = {"type": t}
            for c in cols:
                b = bio[(bio["source"] == c) & (bio["stratum"] == t)]
                if len(b) == 0:
                    r[c] = ""
                    continue
                b = b.iloc[0]
                r[c] = (f"{b['estimate']:.3f}" if np.isnan(b["ci_lo"])
                        else f"{b['estimate']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]")
            rows.append(r)
        out = pd.DataFrame(rows)
        return out.rename(columns={"Rogan-Gladen BASE-T (borrows from MIX)": "Rogan-Gladen BASE-T (borrows from MIX)"})

    rg = bio[(bio["source"] == "Rogan-Gladen BASE-T (borrows from MIX)") & bio["stratum"].isin(types)]
    rg_neg = rg[rg["ci_hi"] < 0]["stratum"].tolist()

    def arm_means_table():
        rows = []
        for t in order[::-1] + ["all"]:
            r = {"type": t}
            for c in ("BASE-T rate", "NB-c p, BASE-T", "MIX p, BASE-T", "LOG-T p, BASE-T",
                      "BASE-O rate", "NB-c p, BASE-O", "MIX p, BASE-O", "LOG-O p, BASE-O"):
                r[c] = float(bio[(bio["source"] == c) & (bio["stratum"] == t)]["estimate"].iloc[0])
            rows.append(r)
        return pd.DataFrame(rows)

    def dev_type_table():
        rows = []
        for t in types:
            r = {"type": t}
            for dname in diffs:
                b = dev[(dev["difference"] == dname) & (dev["stratifier"] == "cell type") & (dev["level"] == t)].iloc[0]
                flag = "" if b["ci_lo"] <= 0 <= b["ci_hi"] else "*"
                r[dname] = f"{b['mean']:+.4f}{flag}"
            rows.append(r)
        return pd.DataFrame(rows)

    def dev_trend_table():
        rows = []
        for dname in diffs:
            r = {"difference": dname}
            for sname, (_, levels) in stratifiers.items():
                if sname == "cell type":
                    continue
                t = dev[(dev["difference"] == dname) & (dev["stratifier"] == sname)
                        & (dev["cells"] > 0)].set_index("level")
                r[sname] = f"{t['mean'].iloc[0]:+.4f} to {t['mean'].iloc[-1]:+.4f}"
            rows.append(r)
        return pd.DataFrame(rows)

    rec_ce = receptor[receptor["stratum"] == "Cancer Epithelial"][
        ["gene", "detected (y >= 1)", "above null, p < 0.05", "above null, p < 0.001",
         "transcripts per 1,000 captured"]]
    rec_ref = receptor[receptor["stratum"] != "Cancer Epithelial"][
        ["gene", "stratum", "detected (y >= 1)", "above null, p < 0.05", "transcripts per 1,000 captured"]]
    report(f"""
## Stage 5 -- biology and deviation ({time.strftime('%Y-%m-%d %H:%M')})

Out-of-fold predictions from Stage 3; intervals are the Stage 4 block bootstrap
(250 blocks, {N_BOOT} replicates). Label rates and mean p are label scale; MIX's
pi and the Rogan-Gladen fraction are state scale; the two are not compared
with each other.

### 5.1 Fractions per cell type

Rows ordered by MIX's mean pi. The Rogan-Gladen row corrects the BASE-T rate
for its false positives (mean alpha_i) and for its sensitivity, which it
borrows from MIX (mean P1(Y >= c^T_i)).

{md_table(frac_table())}

The Rogan-Gladen fraction is negative, with its interval below 0, in
{len(rg_neg)} types ({', '.join(rg_neg) or 'none'}). There the BASE-T label
fires less often than its own mean alpha_i: the technical null, fitted across
all types with one dispersion (alpha0 = 0.83, widened by the cycling cells),
overstates those types' false-positive rate. The correction is not usable
there; it says the label's false positives are type-dependent, not that the
fraction is below zero.

Each arm's mean p against its label (label scale):

{md_table(arm_means_table(), '{:.4f}')}

### 5.2 Anchors

**The section, in situ.** Receptor genes in Cancer Epithelial cells, against a
per-gene technical null ({'weighted log N offset plus theta m for the gene' if basis.technical == 'mistic' else 'log N offset, area spline'},
robustly trimmed, as the labels' null). Descriptive only; no clinical subtype is assigned.

{md_table(rec_ce, '{:.4f}')}

For reference, Normal Epithelial and all other types:

{md_table(rec_ref, '{:.4f}')}

**The closest Wu subtype.** log2 enrichment of each receptor gene in Cancer
Epithelial over Normal Epithelial (pooled transcripts per transcript
captured), here and in each Wu subtype; distance is Euclidean over the three
genes. Closest: **{closest}**. Wu's subtype is the clinical label per patient
(its HER2+ group includes HER2+/ER+ tumours).

{md_table(subtypes, '{:+.3f}')}

**Wu et al. 2021 (GSE176078).** {len(wu):,} cells from {wu['patient'].nunique()}
primary tumours (10x Chromium). Phase from Seurat's CellCycleScoring via
scanpy's port ({len(wu_meta['s_genes'])} S and {len(wu_meta['g2m_genes'])} G2M
genes; MLF1IP as CENPU), on LogNormalize expression. S/G2M fraction per type,
pooled over cells and as the median and IQR over patients with at least 20
cells of the type:

{md_table(wu_tab, '{:.3f}')}

Platform gap: droplet scRNA-seq at much higher depth (median
{wu['nCount_RNA'].median():,.0f} UMIs per cell, against a median N of
{df['N'].median():,.0f} here), and a score-based phase call that is relative
to control genes, so it calls many non-cycling cells S or G2M. Read the Wu
fractions as an ordering of types more than as absolute rates.

**Literature Ki67 index.** As a range only, and approximate (from memory; to
be checked against a cited source before use): roughly 10-20% in luminal-A-like
tumours, rising to 30-60% or more in HER2-positive and triple-negative ones.
Ki67 protein marks G1 as well as S/G2/M, so a transcript S/G2M fraction should
sit below it; it is an upper bound for Cancer Epithelial, not a target.

### 5.3 Qualitative checks

Rules, fixed before any result was seen: Cancer Epithelial highest -- pass if
ranked first, unclear if second. Plasmablasts and T-cells intermediate -- pass
if both are below Cancer Epithelial and above all four low types, unclear if
each is above at least two of them. CAFs, PVL, Endothelial and Normal
Epithelial low -- pass if all four are in the bottom five of nine, unclear if
three are. Consistency with Wu -- Spearman correlation across the nine types
with the closest-subtype Wu fractions: pass at 0.6 or above, unclear from 0.3.

{md_table(checks)}

### 6.1-6.2 Where the arms disagree

Mean out-of-fold difference by cell type (* = 95% interval excludes 0):

{md_table(dev_type_table())}

Along the continuous strata, first level to last (decile 1 to 10; tumour
edge from 100 um or more inside to 300 um or more outside):

{md_table(dev_trend_table())}

The tumour region (100 um bins with Cancer Epithelial share >= 0.5) holds
{100 * in_region.mean():.1f}% of cells. Every level, with intervals and mean
|difference|, is in `deviation.csv`; the trends are drawn in
`deviation_strata.png`.

**X4** (NB-d moves predictions mainly in Cancer Epithelial): largest mean
|NB-c - NB-d| in {x4_t.idxmax()} (BASE-T) and {x4_o.idxmax()} (BASE-O) --
**{'holds' if x4 else 'does not hold'}**.
**X5** (NB-MIX disagreement concentrates in Cancer Epithelial and
Plasmablasts): the two largest mean |MIX - NB-c| are
{', '.join(x5_t.sort_values(ascending=False).index[:2])} (BASE-T) and
{', '.join(x5_o.sort_values(ascending=False).index[:2])} (BASE-O) --
**{'holds' if x5 else 'does not hold'}**.

### 6.3 Residual maps

Local Moran's I of the out-of-fold label residual L - p on 100 um bins, as
`run/spatial_analysis.py` computes it. HH bins are under-prediction (more
labelled cells than predicted), LL bins over-prediction. The reference is the
mixture's count-residual clustering from the spatial analysis: HH by quadrant
{ref_q['HH']}; LL by quadrant {ref_q['LL']} (quadrants in plotted
coordinates, y upward).

{md_table(lisa_tab, '{:.3f}')}

### 6.4 Over-dispersed types

Stage 4's scores for Cancer Epithelial and Plasmablasts:

{md_table(od)}

Figures: `fraction_ladder.png`, `deviation_strata.png`, `residual_maps.png`.
Stage 5 complete in {(time.time() - t0) / 60:.0f} min.
""")
    say(f"\nStage 5 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
