"""BASE-T-trunc calibrated on Wu et al. 2021, thinned to this section's depth.

    python run/division_wu_calibration.py [--seeds 3] [--smoke]

Requested by the user (2026-10-17) as a less circular alternative to the G4
simulation (run/division_trunc_calibration.py), whose truth is the production
mixture's own pi and whose 10% target is the mixture's cycling fraction.
Here the truth comes from Wu's whole-transcriptome Seurat phase scores, and
the label rate on this section is an OUTPUT, not a target.

Data and thinning are A6's (run/division_addendum_e.py; V15 and V16 passed):
Wu restricted to the genes shared with this panel, each cell binomially
thinned so its depth matches the same within-type quantile of this section's
depth. y = Tirosh composite, N = library without Tirosh and blocklist genes,
cells with N >= 10. The null is the plain one V16 checked on this section:
NB with a log N offset and no MisTIC term (thinned scRNA-seq has none). On the
thinned counts every null of the G4 run is refitted -- Stage 1's naive robust
null and the truncated null at each alpha_trim -- and BASE-T is scored at
each alpha_B.

Truth. Seurat's score is max(S, G2M) at whole-transcriptome depth (median
~3,600 UMIs). Its own call (S/G2M where that is > 0) cuts through the
non-cycling bulk in some types: in ER+ T-cells the bulk sits around -0.03 and
33% are called S/G2M, in Cancer Epithelial 6%. So the primary truth uses
margins on the score's own scale:
  positive  max score > 0.1  (the cycling tail in every type)
  negative  max score <= 0   (Seurat's G1)
  excluded  0 < max score <= 0.1, and cells with fewer than 3,000 UMIs
            (shallow cells' scores are noisy: ER+ CAFs are 8.8% S/G2M in the
            shallowest third, 0.4% in the deepest).
Excluded cells go through the thinning and the null fits; they are only left
out of the scoring.

Scope and weights. Primary: ER+ patients (the subtype closest to this
section). Each scored Wu cell is weighted by (this section's fraction of its
type) / (the scored Wu cells' fraction of its type), so the score reflects
this section's type mix. Metric: weighted MCC with the truth.

Selection rule (fixed before results are read): the (null, alpha_B) with the
highest mean weighted MCC over thinning seeds, over the full grid with no
rate constraint. Reported with it: the alpha_B band whose MCC is within 0.005
of the best, for that null; the label rate those settings give on this
section, under the production MisTIC null and under the plain-N null.

Sensitivities (the same rule, other truths and scopes): the positive cut
moved to 0.05 and 0.15; Seurat's own call (cut 0, no excluded band); a
per-type tail truth (robust half-normal on the score per type and depth
tercile: positive at p < 0.001, negative at p > 0.05); no depth floor; all
patients; no type weights; the full-panel BASE-T label (Wu panel genes at full
depth, naive null, alpha_B 0.05) as the target.

Fidelity: alpha0 of every null on thinned Wu against the same plain-N null
on this section's counts, as alpha_trim grows. G4 reproduced this section's
alpha0 up to alpha_trim 0.2; thinned Wu lacks Xenium spill-over, so it may not.

Reads Wu's counts from data/wu2021/wu_counts_panel_csr.npz (the shared genes
only), built on the first run from the full matrix (`--wu-full`, or A6's
cache / the .mtx).

Writes outputs/division_classifier/wu_calibration/{fidelity.csv, grid.csv,
choice.json, mcc_vs_rate.png} and a section of report.md.
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
sys.path.insert(0, HERE)

from division_addendum_e import ALIASES, WU_DIR, load_wu, load_xenium, thin  # noqa: E402
from division_trunc_calibration import AB_GRID, NULLS, PARAM_TOL, TRIMS  # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/division_classifier"
WOUT = f"{OUT}/wu_calibration"
PANEL_CACHE = "data/wu2021/wu_counts_panel_csr.npz"
N_MIN = 10
UMI_FLOOR = 3000
POS_CUT = 0.1
NEAR = 0.005
SEED = 20261017

say = lambda *a: print(*a, flush=True)                                # noqa: E731


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def y_and_N(X, genes, tirosh, block):
    """Stage 1's composite and depth, as A6's `methods` builds them."""
    gi = {g: i for i, g in enumerate(genes)}
    tir = [gi[g] for g in tirosh if g in gi]
    excl = sorted(set(tir) | {gi[g] for g in block if g in gi})
    X = X.tocsr()
    y = np.asarray(X[:, tir].sum(1)).ravel()
    N = np.asarray(X.sum(1)).ravel() - np.asarray(X[:, excl].sum(1)).ravel()
    return y.astype(float), N.astype(float)


def plain_nulls(y, N):
    """Every null with the plain log N offset: {name: p-values, alpha0}."""
    X, off, m = np.ones((len(y), 1)), np.log(N), np.zeros(len(y))
    out = {}
    nv = L.robust_null(y, X, off, m=m)
    out["naive"] = (nb_p_ge(y, nv["mu"], nv["alpha"]), float(nv["alpha"]))
    full = L.fit_nb_additive_null(y, X, off, m)
    for t in TRIMS:
        ft = L.truncated_null(y, X, off, m, t, full, param_tol=PARAM_TOL)
        out[f"trunc {t:g}"] = (nb_p_ge(y, ft["mu"], ft["alpha"]), float(ft["alpha"]))
    return out


def weighted_curve(p, truth, w, grid):
    """Weighted MCC, sensitivity, precision and call rate (over scored cells) of
    the label p < a, for every a in `grid`. truth: 1 / 0 / -1 (not scored)."""
    s = truth >= 0
    p, t, w = p[s], truth[s] == 1, w[s]
    o = np.argsort(p)
    ps, ts, ws = p[o], t[o], w[o]
    cpos = np.concatenate([[0.0], np.cumsum(ws * ts)])
    cneg = np.concatenate([[0.0], np.cumsum(ws * ~ts)])
    k = np.searchsorted(ps, grid, side="left")              # cells with p < a
    tp, fp = cpos[k], cneg[k]
    P, Q = cpos[-1], cneg[-1]
    fn, tn = P - tp, Q - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"MCC": np.where(den > 0, (tp * tn - fp * fn) / den, np.nan),
                "sensitivity": tp / P, "precision": tp / (tp + fp),
                "fpr": fp / Q, "rate": (tp + fp) / (P + Q)}


def mcc_at(sens, fpr, f):
    """MCC of a label with this sensitivity and false-positive rate where a
    fraction f of cells are positive (both rates are free of prevalence)."""
    tp, fn = f * sens, f * (1 - sens)
    fp, tn = (1 - f) * fpr, (1 - f) * (1 - fpr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / den, np.nan)


def rogan_gladen(rate, sens, fpr):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rate - fpr) / (sens - fpr)


def tail_truth(score, wtype, umi, cells):
    """Per type and depth tercile: robust half-normal on the score; 1 at
    p < 0.001, 0 at p > 0.05, -1 between (and outside `cells`)."""
    out = np.full(len(score), -1)
    for t in np.unique(wtype[cells]):
        c = np.flatnonzero(cells & (wtype == t))
        terc = pd.qcut(umi[c], 3, labels=False, duplicates="drop")
        for k in np.unique(terc):
            ix = c[terc == k]
            pu = L.score_threshold(score[ix])["p_upper"]
            out[ix] = np.where(pu < 0.001, 1, np.where(pu > 0.05, 0, -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wu-full", default="", help="the full Wu CSR (A6's wu_counts_csr.npz), "
                    "needed only until the panel cache exists")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = "/tmp/division_wu_cal_smoke" if a.smoke else WOUT
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "y", "N"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    wu = pd.read_parquet("data/wu2021/wu_cells.parquet")
    tirosh_x = list(meta["tirosh_genes"])
    block_x = list(dict.fromkeys(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str)))

    # ---- Wu on the shared panel; this section's depth over the same genes ----------------
    wgenes = pd.read_csv(f"{WU_DIR}/count_matrix_genes.tsv", header=None)[0].astype(str).to_numpy()
    barcodes = pd.read_csv(f"{WU_DIR}/count_matrix_barcodes.tsv", header=None)[0].astype(str).to_numpy()
    if not np.array_equal(barcodes, wu["barcode"].astype(str).to_numpy()):
        sys.exit("wu_cells.parquet is not in the matrix's cell order")
    xcache = f"{WOUT}/section_panel.npz"
    if os.path.exists(xcache):
        z = np.load(xcache, allow_pickle=True)
        xgenes, xdepth_all = z["genes"].astype(str), z["depth_by_gene_set"].item()
    else:
        say("reading this section ...")
        XX, xgenes = load_xenium(cfg, lab["cell_id"].astype(str).to_numpy())
        xdepth_all = {"genes": xgenes, "X": XX}
    wset, inv = set(wgenes), {v: k for k, v in ALIASES.items()}
    x2w = {}
    for g in xgenes:
        if g in wset:
            x2w[g] = g
        elif ALIASES.get(g) in wset:
            x2w[g] = ALIASES[g]
        elif inv.get(g) in wset:
            x2w[g] = inv[g]
    shared = [g for g in xgenes if g in x2w]
    tir_sh = [g for g in tirosh_x if g in x2w]
    if "X" in xdepth_all:
        xi = {g: i for i, g in enumerate(xgenes)}
        xdepth = np.asarray(xdepth_all["X"][:, [xi[g] for g in shared]].sum(1)).ravel()
        os.makedirs(WOUT, exist_ok=True)
        np.savez(xcache, genes=np.asarray(xgenes), depth_by_gene_set=np.array(
            {"shared_depth": xdepth}, dtype=object))
        del xdepth_all
    else:
        xdepth = xdepth_all["shared_depth"]
    # Wu's counts on the shared genes only (columns in `shared` order), cached: the full
    # 100k x 33k matrix (1.4 GB as CSR) is read once, from --wu-full or A6's cache
    if os.path.exists(PANEL_CACHE):
        WP = sp.load_npz(PANEL_CACHE).tocsr()
        if json.load(open(PANEL_CACHE.replace(".npz", ".genes.json"))) != shared:
            sys.exit(f"{PANEL_CACHE} was built on a different gene list")
    else:
        say("reading Wu (full matrix, once) ...")
        if a.wu_full:
            WX = sp.load_npz(a.wu_full).tocsr()
        else:
            WX, _ = load_wu()
        wi = {g: i for i, g in enumerate(wgenes)}
        WP = WX[:, [wi[x2w[g]] for g in shared]].tocsr()
        del WX
        sp.save_npz(PANEL_CACHE, WP, compressed=True)
        json.dump(shared, open(PANEL_CACHE.replace(".npz", ".genes.json"), "w"))
        say(f"wrote {PANEL_CACHE}")
    wtype = wu["celltype_major"].astype(str).to_numpy()
    xtype = lab["cell_type"].astype(str).to_numpy()
    if a.smoke:
        ws = np.sort(np.random.default_rng(1).choice(len(wu), 15000, replace=False))
        WP, wu, wtype = WP[ws], wu.iloc[ws].reset_index(drop=True), wtype[ws]
    say(f"shared genes {len(shared):,}; Tirosh {len(tir_sh)}/{len(tirosh_x)}")
    wdepth = np.asarray(WP.sum(1)).ravel()
    target = np.zeros(len(wu))
    for t in types:
        wc, xc = wtype == t, xtype == t
        q = (pd.Series(wdepth[wc]).rank(method="average").to_numpy() - 0.5) / wc.sum()
        target[wc] = np.quantile(xdepth[xc], q)
    p_thin = np.minimum(1.0, target / np.maximum(wdepth, 1e-12))

    # ---- truths -------------------------------------------------------------------------
    score = np.maximum(wu["S_score"].to_numpy(float), wu["G2M_score"].to_numpy(float))
    umi = wu["nCount_RNA"].to_numpy(float)
    er = (wu["subtype"] == "ER+").to_numpy()
    deep = umi >= UMI_FLOOR

    def cut_truth(pos_cut, floor=True, band=True):
        tr = np.where(score > pos_cut, 1, np.where(score <= 0, 0, -1)) if band else (score > 0).astype(int)
        return np.where(deep | (not floor), tr, -1)

    truths = {"primary (pos > 0.1, neg <= 0, >= 3k UMI)": cut_truth(POS_CUT),
              "pos > 0.05": cut_truth(0.05), "pos > 0.15": cut_truth(0.15),
              "Seurat call (cut 0)": cut_truth(0.0, band=False),
              "per-type tail": tail_truth(score, wtype, umi, deep),
              "no depth floor": cut_truth(POS_CUT, floor=False)}
    # full-panel BASE-T (condition (ii)): naive plain-N null, alpha_B 0.05, on each scope
    y_full, N_full = y_and_N(WP, shared, tir_sh, block_x)
    keep = N_full >= N_MIN
    sec_frac = pd.Series(xtype).value_counts(normalize=True)

    # ---- this section with the plain-N null (fidelity and transfer) ------------------------
    sec_path = f"{WOUT}/section_plain_nulls.npz"
    if os.path.exists(sec_path) and not a.smoke:
        z = np.load(sec_path, allow_pickle=True)
        sec_plain = z["nulls"].item()
    else:
        say("this section, plain-N nulls ...")
        yl, Nl = lab["y"].to_numpy(float), lab["N"].to_numpy(float)
        if a.smoke:
            ss = np.sort(np.random.default_rng(2).choice(len(yl), 40000, replace=False))
            yl, Nl = yl[ss], Nl[ss]
        sec_plain = plain_nulls(yl, Nl)
        if not a.smoke:
            np.savez(sec_path, nulls=np.array(sec_plain, dtype=object))
    prod = np.load(f"{OUT}/trunc_calibration/real_nulls.npz", allow_pickle=True)["p"].item()

    # ---- thinned Wu: nulls and scores ----------------------------------------------------
    scopes = {"ER+": er, "all patients": np.ones(len(wu), bool)}
    rows, fid = [], []
    for sd in range(a.seeds):
        say(f"seed {sd}: thinning ... ({time.time() - t0:.0f}s)")
        Wt = thin(WP, p_thin, np.random.default_rng(SEED + sd))
        y, N = y_and_N(Wt, shared, tir_sh, block_x)
        for scope, sc in scopes.items():
            cells = sc & (N >= N_MIN)
            nulls = plain_nulls(y[cells], N[cells])
            # the full-panel target on the same cells, unthinned
            fk = cells & keep
            fn = L.robust_null(y_full[fk], np.ones((fk.sum(), 1)), np.log(N_full[fk]))
            full_lab = np.full(len(wu), -1)
            full_lab[fk] = L.base_t(y_full[fk], fn["mu"], fn["alpha"], 0.05)[0].astype(int)
            tset = dict(truths, **{"full-panel BASE-T": full_lab})
            ct = wtype[cells]
            Nc = N[cells]
            n_lo, n_hi = np.quantile(Nc, [1 / 3, 2 / 3])
            for k, (p, al) in nulls.items():
                fid.append({"seed": sd, "scope": scope, "null": k, "alpha0": al,
                            "rate 0.05": float(np.mean(p < 0.05)), "rate 0.1": float(np.mean(p < 0.1))})
                for tname, tr_all in tset.items():
                    tr = tr_all[cells]
                    sc_ = tr >= 0
                    wfrac = pd.Series(ct[sc_]).value_counts(normalize=True)
                    wts = {"weighted": np.array([sec_frac.get(t, 0) / wfrac.get(t, np.inf) for t in ct]),
                           "unweighted": np.ones(len(ct))}
                    for wname, w in wts.items():
                        if wname == "unweighted" and not tname.startswith("primary"):
                            continue
                        cur = weighted_curve(p, tr, w, AB_GRID)
                        pos = tr == 1
                        sens_d = np.array([np.mean((p < ab)[pos & (Nc >= n_hi)]) for ab in AB_GRID])
                        sens_s = np.array([np.mean((p < ab)[pos & (Nc <= n_lo)]) for ab in AB_GRID])
                        for i, ab in enumerate(AB_GRID):
                            rows.append({"seed": sd, "scope": scope, "truth": tname, "weights": wname,
                                         "null": k, "alpha_B": float(ab),
                                         **{m_: float(v[i]) for m_, v in cur.items()},
                                         "depth bias": float(sens_d[i] / max(sens_s[i], 1e-6)),
                                         "scored": int(sc_.sum()), "truth rate": float(pos.mean())})
            say(f"  {scope}: {cells.sum():,} cells, nulls fitted and scored ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(f"{out_dir}/grid.csv", index=False)
    grid = pd.DataFrame(rows)
    fid = pd.DataFrame(fid)
    fid.to_csv(f"{out_dir}/fidelity.csv", index=False)

    # ---- selection -----------------------------------------------------------------------------
    def section_rate(null, ab, which):
        p = (prod if which == "MisTIC" else sec_plain)[null]
        p = p[0] if isinstance(p, tuple) else p
        return float(np.mean(p < ab))

    def pick(truth, scope="ER+", weights="weighted"):
        g = grid[(grid["truth"] == truth) & (grid["scope"] == scope) & (grid["weights"] == weights)]
        gm = g.groupby(["null", "alpha_B"], sort=False).agg(
            MCC=("MCC", "mean"), MCC_sd=("MCC", "std"), sens=("sensitivity", "mean"), fpr=("fpr", "mean"),
            prec=("precision", "mean"), rate=("rate", "mean"), depth=("depth bias", "mean"),
            truth_rate=("truth rate", "mean"), scored=("scored", "mean")).reset_index()
        b = gm.loc[gm["MCC"].idxmax()]
        near = gm[(gm["null"] == b["null"]) & (gm["MCC"] >= b["MCC"] - NEAR)]["alpha_B"]
        best_by_null = gm.loc[gm.groupby("null")["MCC"].idxmax()].set_index("null")
        # prevalence-free alternatives: Youden's J, and MCC at this section's own
        # (Rogan-Gladen) cycling fraction, estimated at the Youden point of each null
        gm["J"] = gm["sens"] - gm["fpr"]
        gm["section rate"] = [section_rate(k, ab, "MisTIC") for k, ab in zip(gm["null"], gm["alpha_B"])]
        gm["RG fraction"] = rogan_gladen(gm["section rate"], gm["sens"], gm["fpr"])
        yj = gm.loc[gm["J"].idxmax()]
        f_null = {k: float(gm.loc[gm[gm["null"] == k]["J"].idxmax(), "RG fraction"]) for k in NULLS}
        gm["MCC at section fraction"] = [float(mcc_at(s_, f_, f_null[k])) for k, s_, f_ in
                                         zip(gm["null"], gm["sens"], gm["fpr"])]
        pm = gm.loc[gm["MCC at section fraction"].idxmax()]
        return {"truth": truth, "scope": scope, "weights": weights, "null": b["null"],
                "alpha_B": float(b["alpha_B"]), "MCC": float(b["MCC"]), "MCC sd": float(b["MCC_sd"]),
                "Wu sensitivity": float(b["sens"]), "Wu precision": float(b["prec"]),
                "Wu depth bias": float(b["depth"]), "truth positive rate": float(b["truth_rate"]),
                "scored cells": float(b["scored"]),
                "alpha_B band": [float(near.min()), float(near.max())],
                "section rate (MisTIC null)": section_rate(b["null"], b["alpha_B"], "MisTIC"),
                "section rate band (MisTIC)": [section_rate(b["null"], near.min(), "MisTIC"),
                                               section_rate(b["null"], near.max(), "MisTIC")],
                "section rate (plain-N null)": section_rate(b["null"], b["alpha_B"], "plain"),
                "Youden pick": {"null": yj["null"], "alpha_B": float(yj["alpha_B"]), "J": float(yj["J"]),
                                "section rate": float(yj["section rate"]),
                                "RG section fraction": float(yj["RG fraction"])},
                "section-fraction MCC pick": {"null": pm["null"], "alpha_B": float(pm["alpha_B"]),
                                              "MCC": float(pm["MCC at section fraction"]),
                                              "fraction used": f_null[pm["null"]],
                                              "section rate": float(pm["section rate"])},
                "best MCC by null": {k: float(v) for k, v in best_by_null["MCC"].items()},
                "best alpha_B by null": {k: float(v) for k, v in best_by_null["alpha_B"].items()}}, gm

    primary_name = "primary (pos > 0.1, neg <= 0, >= 3k UMI)"
    prim, gm_prim = pick(primary_name)
    sens = [pick(t)[0] for t in list(truths)[1:] + ["full-panel BASE-T"]]
    sens.append(pick(primary_name, scope="all patients")[0])
    sens.append(pick(primary_name, weights="unweighted")[0])
    g4 = json.load(open(f"{OUT}/trunc_calibration/choice.json"))
    json.dump({"primary": prim, "sensitivities": sens, "g4_choice": {k: g4[k] for k in ("null", "alpha_B", "real rate")}},
              open(f"{out_dir}/choice.json", "w"), indent=1)
    say(json.dumps(prim, indent=1))

    # ---- figure: weighted MCC on thinned Wu against the label rate it gives on this section --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#07244a"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, facecolor=SURF)
    ax.set_facecolor(SURF)
    for k in NULLS:
        s = gm_prim[gm_prim["null"] == k]
        xr = [section_rate(k, ab, "MisTIC") for ab in s["alpha_B"]]
        col = "#898781" if k == "naive" else SEQ[TRIMS.index(float(k.split()[1]))]
        ax.plot(xr, s["MCC"], color=col, lw=2.5 if k == prim["null"] else 1.5,
                ls="--" if k == "naive" else "-", label=k + ("  (chosen)" if k == prim["null"] else ""))
    ax.plot([prim["section rate (MisTIC null)"]], [prim["MCC"]], "o", ms=8, mfc=SEQ[5], mec=SURF, mew=2)
    ax.axvline(0.10, color=INK2, lw=1, ls=":")
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("label rate those settings give on THIS section (MisTIC null)")
    ax.set_ylabel("weighted MCC with Seurat truth, thinned Wu (ER+)")
    ax.grid(color=GRIDC, lw=0.8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    ax.set_title("BASE-T settings scored on Wu thinned to this section's depth", loc="left", fontsize=11)
    fig.savefig(f"{out_dir}/mcc_vs_rate.png", dpi=170, facecolor=SURF)

    # ---- report ---------------------------------------------------------------------------------
    fm = fid.groupby(["scope", "null"], sort=False)[["alpha0", "rate 0.1"]].mean().reset_index()
    ftab = pd.DataFrame({"null": list(NULLS),
                         "section, plain-N alpha0": [sec_plain[k][1] for k in NULLS],
                         "section, plain-N rate 0.1": [float(np.mean(sec_plain[k][0] < 0.1)) for k in NULLS],
                         "section, MisTIC rate 0.1": [float(np.mean(prod[k] < 0.1)) for k in NULLS]})
    for sc in scopes:
        s = fm[fm["scope"] == sc].set_index("null")
        ftab[f"Wu {sc} alpha0"] = ftab["null"].map(s["alpha0"])
        ftab[f"Wu {sc} rate 0.1"] = ftab["null"].map(s["rate 0.1"])
    stab = pd.DataFrame([{"truth / scope / weights": f"{r['truth']} / {r['scope']} / {r['weights']}",
                          "truth pos. rate": r["truth positive rate"], "null": r["null"],
                          "alpha_B": r["alpha_B"], "MCC": r["MCC"],
                          "section rate (MisTIC)": r["section rate (MisTIC null)"],
                          "section rate band": f"{r['section rate band (MisTIC)'][0]:.3f}-{r['section rate band (MisTIC)'][1]:.3f}",
                          "Youden pick": f"{r['Youden pick']['null']} @ {r['Youden pick']['alpha_B']:.3g}",
                          "Youden section rate": r["Youden pick"]["section rate"],
                          "RG section fraction": r["Youden pick"]["RG section fraction"],
                          "MCC-at-fraction pick": f"{r['section-fraction MCC pick']['null']} @ {r['section-fraction MCC pick']['alpha_B']:.3g}",
                          "its section rate": r["section-fraction MCC pick"]["section rate"]}
                         for r in [prim] + sens])
    rg_rows = []
    for k in NULLS:
        g = gm_prim[gm_prim["null"] == k]
        r_ = {"null": k}
        for ab in (0.002, 0.01, 0.05, 0.1, 0.2):
            j = int(np.argmin(np.abs(np.log(g["alpha_B"].to_numpy()) - np.log(ab))))
            r_[f"alpha_B {ab:g}"] = float(rogan_gladen(g["section rate"].iloc[j], g["sens"].iloc[j], g["fpr"].iloc[j]))
        rg_rows.append(r_)
    rgt = pd.DataFrame(rg_rows)
    bn = pd.DataFrame({"null": list(NULLS),
                       "best MCC": [prim["best MCC by null"][k] for k in NULLS],
                       "at alpha_B": [prim["best alpha_B by null"][k] for k in NULLS],
                       "section rate there": [section_rate(k, prim["best alpha_B by null"][k], "MisTIC") for k in NULLS]})
    text = f"""
## BASE-T-trunc calibrated on Wu thinned to this section's depth ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_wu_calibration.py`. Requested by the user (2026-10-17) as a less circular check on
the G4 calibration: the truth is Wu's whole-transcriptome Seurat score, and the label rate on this
section is an output. Wu restricted to the {len(shared):,} shared panel genes and thinned within type
to this section's depth (A6's thinning; V15, V16 passed); {a.seeds} thinning seeds. Every null is the
plain log N null V16 checked (thinned scRNA-seq has no MisTIC term), refitted on the thinned counts.

**Truth.** Seurat's own call (max(S, G2M) > 0) cuts through the non-cycling bulk in some types:
33% of ER+ T-cells are called S/G2M (bulk centred near -0.03) against 6% of Cancer Epithelial. The
primary truth therefore uses margins on the score: positive above {POS_CUT}, negative at or below 0,
the band between not scored, and only cells with at least {UMI_FLOOR:,} UMIs scored. Primary scope
ER+ patients, cells weighted to this section's type mix; metric weighted MCC.

### Fidelity: alpha0 of the plain-N null as alpha_trim grows

Mean over seeds. Rate 0.1 = label rate at alpha_B 0.1.

{md_table(ftab)}

### Best setting per null (primary truth)

{md_table(bn)}

### Choice by the rule, and how it moves with the truth

The primary row is the rule's pick. The section rate band covers the alpha_B values whose MCC is
within {NEAR} of the best, for that null.

{md_table(stab)}

**Pick: {prim['null']}, alpha_B {prim['alpha_B']:.4g}** (band {prim['alpha_B band'][0]:.4g}-{prim['alpha_B band'][1]:.4g}).
Weighted MCC {prim['MCC']:.3f}; on Wu, sensitivity {prim['Wu sensitivity']:.3f}, precision
{prim['Wu precision']:.3f}, depth bias {prim['Wu depth bias']:.2f}. On this section it labels
**{prim['section rate (MisTIC null)']:.1%}** of cells under the production MisTIC null (band
{prim['section rate band (MisTIC)'][0]:.1%}-{prim['section rate band (MisTIC)'][1]:.1%}) and
{prim['section rate (plain-N null)']:.1%} under the plain-N null. The G4 calibration chose
{g4['null']}, alpha_B {g4['alpha_B']:.4g} at a fixed 10% rate.

**Prevalence-free readings (added after the smoke run showed the MCC pick tracks Wu's
prevalence).** The MCC-optimal operating point depends on how common the positives are, and
confident positives are rare in ER+ Wu ({prim['truth positive rate']:.1%} of scored cells), so the
rule's pick partly encodes Wu's biology. Sensitivity and false-positive rate do not depend on
prevalence. Youden's J (sensitivity minus false-positive rate) picks
{prim['Youden pick']['null']} at alpha_B {prim['Youden pick']['alpha_B']:.3g}, labelling
{prim['Youden pick']['section rate']:.1%} of this section. MCC re-evaluated at this section's own
estimated fraction picks {prim['section-fraction MCC pick']['null']} at alpha_B
{prim['section-fraction MCC pick']['alpha_B']:.3g} ({prim['section-fraction MCC pick']['section rate']:.1%}
of this section).

Rogan-Gladen estimate of this section's cycling fraction, (section label rate - Wu FPR) / (Wu
sensitivity - Wu FPR), primary truth. If Wu's error rates transferred, each row would be flat
across alpha_B:

{md_table(rgt, fmt="{:.3f}")}

![MCC against section rate](wu_calibration/mcc_vs_rate.png)

Done in {(time.time() - t0) / 60:.0f} min. Not propagated to Stages 3-7; the user decides.
"""
    if not a.smoke:
        with open(f"{OUT}/report.md", "a") as fh:
            fh.write(text)
    say(text)


if __name__ == "__main__":
    main()
