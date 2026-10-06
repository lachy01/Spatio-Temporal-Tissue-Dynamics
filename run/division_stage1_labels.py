"""Division classifier comparison, Stage 1: build and freeze the labels.

    python run/division_stage1_labels.py

Spec sections 2 and 8 (V2, V3, V10). Writes outputs/division_classifier/
labels.parquet and the Stage 1 note in report.md. Labels are target
definitions: they are fitted once on all analysis cells and frozen here, before
any model is fitted.

Stops (non-zero exit) on any failing V-test, after writing the failure to
report.md, as spec section 8 requires.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.division import labels as L                       # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                  # noqa: E402
from tissue_dynamics.mixture.data import _tirosh_genes, load_config    # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------
HEADER = """# Division classifier comparison

Spec: `instructions/division_classifier_comparison_spec.md`. This file is the run
log: pre-stated rules and expectations first, then one status note per stage.

## Interpretive rule (written before any results)

- Each LOG arm is trained directly on its own label, so it is expected to win
  label-scale scores on that label.
- A label-scale win for a LOG arm is not evidence that it infers division better.
- NB and MIX can beat a LOG arm on the label scale only if their count likelihood
  carries information that binarisation discards.
- Section 7 decides which approach recovers the cycling state, including whether
  the OSDR pipeline (LOG-O) or a score pipeline (LOG-S) matches MIX.
- Label scale, P(L = 1 | x), and state scale, P(Z = 1 | x), are never compared
  with each other.

## Pre-stated expectations (spec section 11; recorded before Stage 4, not edited)

- **X1.** Each LOG arm has the best label-scale log loss on its own label.
- **X2.** NB-c's calibration-in-the-large changes sign across the threshold grids:
  over-prediction where c_i falls in 5 to 10, under-prediction where c_i falls in
  15 to 30. This follows the NB rootogram wave.
- **X3.** MIX is better calibrated than NB-c across the grids, but under-predicts
  at the strictest thresholds (its 30+ tail shortfall).
- **X4.** NB-d moves predictions mainly in Cancer Epithelial.
- **X5.** Disagreement between NB and MIX concentrates in Cancer Epithelial and
  Plasmablasts.
- **X6.** BASE-O's null call rate at Td = 0.5 is several percent, not near zero.
  Where mu0 is below 1, a count of 1 or 2 already clears the threshold, so OSDR's
  [0, 1] robustness range will not transfer.
- **X7.** Among cells with y = 0, the SCORE value has non-trivial spread, coming
  entirely from the control-gene term.
- **X8.** Under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from
  1 than MIX, with the largest distortion in small, low-depth cell types.

## Decisions and deviations from the spec

| item | spec | adopted | why |
| --- | --- | --- | --- |
| NB arms' depth term | fixed offset, theta = 1 | **theta free**, as MIX | parity with MIX, so arms differ only in likelihood; fixing the offset cost ~880 log-likelihood on the division NB. Fixed-offset NB and MIX with theta = 1 are sensitivity rows. Decided by the user. |
| technical null depth term | fixed offset | fixed offset, raw log N | unchanged; labels are definitions |
| NB-d dispersion | type indicators | **log alpha_i = gamma0 + phi_i^T gamma**, 8-dim projected state | matches the earlier dispersion submodel the spec refers to. Decided by the user. |
| Wu et al. reference; nucleus-only matrix | used in 5.2 and stage 7 | code paths built; **runs wait for the user to supply the files** | neither is on this machine. Decided by the user. |
| stage gating | stop after every stage | review after stages 1, 3 and 6; stop on any V-test failure | decided by the user |
| scanpy `score_genes` pool | pool excludes Tirosh and blocklist | pool = panel minus blocklist, plus the Tirosh genes for binning only, `ctrl_as_ref=False` | scanpy 1.12 raises a KeyError if scored genes are absent from the pool; `ctrl_as_ref=False` still keeps them out of the control draw |
| OSDR's SD (BASE-O) | check Supp. Fig. 1D | **per-type s+ primary**; global s+ as the sensitivity BASE-O-global | Supp. Fig. 1D was not supplied, but OSDR's code settles it: `tdm.preprocess.ki67.is_dividing` (github.com/JonathanSomer/osdr, commit 7d04970) sets one cutoff per cell type, `Td * sigma_type + Tn`, with `sigma_type = np.std(x[x > Tn] - Tn)` (ddof 0), and the Danenberg labels are built through it (Tn = 0.5, Td = 0.5). Consistent with the Methods ("to correct for different scaling and division rates between cell lines") and with the function's docstring ("The transformed values should have similar distributions accross different cell types"). |
| primary alpha_B (BASE-T) | 0.001 | **0.05** | at 0.001 the label called 0.18% of cells (9 positives among CAFs); 0.05 has PPV 0.86 under MIX and sits in OSDR's range of label rates. Full grid still reported. Decided by the user after Stage 1. |
| SCORE threshold | robust half-normal, p < 0.001 | unchanged, literal | its zero-count dependence is reported as a finding. Decided by the user. |
| state covariates | -- | **log1p-normalised PhiSpace** (`phispace_scores_major_lognorm.csv`), cells with N >= 10 | the first run (SpaNorm-based scores) stopped in Stage 2: those scores were ~86% depth, an artefact shown by a marker-gene test. That run's report is archived in `archive_spanorm/report.md`. Shallow cells (N < 10, 17.4%) are dropped as analysis cells and left out of neighbour averages; their labels defaulted to Endothelial. Decided by the user. |
| oxygen field source | -- | new major Endothelial score, typed cells | the old source (SpaNorm minor run) gave a field with AUROC 0.515 for marker endothelium against marker-negative cells; the new field 0.788 |
| LOG-O regression | "OSDR's pipeline" | matched covariates, pooled over types | OSDR fits a separate logistic regression per cell type on neighbour counts within 80 um (`tdm.model.Model`, "two models per cell_type"). LOG-O reproduces OSDR's **label** exactly in translation, but its regression is the shared weight block, for covariate parity. It is OSDR's labelling with our regression, not OSDR end to end. |
"""


def report(text):
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(REPORT):
        with open(REPORT, "w") as fh:
            fh.write(HEADER)
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def fail(test, detail):
    report(f"\n### STOPPED: {test} failed\n\n{detail}\n\nPer spec section 8, nothing "
           "further was run.")
    say(f"\n*** {test} FAILED -- stopping. {detail}")
    sys.exit(1)


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in r.to_numpy()]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    phi_names = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = phi_names
    basis = build_basis(df, cfg, verbose=False)
    n = len(df)
    y = df["y_count"].to_numpy(float)
    N = df["N"].to_numpy(float)
    A = df["area"].to_numpy(float)
    ctype = L.cell_type(df, phi_names)
    agree = float(np.mean(ctype == df["archetype"].astype(str).to_numpy()))
    say(f"{n:,} cells; cell type = standardised argmax, agrees with stored label "
        f"for {100 * agree:.2f}%")

    # ---- the raw matrix: recheck the outcome and library size ---------------
    import scanpy as sc

    say("reading the Xenium count matrix ...")
    ad = sc.read_10x_h5(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", gex_only=True)
    ad.var_names_make_unique()
    ad = ad[df["cell_id"].astype(str).to_numpy()].copy()
    genes = np.asarray(ad.var_names)
    tirosh = [g for g in _tirosh_genes() if g in set(genes)]
    block = list(dict.fromkeys(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0]
                               .astype(str)))
    block = [g for g in block if g in set(genes)]
    X = ad.X.tocsr()
    gi = {g: i for i, g in enumerate(genes)}
    y_chk = np.asarray(X[:, [gi[g] for g in tirosh]].sum(1)).ravel()
    excl = sorted({gi[g] for g in tirosh} | {gi[g] for g in block})
    N_chk = np.asarray(X.sum(1)).ravel() - np.asarray(X[:, excl].sum(1)).ravel()
    y_ok = bool(np.array_equal(y_chk, y))
    N_ok = bool(np.array_equal(np.rint(N_chk), N))
    say(f"  Tirosh genes in panel: {len(tirosh)}; y recomputed equal: {y_ok}; "
        f"N recomputed equal: {N_ok}")
    if not (y_ok and N_ok):
        fail("outcome/library identity",
             f"recomputed y equal = {y_ok}, N equal = {N_ok}. The labels must use "
             "the same gene list and library size as the mixture (spec 2.1).")

    # ---- 2.1 technical null --------------------------------------------------
    Xa = np.column_stack([np.ones(n), basis.log_area_basis(df)])
    off = np.log(N)                        # raw log N: the null fixes theta = 1
    say("fitting the robust technical null ...")
    null = L.robust_null(y, Xa, off, say=say)
    mu0, a0 = null["mu"], null["alpha"]
    sd0 = np.sqrt(mu0 + a0 * mu0 ** 2)
    say(f"  alpha0 = {a0:.4f}, mean mu0 = {mu0.mean():.4f}, stopped on "
        f"{null['stopped_on']} after {len(null['trace'])} fits")

    out = pd.DataFrame({"cell_id": df["cell_id"].astype(str).to_numpy(), "cell_type": ctype,
                        "y": y.astype(np.int64), "N": N.astype(np.int64), "A": A,
                        "mu0": mu0, "alpha0": a0, "sd0": sd0})

    # ---- 2.2 BASE-T -----------------------------------------------------------
    for ab in L.ALPHA_B_GRID:
        Lt, c, ai, p = L.base_t(y, mu0, a0, ab)
        tg = L.ab_tag(ab)
        out[f"cT_{tg}"] = c
        out[f"alphai_{tg}"] = ai
        out[f"LT_{tg}"] = Lt
        if ab == L.ALPHA_B_PRIMARY:
            out["p_tail"] = p
        if not np.array_equal(Lt, y >= c):
            fail("BASE-T equivalence", f"1{{p < alpha_B}} != 1{{y >= c}} at alpha_B={ab}")

    # discreteness: realised false-positive rate by depth decile
    dec = pd.qcut(N, 10, labels=False, duplicates="drop")
    disc = []
    for d in np.unique(dec):
        m = dec == d
        row = {"N decile": int(d) + 1, "N median": float(np.median(N[m])),
               "mu0 median": float(np.median(mu0[m]))}
        for ab in L.ALPHA_B_GRID:
            tg = L.ab_tag(ab)
            row[f"c^T median @{ab:g}"] = float(np.median(out.loc[m, f"cT_{tg}"]))
            row[f"mean alpha_i @{ab:g}"] = float(out.loc[m, f"alphai_{tg}"].mean())
        disc.append(row)
    disc = pd.DataFrame(disc)

    # ---- 2.3 BASE-O and variants ---------------------------------------------
    r = (y - mu0) / sd0
    out["r"] = r
    s_glob = L.base_o_scale(r)
    s_type = L.base_o_scale(r, ctype)
    out["a"] = r / s_type                    # primary: per type, as OSDR
    out["a_global"] = r / s_glob
    # BASE-O-neg: decoding false-positive floor from negative-control codewords
    cq = pd.read_csv(f"{cfg['data']['xenium_dir']}/cells.csv.gz",
                     usecols=["cell_id", "control_codeword_counts", "cell_area"])
    cq = cq.set_index(cq["cell_id"].astype(str)).reindex(out["cell_id"])
    import h5py

    with h5py.File(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", "r") as fh:
        ftypes = [x.decode() for x in fh["matrix/features/feature_type"][:]]
    n_negcw = int(sum(t == "Negative Control Codeword" for t in ftypes))
    rho_neg = float(cq["control_codeword_counts"].sum() / cq["cell_area"].sum())
    b_neg = rho_neg * A * len(tirosh) / n_negcw
    sd_neg = np.sqrt(b_neg)
    r_neg = (y - b_neg) / sd_neg
    s_neg = L.base_o_scale(r_neg)
    out["b_neg"] = b_neg
    out["a_neg"] = r_neg / s_neg
    say(f"  BASE-O: s+ per type {s_type.min():.3f} to {s_type.max():.3f} (primary), "
        f"global {s_glob[0]:.4f}; negative-codeword floor: rho "
        f"{rho_neg:.3e} per um^2, {n_negcw} codewords, mean b_i {b_neg.mean():.4f}")

    v10_bad = []
    for td in L.TD_GRID:
        tg = L.td_tag(td)
        for nm, (b, sd, sp_) in {"": (mu0, sd0, s_type), "global": (mu0, sd0, s_glob),
                                 "neg": (b_neg, sd_neg, s_neg)}.items():
            Lo, c, a = L.base_o(y, b, sd, sp_, td)
            sfx = f"_{nm}" if nm else ""
            out[f"cO{sfx}_{tg}"] = c
            out[f"LO{sfx}_{tg}"] = Lo
            mis_a = int(np.sum(Lo != (a > td)))
            mis_c = int(np.sum(Lo != (y >= c)))
            if mis_a or mis_c:
                v10_bad.append(f"BASE-O{('-' + nm) if nm else ''} Td={td}: "
                               f"{mis_a} cells differ from a>Td, {mis_c} from y>=c")
    if v10_bad:
        fail("V10 (BASE-O discrete equivalence)", "; ".join(v10_bad))
    say("  V10 passed: 1{a > Td} == 1{y >= c^O} for every cell at every Td")

    # ---- 2.4 SCORE ------------------------------------------------------------
    say("computing the module score ...")
    score, sinfo = L.module_score(X, genes, N, tirosh, block, seed=0)
    thr = L.score_threshold(score)
    LS = score > thr["threshold"]
    out["score"] = score
    out["LS"] = LS
    say(f"  SCORE: mode {thr['mode']:.4f}, sigma {thr['sigma']:.4f}, threshold "
        f"{thr['threshold']:.4f}, call rate {100 * LS.mean():.2f}%")

    # ---- V3 -------------------------------------------------------------------
    import test_division_survival as tds

    v3 = dict(nb=tds.max_abs_diff_nb(), pois=tds.max_abs_diff_pois())
    if max(v3.values()) >= 1e-12:
        fail("V3 (survivor functions)", json.dumps(v3))

    # ---- V2: null calibration, end to end -------------------------------------
    say("V2: simulating from the technical null and rebuilding the labels ...")
    rng = np.random.default_rng(2026)
    ys = L.sim_nb(mu0, a0, rng)
    null_s = L.robust_null(ys, Xa, off, say=lambda *a: None)
    mu_s, a_s = null_s["mu"], null_s["alpha"]
    sd_s = np.sqrt(mu_s + a_s * mu_s ** 2)
    v2_rows, v2_fail = [], []
    for ab in L.ALPHA_B_GRID:
        Lt_s, _c, ai_s, _p = L.base_t(ys, mu_s, a_s, ab)
        # the frozen labels applied to null data, for comparison
        Lt_f = ys >= out[f"cT_{L.ab_tag(ab)}"].to_numpy()
        rate, target = float(Lt_s.mean()), float(ai_s.mean())
        rel = rate / target - 1.0
        ok = abs(rel) <= 0.30
        v2_rows.append({"alpha_B": ab, "label rate (refit)": rate, "mean alpha_i": target,
                        "relative error": rel, "pass (+/-30%)": "pass" if ok else "FAIL",
                        "label rate (frozen c)": float(Lt_f.mean()),
                        "mean alpha_i (frozen)": float(out[f"alphai_{L.ab_tag(ab)}"].mean())})
        if not ok:
            v2_fail.append(f"alpha_B={ab}: rate {rate:.3e} vs mean alpha_i {target:.3e}")
    r_s = (ys - mu_s) / sd_s
    s_s = L.base_o_scale(r_s, ctype)
    o_rows = []
    for td in L.TD_GRID:
        Lo_s, _c, _a = L.base_o(ys, mu_s, sd_s, s_s, td)
        Lo_f = ys >= out[f"cO_{L.td_tag(td)}"].to_numpy()
        o_rows.append({"Td": td, "null call rate (rebuilt)": float(Lo_s.mean()),
                       "null call rate (frozen c)": float(Lo_f.mean()),
                       "observed call rate": float(out[f"LO_{L.td_tag(td)}"].mean())})
    v2 = pd.DataFrame(v2_rows)
    v2o = pd.DataFrame(o_rows)
    say(v2.to_string(index=False))
    say(v2o.to_string(index=False))

    # ---- SCORE diagnostics ----------------------------------------------------
    LT, LO = out[f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}"].to_numpy(), \
        out[f"LO_{L.td_tag(L.TD_PRIMARY)}"].to_numpy()
    kap = [{"stratum": "all", "n": n, "kappa vs BASE-T": L.cohen_kappa(LS, LT),
            "kappa vs BASE-O": L.cohen_kappa(LS, LO),
            "SCORE rate": float(LS.mean()), "BASE-T rate": float(LT.mean()),
            "BASE-O rate": float(LO.mean())}]
    for t in phi_names:
        m = ctype == t
        kap.append({"stratum": t, "n": int(m.sum()),
                    "kappa vs BASE-T": L.cohen_kappa(LS[m], LT[m]),
                    "kappa vs BASE-O": L.cohen_kappa(LS[m], LO[m]),
                    "SCORE rate": float(LS[m].mean()), "BASE-T rate": float(LT[m].mean()),
                    "BASE-O rate": float(LO[m].mean())})
    kap = pd.DataFrame(kap)
    z0 = y == 0
    zero = {"cells with y = 0": int(z0.sum()),
            "score sd, y = 0": float(score[z0].std()),
            "score sd, all cells": float(score.std()),
            "score IQR, y = 0": float(np.subtract(*np.percentile(score[z0], [75, 25]))),
            "score 1st-99th pct, y = 0": "%.3f to %.3f" % tuple(np.percentile(score[z0], [1, 99])),
            "SCORE call rate among y = 0": float(LS[z0].mean()),
            "share of SCORE calls with y <= 1": float(np.mean(y[LS] <= 1)) if LS.any() else float("nan")}

    # ---- freeze ---------------------------------------------------------------
    out.to_parquet(f"{OUT}/labels.parquet", index=False)
    meta = dict(n_cells=n, tirosh_genes=tirosh, n_tirosh=len(tirosh), n_blocklist=len(block),
                alpha0=a0, null_beta=null["beta"].tolist(), null_trace=null["trace"],
                null_stopped_on=null["stopped_on"], s_plus_global=float(s_glob[0]),
                s_plus_type={t: float(s_type[ctype == t][0]) for t in phi_names},
                rho_neg=rho_neg, n_neg_codewords=n_negcw, s_plus_neg=float(s_neg[0]),
                score=dict(mode=thr["mode"], sigma=thr["sigma"], threshold=thr["threshold"],
                           **sinfo),
                v3=v3, cell_type_agreement=agree,
                logN_residualised_in_basis=basis.logN_ortho_coef is not None)
    json.dump(meta, open(f"{OUT}/labels_meta.json", "w"), indent=2, default=float)
    plot_stage1(out, thr, disc)

    # ---- report ---------------------------------------------------------------
    rates = []
    for t in ["all"] + list(phi_names):
        m = np.ones(n, bool) if t == "all" else ctype == t
        row = {"cell type": t, "cells": int(m.sum())}
        for ab in L.ALPHA_B_GRID:
            row[f"BASE-T {ab:g}"] = float(out.loc[m, f"LT_{L.ab_tag(ab)}"].mean())
        row["BASE-O 0.5"] = float(out.loc[m, "LO_td050"].mean())
        row["BASE-O-global 0.5"] = float(out.loc[m, "LO_global_td050"].mean())
        row["BASE-O-neg 0.5"] = float(out.loc[m, "LO_neg_td050"].mean())
        row["SCORE"] = float(out.loc[m, "LS"].mean())
        rates.append(row)
    rates = pd.DataFrame(rates)
    tr = pd.DataFrame(null["trace"])[["iteration", "n_fit", "alpha0", "flagged",
                                      "flagged_frac", "changed", "converged"]]
    v2_ok = not v2_fail
    report(f"""
## Stage 1 -- labels (built and frozen {time.strftime('%Y-%m-%d %H:%M')})

{n:,} analysis cells. Cell type is the argmax of column-standardised PhiSpace
scores, agreeing with the stored label for {100 * agree:.2f}% of cells.

**Outcome and library size.** y recomputed from the count matrix with the
mixture's {len(tirosh)}-gene Tirosh list equals the table's `y_count` for every
cell; N (excluding the Tirosh genes and the {len(block)}-gene blocklist) equals the
table's `N` for every cell.

### Technical null (spec 2.1)

log mu0 = b0 + log N + s(log A), area spline as the NB arms (df 8), raw log N as
a fixed offset. Robust fit by iterative trimming at P(Y >= y) < 0.001; stopped on
**{null['stopped_on']}** after {len(null['trace'])} fits. Final alpha0 = **{a0:.4f}**.

{md_table(tr)}

### Label rates

{md_table(rates)}

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

{md_table(disc)}

### BASE-O (spec 2.3)

Primary BASE-O scales the residual within each cell type, as OSDR's own code
does (`is_dividing`: one cutoff per type, `Td * sigma_type + Tn`, ddof 0).
Per-type s+: {', '.join(f'{t} {meta["s_plus_type"][t]:.3f}' for t in phi_names)}.
BASE-O-global uses one s+ = {s_glob[0]:.4f} for all cells.
Negative-codeword floor (BASE-O-neg, decoding noise only, no spillover):
rho = {rho_neg:.3e} per um^2 over {n_negcw} negative-control codewords,
|T| = {len(tirosh)}, mean b_i = {b_neg.mean():.2e} against mean mu0 = {mu0.mean():.4f};
s+ = {s_neg[0]:.4f}.

Caveat: per-type scaling raises the effective threshold in types with more
cycling cells, so it partly removes between-type proliferation differences.
That is a property of OSDR's construction, inherited here.

### SCORE (spec 2.4)

scanpy `score_genes`, {len(tirosh)} genes, n_bins 25, ctrl_size 50, control pool
{sinfo['n_pool']:,} genes (panel minus blocklist), log1p(X s / N) with s = median
N = {sinfo['normaliser_s']:.0f}. Robust null: KDE mode {thr['mode']:.4f}, left-half
RMS sigma {thr['sigma']:.4f}, threshold {thr['threshold']:.4f} (upper-tail p <
0.001).

{md_table(kap)}

**Zero-count diagnostic.**

{md_table(pd.DataFrame([zero]))}

### Validation

- **V3** survivor functions vs scipy: max |diff| NB {v3['nb']:.2e}, Poisson
  {v3['pois']:.2e} (< 1e-12): **pass**.
- **V10** BASE-O discrete equivalence, every cell, every Td, all three
  variants: **pass**.
- **V2** null calibration, end to end (simulate from the null, refit the robust
  null, rebuild the labels): **{'pass' if v2_ok else 'FAIL'}**.

{md_table(v2)}

BASE-O call rate on null data (reported; no pass criterion):

{md_table(v2o)}

Stage 1 complete in {(time.time() - t0) / 60:.1f} min. **Review point** (stages 1,
3 and 6 pause for review).
""")
    if not v2_ok:
        fail("V2 (null calibration)", "; ".join(v2_fail))
    say(f"\nStage 1 done in {(time.time() - t0) / 60:.1f} min; wrote {OUT}/")


def plot_stage1(out, thr, disc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    s = out["score"].to_numpy()
    z0 = out["y"].to_numpy() == 0
    bins = np.linspace(np.percentile(s, 0.1), np.percentile(s, 99.9), 160)
    ax.hist(s[~z0], bins=bins, color="#2a78d6", alpha=0.75, label="y >= 1")
    ax.hist(s[z0], bins=bins, color="#8a8985", alpha=0.6, label="y = 0")
    ax.axvline(thr["mode"], color=INK, lw=1, ls=":")
    ax.axvline(thr["threshold"], color="#e34948", lw=1.5,
               label=f"threshold {thr['threshold']:.3f}")
    ax.set_xlabel("module score")
    ax.set_ylabel("cells")
    ax.set_title("SCORE: cells with no Tirosh counts still spread", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for ab, c in zip(L.ALPHA_B_GRID, ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")):
        ax.plot(disc["N decile"], disc[f"mean alpha_i @{ab:g}"], marker="o", color=c, lw=2,
                label=f"alpha_B = {ab:g}")
        ax.axhline(ab, color=c, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("depth (N) decile")
    ax.set_ylabel("realised false-positive rate, mean alpha_i")
    ax.set_title("BASE-T: realised rate sits below alpha_B, varying with depth",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage1_labels.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
