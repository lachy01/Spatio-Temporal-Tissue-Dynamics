"""Division classifier, addendum stage A2 (part 3): the extended grid and TWO, scored.

    python run/division_addendum_a2_grid.py

Addendum section 2.1 on real data, out of fold, on the stored partition:

* LOG-T fitted on every BASE-T label of the extended grid alpha_B in
  {0.001, 0.01, 0.05, 0.1, 0.2}, for the naive null (Stage 1's, as extended in
  A1) and each truncated null (alpha_trim 0.001, 0.01, 0.05): 20 labels. The
  naive 0.05 label is Stage 3's own LOG-T; the other 19 are fitted here exactly
  as Stage 3 fits it (shared weight block plus b0 + theta log N, MIX's penalty,
  three seeds at most), each training fold predicting its held-out fold.
* NB-c, NB-d and MIX pushed through each label (Stage 2/3's out-of-fold
  predictive distributions; nothing refitted), and TWO-P and TWO-PNB
  (two_step.parquet).
* Scored as Stage 4 (`division_stage4_scores.Scorer`): log loss, Brier,
  AUROC, AUPRC, calibration-in-the-large, overall and per type, with the same
  300-replicate block bootstrap; paired differences within replicate.
  TWO-P and TWO-PNB are also scored on the primary BASE-O label.

Resumable: each label's LOG-T predictions are cached in cache_grid/.
Writes oof_grid.parquet, grid_fits.csv, scores_grid.csv and a section of
report.md. V7 (sd(eta) <= 20) is checked on every LOG-T fit.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from division_stage4_scores import N_BOOT, SEED, Scorer, ci           # noqa: E402
from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import mix_p_ge, nb_p_ge       # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data                  # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
CACHE = f"{OUT}/cache_grid"
N_FOLDS = 5
ARMS = {"NB-c": "nbc", "NB-d": "nbd", "MIX": "mix", "TWO-P": "twop", "TWO-PNB": "twopnb"}
METRICS = ("log loss", "Brier", "AUROC", "AUPRC", "mean p / rate", "mean p - rate")
PAIRS = [("MIX", "LOG-T"), ("TWO-P", "LOG-T"), ("MIX", "TWO-P"), ("TWO-PNB", "TWO-P")]


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


def grid_labels():
    """[(null name, alpha_B, label column, threshold column, source)]."""
    out = []
    for a in L.ALPHA_B_GRID_EXT:
        src = "labels" if a in L.ALPHA_B_GRID else "trunc"
        out.append(("naive", a, f"LT_{L.ab_tag(a)}", f"cT_{L.ab_tag(a)}", src))
    for at in L.TRUNC_GRID:
        tt = L.trunc_tag(at)
        for a in L.ALPHA_B_GRID_EXT:
            out.append((f"trunc {at:g}", a, f"LTtr{tt}_{L.ab_tag(a)}", f"cTtr{tt}_{L.ab_tag(a)}", "trunc"))
    return out


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    lt = pd.read_parquet(f"{OUT}/labels_trunc.parquet")
    oof = pd.read_parquet(f"{OUT}/oof_predictions.parquet")
    two = pd.read_parquet(f"{OUT}/two_step.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("labels_trunc", lt), ("oof_predictions", oof),
                        ("two_step", two), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    folds = fm["fold"].to_numpy()
    if not (np.array_equal(oof["fold"].to_numpy(), folds) and np.array_equal(two["fold"].to_numpy(), folds)):
        sys.exit("V5: out-of-fold files disagree with the stored folds")
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    n = len(df)
    ctype = lab["cell_type"].astype(str).to_numpy()
    G = grid_labels()

    def col(src, c):
        return (lab if src == "labels" else lt)[c].to_numpy(float)

    # ---- LOG-T on every grid label, out of fold -----------------------------------
    fit_rows, preds = [], {}
    primary = f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}"
    for null, a, lcol, ccol, src in G:
        if lcol == primary:
            preds[lcol] = oof[f"p_logt_{primary}"].to_numpy()
            continue
        cache = f"{CACHE}/logt_{lcol}.npz"
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            preds[lcol] = z["pred"]
            fit_rows += list(z["rows"])
            continue
        yl_all = col(src, lcol)
        pred, rows = np.zeros(n), []
        for f in range(N_FOLDS):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            Dtr = data.subset(tr).as_jax()
            yl = jnp.asarray(yl_all[tr])
            say(f"LOG-T {lcol} ({null}, alpha_B {a:g}) train fold {f}: label rate {float(yl.mean()):.4f}")
            fobj = arms.logistic_objective(Dtr, yl, len(tr), spec.weight_ridge, spec.n_reference,
                                           spec.ridge_tau, spec.b_penalty)
            for attempt in range(3):
                p, info = arms.minimise(fobj, arms.init_logistic(Dtr, yl, spec.rank, 100 * attempt))
                if info["converged"]:
                    break
            dg = arms.eta_diagnostics(p, Dtr)
            rows.append({"label": lcol, "null": null, "alpha_B": a, "fold": f,
                         "label rate (train)": float(yl.mean()), "seed": 100 * attempt,
                         "converged": "pass" if info["converged"] else "FAIL",
                         "iterations": info["lbfgs_iters"] + info["newton_iters"],
                         "grad_inf": info["grad_inf"], "theta (log N)": float(p["theta"]),
                         "sd(eta)": dg["sd_eta"], "sigma1(B)": dg["sigma1_B"],
                         "minutes": info["wall_time"] / 60})
            pred[te] = np.asarray(jax.nn.sigmoid(arms.logistic_eta(p, data.subset(te).as_jax())))
        np.savez(cache, pred=pred, rows=np.array(rows, dtype=object))
        preds[lcol] = pred
        fit_rows += rows
        say(f"  done ({(time.time() - t0) / 60:.0f} min)")
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(f"{OUT}/grid_fits.csv", index=False)
    v7_ok = bool((fits["sd(eta)"] <= 20).all())
    conv_ok = bool((fits["converged"] == "pass").all())

    # ---- every arm's p on every grid label --------------------------------------------
    og = pd.DataFrame({"cell_id": ids, "fold": folds})
    mu_c, al_c = oof["mu_nbc"].to_numpy(), oof["alpha_nbc"].to_numpy()
    mu_d, al_d = oof["mu_nbd"].to_numpy(), oof["alpha_nbd"].to_numpy()
    pi_m, l0, l1 = (oof[k].to_numpy() for k in ("pi_mix", "lam0_mix", "lam1_mix"))
    for null, a, lcol, ccol, src in G:
        c = col(src, ccol)
        og[f"p_logt_{lcol}"] = preds[lcol]
        og[f"p_nbc_{lcol}"] = nb_p_ge(c, mu_c, al_c)
        og[f"p_nbd_{lcol}"] = nb_p_ge(c, mu_d, al_d)
        og[f"p_mix_{lcol}"] = mix_p_ge(c, pi_m, l0, l1)
        for arm in ("twop", "twopnb"):
            og[f"p_{arm}_{lcol}"] = two[f"p_{arm}_{lcol}"].to_numpy()
    og.to_parquet(f"{OUT}/oof_grid.parquet", index=False)

    # ---- scoring ---------------------------------------------------------------------------
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb = int(bidx.max()) + 1
    rng = np.random.default_rng(SEED)
    M = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]).astype(float)
    strata = {"all": None, **{t: ctype == t for t in types}}
    scorers = {s: Scorer(bidx, nb, M, m) for s, m in strata.items()}
    rows, boots = [], {}

    def run(label, yv, arm, p, s, null="", a=np.nan):
        point, boot, _ = scorers[s].score(p, yv)
        boots[(label, arm, s)] = boot
        for m in METRICS:
            lo, hi = ci(boot[m])
            rows.append({"label": label, "null": null, "alpha_B": a, "arm": arm, "stratum": s,
                         "metric": m, "estimate": point[m], "ci_lo": lo, "ci_hi": hi})

    all_arms = {**ARMS, "LOG-T": "logt"}
    for null, a, lcol, ccol, src in G:
        yv = col(src, lcol)
        for arm, tag in all_arms.items():
            for s in strata:
                run(lcol, yv, arm, og[f"p_{tag}_{lcol}"].to_numpy(), s, null, a)
        say(f"  scored {lcol} ({(time.time() - t0) / 60:.0f} min)")
    lo_col = f"LO_{L.td_tag(L.TD_PRIMARY)}"
    yo = lab[lo_col].to_numpy(float)
    for arm, pc in {"NB-c": f"p_nbc_{lo_col}", "NB-d": f"p_nbd_{lo_col}", "MIX": f"p_mix_{lo_col}",
                    "LOG-O": f"p_logo_{lo_col}"}.items():
        for s in strata:
            run(lo_col, yo, arm, oof[pc].to_numpy(), s, "BASE-O")
    for arm in ("TWO-P", "TWO-PNB"):
        for s in strata:
            run(lo_col, yo, arm, two[f"p_{ARMS[arm]}_{lo_col}"].to_numpy(), s, "BASE-O")
    scores = pd.DataFrame(rows)
    diff_rows = []
    for label in [g[2] for g in G] + [lo_col]:
        own = "LOG-O" if label == lo_col else "LOG-T"
        for a_, b_ in PAIRS:
            a_, b_ = (own if x == "LOG-T" else x for x in (a_, b_))
            for m in ("log loss", "Brier", "AUROC", "AUPRC"):
                pa = scores.query("label == @label and arm == @a_ and stratum == 'all' and metric == @m")
                pb = scores.query("label == @label and arm == @b_ and stratum == 'all' and metric == @m")
                lo, hi = ci(boots[(label, a_, "all")][m] - boots[(label, b_, "all")][m])
                diff_rows.append({"label": label, "null": pa["null"].iloc[0], "alpha_B": pa["alpha_B"].iloc[0],
                                  "arm": f"{a_} - {b_}", "stratum": "all", "metric": m,
                                  "estimate": pa["estimate"].iloc[0] - pb["estimate"].iloc[0],
                                  "ci_lo": lo, "ci_hi": hi})
    diffs = pd.DataFrame(diff_rows)
    pd.concat([scores, diffs]).to_csv(f"{OUT}/scores_grid.csv", index=False)

    # ---- report --------------------------------------------------------------------------------
    def wide(metric, d=4):
        out = []
        for null, a, lcol, _, src in G:
            r = {"null": null, "alpha_B": f"{a:g}", "label rate": f"{col(src, lcol).mean():.4f}"}
            for arm in all_arms:
                q = scores.query("label == @lcol and arm == @arm and stratum == 'all' and metric == @metric").iloc[0]
                r[arm] = f"{q['estimate']:.{d}f}"
            out.append(r)
        return pd.DataFrame(out)

    def diff_wide(metric):
        out = []
        for null, a, lcol, _, _ in G:
            r = {"null": null, "alpha_B": f"{a:g}"}
            for pa, pb in PAIRS:
                name = f"{pa} - {pb}"
                q = diffs.query("label == @lcol and arm == @name and metric == @metric").iloc[0]
                flag = "*" if (q["ci_lo"] > 0 or q["ci_hi"] < 0) else ""
                r[f"{pa} - {pb}"] = f"{q['estimate']:+.5f}{flag}"
            out.append(r)
        return pd.DataFrame(out)

    bo = []
    for arm in ("NB-c", "NB-d", "MIX", "LOG-O", "TWO-P", "TWO-PNB"):
        q = scores.query("label == @lo_col and arm == @arm and stratum == 'all'").set_index("metric")
        bo.append({"arm": arm, **{m: f"{q.loc[m, 'estimate']:.4f} [{q.loc[m, 'ci_lo']:.4f}, {q.loc[m, 'ci_hi']:.4f}]"
                                  for m in ("log loss", "Brier", "AUROC", "AUPRC", "mean p / rate")}})
    bo = pd.DataFrame(bo)
    by_type = []
    prim = f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}"
    for s in strata:
        r = {"stratum": s}
        for arm in all_arms:
            q = scores.query("label == @prim and arm == @arm and stratum == @s and metric == 'mean p / rate'").iloc[0]
            r[arm] = f"{q['estimate']:.3f}" + ("" if q["ci_lo"] <= 1 <= q["ci_hi"] else "*")
        by_type.append(r)
    fit_sum = fits.groupby(["null", "alpha_B"]).agg(
        fits=("fold", "size"), converged=("converged", lambda s: f"{(s == 'pass').sum()}/{len(s)}"),
        **{"max sd(eta)": ("sd(eta)", "max")}, **{"mean theta": ("theta (log N)", "mean")},
        minutes=("minutes", "sum")).reset_index()
    report(f"""
## Addendum stage A2 -- extended grid and TWO, scored out of fold ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_addendum_a2_grid.py`. Every BASE-T label of the extended grid
(alpha_B 0.001-0.2) for the naive null and each truncated null: LOG-T fitted on
each (Stage 3's specification, five training folds), NB-c, NB-d and MIX pushed
through each (their out-of-fold predictive distributions, unchanged), and TWO-P
and TWO-PNB pushed through each (part 1). Scored as Stage 4 on the same 300
block-bootstrap replicates; * marks a 95% interval excluding 0 (differences) or
1 (calibration). Per the interpretive rule, LOG-T is trained on the label it is
scored against. The NB, MIX and TWO fits never see a label: across alpha_B
only their label-scale expression changes.

### LOG-T fits (V7, convergence)

All {len(fits)} new fold fits converged: **{'yes' if conv_ok else 'NO'}**. V7 (sd(eta) <= 20):
**{'pass' if v7_ok else 'FAIL'}** (largest {fits['sd(eta)'].max():.3f}).

{md_table(fit_sum)}

### Log loss (all cells; lower is better)

{md_table(wide('log loss'))}

### AUPRC (all cells)

{md_table(wide('AUPRC'))}

### AUROC (all cells)

{md_table(wide('AUROC'))}

### Calibration-in-the-large, mean p / label rate (all cells)

{md_table(wide('mean p / rate', 3))}

### Paired differences in log loss (negative: the first arm is better)

{md_table(diff_wide('log loss'))}

### Paired differences in AUPRC (positive: the first arm is better)

{md_table(diff_wide('AUPRC'))}

### Calibration by type at the primary label ({prim}; * = interval excludes 1)

{md_table(pd.DataFrame(by_type))}

### TWO on the primary BASE-O label ({lo_col})

{md_table(bo)}

Per-type scores for every label and arm, and the Brier and AUROC differences,
are in `scores_grid.csv`; out-of-fold p for every arm and label in
`oof_grid.parquet`. The primary alpha_B and null are not chosen here: by the
section 2.3 rule they come from the A5 simulation. Part 3 of A2 done in
{(time.time() - t0) / 60:.0f} min.
""")
    if not (v7_ok and conv_ok):
        report("\n### STOPPED in addendum stage A2: a LOG-T grid fit failed V7 or convergence\n\n"
               + md_table(fits[(fits['sd(eta)'] > 20) | (fits['converged'] != 'pass')]))
        sys.exit(1)
    say(f"grid done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
