"""Division classifier, addendum 1, stage A1: BASE-T-trunc.

    python run/division_addendum_a1.py [--smoke N]      # --smoke: N random cells, to a scratch folder

Addendum sections 1 and 2.1 (labels only): the technical null refitted by
iterated right-truncation (`labels.truncated_null`) at alpha_trim in
{0.001, 0.01, 0.05}, from the untrimmed fit; BASE-T labels on the extended grid
alpha_B in {0.001, 0.01, 0.05, 0.1, 0.2} for the naive null (the frozen Stage 1
fit, extended by 0.1 and 0.2) and for every truncated null.

Validation (blocking, addendum section 6):
  V11       simulate y from the naive production null (known beta0, theta_m,
            alpha0) with the real Nt and m; the truncated fit recovers all three
            to 2% at every alpha_trim. The naive-trimmed fit on the same data,
            trimmed at the same level, is reported for contrast.
  V2-trunc  simulate from each fitted truncated null, refit it, rebuild the
            labels: label rate within +/-30% of mean alpha_i at every alpha_B.
Diagnostics (section 1.4): fitted parameters against the naive fit; trimmed
fraction by N decile and cell type; label rates per type at every alpha_B; the
Rogan-Gladen fraction per type with sensitivity borrowed from MIX (Stage 5's
construction and block bootstrap), and how many types have an interval wholly
below 0. X9 and X10 are judged.

Writes labels_trunc.parquet, null_fits_trunc.json, trunc_diagnostics.csv and a
section of report.md, all in outputs/division_classifier/.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.division import labels as L                     # noqa: E402
from tissue_dynamics.division.survival import pois_p_ge               # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_BOOT, SEED = 300, 20261008          # Stage 5's bootstrap, so the intervals compare
GRID = L.ALPHA_B_GRID_EXT
TOL_V11, TOL_V2 = 0.02, 0.30
# V11 missed 2% on theta_m at alpha_trim 0.05 (+2.5%, z = 1.9; the untrimmed MLE of the
# same draw was already +1.1%): accepted by the user on 2026-10-13, to be tested on more
# draws later. A rerun reports the miss and continues instead of stopping.
V11_ACCEPTED = {0.05: 0.03}


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write(text.rstrip() + "\n")


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                       else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def stop(test, detail):
    report(f"\n### STOPPED in addendum stage A1: {test} failed\n\n{detail}\n\nPer the "
           "addendum's section 6, nothing further was run.\n")
    sys.exit(f"{test} failed: {detail}")


def labels_from(y, mu, alpha):
    """BASE-T on the extended grid: {alpha_B: (L, c, alpha_i)}."""
    out = {}
    for ab in GRID:
        Lt, c, ai, _p = L.base_t(y, mu, alpha, ab)
        out[ab] = (Lt, c, ai)
    return out


def untrimmed(y, X, off, m):
    return L.fit_nb_additive_null(y, X, off, m)


def params(fit):
    return {"beta0": float(np.asarray(fit["beta"])[0]), "theta_m": float(fit["theta_m"]),
            "alpha0": float(fit["alpha"])}


def main():
    global OUT, REPORT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--smoke-dir", default="/tmp/division_a1_smoke")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    oof = pd.read_parquet(f"{OUT}/oof_predictions.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    if a.smoke:
        sel = np.sort(np.random.default_rng(0).choice(len(df), a.smoke, replace=False))
        df, lab, oof, fm = (f.iloc[sel].reset_index(drop=True) for f in (df, lab, oof, fm))
        OUT = a.smoke_dir
        REPORT = f"{OUT}/report.md"
        os.makedirs(OUT, exist_ok=True)
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("oof_predictions", oof), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    if meta.get("technical") != "mistic":
        sys.exit("the frozen Stage 1 labels are not the MisTIC null's")
    y = df["y_count"].to_numpy(float)
    X, off, m = L.null_design(df, basis)
    n = len(y)
    ctype = lab["cell_type"].astype(str).to_numpy()
    types = df.attrs["phi_names"]
    say(f"{n:,} cells")

    # ---- the naive null: Stage 1's frozen fit -----------------------------------
    mu_nv, a_nv = lab["mu0"].to_numpy(float), float(lab["alpha0"].iloc[0])
    naive = {"beta0": float(meta["null_beta"][0]), "theta_m": float(meta["null_theta_m"]),
             "alpha0": a_nv}
    lab_nv = labels_from(y, mu_nv, a_nv)
    for ab in (0.05, 0.01, 0.001):          # the frozen labels must be reproduced exactly
        if not np.array_equal(lab_nv[ab][0], lab[f"LT_{L.ab_tag(ab)}"].to_numpy()):
            sys.exit(f"naive BASE-T at {ab} does not reproduce the frozen Stage 1 label")
    say("naive null reproduced: " + json.dumps({k: round(v, 4) for k, v in naive.items()}))

    # ---- truncated nulls on the real data -----------------------------------------
    say("untrimmed fit ...")
    full = untrimmed(y, X, off, m)
    fits, store = {}, {"cell_id": ids}
    for at in L.TRUNC_GRID:
        say(f"truncated null, alpha_trim {at:g} ...")
        fits[at] = L.truncated_null(y, X, off, m, at, full, say=say)
    for ab in (0.1, 0.2):
        Lt, c, ai = lab_nv[ab]
        tg = L.ab_tag(ab)
        store[f"LT_{tg}"], store[f"cT_{tg}"], store[f"alphai_{tg}"] = Lt, c, ai
    for at, f in fits.items():
        tt = L.trunc_tag(at)
        store[f"mu0tr{tt}"] = f["mu"]
        store[f"ctrim{tt}"] = f["c_trim"]
        for ab, (Lt, c, ai) in labels_from(y, f["mu"], f["alpha"]).items():
            tg = L.ab_tag(ab)
            store[f"LTtr{tt}_{tg}"], store[f"cTtr{tt}_{tg}"], store[f"alphaitr{tt}_{tg}"] = Lt, c, ai
    labels_trunc = pd.DataFrame(store)

    # ---- V11: known-null recovery ----------------------------------------------------
    say("V11: simulating from the naive null ...")
    rng = np.random.default_rng(2026_1013)
    ys = L.sim_nb(mu_nv, a_nv, rng)
    full_s = untrimmed(ys, X, off, m)
    v11 = []
    for at in L.TRUNC_GRID:
        ft = L.truncated_null(ys, X, off, m, at, full_s)
        fn = L.robust_null(ys, X, off, trim_p=at, m=m)
        for how, f in (("truncated", ft), ("naive-trimmed (contrast)", fn)):
            p = params(f)
            row = {"alpha_trim": at, "fit": how}
            for k in ("beta0", "theta_m", "alpha0"):
                row[k] = p[k]
                row[f"rel err {k}"] = p[k] / naive[k] - 1.0
            if how == "truncated":       # the same errors in standard errors of the fit
                se = f["se"]
                row["z beta0"] = (p["beta0"] - naive["beta0"]) / se["beta0"]
                row["z log theta_m"] = (np.log(p["theta_m"]) - np.log(naive["theta_m"])) / se["eta"]
                row["z log alpha0"] = (np.log(p["alpha0"]) - np.log(naive["alpha0"])) / se["log_alpha"]
            v11.append(row)
    v11 = pd.DataFrame(v11)
    say(v11.to_string(index=False))
    tr = v11[v11["fit"] == "truncated"]
    worst = float(tr[[c for c in v11.columns if c.startswith("rel err")]].abs().to_numpy().max())
    v11 = v11[[c for c in v11.columns if not c.startswith("z ")] + [c for c in v11.columns if c.startswith("z ")]]

    # ---- V2-trunc ------------------------------------------------------------------
    say("V2-trunc ...")
    v2 = []
    for at, f in fits.items():
        ys2 = L.sim_nb(f["mu"], f["alpha"], rng)
        ft = L.truncated_null(ys2, X, off, m, at, untrimmed(ys2, X, off, m))
        for ab, (Lt, _c, ai) in labels_from(ys2, ft["mu"], ft["alpha"]).items():
            rate, target = float(Lt.mean()), float(ai.mean())
            v2.append({"alpha_trim": at, "alpha_B": ab, "label rate (refit)": rate,
                       "mean alpha_i": target, "relative error": rate / target - 1.0})
    v2 = pd.DataFrame(v2)
    v2["pass (+/-30%)"] = np.where(v2["relative error"].abs() <= TOL_V2, "pass", "FAIL")
    say(v2.to_string(index=False))

    # ---- 1.4 diagnostics -----------------------------------------------------------
    nulls = {"naive (Stage 1)": (mu_nv, a_nv, None, naive)}
    for at, f in fits.items():
        nulls[f"trunc {at:g}"] = (f["mu"], f["alpha"], f["c_trim"], params(f))
    par = pd.DataFrame([{"null": k, **v[3], "trimmed fraction":
                         (float(np.mean(y >= v[2])) if v[2] is not None else
                          float(meta["null_trace"][-1]["flagged_frac"])),
                         "iterations": (len(fits[float(k.split()[1])]["trace"])
                                        if k.startswith("trunc") else len(meta["null_trace"])),
                         "stopped on": (fits[float(k.split()[1])]["stopped_on"]
                                        if k.startswith("trunc") else meta["null_stopped_on"])}
                        for k, v in nulls.items()])
    dec = pd.qcut(lab["N"].to_numpy(float), 10, labels=False, duplicates="drop") + 1
    trim_rows = []
    for at, f in fits.items():
        tr_ = y >= f["c_trim"]
        for d in range(1, int(dec.max()) + 1):
            trim_rows.append({"alpha_trim": at, "stratum": f"N decile {d}", "trimmed": float(tr_[dec == d].mean())})
        for t in types:
            trim_rows.append({"alpha_trim": at, "stratum": t, "trimmed": float(tr_[ctype == t].mean())})
    trim = pd.DataFrame(trim_rows).pivot(index="stratum", columns="alpha_trim", values="trimmed")
    trim = trim.reindex([f"N decile {d}" for d in range(1, int(dec.max()) + 1)] + list(types))

    # label rates per type, and Rogan-Gladen with MIX's sensitivity, block bootstrap
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb = int(bidx.max()) + 1
    brng = np.random.default_rng(SEED)
    M = np.stack([np.bincount(brng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]).astype(float)
    tix = {t: i for i, t in enumerate(types)}
    g = np.array([tix[t] for t in ctype])
    T = len(types)
    lam1 = oof["lam1_mix"].to_numpy()

    def sums(v):
        return np.bincount(bidx * T + g, weights=v, minlength=nb * T).reshape(nb, T)

    s_n = sums(np.ones(n))
    diag = []
    for name, (mu, a, _c, _p) in nulls.items():
        labs = lab_nv if name.startswith("naive") else labels_from(y, mu, a)
        for ab, (Lt, c, ai) in labs.items():
            sens = pois_p_ge(c.astype(float), lam1)
            s_r, s_a, s_s = sums(Lt.astype(float)), sums(ai), sums(sens)
            rate, abar, sen = (s.sum(0) / s_n.sum(0) for s in (s_r, s_a, s_s))
            Mn = M @ s_n
            rb, ab_, sb = ((M @ s) / Mn for s in (s_r, s_a, s_s))
            rg, rgb = (rate - abar) / (sen - abar), (rb - ab_) / (sb - ab_)
            lo, hi = np.percentile(rgb, [2.5, 97.5], axis=0)
            for k, t in enumerate(types):
                diag.append({"null": name, "alpha_B": ab, "type": t, "label rate": rate[k],
                             "mean alpha_i": abar[k], "sensitivity (MIX)": sen[k],
                             "Rogan-Gladen": rg[k], "RG lo": lo[k], "RG hi": hi[k]})
            diag.append({"null": name, "alpha_B": ab, "type": "all", "label rate": float(Lt.mean()),
                         "mean alpha_i": float(ai.mean()), "sensitivity (MIX)": float(sens.mean()),
                         "Rogan-Gladen": np.nan, "RG lo": np.nan, "RG hi": np.nan})
    diag = pd.DataFrame(diag)
    diag.to_csv(f"{OUT}/trunc_diagnostics.csv", index=False)

    # ---- write -----------------------------------------------------------------------
    labels_trunc.to_parquet(f"{OUT}/labels_trunc.parquet", index=False)
    json.dump({"naive": naive, "untrimmed": params(full),
               "truncated": {str(at): {**params(f), "stopped_on": f["stopped_on"],
                                       "trace": f["trace"]} for at, f in fits.items()},
               "v11": v11.to_dict("records"), "v2_trunc": v2.to_dict("records")},
              open(f"{OUT}/null_fits_trunc.json", "w"), indent=2, default=float)

    per = diag[diag["type"] != "all"]
    neg = per.assign(below=per["RG hi"] < 0).groupby(["null", "alpha_B"])["below"].sum().unstack()
    rates = diag[diag["type"] == "all"].pivot(index="null", columns="alpha_B", values="label rate")
    order = list(nulls)
    rates, neg = rates.reindex(order), neg.reindex(order)
    rg05 = per[per["alpha_B"] == 0.05].pivot(index="type", columns="null", values="Rogan-Gladen")[order].reindex(types)
    rg05_ci = per[per["alpha_B"] == 0.05].assign(
        cell=lambda d: d.apply(lambda r: f"{r['Rogan-Gladen']:+.3f} [{r['RG lo']:+.3f}, {r['RG hi']:+.3f}]", axis=1)
    ).pivot(index="type", columns="null", values="cell")[order].reindex(types)
    rate_t = per[per["alpha_B"] == 0.05].pivot(index="type", columns="null", values="label rate")[order].reindex(types)

    # X9 and X10
    a_tr = [nulls[f"trunc {at:g}"][3]["alpha0"] for at in L.TRUNC_GRID]
    x9a = all(a < a_nv for a in a_tr)
    x9b = all((rates.loc[f"trunc {at:g}"] > rates.loc["naive (Stage 1)"]).all() for at in L.TRUNC_GRID)
    x9c = all(neg.loc[f"trunc {at:g}"].sum() < neg.loc["naive (Stage 1)"].sum() for at in L.TRUNC_GRID)
    x9d = bool(np.all(np.diff(a_tr) < 0))
    v2_ok = bool((v2["pass (+/-30%)"] == "pass").all())
    nv05 = v11[(v11["fit"] != "truncated") & (v11["alpha_trim"] == 0.05)]
    bias05 = float(nv05[[c for c in v11.columns if c.startswith("rel err")]].abs().to_numpy().max())

    v11_ok = worst <= TOL_V11 or all(
        float(r[[c for c in v11.columns if c.startswith("rel err")]].abs().max())
        <= V11_ACCEPTED.get(r["alpha_trim"], TOL_V11) for _, r in tr.iterrows())
    text = [f"\n## Addendum stage A1 -- BASE-T-trunc ({time.strftime('%Y-%m-%d %H:%M')})\n",
            "The MisTIC null (mu0 = Nt e^b0 + theta_m m, NB2) refitted by iterated right-truncation "
            "(addendum section 1.2) from the untrimmed fit, at alpha_trim 0.001, 0.01 and 0.05. "
            "Labels on the extended grid alpha_B in {0.001, 0.01, 0.05, 0.1, 0.2} for every null; "
            "the naive Stage 1 labels are unchanged and reproduced exactly at 0.05, 0.01 and 0.001.\n",
            "### V11 -- known-null recovery\n",
            "y simulated from the naive production null (b0 "
            f"{naive['beta0']:.4f}, theta_m {naive['theta_m']:.4f}, alpha0 {naive['alpha0']:.4f}) with the "
            "real Nt and m; each fit started from the untrimmed fit of the simulated data. The "
            "naive-trimmed fit at the same level is the contrast (not part of the criterion).\n",
            md_table(v11), "",
            f"Largest relative error of the truncated fit: {worst:.4f} (criterion {TOL_V11}): "
            f"**{'pass' if worst <= TOL_V11 else 'accepted exception (user, 2026-10-13)' if v11_ok else 'FAIL'}**. The z columns give the same errors in standard "
            "errors of the truncated fit (exact Hessian), so a miss can be told from sampling noise.\n",
            "### V2-trunc -- null calibration, end to end\n",
            md_table(v2), "", f"**{'pass' if v2_ok else 'FAIL'}**.\n"]
    report("\n".join(text))
    if not v11_ok:
        stop("V11 (truncated-likelihood recovery)", f"largest relative error {worst:.4f} > {TOL_V11}")
    if not v2_ok:
        stop("V2-trunc", "; ".join(f"alpha_trim {r['alpha_trim']}, alpha_B {r['alpha_B']}: "
                                   f"{r['relative error']:+.3f}" for _, r in v2[v2['pass (+/-30%)'] != 'pass'].iterrows()))

    text = ["### Fitted nulls\n", md_table(par), "",
            "Trimmed fraction (y >= c_trim at the final fit) by N decile and cell type:\n",
            md_table(trim.reset_index().rename(columns=lambda c: f"alpha_trim {c:g}" if isinstance(c, float) else c)), "",
            "### Label rates, all cells\n",
            md_table(rates.reset_index().rename(columns=lambda c: f"alpha_B {c:g}" if isinstance(c, float) else c)), "",
            "BASE-T rate by type at alpha_B 0.05:\n", md_table(rate_t.reset_index()), "",
            "### Rogan-Gladen, sensitivity borrowed from MIX\n",
            "Fraction (rate - mean alpha_i) / (sens - mean alpha_i) per type, sens = mean P1(Y >= c^T_i) "
            "under MIX's out-of-fold cycling component; 95% block-bootstrap intervals (Stage 5's 300 "
            "replicates). At alpha_B 0.05:\n", md_table(rg05_ci.reset_index()), "",
            "Number of types (of 9) whose interval lies wholly below 0:\n",
            md_table(neg.reset_index().rename(columns=lambda c: f"alpha_B {c:g}" if isinstance(c, float) else c)), "",
            "Every type, null and alpha_B is in `trunc_diagnostics.csv`; the labels in "
            "`labels_trunc.parquet` (`LTtr{trim}_{alpha_B}`, with `cTtr...` and `alphaitr...`; the "
            "naive grid extension as `LT_a1`, `LT_a2`); the iteration traces in `null_fits_trunc.json`.\n",
            "### X9 and X10\n",
            f"- **X9** (lower alpha0, higher label rate at each alpha_B, fewer negative Rogan-Gladen "
            f"types, gains growing with alpha_trim): alpha0 lower at every alpha_trim: "
            f"**{'yes' if x9a else 'no'}** ({', '.join(f'{a:.4f}' for a in a_tr)} against {a_nv:.4f}); "
            f"label rate higher at every alpha_B: **{'yes' if x9b else 'no'}**; fewer types below 0 "
            f"(summed over alpha_B): **{'yes' if x9c else 'no'}**; gain grows with alpha_trim (alpha0 "
            f"falls monotonically): **{'yes' if x9d else 'no'}**. Overall: "
            f"**{'holds' if all((x9a, x9b, x9c, x9d)) else 'partly holds' if any((x9a, x9b, x9c, x9d)) else 'does not hold'}**.",
            f"- **X10** (V2-trunc passes at every alpha_trim; truncation removes the bias naive "
            f"trimming shows at 0.05): V2-trunc **{'passes' if v2_ok else 'fails'}**; in V11 the "
            f"naive-trimmed fit at 0.05 is off by up to {100 * bias05:.1f}% against "
            f"{100 * worst:.2f}% for the truncated fit. Overall: "
            f"**{'holds' if v2_ok and bias05 > TOL_V11 >= worst else 'partly holds'}**.\n",
            f"Stage A1 complete in {(time.time() - t0) / 60:.0f} min. **Review point** (A1, A3 and A5 "
            "pause for review).\n"]
    report("\n".join(text))
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
