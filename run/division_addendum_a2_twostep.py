"""Division classifier, addendum stage A2 (part 1): the two-step arm TWO.

    python run/division_addendum_a2_twostep.py [--smoke N]

Addendum section 3 (`tissue_dynamics.division.twostep`):

* Step 1, constant-pi mixtures on MIX's rate block, all cells and each
  training fold: TWO-P (Poisson + Poisson, MIX's components) from MIX's own
  rate block, and TWO-PNB (Poisson null + NB cycling component) from 10 random
  starts on all cells (V13), each fold starting from the all-cell optimum.
* Section 3.1b: Poisson + Poisson against Poisson + NB on all cells: the
  likelihood-ratio statistic, hanging rootograms (overall, Cancer Epithelial,
  Plasmablasts; the NB-vs-mixture write-up's axes, 0-29 and 30+), fitted
  (pi, e^delta, alpha1) and predicted against observed variance-to-mean ratio
  by type. MIX is not changed.
* BASE-M = 1{w > 0.5} from TWO-P's all-cell step 1, beside BASE-T.
* Step 2, penalised fractional logistic regression of w on MIX's weight block
  (a0 + the shared eta_state, MIX's penalty with weight_ridge scaled to n), on
  all cells (the start for V12) and on each training fold; V6 parity, V7
  separation, V8 determinism.
* Out of fold: pi_TWO on the held-out fold, pushed through every count label
  (Stage 1's alpha_B and Td grids, the extended alpha_B grid and every
  truncated null) with step 1's mu_0, mu_1 (and alpha1 for TWO-PNB); V4.

Writes two_step.parquet, two_step_params.json, two_step_fits.csv,
component_family.csv, rootogram_pp_vs_pnb.png, base_m.csv and a section of
report.md in outputs/division_classifier/. Stops on a failing V-test.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from scipy import stats                                               # noqa: E402

from tissue_dynamics.division import arms, twostep as T               # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge, pois_p_ge      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import (ModelSpec, _logit_pi, build_data,  # noqa: E402
                                           loglik_terms)

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_FOLDS, N_STARTS, SEED = 5, 10, 20261014
N_V4, KMAX = 50_000, 30
FAMILIES = {"TWO-P": "pp", "TWO-PNB": "pnb"}
COL = {"TWO-P": "#2a78d6", "TWO-PNB": "#eb6834"}
INK, INK2 = "#0b0b0b", "#52514e"


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


def fail(test, detail):
    report(f"\n### STOPPED in addendum stage A2 (two-step arm): {test} failed\n\n{detail}\n\n"
           "Per the addendum's section 7, nothing further was run.")
    say(f"\n*** {test} FAILED -- stopping. {detail}")
    sys.exit(1)


def to_json(p):
    return {k: np.asarray(v).tolist() for k, v in p.items()}


def summary(p, family):
    out = {"pi": float(jax.nn.sigmoid(p["a0"])), "e^delta": float(jnp.exp(p["delta"])),
           "rate ratio": float(jnp.exp(jnp.exp(p["delta"]))), "b0": float(p["b0"]),
           "theta_m": float(jnp.exp(p["eta_m"]))}
    out["alpha1"] = float(jnp.exp(p["la1"])) if family == "pnb" else np.nan
    return out


def thresholds(lab, lt):
    """Every count threshold: {label column: c array}."""
    out = {}
    for a in L.ALPHA_B_GRID:
        out[f"LT_{L.ab_tag(a)}"] = lab[f"cT_{L.ab_tag(a)}"].to_numpy(float)
    for t in L.TD_GRID:
        out[f"LO_{L.td_tag(t)}"] = lab[f"cO_{L.td_tag(t)}"].to_numpy(float)
    for c in lt.columns:
        if c.startswith("cT"):
            out["L" + c[1:]] = lt[c].to_numpy(float)
    return out


def p_ge(family, c, pi, mu0, mu1, alpha1):
    """Push-through with the prior pi (never w): pi P1(Y >= c) + (1 - pi) P0(Y >= c)."""
    p1 = pois_p_ge(c, mu1) if family == "pp" else nb_p_ge(c, mu1, alpha1)
    return pi * p1 + (1.0 - pi) * pois_p_ge(c, mu0)


def pmf_matrix(family, y_len, mu0, mu1, pi, alpha1):
    """(n, KMAX) marginal pmf on 0..KMAX-1."""
    k = np.arange(KMAX)[None, :]
    f0 = stats.poisson.pmf(k, mu0[:, None])
    if family == "pp":
        f1 = stats.poisson.pmf(k, mu1[:, None])
    else:
        r = 1.0 / alpha1
        f1 = stats.nbinom.pmf(k, r, r / (r + mu1[:, None]))
    return pi * f1 + (1 - pi) * f0


def rootogram_counts(y, pmf):
    obs = np.array([np.sum(y == k) for k in range(KMAX)] + [np.sum(y >= KMAX)], float)
    exp_ = pmf.sum(0)
    return obs, np.append(exp_, max(len(y) - exp_.sum(), 0.0))


def marginal_moments(family, mu0, mu1, pi, alpha1):
    """Per-cell E[y] and E[y^2] under the constant-pi mixture."""
    v1 = mu1 + (alpha1 * mu1 ** 2 if family == "pnb" else 0.0)
    ey = (1 - pi) * mu0 + pi * mu1
    ey2 = (1 - pi) * (mu0 + mu0 ** 2) + pi * (v1 + mu1 ** 2)
    return ey, ey2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="N random cells; outputs to a scratch folder")
    args = ap.parse_args()
    global OUT, REPORT
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    lt = pd.read_parquet(f"{OUT}/labels_trunc.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("labels_trunc", lt), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    if not (spec.mistic and not spec.theta_free):
        sys.exit("the two-step arm is written for MIX's production rate block "
                 "(technical: mistic, theta fixed at 1)")
    if args.smoke:
        keep = np.sort(np.random.default_rng(0).choice(len(df), args.smoke, replace=False))
        df, lab, lt, fm = (x.iloc[keep].reset_index(drop=True) for x in (df, lab, lt, fm))
        ids = ids[keep]
        OUT = os.environ.get("SMOKE_OUT", "/tmp/a2_smoke")
        os.makedirs(OUT, exist_ok=True)
        REPORT = f"{OUT}/report.md"
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    n = len(df)
    y = np.asarray(D["y"])
    ctype = lab["cell_type"].astype(str).to_numpy()
    folds = fm["fold"].to_numpy()
    thr = thresholds(lab, lt)
    say(f"{n:,} cells; {len(thr)} count thresholds")
    rows, params = [], {}

    def fit_row(arm, tag, p, info, family=None, step=1):
        r = {"arm": arm, "step": step, "data": tag, "converged": "pass" if info["converged"] else "FAIL",
             "iterations": info["lbfgs_iters"] + info["newton_iters"], "grad_inf": info["grad_inf"],
             "objective": info["f"], "minutes": info["wall_time"] / 60}
        if step == 1:
            r.update(summary(p, family))
        else:
            d = arms.eta_diagnostics(p, D if tag == "all cells" else data.subset(np.where(folds != int(tag[-1]))[0]).as_jax())
            r.update({"seed": info["seed"], "sd(eta)": d["sd_eta"], "sigma1(B)": d["sigma1_B"],
                      "sd(eta) > 20 (V7)": "FLAG" if d["flag_sd_eta"] else "no"})
        rows.append(r)

    # ---- step 1 on all cells ---------------------------------------------------
    pmix = pe["params"]
    say("TWO-P step 1, all cells (from MIX's rate block and a0)")
    p_pp, info = T.fit_step1(D, "pp", T.start_from_mix(pmix, "pp"), say=say)
    fit_row("TWO-P", "all cells", p_pp, info, "pp")
    # the constant-pi Poisson mixture is MIX's own likelihood at stage A: check
    specA = ModelSpec(stage="A", rank=spec.rank, theta_free=False, d_own=spec.d_own,
                      d_nbr=spec.d_nbr, df_area=0, df_weight=0, mistic=True,
                      technical=spec.technical, oxygen=False)
    ll_mine = float(jnp.sum(T.const_pi_terms(p_pp, D, "pp")))
    ll_mix = float(jnp.sum(loglik_terms(p_pp, D, specA)))
    say(f"  log-likelihood {ll_mine:,.6f}; MIX's stage-A code at the same point {ll_mix:,.6f}")
    if abs(ll_mine - ll_mix) > 1e-6 * abs(ll_mix):
        fail("step 1 parity", f"constant-pi log-likelihood {ll_mine} against MIX's stage A {ll_mix}")

    say(f"TWO-PNB step 1, all cells, {N_STARTS} random starts (V13)")
    rng = np.random.default_rng(SEED)
    starts = []
    for s in range(N_STARTS):
        p, info = T.fit_step1(D, "pnb", T.random_start(pmix, "pnb", rng), say=say)
        starts.append({"start": s, "converged": info["converged"], "objective": info["f"],
                       "grad_inf": info["grad_inf"], **summary(p, "pnb"), "_p": p, "_info": info})
    best = min((s for s in starts if np.isfinite(s["objective"])), key=lambda s: s["objective"])
    p_pnb = best["_p"]
    fit_row("TWO-PNB", "all cells", p_pnb, best["_info"], "pnb")
    v13 = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in starts])
    v13["rel. gap to best"] = (v13["objective"] - best["objective"]) / abs(best["objective"])
    v13_agree = bool((v13["rel. gap to best"].abs() <= 1e-4).all())
    distinct = v13.loc[v13["rel. gap to best"].abs() > 1e-4]
    say(v13.to_string(index=False))

    # ---- 3.1b: Poisson + Poisson against Poisson + NB --------------------------------
    fam = {}
    for arm, family in FAMILIES.items():
        p = p_pp if family == "pp" else p_pnb
        mu0, mu1 = (np.asarray(v) for v in T.means(p, D))
        pi = float(jax.nn.sigmoid(p["a0"]))
        a1 = float(jnp.exp(p["la1"])) if family == "pnb" else 0.0
        fam[arm] = dict(mu0=mu0, mu1=mu1, pi=pi, alpha1=a1,
                        ll=float(jnp.sum(T.const_pi_terms(p, D, family))),
                        pmf=pmf_matrix(family, n, mu0, mu1, pi, a1))
    lr = 2 * (fam["TWO-PNB"]["ll"] - fam["TWO-P"]["ll"])
    cf_rows, roots = [], {}
    for stratum in ["all"] + types:
        m = np.ones(n, bool) if stratum == "all" else ctype == stratum
        row = {"stratum": stratum, "cells": int(m.sum()), "observed mean": float(y[m].mean()),
               "observed VMR": float(y[m].var() / y[m].mean())}
        for arm, family in FAMILIES.items():
            f = fam[arm]
            ey, ey2 = marginal_moments(family, f["mu0"][m], f["mu1"][m], f["pi"], f["alpha1"])
            mean = ey.mean()
            row[f"{arm} mean"] = float(mean)
            row[f"{arm} VMR"] = float((ey2.mean() - mean ** 2) / mean)
            obs, exp_ = rootogram_counts(y[m], f["pmf"][m])
            roots[(stratum, arm)] = (obs, exp_)
            row[f"{arm} rootogram misfit"] = float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2))
            row[f"{arm} loglik"] = float(np.sum(np.asarray(T.const_pi_terms(
                p_pp if family == "pp" else p_pnb, data.subset(np.where(m)[0]).as_jax(), family))))
        row["loglik gain (PNB - PP)"] = row["TWO-PNB loglik"] - row["TWO-P loglik"]
        cf_rows.append(row)
    cf = pd.DataFrame(cf_rows)
    # where the gain sits on the count axis: per-cell loglik difference by y band
    dll = (np.asarray(T.const_pi_terms(p_pnb, D, "pnb")) - np.asarray(T.const_pi_terms(p_pp, D, "pp")))
    bands = [(0, 4), (5, 14), (15, 29), (30, 10 ** 9)]
    band_rows = []
    for stratum in ("all", "Cancer Epithelial", "Plasmablasts"):
        m = np.ones(n, bool) if stratum == "all" else ctype == stratum
        band_rows.append({"stratum": stratum, **{
            (f"y {lo}-{hi}" if hi < 10 ** 9 else f"y >= {lo}"): float(dll[m & (y >= lo) & (y <= hi)].sum())
            for lo, hi in bands}})
    bands_df = pd.DataFrame(band_rows)
    cf.to_csv(f"{OUT}/component_family.csv", index=False)

    ks = np.arange(KMAX + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for j, stratum in enumerate(("all", "Cancer Epithelial", "Plasmablasts")):
        for i, arm in enumerate(FAMILIES):
            ax = axes[i, j]
            obs, exp_ = roots[(stratum, arm)]
            so, se = np.sqrt(obs), np.sqrt(exp_)
            ax.bar(ks, so, bottom=se - so, width=0.8, color=COL[arm], alpha=0.85,
                   edgecolor="white", linewidth=1)
            ax.plot(ks, se, color=INK, lw=1.6, marker="o", ms=3)
            ax.axhline(0, color=INK2, lw=1)
            mis = float(np.sum((so - se) ** 2))
            ax.set_title(f"{arm}, constant pi: {stratum}\nmisfit {mis:,.0f}", color=INK, fontsize=10)
            ax.set_xlabel(f"division count (last bar = {KMAX}+)")
            ax.grid(True, color="#e4e3df", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0, j].sharey(axes[1, j])
    for ax in axes[:, 0]:
        ax.set_ylabel("sqrt(frequency)")
    fig.suptitle("Hanging rootograms, Poisson + Poisson (top) and Poisson + NB (bottom): bars hang "
                 "from the expected frequency; a bar ending off zero is misfit",
                 fontsize=10, color=INK2, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rootogram_pp_vs_pnb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- BASE-M ----------------------------------------------------------------------
    w_pp = np.asarray(T.posterior(p_pp, D, "pp"))
    base_m = w_pp > 0.5
    bt = lab[f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}"].to_numpy(bool)
    bm_rows = [{"stratum": s, "BASE-M rate": float(base_m[m].mean()), "BASE-T (0.05) rate": float(bt[m].mean()),
                "mean w (TWO-P)": float(w_pp[m].mean()),
                "agreement": float(np.mean(base_m[m] == bt[m]))}
               for s, m in [("all", np.ones(n, bool))] + [(t, ctype == t) for t in types]]
    bm = pd.DataFrame(bm_rows)
    bm.to_csv(f"{OUT}/base_m.csv", index=False)
    x12 = bm.iloc[0]["BASE-M rate"] < bm.iloc[0]["BASE-T (0.05) rate"]

    # ---- V6 parity ----------------------------------------------------------------------
    Pmix = {k: jnp.asarray(np.asarray(v)) for k, v in pmix.items()}
    v6_num = float(np.max(np.abs(np.asarray(T.weight_eta(Pmix, D) - _logit_pi(Pmix, D, spec)))))
    probe = T.init_step2(D, jnp.asarray(w_pp), spec.rank, 0)
    v6_names = arms.eta_column_names(probe) == arms.eta_column_names(Pmix)
    v6_ok = v6_names and v6_num < 1e-10
    say(f"V6: names identical {v6_names}; numeric parity with _logit_pi {v6_num:.2e}")
    if not v6_ok:
        fail("V6 (parity)", f"names identical: {v6_names}; numeric max|diff| {v6_num:.2e}")

    # ---- step 2 on all cells (V12's start) and per fold; step 1 per fold ---------------
    oof = pd.DataFrame({"cell_id": ids, "fold": folds, "cell_type": ctype})
    full_two = {}
    for arm, family in FAMILIES.items():
        p1 = p_pp if family == "pp" else p_pnb
        w = T.posterior(p1, D, family)
        say(f"{arm} step 2, all cells (mean w {float(w.mean()):.4f})")
        p0 = {k: Pmix[k] for k in ("a0", "a", "c", "U", "V", "h", "K")}
        p2, info = T.fit_step2(D, w, n, spec, seed=0, say=say, p0=p0)
        fit_row(arm, "all cells", p2, info, step=2)
        full_two[arm] = {"step1": to_json(p1), "step2": to_json(p2),
                         "step2 objective": info["f"], "step1 loglik": fam[arm]["ll"]}
        params[(arm, "full", 1)], params[(arm, "full", 2)] = p1, p2
        cols = {k: np.zeros(n) for k in ("w", "pi", "mu0", "mu1")}
        for f in range(N_FOLDS):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            Dtr, Dte = data.subset(tr).as_jax(), data.subset(te).as_jax()
            say(f"{arm} train fold {f} (n = {len(tr):,})")
            q1, info = T.fit_step1(Dtr, family, p1, say=say)
            fit_row(arm, f"train fold {f}", q1, info, family)
            wtr = T.posterior(q1, Dtr, family)
            mixf = json.load(open(f"outputs/division_classifier/mix_params_fold{f}.json")) \
                if not args.smoke else pmix
            q0 = {k: jnp.asarray(np.asarray(mixf[k])) for k in ("a0", "a", "c", "U", "V", "h", "K")}
            q2, info = T.fit_step2(Dtr, wtr, len(tr), spec, seed=0, say=say, p0=q0)
            fit_row(arm, f"train fold {f}", q2, info, step=2)
            params[(arm, f, 1)], params[(arm, f, 2)] = q1, q2
            cols["w"][te] = np.asarray(T.posterior(q1, Dte, family))
            cols["pi"][te] = np.asarray(T.pi_two(q2, Dte))
            m0, m1 = T.means(q1, Dte)
            cols["mu0"][te], cols["mu1"][te] = np.asarray(m0), np.asarray(m1)
            if family == "pnb":
                cols.setdefault("alpha1", np.zeros(n))[te] = float(jnp.exp(q1["la1"]))
        tag = arm.replace("-", "").lower()
        for k, v in cols.items():
            oof[f"{k}_{tag}"] = v
        for lcol, c in thr.items():
            oof[f"p_{tag}_{lcol}"] = p_ge(family, c, cols["pi"], cols["mu0"], cols["mu1"],
                                          cols.get("alpha1"))
    fits = pd.DataFrame(rows)
    fits.to_csv(f"{OUT}/two_step_fits.csv", index=False)
    json.dump({"TWO-P": full_two["TWO-P"], "TWO-PNB": full_two["TWO-PNB"],
               "folds": {f"{a} fold {f}": {"step1": to_json(params[(a, f, 1)]),
                                          "step2": to_json(params[(a, f, 2)])}
                         for a in FAMILIES for f in range(N_FOLDS)},
               "v13": v13.to_dict(orient="list"), "lr_statistic": lr},
              open(f"{OUT}/two_step_params.json", "w"))
    oof.to_parquet(f"{OUT}/two_step.parquet", index=False)
    conv_ok = bool((fits["converged"] == "pass").all())
    s2 = fits[fits["step"] == 2]
    v7_ok = bool((s2["sd(eta) > 20 (V7)"] == "no").all())
    if not conv_ok:
        fail("convergence (V1)", md_table(fits[["arm", "step", "data", "converged", "grad_inf"]]))
    if not v7_ok:
        fail("V7 (separation)", md_table(s2[["arm", "data", "sd(eta)"]]))

    # ---- V8: determinism, step 2 fold 0 refitted -------------------------------------
    tr0 = np.where(folds != 0)[0]
    D0 = data.subset(tr0).as_jax()
    w0 = T.posterior(params[("TWO-P", 0, 1)], D0, "pp")
    mix0 = json.load(open(f"{OUT}/mix_params_fold0.json")) if not args.smoke else pmix
    q0 = {k: jnp.asarray(np.asarray(mix0[k])) for k in ("a0", "a", "c", "U", "V", "h", "K")}
    again, _ = T.fit_step2(D0, w0, len(tr0), spec, seed=0, p0=q0)
    d_v8 = max(float(np.max(np.abs(np.asarray(again[k]) - np.asarray(params[("TWO-P", 0, 2)][k]))))
               for k in again)
    v8_ok = d_v8 == 0.0
    say(f"V8: TWO-P step 2 fold 0 refit max|diff| {d_v8:.1e}")
    if not v8_ok:
        fail("V8 (determinism)", f"TWO-P step 2 fold 0 refit differs by {d_v8:.3e}")

    # ---- V4: push-through identity ----------------------------------------------------
    rng4 = np.random.default_rng(SEED + 4)
    idx = np.sort(rng4.choice(n, min(N_V4, n), replace=False))
    primary = (f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}", f"LO_{L.td_tag(L.TD_PRIMARY)}")
    v4_rows = []
    for arm, family in FAMILIES.items():
        tag = arm.replace("-", "").lower()
        z = rng4.uniform(size=len(idx)) < oof[f"pi_{tag}"].to_numpy()[idx]
        mu0, mu1 = oof[f"mu0_{tag}"].to_numpy()[idx], oof[f"mu1_{tag}"].to_numpy()[idx]
        y1 = (rng4.poisson(mu1).astype(float) if family == "pp"
              else L.sim_nb(mu1, oof["alpha1_twopnb"].to_numpy()[idx], rng4))
        ysim = np.where(z, y1, rng4.poisson(mu0).astype(float))
        for lcol, c in thr.items():
            p = oof[f"p_{tag}_{lcol}"].to_numpy()[idx]
            rate = float(np.mean(ysim >= c[idx]))
            se = float(np.sqrt(np.sum(p * (1 - p)))) / len(idx)
            v4_rows.append({"arm": arm, "threshold": lcol, "primary": lcol in primary,
                            "mean p": float(p.mean()), "simulated rate": rate, "MC SE": se,
                            "z": (rate - p.mean()) / se})
    v4 = pd.DataFrame(v4_rows)
    v4_ok = bool((v4.loc[v4["primary"], "z"].abs() <= 3).all())
    n_other = int((~v4["primary"]).sum())
    n_other_out = int((v4.loc[~v4["primary"], "z"].abs() > 3).sum())
    say(v4.round(5).to_string(index=False))
    if not v4_ok:
        fail("V4 (push-through identity)", md_table(v4))

    # ---- report ---------------------------------------------------------------------------
    s1 = fits[fits["step"] == 1][["arm", "data", "converged", "iterations", "grad_inf", "pi",
                                  "e^delta", "rate ratio", "theta_m", "alpha1", "minutes"]]
    s2t = s2[["arm", "data", "converged", "iterations", "grad_inf", "objective", "sd(eta)",
              "sigma1(B)", "minutes"]]
    v13_show = v13[["start", "converged", "objective", "rel. gap to best", "pi", "e^delta", "alpha1",
                    "theta_m"]]
    cf_show = cf[["stratum", "cells", "observed mean", "observed VMR", "TWO-P VMR", "TWO-PNB VMR",
                  "TWO-P rootogram misfit", "TWO-PNB rootogram misfit", "loglik gain (PNB - PP)"]]
    ce = cf.set_index("stratum")
    gain_ce = float(bands_df.set_index("stratum").loc["Cancer Epithelial"].iloc[2:].sum())
    gain_ce_all = float(ce.loc["Cancer Epithelial", "loglik gain (PNB - PP)"])
    pp, pnb = fam["TWO-P"], fam["TWO-PNB"]
    report(f"""
## Addendum stage A2 -- the two-step arm TWO ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_addendum_a2_twostep.py`, `tissue_dynamics.division.twostep`.
Step 1 is a constant-pi two-component mixture on MIX's production rate block
(Nt offset with coefficient 1, theta_m m added to both component means and kept
outside the rate ratio, lambda_1 = exp(e^delta) lambda_0). TWO-P has Poisson
components, MIX's own; TWO-PNB swaps the cycling component for an NB with one
shared dispersion alpha1. Step 2 regresses step 1's posterior w on MIX's
weight block (a0 plus all {len(arms.eta_column_names(probe))} eta columns),
with MIX's penalty and weight_ridge scaled to the fitting n; nothing technical
enters it. **TWO-P is not independent of MIX:** its components are MIX's, so
it differs from MIX only in not estimating rates and weights jointly.

As a check on step 1, the constant-pi Poisson log-likelihood written here
equals MIX's own code at stage A (weight = intercept) at the TWO-P optimum:
{ll_mine:,.6f} against {ll_mix:,.6f}.

### Step 1 fits

{md_table(s1)}

### V13 -- TWO-PNB multi-start ({N_STARTS} random starts, all cells)

Starts perturb b0, delta and eta_m about MIX's values and draw pi from
U(0.02, 0.30) and log alpha1 from N(log 0.3, 1). Criterion (reported, not
blocking): every start within 1e-4 (relative) of the best objective.
**{'agree' if v13_agree else 'DISAGREE: weak identifiability flagged'}**
({len(distinct)} of {N_STARTS} starts elsewhere).

{md_table(v13_show, '{:.6g}')}

### Section 3.1b -- Poisson + Poisson against Poisson + NB (all cells, constant pi)

| fit | log-likelihood | pi | e^delta | rate ratio | alpha1 |
| --- | --- | --- | --- | --- | --- |
| Poisson + Poisson | {pp['ll']:,.2f} | {pp['pi']:.4f} | {summary(p_pp, 'pp')['e^delta']:.4f} | {summary(p_pp, 'pp')['rate ratio']:.3f} | -- |
| Poisson + NB | {pnb['ll']:,.2f} | {pnb['pi']:.4f} | {summary(p_pnb, 'pnb')['e^delta']:.4f} | {summary(p_pnb, 'pnb')['rate ratio']:.3f} | {pnb['alpha1']:.4f} |

Likelihood-ratio statistic 2(ll_PNB - ll_PP) = **{lr:,.1f}** for one extra
parameter (alpha1). The boundary at alpha1 = 0 makes a chi-square(1) reference
conservative, so no p-value is given.

By type: observed and predicted variance-to-mean ratio of y, rootogram misfit
(sum of squared hanging-bar residuals on 0-29 and 30+) and the log-likelihood
gain:

{md_table(cf_show, '{:.4g}')}

Where the gain sits on the count axis (sum of per-cell log-likelihood
differences, PNB - PP, by observed y):

{md_table(bands_df, '{:.1f}')}

Rootograms: `rootogram_pp_vs_pnb.png` (same axes as the NB-against-mixture
write-up: 0-29 and a pooled 30+). Everything is in `component_family.csv`.
MIX is not changed. {"Poisson + NB fits clearly better on these numbers; whether MIX's cycling component should become NB is an open specification question for Lachy." if lr > 100 else "Poisson + NB does not clearly improve on Poisson + Poisson."}

Known risks of TWO-PNB, as the addendum asks: any extra-Poisson variance among
non-cycling cells (segmentation, spillover variance the MisTIC mean term does
not remove) can only go into the NB cycling component, inflating pi; G3 will
expose this, since a single overdispersed NB with no latent classes will still
be given a cycling fraction; and G2 simulates exactly an NB cycling component
on a Poisson null, so TWO-PNB's G2 results are a home fixture, to be read as
MIX's G1 results are.

### BASE-M (hard MAP label from TWO-P's step 1, all cells; not a target)

{md_table(bm, '{:.4f}')}

### Step 2 fits, V6, V7, V8

V6: the step 2 predictor a0 + eta_state has the same {len(arms.eta_column_names(probe))}
eta columns as MIX in the same order, and equals MIX's `_logit_pi` at the
production estimate to {v6_num:.1e}. **{'pass' if v6_ok else 'FAIL'}**.
V7: largest sd(eta) {s2['sd(eta)'].max():.3f} (criterion 20). **{'pass' if v7_ok else 'FAIL'}**.
V8: refitting TWO-P's step 2 on training fold 0 reproduces it to {d_v8:.1e}.
**{'pass' if v8_ok else 'FAIL'}**. Every step 2 fit started from MIX's weight
block for the same data (production estimate on all cells, Stage 3's fold fit
per fold) and ran to the mixture's convergence criteria.

{md_table(s2t)}

### V4 -- push-through identity

{len(idx):,} random cells, y simulated from each arm's out-of-fold predictive
distribution (prior pi, step 1's components). Formal criterion on the primary
thresholds ({', '.join(primary)}): within 3 Monte Carlo SE. **{'pass' if v4_ok else 'FAIL'}**.
Of the other {n_other} comparisons {n_other_out} are beyond 3 SE (about
{0.0027 * n_other:.2f} expected by chance).

{md_table(v4[v4['primary']].drop(columns=['primary']), '{:.5g}')}

Out of fold w, pi_TWO and the pushed-through p for every threshold are in
`two_step.parquet`; parameters in `two_step_params.json`; fits in
`two_step_fits.csv`. X12 (BASE-M below BASE-T at 0.05): **{'holds' if x12 else 'does not hold'}**
({bm.iloc[0]['BASE-M rate']:.4f} against {bm.iloc[0]['BASE-T (0.05) rate']:.4f}).
X14's first clause (likelihood-ratio statistic in the hundreds or more, most of
the gain in Cancer Epithelial's 15-30+ range): statistic {lr:,.0f}; Cancer
Epithelial gains {gain_ce_all:,.1f} of {(pnb['ll'] - pp['ll']):,.1f}, {gain_ce:,.1f} of it
from cells with y >= 15. Its second clause (V13 agreement):
**{'holds' if v13_agree else 'does not hold'}**. Its third clause (G3) waits
for A5. Part 1 of A2 done in {(time.time() - t0) / 60:.0f} min.
""")
    say(f"\ntwo-step done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
