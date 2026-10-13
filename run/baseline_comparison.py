"""The MisTIC model against the plain baseline: state, density and depth only.

    python run/baseline_comparison.py [--mistic outputs/hpc] [--plain outputs/hpc_plain]

Two models of the same counts on the same 561,897 cells, identical in every
term but the technical covariates:

    MisTIC   log Nt offset (MisTIC-weighted library size) and theta_m m added to
             every component's mean (model.technical: mistic, production)
    plain    log N offset (raw library size), nothing else (technical: plain)

Both have the PhiSpace own state and the split neighbourhood, the density
terms, the same penalty and rank, and neither has an area term.

Mixture, from each fit directory (the plain one written on Spartan with
MIX_TECHNICAL=plain MIX_OUT=outputs/hpc_plain):
  * held out -- the comparison that counts: job cv_production, the 5 spatial
    folds at the production penalty and rank. Log-likelihood summed per fold,
    and paired by spatial block (the units the project's CV rules resample),
    mean difference with its standard error over blocks;
  * in sample: the data log-likelihood at each point estimate, its parameter
    count, and AIC/BIC on the nominal count (the ridge makes the effective
    count smaller, so these are indicative only);
  * estimands with 95% spatial block bootstrap intervals, as T6: e^delta and
    the rate ratio exp(e^delta), cycling fraction, theta_m, entropy, and the
    priority-pair neighbourhood odds ratios;
  * per-type cycling fraction (mean prior pi by archetype), point estimates.
Null level: BASE-T and BASE-O call rates per archetype under both nulls, and
T3 / T4 (outputs/mistic), which run both nulls.

Writes outputs/baseline/report.md and CSVs. Missing fits show as pending.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics import mistic as M                               # noqa: E402

OUT = "outputs/baseline"
MODELS = ("MisTIC", "plain")
TECH = {"MisTIC": "mistic", "plain": "plain"}


def t6():
    """run/mistic_t6_old_vs_new.py as a module, for its estimand summary."""
    spec = importlib.util.spec_from_file_location("t6", os.path.join(HERE, "mistic_t6_old_vs_new.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def in_sample(d, df, cfg, technical):
    """Data log-likelihood at the point estimate, and its parameter count."""
    import jax.numpy as jnp

    from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
    from tissue_dynamics.mixture.model import build_data, loglik_terms, pi_hat

    pe = json.load(open(os.path.join(d, "point_estimate.json")))
    c = {**cfg, "model": {**cfg["model"], "technical": technical}}
    basis = build_basis(df, c, verbose=False)
    spec = spec_from_basis(basis, c, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, c, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    ll = float(np.asarray(loglik_terms(P, D, spec)).sum())
    k = int(sum(np.size(np.asarray(v)) for v in pe["params"].values()))
    pi = np.asarray(pi_hat(P, D, spec))
    return {"loglik_data": ll, "loglik_penalised": float(pe["loglik"]), "k": k,
            "aic": -2 * ll + 2 * k, "bic": -2 * ll + k * np.log(len(df)),
            "restart_agreement": pe.get("restart_agreement"), "pi": pi}


def heldout(d):
    """cv_production: per fold and per spatial block held-out log-likelihood."""
    rows, blocks = [], {}
    for f in sorted(glob.glob(os.path.join(d, "cv_production", "*.json"))):
        r = json.load(open(f))
        rows.append({"fold": int(r["fold"]), "n_test": int(r["n_test"]),
                     "heldout": float(r["heldout_loglik_per_cell"]) * int(r["n_test"]),
                     "converged": bool(r["converged"]), "technical": r.get("technical", "area")})
        for b, (ll, n) in r["heldout_by_block"].items():
            blocks[int(b)] = (float(ll), int(n))
    return pd.DataFrame(rows), blocks


def md(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(f"{v:,.4f}" if isinstance(v, float) else str(v)
                                       for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mistic", default="outputs/hpc")
    ap.add_argument("--plain", default="outputs/hpc_plain")
    a = ap.parse_args()
    dirs = {"MisTIC": a.mistic, "plain": a.plain}
    t0 = time.time()
    cfg, df, _out = M.test_setup()
    os.makedirs(OUT, exist_ok=True)
    T6 = t6()
    names = df.attrs["phi_names"]
    arche = df["archetype"].astype(str).to_numpy()
    have = {m: os.path.exists(os.path.join(d, "point_estimate.json")) for m, d in dirs.items()}
    for m, d in dirs.items():
        if have[m]:
            tech = json.load(open(os.path.join(d, "point_estimate.json"))).get("technical", "area")
            if tech != TECH[m]:
                sys.exit(f"{d}/point_estimate.json is a {tech!r} fit, not {TECH[m]!r}")
    text = ["# The MisTIC model against the plain baseline", "",
            "Two mixtures of the same counts on the same cells, identical except for the "
            "technical covariates: **MisTIC** (offset log Nt, the MisTIC-weighted library "
            "size, plus theta_m m in every component's mean) and **plain** (offset log N, the "
            "raw library size, nothing else). Neither has an area term.", "",
            f"_Fits: MisTIC `{a.mistic}`, plain `{a.plain}`; assembled "
            f"{time.strftime('%Y-%m-%d %H:%M')}._", ""]

    # ---- held out --------------------------------------------------------------
    text += ["## Held-out log-likelihood (cv_production: 5 spatial folds, production penalty)", ""]
    cv = {m: heldout(d) for m, d in dirs.items()}
    if all(len(cv[m][0]) == 5 for m in MODELS):
        F = cv["MisTIC"][0].merge(cv["plain"][0], on=["fold", "n_test"], suffixes=(" MisTIC", " plain"))
        F["MisTIC - plain"] = F["heldout MisTIC"] - F["heldout plain"]
        bm, bp = cv["MisTIC"][1], cv["plain"][1]
        common = sorted(set(bm) & set(bp))
        dif = np.array([bm[b][0] - bp[b][0] for b in common])
        F.to_csv(f"{OUT}/heldout_by_fold.csv", index=False)
        text += [md(F[["fold", "n_test", "heldout MisTIC", "heldout plain", "MisTIC - plain"]]), "",
                 f"Total over folds: **{F['MisTIC - plain'].sum():+,.1f}** log-likelihood in "
                 f"favour of {'MisTIC' if F['MisTIC - plain'].sum() > 0 else 'plain'}; better on "
                 f"{int((F['MisTIC - plain'] > 0).sum())} of 5 folds. Paired by spatial block "
                 f"({len(common)} blocks): mean {dif.mean():+.2f} per block, SE "
                 f"{dif.std(ddof=1) / np.sqrt(len(dif)):.2f}, so the total is "
                 f"{dif.sum():+,.1f} +/- {dif.std(ddof=1) * np.sqrt(len(dif)):,.1f}.", ""]
    else:
        text += ["_Pending: job cv_production for both models "
                 f"(found {len(cv['MisTIC'][0])} and {len(cv['plain'][0])} of 5 folds)._", ""]

    # ---- in sample ----------------------------------------------------------------
    text += ["## In sample, at each point estimate", ""]
    ins = {m: in_sample(dirs[m], df, cfg, TECH[m]) for m in MODELS if have[m]}
    if len(ins) == 2:
        S = pd.DataFrame([{"model": m, "data log-likelihood": v["loglik_data"],
                           "penalised objective": v["loglik_penalised"], "parameters": v["k"],
                           "AIC": v["aic"], "BIC": v["bic"],
                           "restart agreement": v["restart_agreement"]} for m, v in ins.items()])
        S.to_csv(f"{OUT}/in_sample.csv", index=False)
        dll = ins["MisTIC"]["loglik_data"] - ins["plain"]["loglik_data"]
        text += [md(S), "", f"MisTIC minus plain: data log-likelihood **{dll:+,.1f}** with "
                 f"{ins['MisTIC']['k'] - ins['plain']['k']:+d} parameters. AIC and BIC use the "
                 "nominal parameter count; the ridge shrinks the weight model, so the effective "
                 "count is smaller and these understate both models equally.", ""]
    else:
        text += [f"_Pending: point estimates found for {sorted(ins) or 'neither model'}._", ""]

    # ---- estimands -------------------------------------------------------------------
    text += ["## Estimands, with 95% spatial block bootstrap intervals", ""]
    summ = {m: (T6.mixture_summary(dirs[m], df, cfg, TECH[m]) if have[m] else None) for m in MODELS}
    rows = []
    for key, lab in (("e_delta", "e^delta"), ("theta_m", "theta_m"),
                     ("mean_pi", "cycling fraction, mean prior pi"),
                     ("mean_w", "cycling fraction, mean posterior w")):
        rows.append({"quantity": lab, **{m: (T6.fmt(*s[key]) if s else "pending")
                                         for m, s in summ.items()}})
    rows.append({"quantity": "rate ratio exp(e^delta)",
                 **{m: (f"{np.exp(s['e_delta'][0]):.3f}" if s else "pending") for m, s in summ.items()}})
    E = pd.DataFrame(rows)
    E.to_csv(f"{OUT}/estimands.csv", index=False)
    ref = next((s for s in summ.values() if s), None)
    eff = pd.DataFrame([{"own archetype": k[0], "neighbour component": f"{k[2]}: {k[1]}",
                         **{m: (T6.fmt(*s["effects"][k]) if s else "pending") for m, s in summ.items()}}
                        for k in (ref["effects"] if ref else [])])
    eff.to_csv(f"{OUT}/priority_pair_effects.csv", index=False)
    text += [md(E), "", "Odds ratio of cycling per IQR of each neighbourhood component, "
             "priority pairs both ways round:", "", md(eff) if len(eff) else "_pending_", ""]

    # ---- paired: replicate r resamples the same blocks in both bootstraps --------------
    text += ["## MisTIC against plain, paired by bootstrap replicate", ""]
    if all(summ.values()):
        bm, bp = summ["MisTIC"]["boot"], summ["plain"]["boot"]
        reps = sorted(set(bm) & set(bp))
        unpaired = [r for r in reps if bm[r]["n_cells"] != bp[r]["n_cells"]]
        if unpaired:
            text += [f"_Not paired: replicates {unpaired[:5]} resampled different cells._", ""]
        else:
            def row(lab, pt, per_rep, null):
                lo, hi = np.percentile(per_rep, [2.5, 97.5])
                return {"quantity": lab, "estimate": pt, "95% low": lo, "95% high": hi,
                        "interval excludes no difference": "no" if lo <= null <= hi else "yes"}
            sm, sp = summ["MisTIC"], summ["plain"]
            rows = [row("e^delta, MisTIC - plain", sm["e_delta"][0] - sp["e_delta"][0],
                        [bm[r]["e_delta"] - bp[r]["e_delta"] for r in reps], 0.0),
                    row("cycling fraction (mean w), MisTIC - plain", sm["mean_w"][0] - sp["mean_w"][0],
                        [bm[r]["mean_w"] - bp[r]["mean_w"] for r in reps], 0.0)]
            for k in sm["effects"]:
                rows.append(row(f"OR ratio MisTIC / plain: {k[0]} <- {k[2]}: {k[1]}",
                                sm["effects"][k][0] / sp["effects"][k][0],
                                [bm[r]["effects"][k] / bp[r]["effects"][k] for r in reps], 1.0))
            Pd = pd.DataFrame(rows)
            Pd.to_csv(f"{OUT}/paired_differences.csv", index=False)
            n_diff = int((Pd["interval excludes no difference"] == "yes").sum())
            text += [f"{len(reps)} replicates. Replicate r draws the same spatial blocks in both "
                     "bootstraps (seeded by r, same cells), so the difference is computed within "
                     "each replicate; the intervals are its 2.5th and 97.5th percentiles. For the "
                     "odds ratios the comparison is their ratio.", "",
                     md(Pd), "",
                     f"{n_diff} of {len(Pd)} intervals exclude no difference.", ""]
    else:
        text += ["_Pending: both bootstraps._", ""]

    # ---- per-type cycling fraction ---------------------------------------------------
    if len(ins) == 2:
        P = pd.DataFrame({"archetype": ["all"] + list(names)})
        for m in MODELS:
            pi = ins[m]["pi"]
            P[f"mean pi, {m}"] = [pi.mean()] + [pi[arche == t].mean() for t in names]
        P["MisTIC / plain"] = P["mean pi, MisTIC"] / P["mean pi, plain"]
        P.to_csv(f"{OUT}/cycling_fraction_by_type.csv", index=False)
        text += ["## Cycling fraction by cell type (mean prior pi, point estimates)", "", md(P), ""]

    # ---- null level ---------------------------------------------------------------------
    rows, lab = [], {}
    for m in MODELS:
        LT, LO, ctype, null = T6.label_rates(df, cfg, TECH[m])
        lab[m] = (LT, LO, null)
    for t in ["all"] + list(names):
        mm = np.ones(len(df), bool) if t == "all" else ctype == t
        rows.append({"archetype": t, "cells": int(mm.sum()),
                     **{f"{nm}, {m}": float(lab[m][k][mm].mean())
                        for nm, k in (("BASE-T", 0), ("BASE-O", 1)) for m in MODELS}})
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/label_rates.csv", index=False)
    text += ["## Division labels: call rate per archetype under each null", "",
             f"Plain null: NB, mean N e^b0 (alpha0 {lab['plain'][2]['alpha']:.4f}). MisTIC null: "
             f"mean Nt e^b0 + theta m (theta {lab['MisTIC'][2]['theta_m']:.4f}, alpha0 "
             f"{lab['MisTIC'][2]['alpha']:.4f}). BASE-T at alpha_B 0.05, BASE-O at Td 0.5.", "",
             md(R), "",
             "The nulls' calibration and spillover tests run both forms: "
             "`outputs/mistic/t3.md` (MisTIC) and `t3_plain.md` (plain) on 20 matched random "
             "gene sets; `outputs/mistic/t4.md` sets the plain null's spillover curve beside the "
             "MisTIC null's.", ""]
    open(f"{OUT}/report.md", "w").write("\n".join(text) + "\n")
    print("\n".join(text[:40]))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
