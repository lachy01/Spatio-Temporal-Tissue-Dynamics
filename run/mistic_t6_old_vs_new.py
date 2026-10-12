"""MisTIC T6 (report): the area-based fits against the MisTIC-based fits.

    python run/mistic_t6_old_vs_new.py [--old outputs/hpc_area] [--new outputs/hpc]

Mixture, from each directory's point_estimate.json (CIs from its bootstrap/
replicates, 2.5th-97.5th percentiles, where they exist):
  * e^delta, the cycling rate ratio being exp(e^delta);
  * theta_m (MisTIC fit only; the area fit pins theta = 1 and has no m);
  * the overall cycling fraction: mean prior pi and mean posterior w;
  * for the priority archetype pairs of the phase portrait -- (CAFs, Myeloid),
    (Cancer Epithelial, CAFs), (Cancer Epithelial, Myeloid), (Cancer
    Epithelial, T-cells) -- the odds ratio of cycling per IQR of each
    neighbourhood component, both ways round, contact and regional blocks
    (run/estimands_report.py's estimand: c + B' phi_a for own archetype a).
    The neighbourhood acts on the odds of cycling, not on the rates, so these
    are odds ratios; the spec's "rate ratios" are read as these effects.
BASE-T (alpha_B 0.05) and BASE-O (Td 0.5): call rates per archetype under the
area null and the MisTIC null, both fitted here.

No pass/fail. Writes outputs/mistic/t6.md and t6_*.csv.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics import mistic as M                               # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402

PAIRS = [("CAFs", "Myeloid"), ("Cancer Epithelial", "CAFs"),
         ("Cancer Epithelial", "Myeloid"), ("Cancer Epithelial", "T-cells")]


def ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (np.nan, np.nan) if len(v) < 10 else tuple(np.percentile(v, [2.5, 97.5]))


def fmt(est, lo_hi):
    lo, hi = lo_hi
    if not np.isfinite(est):
        return "--"
    return f"{est:.4f}" + (f" ({lo:.4f}, {hi:.4f})" if np.isfinite(lo) else "")


def mixture_summary(d, df, cfg, technical):
    """Point estimates and bootstrap CIs for one result directory, or None."""
    import jax.numpy as jnp

    from tissue_dynamics.mixture.model import build_data, pi_hat, responsibilities

    pt = os.path.join(d, "point_estimate.json")
    if not os.path.exists(pt):
        return None
    pe = json.load(open(pt))
    if pe.get("technical", "area") != technical:
        raise SystemExit(f"{pt} is a {pe.get('technical', 'area')!r} fit, not {technical!r}")
    c = {**cfg, "model": {**cfg["model"], "technical": technical}}
    basis = build_basis(df, c, verbose=False)
    spec = spec_from_basis(basis, c, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, c, spec)
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    D = data.as_jax()
    boot = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "bootstrap", "*.json")))]
    boot = [b for b in boot if b.get("converged", True) and b.get("technical", "area") == technical]

    names = df.attrs["phi_names"]
    own = basis.own(df)
    arche = df["archetype"].astype(str).to_numpy()
    nbr_raw = df[basis.nbr_cols].to_numpy(float)
    iqr = np.percentile(nbr_raw, 75, axis=0) - np.percentile(nbr_raw, 25, axis=0)
    blocks = [b for b, _ in basis.nbr_blocks]

    def effects(c_vec, B):
        out = {}
        for a, b in PAIRS + [(q, p) for p, q in PAIRS]:
            phi_a = own[arche == a].mean(0)
            eff = basis.Q_nbr @ (np.asarray(c_vec) + np.asarray(B).T @ phi_a)
            for k, blk in enumerate(blocks):
                j = k * len(names) + names.index(b)
                out[(a, b, blk)] = float(np.exp(eff[j] * iqr[j]))
        return out

    B_hat = np.asarray(pe["params"]["U"]) @ np.asarray(pe["params"]["V"]).T
    eff_pt = effects(pe["params"]["c"], B_hat)
    eff_bt = [effects(b["c"], b["B"]) for b in boot if b.get("B") is not None]
    out = {"n_boot": len(boot),
           "e_delta": (float(np.exp(np.asarray(pe["params"]["delta"]))), ci([b["e_delta"] for b in boot])),
           "theta_m": ((float(np.exp(np.asarray(pe["params"]["eta_m"]))) if "eta_m" in pe["params"]
                        else np.nan), ci([b.get("theta_m") for b in boot])),
           "mean_pi": (float(np.asarray(pi_hat(P, D, spec)).mean()), (np.nan, np.nan)),
           "mean_w": (float(np.asarray(responsibilities(P, D, spec)).mean()),
                      ci([b["mean_w"] for b in boot])),
           "effects": {k: (v, ci([e[k] for e in eff_bt])) for k, v in eff_pt.items()}}
    return out


def label_rates(df, cfg, technical):
    c = {**cfg, "model": {**cfg["model"], "technical": technical}}
    basis = build_basis(df, c, verbose=False)
    X, off, m = L.null_design(df, basis)
    y = df["y_count"].to_numpy(float)
    null = L.robust_null(y, X, off, m=m)
    mu0, a0 = null["mu"], null["alpha"]
    sd0 = np.sqrt(mu0 + a0 * mu0 ** 2)
    ctype = L.cell_type(df, df.attrs["phi_names"])
    LT, _c, _ai, _p = L.base_t(y, mu0, a0, L.ALPHA_B_PRIMARY)
    LO, _cO, _a = L.base_o(y, mu0, sd0, L.base_o_scale((y - mu0) / sd0, ctype), L.TD_PRIMARY)
    return LT, LO, ctype, null


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", default="outputs/hpc_area")
    ap.add_argument("--new", default="outputs/hpc")
    a = ap.parse_args()
    t0 = time.time()
    cfg, df, out = M.test_setup()
    names = df.attrs["phi_names"]

    # ---- labels -------------------------------------------------------------------
    lab = {}
    for tech in ("area", "mistic"):
        LT, LO, ctype, null = label_rates(df, cfg, tech)
        lab[tech] = (LT, LO, null)
        print(f"  {tech} null: alpha0 {null['alpha']:.4f}"
              + (f", theta_m {null['theta_m']:.4f}" if tech == "mistic" else "")
              + f"; BASE-T {LT.mean():.4f}, BASE-O {LO.mean():.4f}", flush=True)
    rows = []
    for t in ["all"] + list(names):
        m = np.ones(len(df), bool) if t == "all" else ctype == t
        r = {"archetype": t, "cells": int(m.sum())}
        for nm, k in (("BASE-T", 0), ("BASE-O", 1)):
            r[f"{nm}, area"] = float(lab["area"][k][m].mean())
            r[f"{nm}, MisTIC"] = float(lab["mistic"][k][m].mean())
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(f"{out}/t6_label_rates.csv", index=False)

    # ---- mixture -------------------------------------------------------------------
    mix = {"area": mixture_summary(a.old, df, cfg, "area"),
           "MisTIC": mixture_summary(a.new, df, cfg, "mistic")}
    scal = []
    for key, label in (("e_delta", "e^delta"), ("theta_m", "theta_m"),
                       ("mean_pi", "cycling fraction, mean prior pi"),
                       ("mean_w", "cycling fraction, mean posterior w")):
        scal.append({"quantity": label, **{nm: (fmt(*s[key]) if s else "pending")
                                         for nm, s in mix.items()}})
    S = pd.DataFrame(scal)
    eff_rows = []
    ref = next((s for s in mix.values() if s), None)
    for k in (ref["effects"] if ref else []):
        own_a, nb, blk = k
        eff_rows.append({"own archetype": own_a, "neighbour component": f"{blk}: {nb}",
                         **{nm: (fmt(*s["effects"][k]) if s else "pending") for nm, s in mix.items()}})
    E = pd.DataFrame(eff_rows)
    S.to_csv(f"{out}/t6_mixture.csv", index=False)
    E.to_csv(f"{out}/t6_effects.csv", index=False)

    def md(frame):
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in frame.iterrows():
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                                           for v in r.to_numpy()) + " |")
        return "\n".join(lines)

    nm_ = lab["mistic"][2]
    nb_ = {k: (v["n_boot"] if v else 0) for k, v in mix.items()}
    text = ["## T6. Old (area) against new (MisTIC) fits (report)", "",
            f"Mixture point estimates from `{a.old}` (area) and `{a.new}` (MisTIC); "
            f"bracketed: 95% spatial block bootstrap intervals ({nb_['area']} and "
            f"{nb_['MisTIC']} replicates).", "",
            md(S), "",
            "Odds ratio of cycling per IQR of each neighbourhood component, priority pairs "
            "both ways round (own archetype; contact = 0-15 um, regional = 15-80 um):", "",
            md(E) if len(E) else "_pending: no point estimate yet_", "",
            "Division labels, call rate per archetype (area null: log N offset + area spline; "
            f"MisTIC null: weighted N offset + theta m, theta = {nm_['theta_m']:.4f}):", "",
            md(R)]
    open(f"{out}/t6.md", "w").write("\n".join(text) + "\n")
    print(R.to_string(index=False))
    print(S.to_string(index=False))
    print(f"T6 done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
