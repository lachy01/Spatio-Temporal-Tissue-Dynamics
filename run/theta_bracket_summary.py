"""The depth exponent: the mixture with theta pinned at 1 (F) and free (V).

    python run/theta_bracket_summary.py [--fits outputs/hpc]

Spec 2.2 / 6.12: F and V are a bracket, reported side by side and never chosen
between on fit (spec 9.11). F is production (`model.theta_free: false`), so its
estimands carry the spatial block bootstrap; V is the best of the
`theta_bracket` job's free-theta restarts and is a point estimate only. Each
shift V - F is also given in units of F's bootstrap standard error (on the log
scale for the odds ratios), the yardstick the NB baseline uses for the same
question.

Writes outputs/theta_bracket/{report.md, restarts.csv, estimands.csv,
priority_pair_effects.csv, cycling_fraction_by_type.csv}.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics import mistic as M                               # noqa: E402

OUT = "outputs/theta_bracket"
Z = 1.959964


def module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def se(lo_hi, log=False):
    """Standard error implied by a 95% percentile interval, or NaN without one."""
    lo, hi = lo_hi
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return ((np.log(hi) - np.log(lo)) if log else (hi - lo)) / (2 * Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fits", default="outputs/hpc")
    a = ap.parse_args()
    t0 = time.time()
    cfg, df, _out = M.test_setup()
    BC = module("baseline_comparison")
    T6 = BC.t6()
    md = BC.md
    mc = cfg["model"]
    tech = str(mc.get("technical", "area"))
    if mc.get("theta_free"):
        sys.exit("production fits theta freely, so F is not the production fit; this "
                 "summary reads F's intervals from the production bootstrap")
    names = df.attrs["phi_names"]
    arche = df["archetype"].astype(str).to_numpy()

    rows = [json.load(open(f)) for f in
            sorted(glob.glob(os.path.join(a.fits, "theta_bracket", "*.json")))]
    if not rows:
        sys.exit(f"no results in {os.path.join(a.fits, 'theta_bracket')}")
    for r in rows:
        if r.get("technical", "area") != tech or float(r["weight_ridge"]) != float(mc["weight_ridge"]) \
                or int(r["rank"]) != int(mc["rank"]):
            sys.exit(f"theta_bracket task {r['index']} was fitted under technical "
                     f"{r.get('technical', 'area')!r}, lambda {r['weight_ridge']}, rank "
                     f"{r['rank']}; the configuration is {tech!r}, {mc['weight_ridge']}, "
                     f"{mc['rank']}")
    R = pd.DataFrame([{"arm": "V" if r["theta_free"] else "F", "restart": int(r["restart"]),
                       "theta": float(np.ravel(r["theta"])[0]) if r["theta_free"] else 1.0,
                       "penalised objective": float(r["loglik"]), "e^delta": float(r["e_delta"]),
                       "theta_m": float(r.get("theta_m", np.nan)), "mean w": float(r["mean_w"]),
                       "entropy": float(r["entropy"]), "grad_inf": float(r["grad_inf"]),
                       "converged": bool(r["converged"])} for r in rows]
                     ).sort_values(["arm", "restart"])
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f"{OUT}/restarts.csv", index=False)
    best = {arm: max((r for r in rows if bool(r["theta_free"]) == (arm == "V")),
                     key=lambda r: r["loglik"]) for arm in ("F", "V")}
    agree = {arm: int((g["penalised objective"] > g["penalised objective"].max() - 1e-3).sum())
             for arm, g in R.groupby("arm")}
    n_arm = R.groupby("arm").size().to_dict()

    # F is the production fit: check the bracket found the same optimum
    pe = json.load(open(os.path.join(a.fits, "point_estimate.json")))
    gap_F = float(best["F"]["loglik"]) - float(pe["loglik"])

    cfg_F = {**cfg, "model": {**mc, "theta_free": False}}
    cfg_V = {**cfg, "model": {**mc, "theta_free": True}}
    with tempfile.TemporaryDirectory() as tmp:
        bv = best["V"]
        json.dump({"params": bv["params"], "loglik": bv["loglik"],
                   "weight_ridge": bv["weight_ridge"], "technical": bv.get("technical", "area")},
                  open(os.path.join(tmp, "point_estimate.json"), "w"))
        sF = T6.mixture_summary(a.fits, df, cfg_F, tech)
        sV = T6.mixture_summary(tmp, df, cfg_V, tech)
        iF = BC.in_sample(a.fits, df, cfg_F, tech)
        iV = BC.in_sample(tmp, df, cfg_V, tech)
    theta_V = float(np.ravel(bv["theta"])[0])

    # ---- estimands ---------------------------------------------------------------
    est = [{"quantity": "theta (depth exponent)", "F (theta = 1)": "1 (fixed)",
            "V (theta free)": f"{theta_V:.4f}", "V - F": theta_V - 1.0, "in F SEs": np.nan}]
    for key, lab in (("e_delta", "e^delta"), ("theta_m", "theta_m"),
                     ("mean_pi", "cycling fraction, mean prior pi"),
                     ("mean_w", "cycling fraction, mean posterior w")):
        f, f_ci = sF[key]
        v = sV[key][0]
        s = se(f_ci)
        est.append({"quantity": lab, "F (theta = 1)": T6.fmt(f, f_ci), "V (theta free)": f"{v:.4f}",
                    "V - F": v - f, "in F SEs": (v - f) / s if s > 0 else np.nan})
    est.append({"quantity": "rate ratio exp(e^delta)",
                "F (theta = 1)": f"{np.exp(sF['e_delta'][0]):.3f}",
                "V (theta free)": f"{np.exp(sV['e_delta'][0]):.3f}",
                "V - F": np.exp(sV["e_delta"][0]) - np.exp(sF["e_delta"][0]), "in F SEs": np.nan})
    E = pd.DataFrame(est)
    E.to_csv(f"{OUT}/estimands.csv", index=False)

    eff = []
    for k, (f, f_ci) in sF["effects"].items():
        v = sV["effects"][k][0]
        s = se(f_ci, log=True)
        eff.append({"own archetype": k[0], "neighbour component": f"{k[2]}: {k[1]}",
                    "F (theta = 1)": T6.fmt(f, f_ci), "V (theta free)": f"{v:.4f}",
                    "shift in F SEs": (np.log(v) - np.log(f)) / s if s > 0 else np.nan,
                    "V inside F's 95% CI": "yes" if f_ci[0] <= v <= f_ci[1] else "**no**"})
    X = pd.DataFrame(eff)
    X.to_csv(f"{OUT}/priority_pair_effects.csv", index=False)

    P = pd.DataFrame({"archetype": ["all"] + list(names)})
    for arm, ins in (("F", iF), ("V", iV)):
        pi = ins["pi"]
        P[f"mean pi, {arm}"] = [pi.mean()] + [pi[arche == t].mean() for t in names]
    P["V / F"] = P["mean pi, V"] / P["mean pi, F"]
    P.to_csv(f"{OUT}/cycling_fraction_by_type.csv", index=False)

    # ---- report ------------------------------------------------------------------
    dll = iV["loglik_data"] - iF["loglik_data"]
    shifts = X["shift in F SEs"].abs()
    text = ["# The depth exponent: theta pinned at 1 (F) and free (V)", "",
            f"Mixture under `technical: {tech}`, lambda {mc['weight_ridge']:g}, rank {mc['rank']}. "
            "F is production; V frees theta, the exponent on the library size. Spec 9.11: the "
            "two are a bracket, reported side by side, and are not chosen between on fit.", "",
            f"_Fits: `{a.fits}/theta_bracket` (V) and `{a.fits}/point_estimate.json` with its "
            f"bootstrap (F); assembled {time.strftime('%Y-%m-%d %H:%M')}._", "",
            "## Restarts", "",
            f"Restarts at the best optimum: F {agree['F']} of {n_arm['F']}, V {agree['V']} of "
            f"{n_arm['V']}. The bracket's best F is {gap_F:+.2f} from the production point "
            f"estimate{' (the same optimum)' if abs(gap_F) < 0.5 else ' -- NOT the same optimum'}.",
            "", md(R), "",
            "## Estimands", "",
            "F with its 95% spatial block bootstrap interval; V is a point estimate. 'In F SEs' "
            "is the shift over the standard error implied by F's interval.", "",
            md(E), "",
            f"Data log-likelihood: V minus F **{dll:+,.1f}** for one extra parameter.", "",
            "Odds ratio of cycling per IQR of each neighbourhood component, priority pairs both "
            "ways round. The shift is on the log scale.", "",
            md(X), "",
            f"Largest shift {shifts.max():.2f} F SEs; median {shifts.median():.2f}; "
            f"{int((X['V inside F' + chr(39) + 's 95% CI'] == 'yes').sum())} of {len(X)} V "
            "estimates inside F's interval.", "",
            "## Cycling fraction by cell type (mean prior pi, point estimates)", "",
            md(P), ""]
    open(f"{OUT}/report.md", "w").write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
