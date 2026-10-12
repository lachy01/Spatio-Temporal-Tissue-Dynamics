"""MisTIC T1 (blocking): likelihood recovery for the new technical null.

    python run/mistic_t1_recovery.py

200,000 analysis cells drawn at random keep their real MisTIC-weighted library
size Nt and expected misassigned cycle count m. Counts are simulated from the
new null,

    y ~ NB(mu, alpha),   mu = Nt e^b0 + theta m,   theta = exp(eta),

with (b0, theta, alpha) set to the null fitted to the real counts (theta = 1
instead if that fit puts theta at its floor, where eta is not identified). The
null is refitted by maximum likelihood (no trimming: this checks the
likelihood and its optimiser) and must recover every parameter -- b0, eta and
log alpha -- within 2 standard errors from the observed information. That is
the spec's pass rule, on the one primary draw.

One draw can miss 2 SE by chance (about 14% of the time for three
parameters). Fifty further draws therefore report how often each parameter's
+/- 2 SE interval covers the truth (about 95% if the fit and its standard
errors are right) and the mean z-score (near 0 if unbiased). These are
reported, not used to pass or fail.

The mixture half of T1 -- Test 01's known-parameter recovery with the theta m
term added -- is tests/test_01_recovery.py::test_01_recovery_mistic.

Writes outputs/mistic/t1.md and t1_null_recovery.json; exits non-zero on a
failure.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.additive_glm import sim_nb                        # noqa: E402
from tissue_dynamics.division import labels as L                       # noqa: E402
from tissue_dynamics.mistic import test_setup                          # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                  # noqa: E402

N_SIM, N_REPS, SEED = 200_000, 50, 20261012
PARAMS = ("b0", "eta", "log_alpha")


def estimates(fit):
    return np.array([fit["beta"][0], fit["eta"], np.log(fit["alpha"])])


def ses(fit):
    return np.array([fit["se"]["b0"], fit["se"]["eta"], fit["se"]["log_alpha"]])


def main():
    t0 = time.time()
    cfg, df, OUT = test_setup()
    basis = build_basis(df, cfg, verbose=False)
    X, off, m = L.null_design(df, basis)
    y = df["y_count"].to_numpy(float)

    real = L.fit_nb_additive_null(y, X, off, m)
    b0, eta, la = estimates(real)
    floor = real["theta_m_at_floor"] or real["theta_m"] < 1e-3
    if floor:
        eta = 0.0
    truth = np.array([b0, eta, la])
    print(f"truth (null fitted to the real counts): b0 {b0:.4f}, theta {np.exp(eta):.4f}"
          f"{' (floor; set to 1)' if floor else ''}, alpha {np.exp(la):.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for r in range(N_REPS + 1):
        idx = rng.choice(len(df), N_SIM, replace=False)
        Xs, os_, ms = X[idx], off[idx], m[idx]
        mu = np.exp(b0 + os_) + np.exp(eta) * ms
        ys = sim_nb(mu, np.exp(la), rng)
        fit = L.fit_nb_additive_null(ys, Xs, os_, ms)
        est, se = estimates(fit), ses(fit)
        rows.append({"rep": r, "converged": fit["converged"],
                     **{f"est_{k}": v for k, v in zip(PARAMS, est)},
                     **{f"se_{k}": v for k, v in zip(PARAMS, se)},
                     **{f"z_{k}": v for k, v in zip(PARAMS, (est - truth) / se)}})
    R = pd.DataFrame(rows)
    prim = R.iloc[0]
    z = np.array([prim[f"z_{k}"] for k in PARAMS])
    ok = bool(np.all(np.abs(z) < 2) and prim["converged"])
    reps = R.iloc[1:]
    cover = {k: float(np.mean(np.abs(reps[f"z_{k}"]) < 2)) for k in PARAMS}
    zbar = {k: float(reps[f"z_{k}"].mean()) for k in PARAMS}
    sdz = {k: float(reps[f"z_{k}"].std(ddof=1)) for k in PARAMS}

    json.dump({"truth": dict(zip(PARAMS, truth.tolist())), "theta_floor": bool(floor),
               "real_fit": {"theta_m": real["theta_m"], "alpha": real["alpha"],
                            "b0": float(real["beta"][0]), "se": real["se"]},
               "primary_z": dict(zip(PARAMS, z.tolist())), "pass": ok,
               "coverage_2se": cover, "mean_z": zbar, "sd_z": sdz,
               "replicates": R.to_dict("records")},
              open(f"{OUT}/t1_null_recovery.json", "w"), indent=1, default=float)
    tab = ["| parameter | truth | estimate (primary) | SE | z | coverage of +/- 2 SE (50 draws) | mean z | sd z |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for j, k in enumerate(PARAMS):
        tab.append(f"| `{k}` | {truth[j]:.4f} | {prim[f'est_{k}']:.4f} | {prim[f'se_{k}']:.4f} | "
                   f"{z[j]:+.2f} | {100 * cover[k]:.0f}% | {zbar[k]:+.2f} | {sdz[k]:.2f} |")
    md = [f"## T1. Likelihood recovery, technical null — **{'PASS' if ok else 'FAIL'}**", "",
          f"{N_SIM:,} analysis cells with their real Nt ({basis.logN_column}) and m "
          f"({basis.m_column}); y ~ NB(Nt e^b0 + theta m, alpha) with the truth set to the null "
          f"fitted to the real counts{' (theta at its floor there, so set to 1)' if floor else ''}: "
          f"b0 {b0:.4f}, theta {np.exp(eta):.4f}, alpha {np.exp(la):.4f}. Refitted by maximum "
          "likelihood (scipy L-BFGS-B, analytic gradients); standard errors from the observed "
          "information. Pass: every parameter of the primary draw within 2 SE.", "",
          *tab, "",
          "Coverage, mean z and sd z are over 50 further draws, reported rather than "
          "tested: about 95%, 0 and 1 if the estimator and its standard errors are right.",
          "",
          "The mixture half of T1 is Test 01 with the theta m term "
          "(`tests/test_01_recovery.py::test_01_recovery_mistic`, in "
          "`outputs/validation_report.md`)."]
    open(f"{OUT}/t1.md", "w").write("\n".join(md) + "\n")
    print("\n".join(tab))
    print(f"T1 null recovery: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
