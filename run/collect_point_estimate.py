"""Turn the finished `multistart` array into the point estimate (spec 4.4).

    python run/collect_point_estimate.py

Reads every `outputs/hpc/multistart/*.json`, picks the restart with the highest
log-likelihood, and writes `outputs/hpc/point_estimate.json` -- the file the
`bootstrap` array warm-starts from.

Spec 4.4 requires reporting the fraction of restarts reaching the same optimum
and, below 60%, declaring the surface multimodal and saying so rather than
silently taking the best. That judgement is printed and recorded in the file.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_dynamics.mixture.data import load_config          # noqa: E402

OUT = os.environ.get("MIX_OUT", "outputs/hpc")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="multistart",
                    help="which array to collect from (e.g. weight_variants)")
    ap.add_argument("--variant", default=None,
                    help="restrict to one variant of that job")
    a = ap.parse_args()

    cfg = load_config()
    files = sorted(glob.glob(os.path.join(OUT, a.job, "*.json")))
    if not files:
        raise SystemExit(f"no multistart results in {os.path.join(OUT, 'multistart')} "
                         f"-- run ./hpc/submit.sh {a.job} first")
    rows = [json.load(open(f)) for f in files]
    if a.variant:
        rows = [r for r in rows if r.get("variant") == a.variant]
        print(f"collecting from {a.job}, variant {a.variant!r}")
    n_expected = int(cfg["init"]["n_restarts"])
    # the collected fit's own settings, not the config's, decide what it is
    if rows:
        for key in ("density_interaction", "density_interaction_spline", "variant"):
            if key in rows[0]:
                print(f"  {key}: {rows[0][key]}")
    ok = [r for r in rows if not r.get("failed") and np.isfinite(r.get("loglik", -np.inf))]
    print(f"restarts: {len(rows)}/{n_expected} present, {len(ok)} usable")
    if not ok:
        raise SystemExit("every restart failed or diverged; nothing to collect")

    lls = np.array([r["loglik"] for r in ok])
    best = ok[int(np.argmax(lls))]
    agree = float(np.mean(lls > lls.max() - 1e-6))
    multimodal = bool(agree < 0.60)

    print(f"best log-likelihood {lls.max():.6f} (restart {best['restart']}), "
          f"||grad||inf {best['grad_inf']:.2e}, converged {best['converged']}")
    print(f"restart agreement: {100 * agree:.0f}% within 1e-6 of the best")
    if multimodal:
        print("  ** below 60%: spec 4.4 declares this surface MULTIMODAL. The "
              "best-of-N estimate below carries that caveat and must be reported "
              "with it, not presented as the optimum. **")
    print(f"e^delta {best['e_delta']:.4f} (rate ratio {np.exp(best['e_delta']):.3f}), "
          f"mean w {best['mean_w']:.4f}, entropy {best['entropy']:.4f}")
    if best["entropy"] < 0.30:
        print("  ** entropy below 0.30: spec 8.7 forbids presenting w as "
              "P(cycling) for this fit. **")

    payload = {
        "params": best["params"],
        "loglik": best["loglik"],
        "grad_inf": best["grad_inf"],
        "converged": best["converged"],
        "restart": best["restart"],
        "n_restarts_present": len(rows),
        "n_restarts_expected": n_expected,
        "restart_agreement": agree,
        "multimodal": multimodal,
        "entropy": best["entropy"],
        "e_delta": best["e_delta"],
        "mean_w": best["mean_w"],
        # taken from the fit itself, not from the config -- the config may have
        # been changed since the array ran, and the estimate belongs to the
        # settings that produced it
        "weight_ridge": best.get("weight_ridge", cfg["model"].get("weight_ridge")),
        "area_spline_df": best.get("df_area", cfg["model"]["area_spline_df"]),
        "technical": best.get("technical", "area"),
        "theta_m": best.get("theta_m"),
        "rank": best.get("rank", cfg["model"]["rank"]),
        "stage": best.get("stage"), "theta_free": best.get("theta_free"),
    }
    path = os.path.join(OUT, "point_estimate.json")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path} -- the bootstrap array warm-starts from this")


if __name__ == "__main__":
    main()
