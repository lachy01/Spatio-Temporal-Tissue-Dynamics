"""Collect finished array-task results into one table.

    python hpc/gather.py cv_penalty        # held-out log-likelihood per lambda
    python hpc/gather.py bootstrap         # percentile CIs on the estimands
    python hpc/gather.py sim_recovery      # bias / MC SE against truth

Reports how many tasks are missing rather than silently averaging over whatever
happens to be present -- spec 9.13 forbids dropping non-converged replicates
without recording the rate, and the same logic applies to missing ones.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_dynamics.mixture.data import load_config              # noqa: E402
from tissue_dynamics.mixture.tasks import enumerate_tasks         # noqa: E402


def load(job, out):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, job, "*.json")))]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--out", default=os.environ.get("MIX_OUT", "outputs/hpc"))
    a = ap.parse_args()

    cfg = load_config()
    expected = len(enumerate_tasks(a.job, cfg))
    df = load(a.job, a.out)
    if df.empty:
        raise SystemExit(f"no results found in {os.path.join(a.out, a.job)}")
    missing = expected - len(df)
    print(f"{a.job}: {len(df)}/{expected} tasks present"
          + (f"  ** {missing} MISSING **" if missing else ""))
    if "converged" in df:
        bad = int((~df["converged"].astype(bool)).sum())
        print(f"  not converged: {bad} ({100 * bad / len(df):.1f}%)")
    if "diverged" in df:
        print(f"  diverged: {int(df['diverged'].astype(bool).sum())}")
    if "failed" in df:
        nf = int(df["failed"].notna().sum())
        if nf:
            print(f"  could not be fit at all: {nf} "
                  f"(recorded, not missing -- see the 'failed' column)")
        df = df[df["failed"].isna()].copy()
        if df.empty:
            raise SystemExit("every task failed to fit; nothing to summarise")

    if a.job in ("cv_penalty", "cv_rank"):
        key = "lam" if a.job == "cv_penalty" else "rank"
        df["sigma1"] = [s[0] if s else np.nan for s in df.get("sigma", [])]
        g = df.groupby(key).agg(
            heldout=("heldout_loglik_per_cell", "mean"),
            n_converged=("converged", "sum"), n=("converged", "size"),
            grad=("grad_inf", "median"), sigma1=("sigma1", "median"),
            entropy=("entropy", "median"), e_delta=("e_delta", "median"))
        print("\nper-%s summary:" % key)
        print(g.round(6).to_string())

        # Paired comparison. Every value of `key` is fit on the SAME folds, so
        # the between-fold variance -- which is several times larger than the
        # effect -- cancels. Comparing group means hides the signal entirely.
        piv = df.pivot(index="fold", columns=key, values="heldout_loglik_per_cell")
        base = piv[piv.columns[0]]
        print("\npaired difference vs %s=%g, per cell (x1e6):" % (key, piv.columns[0]))
        for k in piv.columns:
            d_ = (piv[k] - base) * 1e6
            sd = d_.std(ddof=1)
            t = d_.mean() / (sd / np.sqrt(len(d_))) if sd > 0 else float("nan")
            print(f"  {key} {k:<8g} {d_.mean():+10.2f}  paired t = "
                  + (f"{t:+.1f}" if np.isfinite(t) else "-"))

        # Selection is restricted to settings that actually converged. A held-out
        # score from a fit that stopped at the iteration cap is not an estimate
        # of that setting's predictive performance.
        ok = g[g["n_converged"] == g["n"]]
        if ok.empty:
            print(f"\n** No {key} value converged on every fold, so none can be "
                  "selected on held-out likelihood. Raise `optim.maxiter` and "
                  "re-run before reading anything from the table above. **")
        else:
            best = ok["heldout"].idxmax()
            print(f"\nselected {key} = {best} "
                  f"(best held-out score among settings that converged on all "
                  f"{int(g['n'].iloc[0])} folds)")
            if best != g["heldout"].idxmax():
                print(f"  note: {key}={g['heldout'].idxmax()} scores higher but did "
                      "not converge, so it is not eligible.")

    elif a.job == "bootstrap":
        print("\npercentile CIs (spec 7.1):")
        for col in ("e_delta", "mean_w", "entropy"):
            if col in df:
                v = df[col].to_numpy(float)
                lo, hi = np.percentile(v, [2.5, 97.5])
                print(f"  {col:<12} {v.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        if "sigma" in df:
            s1 = np.array([s[0] for s in df["sigma"] if s])
            lo, hi = np.percentile(s1, [2.5, 97.5])
            print(f"  {'sigma_1(B)':<12} {s1.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    elif a.job.startswith("sim_"):
        if "truth" in df.columns:
            print("\nrecovery against truth:")
            print("| parameter | truth | mean est | MC SE | bias/SE |")
            for k in ("b0", "theta", "delta", "a0", "psi"):
                if k in df and k in df["truth"].iloc[0]:
                    est = df[k].astype(float).to_numpy()
                    tru = float(df["truth"].iloc[0][k])
                    se = est.std(ddof=1) / np.sqrt(len(est))
                    print(f"| {k} | {tru:.4f} | {est.mean():.4f} | {se:.4f} "
                          f"| {abs(est.mean() - tru) / max(se, 1e-12):.2f} |")
        for col in ("entropy", "e_delta"):
            if col in df:
                print(f"  {col}: median {df[col].median():.4f} "
                      f"[{df[col].quantile(0.05):.4f}, {df[col].quantile(0.95):.4f}]")
    df.to_csv(os.path.join(a.out, f"{a.job}.csv"), index=False)
    print(f"\nwrote {os.path.join(a.out, f'{a.job}.csv')}")


if __name__ == "__main__":
    main()
