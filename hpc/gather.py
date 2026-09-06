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
        # Convergence alone is not enough. On the real data lambda=0.3 converged
        # on every fold yet returned sigma_1(B) = 26.6 -- 60x the true value in
        # the Test 01 simulation (0.440) -- because a converged fit of an
        # overfit weight model is still an overfit weight model. A genuine
        # interaction is stable against the penalty; a variance-dominated one
        # tracks it. So require sigma_1 to have entered its plateau: the change
        # to the next-larger setting must be modest.
        if key == "lam" and "sigma1" in g:
            sig = g["sigma1"]
            rel = (sig / sig.shift(-1) - 1.0).abs()      # change vs next lambda up
            g["sigma1_drift"] = rel
            stable = rel < 0.5
            print("\nsigma_1(B) stability (|change| to the next setting up; a real"
                  "\ninteraction plateaus, an overfit one keeps falling):")
            for k in g.index:
                mark = "plateau" if bool(stable.get(k, False)) else "still falling"
                print(f"  {key} {k:<8g} sigma_1 {sig[k]:8.3f}   {mark}")
            g = g[stable.fillna(True)]
            if g.empty:
                print("\n** sigma_1 never stabilises across this grid; extend it "
                      "upward before selecting. **")

        ok = g[g["n_converged"] == g["n"]]
        if ok.empty:
            print(f"\n** No {key} value converged on every fold, so none can be "
                  "selected on held-out likelihood. Raise `optim.maxiter` and "
                  "re-run before reading anything from the table above. **")
        else:
            if key == "rank":
                # Spec 6.11 selects the rank by held-out log-likelihood on
                # spatial blocks. Take the smallest rank that is not
                # significantly worse than the best -- the parsimonious end of
                # the plateau, which is what "a clear maximum or plateau" means.
                top = ok["heldout"].idxmax()
                best = top
                for k in sorted(ok.index):
                    d_ = (piv[k] - piv[top]) * 1e6
                    sd = d_.std(ddof=1)
                    t = d_.mean() / (sd / np.sqrt(len(d_))) if sd > 0 else 0.0
                    if not np.isfinite(t) or t > -2.0:      # indistinguishable
                        best = k
                        break
                print(f"\nselected {key} = {best} (smallest rank not significantly "
                      f"worse than the best-scoring rank {top}; spec 6.11 selects "
                      "the rank on held-out log-likelihood)")
                ties = [k for k in ok.index
                        if k != best and abs(ok.loc[k, "heldout"]
                                             - ok.loc[best, "heldout"]) < 1e-12]
                if ties:
                    print(f"  ranks {ties} are numerically identical to {best}: the "
                          "extra components carry no weight, so B is effectively "
                          f"rank {best}.")
            else:
                # For the penalty, held-out likelihood is monotone toward
                # lambda=0 and cannot score the parameter (see config.yaml), so
                # take the smallest eligible setting instead.
                best = ok.index.min()
                print(f"\nselected {key} = {best} (smallest setting that both "
                      f"converged on all folds and has a stable sigma_1)")
                top = g["heldout"].idxmax()
                if best != top and int(g.loc[top, "n_converged"]) < int(g.loc[top, "n"]):
                    print(f"  note: {key}={top} scores higher but did not converge "
                          "on every fold, so it is not eligible.")

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

    elif a.job == "heldout_genes":
        print("\nspec 6.8: does w predict the held-out half of the signature,")
        print("beyond depth/area/density, against a spatially-permuted null?\n")
        print("| split | coef w | delta-dev | permuted mean | ratio | p |")
        print("|---|---:|---:|---:|---:|---:|")
        for _i, r in df.sort_values("split").iterrows():
            print(f"| {int(r['split'])} | {r['coef_w']:+.4f} | "
                  f"{r['delta_deviance']:.1f} | {r['null_mean']:.1f} | "
                  f"{r['ratio_to_null_mean']:.1f}x | {r['perm_p_value']:.3f} |")
        ratio, coef = df["ratio_to_null_mean"], df["coef_w"]
        ok = bool((ratio > 10).all() and (coef > 0).all())
        print(f"\nratio to permuted null: median {ratio.median():.1f}x, "
              f"min {ratio.min():.1f}x (spec 6.8 wants >= 10x on every split)")
        print(f"coefficient on w: median {coef.median():+.4f}, "
              f"all positive: {bool((coef > 0).all())}")
        print(f"-> {'PASS' if ok else 'FAIL'}")

    elif a.job.startswith("sim_"):
        # the pass/fail criteria the spec names, which the recovery table alone
        # does not cover
        if "B" in df.columns and df["B"].notna().any():
            sig = np.array([s[0] for s in df["sigma"] if s])
            print(f"\n  sigma_1(B_hat): median {np.median(sig):.4f} "
                  f"[{np.percentile(sig, 5):.4f}, {np.percentile(sig, 95):.4f}]")
            if a.job == "sim_recovery":
                fro = np.array([
                    np.linalg.norm(np.array(b) - np.array(t["B"]))
                    / np.linalg.norm(np.array(t["B"]))
                    for b, t in zip(df["B"], df["truth"])])
                ok = fro.mean() < 0.15
                print(f"  spec 6.1  ||B_hat-B||_F/||B||_F = {fro.mean():.4f} "
                      f"(max {fro.max():.4f}, threshold 0.15) -> "
                      f"{'PASS' if ok else 'FAIL'}")
            if a.job == "sim_size_confound":
                fold = np.exp(df["e_delta"].to_numpy(float))
                ent = df["entropy"].to_numpy(float)
                print(f"  spec 6.7  fold change median {np.median(fold):.3f} "
                      f"(threshold 1.5) -> {'PASS' if np.median(fold) < 1.5 else 'FAIL'}")
                print(f"  spec 6.7  entropy median {np.median(ent):.4f} "
                      f"(threshold 0.25) -> {'PASS' if np.median(ent) < 0.25 else 'FAIL'}")
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
