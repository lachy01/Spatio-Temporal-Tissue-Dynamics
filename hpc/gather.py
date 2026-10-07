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


def select_lambda(df, ok):
    """The largest lambda whose held-out log-likelihood is within noise of the best.

    Held-out log-likelihood is a proper scoring rule and, with contiguous
    spatial folds, a valid criterion on spatially dependent data. For the
    penalty it is also blunt: the weight model reaches the marginal likelihood
    only through a latent class, so the curve is flat over a wide range of
    lambda. The rule therefore takes the MOST regularised setting that is
    statistically indistinguishable from the best (the one-standard-error rule),
    not the best point, and not -- as before 2026-10-06 -- the smallest lambda
    whose sigma_1(B) had "plateaued", which on the log1p table picked the
    unpenalised end of the grid.

    Uncertainty comes from the spatial blocks: each fit stores its held-out
    log-likelihood per block (`heldout_by_block`), the per-block differences
    between two lambdas are paired, and the SE of their sum is
    sqrt(n_blocks) x sd(block differences). With ~250 blocks this is far
    better determined than a t on 5 folds. Runs without per-block scores fall
    back to the fold-level paired t.

    Only settings converged on every fold are eligible.
    """
    lams = sorted(ok.index)
    has_blocks = "heldout_by_block" in df and df["heldout_by_block"].notna().all()

    def block_scores(lam):
        out = {}
        for _, r in df[df["lam"] == lam].iterrows():
            for b, (ll, _n) in r["heldout_by_block"].items():
                out[int(b)] = out.get(int(b), 0.0) + float(ll)
        return pd.Series(out).sort_index()

    if has_blocks:
        S = pd.DataFrame({lam: block_scores(lam) for lam in lams})
        tot = S.sum()
        top = tot.idxmax()
        print(f"\nheld-out log-likelihood, total over {len(S)} spatial blocks; "
              f"difference from the best (lam={top:g}) with its block SE:")
        within = {}
        for lam in lams:
            d = S[lam] - S[top]
            se = float(d.std(ddof=1) * np.sqrt(len(d))) if lam != top else 0.0
            diff = float(d.sum())
            within[lam] = lam == top or diff > -2.0 * se
            print(f"  lam {lam:<8g} total {tot[lam]:14.1f}   diff {diff:+10.1f} "
                  f"+/- {se:7.1f}   {'within noise' if within[lam] else 'worse'}")
    else:
        piv = df.pivot(index="fold", columns="lam", values="heldout_loglik_per_cell")
        top = ok["heldout"].idxmax()
        print(f"\nno per-block scores in these results; fold-level paired t vs the "
              f"best (lam={top:g}):")
        within = {}
        for lam in lams:
            d = (piv[lam] - piv[top]) * 1e6
            sd = d.std(ddof=1)
            t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
            within[lam] = lam == top or t > -2.0
            print(f"  lam {lam:<8g} {d.mean():+10.2f} x1e-6/cell  t = {t:+.1f}   "
                  f"{'within noise' if within[lam] else 'worse'}")
    return max(lam for lam in lams if within[lam])


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
    # A directory holding fits from two model versions is worse than a missing
    # one, because the numbers look plausible. Detect it from the parameter
    # shapes, which change whenever the bases do.
    if "params" in df.columns and df["params"].notna().any():
        shapes = df["params"].dropna().apply(
            lambda p: tuple(sorted((k, np.shape(np.asarray(v))) for k, v in p.items())))
        if shapes.nunique() > 1:
            print(f"\n  ** {shapes.nunique()} DIFFERENT parameter shapes in this "
                  "directory: it holds fits from more than one model version. "
                  "Delete it and resubmit -- the results below mix them. **\n")

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

    # A job that refits the same specification must land on the same optimum.
    # Silent failure to do so is what made the first subsample_stability and
    # density_sensitivity runs uninterpretable: every task found a local optimum
    # 1,089 log-likelihood units worse and reported it as a result.
    pt = os.path.join(a.out, "point_estimate.json")
    if os.path.exists(pt) and "loglik" in df.columns:
        ref = json.load(open(pt))["loglik"]
        best = float(df["loglik"].max())
        if best < ref - 1.0:
            print(f"\n  ** best log-likelihood here ({best:.1f}) is {ref - best:.1f} "
                  f"units WORSE than the point estimate ({ref:.1f}). These fits are "
                  "in a different, worse optimum -- check the restart count before "
                  "reading anything below. **")

    if a.job in ("cv_penalty", "cv_penalty_unwhitened", "cv_penalty_buffered", "cv_rank"):
        key = "rank" if a.job == "cv_rank" else "lam"
        df["sigma1"] = [s[0] if s else np.nan for s in df.get("sigma", [])]
        agg = dict(heldout=("heldout_loglik_per_cell", "mean"),
                   n_converged=("converged", "sum"), n=("converged", "size"),
                   grad=("grad_inf", "median"), sigma1=("sigma1", "median"),
                   entropy=("entropy", "median"), e_delta=("e_delta", "median"))
        for c in ("sd_eta_own", "sd_eta_nbr", "sd_eta_bilinear", "sd_eta_density",
                  "sd_eta_density_by_type", "sd_eta"):
            if c in df:
                agg[c] = (c, "median")
        g = df.groupby(key).agg(**agg)
        print("\nper-%s summary (sd_eta_* = each weight block's contribution to the "
              "spread of logit pi, median over folds):" % key)
        print(g.round(4).to_string())

        # Paired comparison. Every value of `key` is fit on the SAME folds, so
        # the between-fold variance -- which is several times larger than the
        # effect -- cancels. Comparing group means hides the signal entirely.
        piv = df.pivot(index="fold", columns=key, values="heldout_loglik_per_cell")
        ok = g[g["n_converged"] == g["n"]]
        if ok.empty:
            print(f"\n** No {key} value converged on every fold, so none can be "
                  "selected on held-out likelihood. Raise `optim.maxiter` and "
                  "re-run before reading anything from the table above. **")
        elif key == "rank":
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
            best = select_lambda(df, ok)
            print(f"\nselected lam = {best}")
            g_ = g.loc[best]
            print(f"  at the selected lam: sigma_1(B) {g_['sigma1']:.3f}, entropy "
                  f"{g_['entropy']:.4f}, e^delta {g_['e_delta']:.4f}")
            if best == max(ok.index):
                print("  ** the selection is at the top of the grid: extend the grid "
                      "upward before trusting it **")

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

    elif a.job == "weight_variants":
        df["sigma1"] = [x[0] if isinstance(x, list) and x else np.nan for x in df["sigma"]]
        best = df.sort_values("loglik", ascending=False).groupby("variant").head(1)
        order = ["knots_only", "linear_interaction", "spline_interaction"]
        best = best.set_index("variant").reindex([v for v in order if v in set(best["variant"])])
        npar = {"knots_only": 65, "linear_interaction": 73, "spline_interaction": 105}
        print("\nWhat does letting the density response vary by cell type buy,")
        print("over simply giving h() more knots? All three share the same knots.\n")
        print(f"{'variant':<20}{'params':>8}{'loglik':>13}{'vs knots_only':>15}"
              f"{'per param':>11}{'fold':>8}{'mean w':>9}{'sigma1':>9}")
        ref = best.loc["knots_only", "loglik"] if "knots_only" in best.index else np.nan
        for v, r in best.iterrows():
            gain = r["loglik"] - ref
            dpar = npar.get(v, np.nan) - npar["knots_only"]
            pp = gain / dpar if dpar else np.nan
            print(f"{v:<20}{npar.get(v, 0):>8}{r['loglik']:>13.1f}{gain:>15.1f}"
                  + (f"{pp:>11.1f}" if np.isfinite(pp) and dpar else f"{'-':>11}")
                  + f"{np.exp(r['e_delta']):>8.4f}{r['mean_w']:>9.4f}{r['sigma1']:>9.4f}")
        print("\nA gain of ~0.5 per parameter is what pure overfitting yields, so anything")
        print("near that is not evidence for the extra block. Restart agreement per variant:")
        for v in best.index:
            ll = df[df["variant"] == v]["loglik"].to_numpy()
            print(f"  {v:<20}{100 * np.mean(ll > ll.max() - 1e-6):>5.0f}%  "
                  f"({df[df['variant'] == v]['converged'].sum()}/{len(ll)} converged)")

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
            if a.job.startswith("sim_single_component"):
                # Test 03 (criterion of 2026-10-07): the model reports no
                # structure. A replicate passes if EITHER the mixture collapses
                # to one component (e^delta < 0.05, or mean pi within 0.01 of a
                # boundary -- the spec's own branch) OR its two components
                # barely differ and cells are not split between them (rate
                # ratio < 1.5, the Test 07 threshold, and entropy < 0.10).
                # Entropy alone cannot judge a collapse: with pi ~ 0 for every
                # cell it is ~1, the same reading as sharply separated classes.
                d_ = df.assign(ratio=np.exp(df["e_delta"].astype(float)),
                               ent=df["entropy"].astype(float),
                               mw=df["mean_w"].astype(float))
                d_["collapsed"] = (d_["e_delta"] < 0.05) | (np.minimum(d_["mw"], 1 - d_["mw"]) < 0.01)
                d_["unsplit"] = (d_["ratio"] < 1.5) & (d_["ent"] < 0.10)
                d_["pass"] = d_["collapsed"] | d_["unsplit"]
                by = d_.groupby("lam") if "lam" in d_ else [(None, d_)]
                for lam_, g_ in by:
                    ok_ = bool(g_["pass"].all())
                    print(f"  spec 6.3{'' if lam_ is None else f'  lam {lam_:<8g}'}  "
                          f"{int(g_['pass'].sum())}/{len(g_)} replicates pass "
                          f"({int(g_['collapsed'].sum())} collapsed to one component, "
                          f"{int((g_['unsplit'] & ~g_['collapsed']).sum())} two components unsplit); "
                          f"failing entropy {g_.loc[~g_['pass'], 'ent'].round(3).tolist()} -> "
                          f"{'PASS' if ok_ else 'FAIL'}")
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
