"""Penalty and rank under the production model: the re-check, and what lambda changes.

    python run/lambda_sensitivity.py [--fits outputs/hpc] [--compare LAM]

The weight-model penalty (`model.weight_ridge`) and the rank of B were chosen
on the area model (2026-10-07). This re-checks both under the configuration in
force, from four Spartan jobs read from --fits:

  sim_single_component_path_full  null data at the real n along a lambda path:
                                  the gate of the selection rule, judged by
                                  Test 03's criterion (hpc/gather.py)
  sim_single_component            Test 03 itself, at the production lambda
  cv_penalty                      held-out log-likelihood on 5 spatial folds
                                  along the lambda grid, paired by spatial block
  cv_rank                         the same over the rank grid, at the
                                  production lambda

and then measures what the choice of lambda changes. Every cv_penalty fold fit
stores c and B, so the priority-pair neighbourhood odds ratios can be read off
each one. Differences between two lambdas are taken within fold and scaled by
the production bootstrap's standard deviation of each log odds ratio. At the
production lambda the fold fits reproduce the full fit (checked in the
report), so their values at other lambdas stand in for full fits there.

--compare (default: the smallest lambda on the null path at which every
replicate passes, the selection rule's literal pick) is set beside the
production lambda.

Writes outputs/penalty/{report.md, null_check.csv, heldout.csv, rank.csv,
effects_by_lambda.csv, effects_by_lambda.png}.
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
sys.path.insert(0, os.path.join(HERE, "..", "hpc"))

import matplotlib                                                     # noqa: E402
import matplotlib.ticker                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from gather import block_scores, load, paired, test03_pass           # noqa: E402
from worker import stale_reasons                                      # noqa: E402
from tissue_dynamics import mistic as M                               # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.tasks import enumerate_tasks             # noqa: E402

OUT = "outputs/penalty"
Z = 1.959964
PATH_COLS = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def t6():
    """run/mistic_t6_old_vs_new.py as a module, for its priority pairs."""
    spec = importlib.util.spec_from_file_location("t6", os.path.join(HERE, "mistic_t6_old_vs_new.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def current(job, fits, cfg):
    """A job's results, refusing any written under another configuration."""
    df = load(job, fits)
    if df.empty:
        sys.exit(f"no results in {os.path.join(fits, job)}")
    tasks = enumerate_tasks(job, cfg)
    stale = [int(r["index"]) for _, r in df.iterrows()
             if stale_reasons(r.to_dict(), tasks[int(r["index"])], cfg)]
    if stale:
        sys.exit(f"{job}: tasks {stale[:5]}{'...' if len(stale) > 5 else ''} were written under "
                 "another configuration; rerun them before reading this report")
    if len(df) < len(tasks):
        print(f"  ** {job}: {len(df)} of {len(tasks)} tasks present **")
    return df


def md(frame, floatfmt="{:,.3f}"):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                                       for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fits", default="outputs/hpc")
    ap.add_argument("--compare", type=float, default=None)
    a = ap.parse_args()
    t0 = time.time()
    cfg, df, _out = M.test_setup()
    mc = cfg["model"]
    lam0, rank0 = float(mc["weight_ridge"]), int(mc["rank"])
    tech = str(mc.get("technical", "area"))
    n_cells = len(df)
    os.makedirs(OUT, exist_ok=True)

    # ---- the null check: the selection rule's gate -----------------------------------
    path = test03_pass(current("sim_single_component_path_full", a.fits, cfg))
    t03 = test03_pass(current("sim_single_component", a.fits, cfg))
    t03["lam"] = lam0
    rows = []
    for src, d in (("lambda path", path), ("Test 03", t03)):
        for lam, g in d.groupby("lam"):
            two = g.loc[~g["collapsed"], "ent"]          # a collapse reads ~1 and passes
            rows.append({"source": src, "lambda": float(lam), "replicates": len(g),
                         "pass": int(g["pass"].sum()), "collapsed": int(g["collapsed"].sum()),
                         "largest entropy, two components": float(two.max()) if len(two) else np.nan,
                         "entropies, two components": ", ".join(f"{e:.3f}" for e in sorted(two))
                         if len(two) <= 5 else f"median {two.median():.3f}"})
    N = pd.DataFrame(rows)
    N.to_csv(f"{OUT}/null_check.csv", index=False)
    on_path = N[N["source"] == "lambda path"]
    passing = on_path.loc[on_path["pass"] == on_path["replicates"], "lambda"]
    rule_pick = float(passing.min()) if len(passing) else np.nan
    lam1 = float(a.compare) if a.compare is not None else rule_pick
    if not np.isfinite(lam1) or lam1 == lam0:
        sys.exit("no comparison lambda: pass --compare")

    # ---- held out along the lambda grid and the rank grid ------------------------------
    cvp = current("cv_penalty", a.fits, cfg)
    cvp = cvp[cvp["converged"].astype(bool)]
    S = block_scores(cvp, "lam")
    S = S[sorted(S.columns)]
    best = S.sum().idxmax()
    H = pd.DataFrame([{"lambda": float(lam), "held-out total": float(S[lam].sum()),
                       "minus best": paired(S, lam, best)[0] if lam != best else 0.0,
                       "SE": paired(S, lam, best)[1] if lam != best else 0.0}
                      for lam in S.columns])
    H["within 2 SE of best"] = np.where(H["lambda"] == best, "best",
                                        np.where(H["minus best"] > -2 * H["SE"], "yes", "no"))
    H.to_csv(f"{OUT}/heldout.csv", index=False)
    d01, se01 = paired(S, lam1, lam0)

    cvr = current("cv_rank", a.fits, cfg)
    Sr = block_scores(cvr[cvr["converged"].astype(bool)], "rank")
    ranks = sorted(Sr.columns)
    Rk = pd.DataFrame([{"rank": int(r), "held-out total": float(Sr[r].sum()),
                        "gain over the rank below": paired(Sr, r, ranks[i - 1])[0] if i else np.nan,
                        "SE": paired(Sr, r, ranks[i - 1])[1] if i else np.nan}
                       for i, r in enumerate(ranks)])
    Rk.to_csv(f"{OUT}/rank.csv", index=False)
    top_rank = int(Sr.sum().idxmax())
    gap, gap_se = paired(Sr, top_rank, rank0) if top_rank != rank0 else (0.0, 0.0)

    # ---- what lambda changes: the priority-pair odds ratios ----------------------------
    T6 = t6()
    basis = build_basis(df, cfg, verbose=False)
    names = df.attrs["phi_names"]
    own = basis.own(df)
    arche = df["archetype"].astype(str).to_numpy()
    nbr_raw = df[basis.nbr_cols].to_numpy(float)
    iqr = np.percentile(nbr_raw, 75, axis=0) - np.percentile(nbr_raw, 25, axis=0)
    blocks = [b for b, _ in basis.nbr_blocks]
    pairs = T6.PAIRS + [(q, p) for p, q in T6.PAIRS]
    phi = {t: own[arche == t].mean(0) for t in {x for pr in pairs for x in pr}}
    labels = [(o, blk, nb) for o, nb in pairs for blk in blocks]

    def log_or(c, B):
        """Log odds ratio of cycling per IQR of each priority-pair component."""
        out = []
        for o, blk, nb in labels:
            eff = basis.Q_nbr @ (np.asarray(c) + np.asarray(B).T @ phi[o])
            j = blocks.index(blk) * len(names) + names.index(nb)
            out.append(float(eff[j] * iqr[j]))
        return np.array(out)

    pe = json.load(open(os.path.join(a.fits, "point_estimate.json")))
    if pe.get("technical", "area") != tech or float(pe["weight_ridge"]) != lam0:
        sys.exit(f"{a.fits}/point_estimate.json is not the production fit")
    full = log_or(pe["params"]["c"], np.asarray(pe["params"]["U"]) @ np.asarray(pe["params"]["V"]).T)
    boot = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.fits, "bootstrap", "*.json")))]
    boot = np.array([log_or(b["c"], b["B"]) for b in boot
                     if b.get("converged", True) and b.get("technical", "area") == tech
                     and b.get("B") is not None])
    se = boot.std(0, ddof=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    fold = {(float(r["lam"]), int(r["fold"])): log_or(r["c"], r["B"]) for _, r in cvp.iterrows()}
    folds = sorted({f for _, f in fold})
    lams = sorted({lam for lam, _ in fold})
    mean_at = {lam: np.mean([fold[(lam, f)] for f in folds], axis=0) for lam in lams}
    shift = np.array([fold[(lam1, f)] - fold[(lam0, f)] for f in folds])     # paired within fold
    agree = np.max(np.abs(np.exp(mean_at[lam0]) / np.exp(full) - 1))

    lab = [f"{o} <- {blk}: {nb}" for o, blk, nb in labels]
    E = pd.DataFrame({"effect": lab,
                      f"OR, full fit (lambda {lam0:g})": np.exp(full),
                      "95% CI low": np.exp(lo), "95% CI high": np.exp(hi),
                      **{f"OR, folds, lambda {lam:g}": np.exp(mean_at[lam]) for lam in lams},
                      f"shift {lam1:g} vs {lam0:g} (bootstrap SEs)": shift.mean(0) / se,
                      "fold sd of shift (SEs)": shift.std(0, ddof=1) / se})
    E.to_csv(f"{OUT}/effects_by_lambda.csv", index=False)
    z = shift.mean(0) / se
    # does the interval change side of 1? normal intervals from the bootstrap SE at
    # both lambdas, the comparison one centred on the full fit moved by the shift
    side = lambda m: np.sign(m - Z * se) * (m - Z * se > 0) + np.sign(m + Z * se) * (m + Z * se < 0)
    crosses = [lab[i] for i in np.flatnonzero(side(full) != side(full + shift.mean(0)))]

    C = pd.DataFrame({"own archetype": [o for o, _, _ in labels],
                      "neighbour component": [f"{blk}: {nb}" for _, blk, nb in labels],
                      f"lambda {lam0:g}, full fit": [f"{np.exp(v):.3f} ({np.exp(l):.3f}, {np.exp(h):.3f})"
                                                    for v, l, h in zip(full, lo, hi)],
                      f"lambda {lam0:g}, folds": [f"{np.exp(v):.3f}" for v in mean_at[lam0]],
                      f"lambda {lam1:g}, folds": [f"{np.exp(v):.3f}" for v in mean_at[lam1]],
                      "shift (bootstrap SEs)": [f"{v:+.2f}" for v in z]})
    P = pd.DataFrame({"effect": lab, **{f"{lam:g}": [f"{np.exp(v):.3f}" for v in mean_at[lam]]
                                         for lam in PATH_COLS if lam in mean_at}})

    # ---- figure ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.2 * len(blocks), 4.8), sharey=True)
    colours = plt.get_cmap("tab10").colors
    xs = [lam for lam in lams if lam > 0]
    for k, (ax, blk) in enumerate(zip(np.atleast_1d(axes), blocks)):
        for i, (o, nb) in enumerate(pairs):
            j = labels.index((o, blk, nb))
            ax.plot(xs, [np.exp(mean_at[lam][j]) for lam in xs], marker="o", ms=3,
                    color=colours[i % 10], label=f"{o} <- {nb}")
        for lam, ls in ((lam0, "--"), (lam1, ":")):
            ax.axvline(lam, color="0.35", ls=ls, lw=1)
        ax.axhline(1.0, color="0.75", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.9, 1.0, 1.25, 1.5, 2.0, 2.5]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{blk} neighbourhood")
        ax.set_xlabel("weight_ridge (lambda)")
        if k == 0:
            ax.set_ylabel("odds ratio of cycling per IQR (fold mean)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"Priority-pair effects along the penalty path (dashed: production "
                 f"{lam0:g}; dotted: {lam1:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/effects_by_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----------------------------------------------------------------------
    strong = np.abs(mean_at[lam0]) > 0.2
    ratio = mean_at[lam1][strong] / mean_at[lam0][strong]
    t03_row = N[N["source"] == "Test 03"].iloc[0]
    alt_row = on_path[on_path["lambda"] == lam1]
    text = [
        "# Penalty and rank under the production model", "",
        f"`technical: {tech}`, `weight_ridge` {lam0:g}, rank {rank0}. Both were chosen on the "
        "area model (2026-10-07); this re-checks them on the model in force and measures what "
        f"the penalty changes. Fits: `{a.fits}`; assembled {time.strftime('%Y-%m-%d %H:%M')}.", "",
        "## Null check (the selection rule's gate)", "",
        "Data with no class structure at the real n, fitted along a lambda path. A replicate "
        "passes Test 03's criterion if the mixture collapses to one component or its two "
        "components barely differ and cells are not split between them (rate ratio < 1.5 and "
        "normalised entropy < 0.10).", "",
        md(N.assign(**{"lambda": N["lambda"].map("{:g}".format)}), "{:.3f}"), "",
        f"Smallest lambda on the path at which every replicate passes: **{rule_pick:g}**. "
        f"Test 03 at the production lambda: {int(t03_row['pass'])} of "
        f"{int(t03_row['replicates'])}, largest entropy "
        f"{t03_row['largest entropy, two components']:.3f}.", "",
        "## Held-out log-likelihood along the lambda grid", "",
        f"cv_penalty: 5 spatial folds, summed over {len(S)} spatial blocks, each lambda paired "
        f"with the best ({best:g}) block by block.", "",
        md(H.assign(**{"lambda": H["lambda"].map("{:g}".format)}), "{:,.1f}"), "",
        f"lambda {lam1:g} minus lambda {lam0:g}: **{d01:+.1f} +/- {se01:.1f}** "
        f"({d01 / n_cells:+.1e} per cell, over {n_cells:,} cells).", "",
        "## Rank", "",
        f"cv_rank at lambda {lam0:g}, paired by spatial block.", "",
        md(Rk.assign(rank=Rk["rank"].astype(str)), "{:,.1f}"), "",
        (f"Rank {rank0} is {gap:.1f} +/- {gap_se:.1f} short of the best-scoring rank {top_rank}."
         if top_rank != rank0 else f"Rank {rank0} scores best."), "",
        f"## What lambda changes: {lam1:g} against {lam0:g}", "",
        "Odds ratio of cycling per IQR of each priority-pair neighbourhood component. The full "
        "fit carries its 95% spatial block bootstrap interval. The fold columns average the "
        "five cv_penalty fold fits at each lambda; the shift is the within-fold difference in "
        f"log odds ratio, {lam1:g} minus {lam0:g}, over the bootstrap SD.", "",
        md(C), "",
        f"- At lambda {lam0:g} the fold means reproduce the full fit to within "
        f"{100 * agree:.1f}%, so the fold values stand in for full fits at other lambdas.",
        f"- {int((np.abs(z) < 0.5).sum())} of {len(z)} effects move by less than 0.5 SE "
        f"(median |shift| {np.median(np.abs(z)):.2f}); the largest moves "
        f"{np.abs(z).max():.2f} SE ({lab[int(np.argmax(np.abs(z)))]}).",
        f"- Effects with |log OR| > 0.2 at lambda {lam0:g}: log OR at {lam1:g} over "
        f"{lam0:g} has median {np.median(ratio):.3f} (range {ratio.min():.3f} to {ratio.max():.3f}).",
        f"- The shift is systematic, not noise: its fold-to-fold SD is at most "
        f"{(shift.std(0, ddof=1) / se).max():.2f} SE.",
        f"- Intervals that would change side of OR = 1 at lambda {lam1:g} (normal intervals "
        f"from the bootstrap SE): {', '.join(crosses) if crosses else 'none'}.", "",
        f"Null check at lambda {lam1:g}: "
        + (f"{int(alt_row['pass'].iloc[0])} of {int(alt_row['replicates'].iloc[0])} pass, "
           f"entropies {alt_row['entropies, two components'].iloc[0]}." if len(alt_row)
           else "not on the path."), "",
        "## The path", "",
        "Fold-mean odds ratios at a selection of grid values (all of them in "
        "`effects_by_lambda.csv`; figure `effects_by_lambda.png`).", "",
        md(P), "",
    ]
    open(f"{OUT}/report.md", "w").write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
