"""Which weight block grows as the penalty is relaxed? The lambda path of a CV run.

    python run/lambda_path.py --run log1p=outputs/hpc/cv_penalty
    python run/lambda_path.py --run log1p=outputs/hpc/cv_penalty \
        --run "SpaNorm=outputs/hpc_spanorm/cv_penalty:path/to/spanorm_table.parquet" \
        --unwhitened --name runs_before_whitening

A cv_penalty run is a decreasing-lambda path, five times over (one fit per
lambda x fold). For each fit this measures every weight block's contribution
to the spread of logit(pi) across the table's cells:

    own        sd(phi' a)            neighbour  sd(phibar' c)
    bilinear   sd(phi' B phibar)     density    sd(H' h)
    oxygen     sd(psi u)             (when the run had it)

These are on the logit scale, so they compare blocks whatever their
covariates' scales -- unlike coefficient norms, which on unwhitened data mostly
measure how small a block's covariates are. A block whose contribution keeps
growing as lambda -> 0 while the others settle is the one the penalty is
restraining: that is where any separation comes from, and the honest basis for
any differential penalty.

Runs written after 2026-10-06 store these contributions themselves (sd_eta_*,
including the per-type density curves K); older runs stored only a, c, B and h,
so K's contribution is missing for them and the rest is recomputed here from
the table. `--unwhitened` rebuilds the basis the way runs before the whitening
change were fit.

Writes outputs/penalty/<name>.csv and <name>.png.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/penalty"
BLOCKS = ("own", "nbr", "bilinear", "density", "density_by_type", "oxygen")
LABEL = {"own": "own state  a", "nbr": "neighbour state  c", "bilinear": "bilinear  B",
         "density": "density  h", "density_by_type": "density by type  K",
         "oxygen": "oxygen  psi"}
COLOUR = {"own": "#2a78d6", "nbr": "#1baf7a", "bilinear": "#eb6834", "density": "#8a8984",
          "density_by_type": "#b8b7b2", "oxygen": "#7a4fc9"}


def design(table, cfg):
    df = pd.read_parquet(table)
    side = os.path.splitext(table)[0] + ".phi_names.json"
    names = (json.load(open(side)) if os.path.exists(side)
             else [f"phi_{j:02d}" for j in range(sum(c.startswith("phi_") for c in df))])
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    X = dict(own=basis.own(df), nbr=basis.nbr(df), hbasis=basis.density_weight_basis(df))
    if "oxygen" in df:
        X["oxygen"] = df["oxygen"].to_numpy(float)
    return X


def contributions(r, X):
    """sd of each block's term in logit(pi), from a fit's stored coefficients."""
    if any(k.startswith("sd_eta_") for k in r):
        return {b: r.get(f"sd_eta_{b}") for b in BLOCKS}
    out = {"own": float(np.std(X["own"] @ np.asarray(r["a"]))),
           "nbr": float(np.std(X["nbr"] @ np.asarray(r["c"])))}
    B = np.asarray(r["B"])
    out["bilinear"] = float(np.std(np.einsum("ij,jk,ik->i", X["own"], B, X["nbr"])))
    # a run fit before the density knots changed has a different h; skip it
    # rather than guess the basis it used
    if r.get("h") is not None and len(r["h"]) == X["hbasis"].shape[1]:
        out["density"] = float(np.std(X["hbasis"] @ np.asarray(r["h"])))
    if r.get("psi") is not None and "oxygen" in X:
        out["oxygen"] = float(np.std(float(r["psi"]) * X["oxygen"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=cv_dir[:table.parquet]; the table defaults to the config's")
    ap.add_argument("--unwhitened", action="store_true",
                    help="rebuild the basis as runs before the whitening change were fit")
    ap.add_argument("--name", default="lambda_path")
    a = ap.parse_args()

    cfg = load_config()
    if a.unwhitened:
        cfg = {**cfg, "state": {**cfg["state"], "whiten": False},
               "model": {**cfg["model"], "scale_weight_splines": False}}
    rows = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        cv_dir, _, table = rest.partition(":")
        table = table or cfg["data"]["table_parquet"]
        X = design(table, cfg)
        for f in sorted(glob.glob(os.path.join(cv_dir, "*.json"))):
            r = json.load(open(f))
            if r.get("failed"):
                continue
            sv = np.linalg.svd(np.asarray(r["B"]), compute_uv=False)
            rows.append({"run": label, "lam": float(r["lam"]),
                         "fold": int(r.get("fold", r.get("rep", 0))),       # sims have reps
                         "converged": bool(r["converged"]),
                         "heldout": r.get("heldout_loglik_per_cell"),
                         "norm_a": float(np.linalg.norm(r["a"])),
                         "norm_c": float(np.linalg.norm(r["c"])),
                         "norm_B": float(np.linalg.norm(r["B"])),
                         "sigma1": float(sv[0]), "sigma2": float(sv[1]) if len(sv) > 1 else 0.0,
                         **{f"sd_{b}": v for b, v in contributions(r, X).items()}})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/{a.name}.csv", index=False)

    runs = list(dict.fromkeys(df["run"]))
    fig, axes = plt.subplots(1, len(runs), figsize=(6.6 * len(runs), 4.6), squeeze=False)
    for ax, run in zip(axes[0], runs):
        d = df[df["run"] == run]
        lams = np.array(sorted(d["lam"].unique()))
        xpos = np.where(lams > 0, lams, lams[lams > 0].min() / 3 if (lams > 0).any() else 1)
        for b in BLOCKS:
            col = f"sd_{b}"
            if col not in d or d[col].isna().all():
                continue
            g = d.groupby("lam")[col]
            med, lo, hi = g.median(), g.min(), g.max()
            ax.fill_between(xpos, lo.reindex(lams), hi.reindex(lams), color=COLOUR[b], alpha=0.15, lw=0)
            ax.plot(xpos, med.reindex(lams), "o-", color=COLOUR[b], lw=2, ms=4, label=LABEL[b])
        nc = d.groupby("lam")["converged"].all().reindex(lams)
        for x, ok in zip(xpos, nc):
            if not ok:
                ax.axvline(x, color="#c4342d", lw=1, ls=":")
        ax.set_xscale("log")
        sds = d[[c for c in d if c.startswith("sd_")]].to_numpy(float)
        if np.nanmax(sds) > 20 * np.nanmedian(sds):      # null fits at lambda = 0 run away
            ax.set_yscale("log")
            ax.set_ylim(1e-3, None)        # collapsed replicates sit at ~0; keep the scale readable
        ax.set_xticks(xpos)
        ax.set_xticklabels(["0" if l_ == 0 else f"{l_:g}" for l_ in lams], fontsize=8)
        ax.set_xlabel("weight_ridge lambda (0 plotted at the left; dotted = not converged on every fold)")
        ax.set_ylabel("sd across cells of the block's term in logit(pi)")
        ax.set_title(f"{run}: weight blocks along the lambda path")
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{a.name}.png", dpi=150, bbox_inches="tight")
    summary = df.groupby(["run", "lam"])[[c for c in df if c.startswith(("sd_", "norm_", "sigma"))]].median()
    print(summary.round(3).to_string())
    print(f"\nwrote {OUT}/{a.name}.csv and .png")


if __name__ == "__main__":
    main()
