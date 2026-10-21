"""Residual correlograms of the two production models, and the bootstrap blocks
measured against the residual correlation range (mixture spec 7.1).

    python run/spatial_residual_correlogram.py

Models (both MisTIC technical terms, depth offset, the same weight block and penalty):
  production Poisson mixture   outputs/hpc/point_estimate.json
  production NB (NB-c)         outputs/division_classifier/nb_params_nbc_full.json,
                               one constant dispersion -- the covariate-matched single NB
(`run/spatial_analysis.py` §2 used an unpenalised refit with a free depth exponent
as its "matched NB", beside the earlier NB; this replaces both with NB-c.)

Correlogram: Moran's I with binary distance-band weights, `run/spatial_analysis.py`'s
estimator (focal sample against all cells), out to 1 mm. Pearson and mid-quantile
residuals; the raw count for scale. Noise envelope: each residual series permuted
over cells N_PERM times and run through the same bands; +/- 2 SD of those.
Two correlation ranges per series, each the smallest band midpoint from which every
band onward satisfies the rule:
  Test 14 (spec 6.14, `run/diagnostics_report.py`)   |I| < 0.01
  significance                                        |I| within the +/- 2 SD envelope
With millions of pairs per band the envelope is ~1e-4 wide, so the second finds
correlation far too weak to matter; spec 7.1's "5x the range" reads the first.

Blocks: the bootstrap's own partition (`spatial_blocks(xy, 250)`, seed 0). Each
block's side is sqrt(area), its area the 20 um squares its cells occupy, so the
tissue's holes and edges are not counted.

Writes outputs/spatial/production_correlogram.csv, production_blocks.json and
residual_correlogram_{removed,pearson,midq}.png.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from spatial_analysis import mix_moments_cdf, nb_moments_cdf, residuals  # noqa: E402

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.inference import spatial_blocks          # noqa: E402
from tissue_dynamics.mixture.model import build_data, component_means, pi_hat  # noqa: E402

OUT = "outputs/spatial"
EDGES = np.array([0, 15, 25, 35, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000], float)
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
N_FOCAL = 15_000
N_PERM = 8
TEST14 = 0.01
CHUNK = 60
SQ_UM = 20.0
MODELS = {"mixture": "Poisson mixture (production)", "nbc": "NB-c (production NB)"}
COL = {"mixture": "#2a78d6", "nbc": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
ENV = "#e4e3df"


def say(*a):
    print(*a, flush=True)


def band_sums(xy, focal, Z, edges=EDGES, chunk=CHUNK):
    """`spatial_analysis.band_sums`, one column at a time (memory: pairs to 1 mm)."""
    tree = cKDTree(xy)
    nb = len(edges) - 1
    num = np.zeros((nb, Z.shape[1]))
    cnt = np.zeros(nb)
    for s in range(0, len(focal), chunk):
        fc = focal[s:s + chunk]
        sp = cKDTree(xy[fc]).sparse_distance_matrix(tree, float(edges[-1]), output_type="ndarray")
        i, j, d = fc[sp["i"]], sp["j"], sp["v"]
        keep = i != j
        i, j, d = i[keep], j[keep], d[keep]
        b = np.searchsorted(edges, d, side="right") - 1
        ok = (b >= 0) & (b < nb)
        i, j, b = i[ok], j[ok], b[ok]
        cnt += np.bincount(b, minlength=nb)
        for c in range(Z.shape[1]):
            num[:, c] += np.bincount(b, weights=Z[i, c] * Z[j, c], minlength=nb)
    return num, cnt


def corr_range(I, bound):
    """Smallest band midpoint from which every band onward has |I| <= bound."""
    inside = np.abs(I) <= bound
    for b in range(len(I)):
        if inside[b:].all():
            return float(MID[b])
    return float("inf")


def block_sides(xy, labels):
    sq = np.floor((xy - xy.min(0)) / SQ_UM).astype(np.int64)
    key = sq[:, 0] * (sq[:, 1].max() + 1) + sq[:, 1]
    sides = []
    for b in np.unique(labels):
        sides.append(np.sqrt(len(np.unique(key[labels == b]))) * SQ_UM)
    return np.array(sides)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("distance band midpoint (µm, log scale)")


def main():
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open(cfg["data"]["table_parquet"].replace(".parquet", ".phi_names.json")))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    y = np.asarray(D["y"], float)
    xy = df[["x", "y"]].to_numpy(float)
    n = len(y)

    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    lam0, lam1 = (np.asarray(x) for x in component_means(P, D, spec))
    pi = np.asarray(pi_hat(P, D, spec))
    nbc = {k: jnp.asarray(np.asarray(v)) for k, v in
           json.load(open("outputs/division_classifier/nb_params_nbc_full.json")).items()}
    mu = np.exp(np.asarray(arms.nb_log_mu(nbc, D)))
    alpha = float(np.exp(np.asarray(arms.nb_log_alpha(nbc, D))))
    res = {"mixture": residuals(y, *mix_moments_cdf(y, lam0, lam1, pi)),
           "nbc": residuals(y, *nb_moments_cdf(y, mu, alpha))}
    say(f"{n:,} cells; NB-c alpha {alpha:.4f}; residuals in {time.time() - t0:.0f}s")

    rng = np.random.default_rng(0)
    cols, Vs = ["raw:count"], [(y - y.mean()) / y.std()]
    for m in MODELS:
        for t, r in zip(("pearson", "midq"), res[m]):
            z = (r - r.mean()) / r.std()
            cols.append(f"{t}:{m}")
            Vs.append(z)
            for p in range(N_PERM):
                cols.append(f"perm{p}:{t}:{m}")
                Vs.append(rng.permutation(z))
    V = np.column_stack(Vs)
    focal = rng.choice(n, N_FOCAL, replace=False)
    say(f"correlogram: {V.shape[1]} series, {N_FOCAL:,} focal cells, bands to {EDGES[-1]:.0f} um")
    num, cnt = band_sums(xy, focal, V)
    I = num / cnt[:, None] / np.mean(V ** 2, axis=0)[None, :]
    tab = pd.DataFrame(I, columns=cols)
    tab.insert(0, "pairs", cnt.astype(np.int64))
    tab.insert(0, "band_hi", EDGES[1:])
    tab.insert(0, "band_lo", EDGES[:-1])
    tab.insert(0, "dist_um", MID)
    rng_out = {}
    for m in MODELS:
        for t in ("pearson", "midq"):
            sd = tab[[f"perm{p}:{t}:{m}" for p in range(N_PERM)]].std(axis=1, ddof=1).to_numpy()
            tab[f"envelope_sd:{t}:{m}"] = sd
            rng_out[f"{t}:{m}"] = {"test14": corr_range(tab[f"{t}:{m}"].to_numpy(), TEST14),
                                   "significance": corr_range(tab[f"{t}:{m}"].to_numpy(), 2 * sd)}
    keep = [c for c in tab.columns if not c.startswith("perm")]
    tab[keep].to_csv(f"{OUT}/production_correlogram.csv", index=False)
    say(f"correlogram done ({time.time() - t0:.0f}s)")

    labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
    sides = block_sides(xy, labels)
    sizes = np.bincount(labels)
    worst = max(v["test14"] for v in rng_out.values())
    blocks = {"n_blocks": int(len(sides)), "cells_per_block_median": float(np.median(sizes)),
              "side_um_median": float(np.median(sides)),
              "side_um_p5": float(np.percentile(sides, 5)), "side_um_min": float(sides.min()),
              "side_um_p95": float(np.percentile(sides, 95)),
              "residual_range_um": rng_out,
              "5x_widest_test14_range_um": 5 * worst,
              "blocks_with_side_ge_5x_test14_range": float(np.mean(sides >= 5 * worst)),
              "raw_count_I_at_widest_band": float(tab["raw:count"].iloc[-1])}
    json.dump(blocks, open(f"{OUT}/production_blocks.json", "w"), indent=1)

    # ---- figures -------------------------------------------------------------
    d = tab["dist_um"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    ax.plot(d, tab["raw:count"], "--o", color=MUTED, lw=1.8, ms=3.5, label="raw Tirosh count")
    for m, lab in MODELS.items():
        ax.plot(d, tab[f"pearson:{m}"], "-o", color=COL[m], lw=2, ms=3.5, label=f"{lab}, Pearson residual")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(blocks["side_um_p5"], blocks["side_um_p95"], color=ENV, alpha=0.6, lw=0)
    ax.axvline(blocks["side_um_median"], color=INK2, lw=1, ls=":")
    ax.text(blocks["side_um_median"] * 0.97, ax.get_ylim()[1] * 0.92,
            f"bootstrap block side\nmedian {blocks['side_um_median']:.0f} µm\n(shade: 5th-95th pct)",
            ha="right", va="top", fontsize=8, color=INK2)
    style(ax)
    ax.set_ylabel("Moran's I")
    ax.set_title("What the production models remove: spatial autocorrelation of the count and its residuals",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    fig.savefig(f"{OUT}/residual_correlogram_removed.png", dpi=170)
    plt.close(fig)

    for t, ttl in (("pearson", "Pearson residuals"), ("midq", "Mid-quantile residuals")):
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        sd = np.max([tab[f"envelope_sd:{t}:{m}"] for m in MODELS], axis=0)
        ax.fill_between(d, -2 * sd, 2 * sd, color=ENV, lw=0, label="permutation noise, ±2 SD")
        for m, lab in MODELS.items():
            r = rng_out[f"{t}:{m}"]
            sig = f"{r['significance']:.0f} µm" if np.isfinite(r["significance"]) else "> 1 mm"
            ax.plot(d, tab[f"{t}:{m}"], "-o", color=COL[m], lw=2, ms=4, mec="white", mew=0.6,
                    label=f"{lab}: |I| < 0.01 from {r['test14']:.0f} µm; within noise from {sig}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axhline(TEST14, color=INK2, lw=0.8, ls=(0, (2, 2)))
        rt = max(rng_out[f"{t}:{m}"]["test14"] for m in MODELS)
        ax.axvspan(blocks["side_um_p5"], blocks["side_um_p95"], color=ENV, alpha=0.6, lw=0)
        ax.axvline(blocks["side_um_median"], color=INK2, lw=1, ls=":")
        ax.axvline(5 * rt, color=INK2, lw=1, ls="--")
        top = ax.get_ylim()[1]
        ax.text(blocks["side_um_median"] * 0.97, top * 0.97,
                f"block side\nmedian {blocks['side_um_median']:.0f} µm\n(shade: 5th-95th pct)",
                ha="right", va="top", fontsize=8, color=INK2)
        ax.text(5 * rt * 0.97, top * 0.97, f"5 × range\n= {5 * rt:.0f} µm", ha="right", va="top",
                fontsize=8, color=INK2)
        ax.text(1100, TEST14, "Test 14\n|I| = 0.01", ha="right", va="bottom", fontsize=7.5, color=INK2)
        style(ax)
        ax.set_ylabel("Moran's I")
        ax.set_title(f"{ttl}: production Poisson mixture and production NB", loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.savefig(f"{OUT}/residual_correlogram_{t}.png", dpi=170)
        plt.close(fig)

    with pd.option_context("display.width", 200):
        say(tab[["dist_um", "pairs", "raw:count"] + [f"{t}:{m}" for m in MODELS for t in ("pearson", "midq")]
                + [f"envelope_sd:pearson:{m}" for m in MODELS]].round(4).to_string(index=False))
    say(json.dumps(blocks, indent=1))
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
