"""The two latent components of each fitted mixture, as count distributions, and what
they do to the posterior probability of cycling.

    python run/plot_mixture_components.py [--band 45 55]

Models (all MisTIC technical terms, the production weight model, lambda 300):
  Poisson + Poisson   outputs/hpc/point_estimate.json (production)
  Poisson + NB        outputs/poisson_nb/params.json (cycling component NB2)
  NB + NB             outputs/nb_nb/params.json (both components NB2)

Each component's mean scales with the cell's depth, so the components are compared on
the same cells: those whose library size N lies between the given percentiles (the
90th-95th by default: at median depth the counts run only 0-4, too few to show the
components' shapes). Over those cells:

  top     the weighted component pmfs, mean_i (1 - pi_i) f0_i(k) and mean_i pi_i f1_i(k),
          and the observed share of cells at each count (points); their sum is the
          mixture's predicted distribution
  bottom  P(cycling | y = k) = mean_i pi_i f1_i(k) / (pi_i f1_i(k) + (1 - pi_i) f0_i(k)),
          the responsibility a cell of the band would get if its count were k

Writes outputs/figures/model_fit/mixture_components_by_family_N{lo}-{hi}pct.png.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.count_diagnostics import mixture_components, nb_parts  # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data                  # noqa: E402

OUT = "outputs/figures/model_fit"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
C_NON, C_CYC = "#2a78d6", "#eb6834"                      # non-cycling, cycling component
MODELS = {"Poisson + Poisson": ("outputs/hpc/point_estimate.json", False, False, "#2a78d6"),
          "Poisson + NB": ("outputs/poisson_nb/params.json", True, False, "#6250d6"),
          "NB + NB": ("outputs/nb_nb/params.json", True, True, "#1baf7a")}


def pmf(k, fam, mu, alpha):
    if fam == "pois":
        return stats.poisson.pmf(k[None, :], mu[:, None])
    r, p = nb_parts(mu, alpha)
    return stats.nbinom.pmf(k[None, :], r[:, None], p[:, None])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, nargs=2, default=(90, 95))
    a = ap.parse_args()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open(cfg["data"]["table_parquet"].replace(".parquet", ".phi_names.json")))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    c = copy.deepcopy(cfg)
    basis = build_basis(df, c, verbose=False)
    spec = spec_from_basis(basis, c, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, c, spec).as_jax()
    y = np.asarray(D["y"], float)
    N = df["N"].to_numpy(float)
    lo, hi = np.percentile(N, a.band)
    band = (N >= lo) & (N <= hi)
    yb = y[band]
    kmax = int(np.percentile(yb, 99.5))
    k = np.arange(kmax + 1)
    obs = np.array([np.mean(yb == j) for j in k])
    print(f"band: N in [{lo:.0f}, {hi:.0f}], {band.sum():,} cells, mean count {yb.mean():.2f}")

    res = {}
    for name, (path, cyc, non, _col) in MODELS.items():
        P = json.load(open(path))
        P = P.get("params", P)
        comps, pi = mixture_components(P, D, spec, cycling_nb=cyc, noncycling_nb=non)
        (w0, f0, m0, a0), (w1, f1, m1, a1) = comps
        p0, p1 = pmf(k, f0, m0[band], a0), pmf(k, f1, m1[band], a1)
        wb = pi[band][:, None]
        part0, part1 = ((1 - wb) * p0).mean(0), (wb * p1).mean(0)
        post = (wb * p1 / np.clip(wb * p1 + (1 - wb) * p0, 1e-300, None)).mean(0)
        mom = {}
        for lab, mm, fam, al in (("non", m0[band], f0, a0), ("cyc", m1[band], f1, a1)):
            var = mm + (al * mm ** 2 if fam == "nb" else 0.0)
            mom[lab] = (float(mm.mean()), float(var.mean() / mm.mean()), al)
        res[name] = dict(part0=part0, part1=part1, post=post, pi=float(pi[band].mean()), mom=mom)
        print(f"{name}: mean pi {res[name]['pi']:.3f} | non mean {mom['non'][0]:.2f} var/mean {mom['non'][1]:.2f} | "
              f"cyc mean {mom['cyc'][0]:.2f} var/mean {mom['cyc'][1]:.2f} | P(cyc|y=0..4) "
              + " ".join(f"{v:.2f}" for v in post[:5]))

    fig = plt.figure(figsize=(15, 9.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.18)
    ymax = max(max(r["part0"].max(), r["part1"].max(), obs.max()) for r in res.values()) * 1.08
    for j, (name, r) in enumerate(res.items()):
        ax = fig.add_subplot(gs[0, j], facecolor=SURF)
        (mn0, vm0, al0), (mn1, vm1, al1) = r["mom"]["non"], r["mom"]["cyc"]
        d0 = "Poisson" if al0 is None else f"NB, α₀ = {al0:.2f}"
        d1 = "Poisson" if al1 is None else f"NB, α₁ = {al1:.2f}"
        ax.bar(k - 0.2, r["part0"], width=0.4, color=C_NON, alpha=0.85, label=f"non-cycling ({d0})")
        ax.bar(k + 0.2, r["part1"], width=0.4, color=C_CYC, alpha=0.85, label=f"cycling ({d1})")
        ax.plot(k, obs, "o", ms=4, color=INK, mfc="white", mew=1.1, label="observed")
        ax.text(0.98, 0.62, f"non-cycling: weight {1 - r['pi']:.2f}\n  mean {mn0:.1f}, var/mean {vm0:.1f}\n"
                f"cycling: weight {r['pi']:.2f}\n  mean {mn1:.1f}, var/mean {vm1:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2, family="monospace")
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.8, kmax + 0.8)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tirosh count")
        if j == 0:
            ax.set_ylabel("share of cells (weight × pmf)")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        style(ax)

    ax = fig.add_subplot(gs[1, :], facecolor=SURF)
    for name, r in res.items():
        col = MODELS[name][3]
        cross = np.flatnonzero(r["post"] >= 0.5)
        lab = name + (f"  (≥ 0.5 from count {cross[0]})" if len(cross) else "")
        ax.plot(k, r["post"], "-o", color=col, lw=2, ms=4, mec=SURF, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, kmax + 0.5)
    ax.set_xlabel("Tirosh count")
    ax.set_ylabel("P(cycling | count)")
    ax.set_title("Posterior probability of the cycling component for a cell of this depth, by its count",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style(ax)
    fig.suptitle(f"The two latent components of each mixture, for cells at the {a.band[0]:.0f}th–{a.band[1]:.0f}th "
                 f"percentile of depth (library size {lo:.0f}–{hi:.0f}, {band.sum():,} cells)",
                 x=0.01, ha="left", fontsize=12.5)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/mixture_components_by_family_N{a.band[0]:.0f}-{a.band[1]:.0f}pct.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
