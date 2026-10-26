"""Pairwise slices with the other seven densities held where the data put them.

    python run/portrait_conditional_slices.py [--bound none|osdr] [--min-cells 5] [--bw 0.25] [--simple]

--simple draws q50 and conditional only, with short titles, to portraits_<bound>_simple.png
(the fixed-point table is not rewritten); with --rows the references are rows and the pairs
columns, to portraits_<bound>_simple_rows.png.

The production slices (portrait_preliminary.py) hold every off-axis archetype at its own
median over all neighbourhoods. Those medians are taken type by type, so together they
need not be any neighbourhood that occurs, and they ignore that the other types are
distributed differently where the pair of interest is present. Count-based state; three
references, side by side:

  q50          (as before) each other type at its median over all observed states
  co-present   each other type at its median over the neighbourhoods where BOTH axis
               types have at least --min-cells cells in the 80 um disc
  conditional  the reference moves with the grid: at (X_a, X_b), each other type is at
               its kernel-weighted mean over observed states near (log2 X_a, log2 X_b)
               (Gaussian kernel, bandwidth --bw in log2 units; a mean, not a median, so
               the field is smooth enough for root finding). The slice then follows the
               data surface E[X_others | X_a, X_b].

Fixed points: roots of (r_a, r_b) in log X from the slowest grid points (`hybr`), with
the reference evaluated at the point. Stability from the 2 x 2 Jacobian of
(r_a, r_b) in log X by central differences, which includes how the conditional
reference moves; for an interior point this has the same eigenvalue signs as the
Jacobian of F. Support: as the production slices (q95 caps, convex hull, >= 20 cells per
coarse bin), counting for q50 and co-present only cells whose held densities lie within
half an IQR of the reference, and for conditional every cell (its reference is local).

Writes outputs/portrait/conditional_slices/{portraits_<bound>.png, fixed_points_<bound>.csv}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import context                              # noqa: E402

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.conditional import AREA, NGRID, Reference, fixed_points, rates  # noqa: E402
from tissue_dynamics.portrait.core import build_archetypes, fit_osdr_bound  # noqa: E402
from tissue_dynamics.portrait.summary import PAIRS, support_mask      # noqa: E402

OUT = "outputs/portrait/conditional_slices"
MODES = ("q50", "co-present", "conditional")
INK, MUTED = "#1f1f1e", "#8a8985"
C_A, C_B, C_OBS = "#2a78d6", "#e34948", "#1baf7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound", choices=("none", "osdr"), default="none")
    ap.add_argument("--min-cells", type=float, default=5)
    ap.add_argument("--bw", type=float, default=0.25)
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--rows", action="store_true", help="with --simple: references as rows, pairs as columns")
    a = ap.parse_args()
    modes = ("q50", "conditional") if a.simple else MODES
    os.makedirs(OUT, exist_ok=True)
    ctx = context(load_config())
    arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state="counts", radius_um=ctx["radius"])
    model, ox, names = ctx["model"], ctx["ox"], arch.names
    bound = (fit_osdr_bound(model, arch, ox, cell_archetype=ctx["arche"], power=8)
             if a.bound == "osdr" else None)
    Xo = np.clip(arch.X_obs, 0, None)
    min_x = a.min_cells / AREA
    tag = "unbounded" if bound is None else "osdr_p8"

    if a.rows:
        fig, axes = plt.subplots(len(modes), len(PAIRS), figsize=(5.2 * len(PAIRS), 4.9 * len(modes)))
        axes = axes.T                                     # still indexed [pair, reference]
    else:
        fig, axes = plt.subplots(len(PAIRS), len(modes), figsize=(5.2 * len(modes), 4.9 * len(PAIRS)))
    rows = []
    for p, (na, nb) in enumerate(PAIRS):
        ka, kb = names.index(na), names.index(nb)
        both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
        obs = Xo[both][:, [ka, kb]]
        lo = np.percentile(obs, 1, axis=0)
        hi = np.array([np.percentile(Xo[:, ka], 95), np.percentile(Xo[:, kb], 95)])
        ga, gb = np.logspace(*np.log10([lo[0], hi[0]]), NGRID), np.logspace(*np.log10([lo[1], hi[1]]), NGRID)
        GA, GB = np.meshgrid(ga, gb)
        held = [m for m in range(len(names)) if m not in (ka, kb)]
        tol = np.array([0.5 * np.subtract(*np.percentile(Xo[:, m], [75, 25])) for m in held])
        for c, mode in enumerate(modes):
            R = Reference(mode, Xo, ka, kb, min_x, a.bw, lo, hi)
            if mode == "conditional":
                held_ok = np.ones(both.sum(), bool)
            else:
                held_ok = np.all(np.abs(Xo[both][:, held] - R.ref[held]) < np.abs(tol), axis=1)
            M, _n = support_mask(GA, GB, obs, held_ok, ka, kb, Xo)
            ra, rb = rates(R, GA.ravel(), GB.ravel(), model, arch, ox, bound)
            U, V = (ra / np.log(2)).reshape(GA.shape), (rb / np.log(2)).reshape(GA.shape)
            fps = fixed_points(R, U, V, GA, GB, lo, hi, model, arch, ox, bound)
            ax = axes[p, c]
            La, Lb = np.log2(ga), np.log2(gb)
            ax.pcolormesh(La, Lb, np.where(M, np.nan, 1.0), cmap="Greys", vmin=0, vmax=3,
                          shading="auto", zorder=0, alpha=.55)
            sub = obs[np.random.default_rng(0).choice(len(obs), min(4000, len(obs)), replace=False)]
            ax.plot(np.log2(sub[:, 0]), np.log2(sub[:, 1]), ".", ms=1.4, color=C_OBS, alpha=.25, zorder=1)
            Um, Vm = np.where(M, U, np.nan), np.where(M, V, np.nan)
            ax.streamplot(La, Lb, Um, Vm, density=1.1, color="0.35", linewidth=.7, arrowsize=.85, zorder=2)
            ax.contour(La, Lb, Um, [0], colors=C_A, linewidths=2.1, zorder=3)
            ax.contour(La, Lb, Vm, [0], colors=C_B, linewidths=2.1, zorder=3)
            n_in = 0
            for xf, cls in fps:
                gi = (np.abs(ga - xf[0]).argmin(), np.abs(gb - xf[1]).argmin())
                on_grid = lo[0] <= xf[0] <= hi[0] and lo[1] <= xf[1] <= hi[1]
                inside = on_grid and bool(M[gi[1], gi[0]])
                n_in += inside
                ax.plot(np.log2(xf[0]), np.log2(xf[1]), {"stable": "o", "saddle": "s", "unstable": "^"}[cls],
                        ms=10, mfc=INK if cls == "stable" else "white", mec=INK, mew=1.6,
                        alpha=1.0 if inside else 0.35, zorder=6)
                Z = R.state(xf[0], xf[1])[0]
                rows.append({"pair": f"{na} x {nb}", "reference": mode, "X_a": xf[0], "X_b": xf[1],
                             "n": float(Z.sum()), "classification": cls, "in_support": inside,
                             **{f"held_{names[m]}": float(Z[m]) for m in held}})
            ax.set_xlim(La[0], La[-1])
            ax.set_ylim(Lb[0], Lb[-1])
            note = {"q50": "others at overall medians",
                    "co-present": (f"others at medians where both >= {a.min_cells:g} cells\n"
                                   f"({R.n_sel:,} neighbourhoods)" if mode == "co-present" else ""),
                    "conditional": f"others at local mean given (X_a, X_b)\n(bandwidth {a.bw:g} log2)"}[mode]
            if a.simple:
                note = None
            ax.set_title(f"{na} x {nb}: {mode}\n" + (f"{note}\n" if note else "")
                         + f"{len(fps)} fixed point{'' if len(fps) == 1 else 's'}, {n_in} in support",
                         fontsize=9, loc="left")
            ax.set_xlabel(f"log2 {na} density")
            ax.set_ylabel(f"log2 {nb} density")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    if a.simple:
        fig.suptitle("Pairwise phase-portrait slices, " + ("unbounded" if bound is None else "OSDR bound (p = 8)"),
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fig.savefig(f"{OUT}/portraits_{tag}_simple{'_rows' if a.rows else ''}.png", dpi=130)
        plt.close(fig)
        return
    fig.suptitle(f"Pairwise slices of the full portrait, count-based state, {tag}: three ways to hold the "
                 "other seven types\nblue = x-nullcline, red = y-nullcline; filled circle stable, square "
                 "saddle, triangle unstable (faded = outside support); grey = outside support",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(f"{OUT}/portraits_{tag}.png", dpi=130)
    plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/fixed_points_{tag}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 8):
        print(tab[["pair", "reference", "X_a", "X_b", "n", "classification", "in_support"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
