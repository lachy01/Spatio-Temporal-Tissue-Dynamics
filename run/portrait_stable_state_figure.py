"""Figures of the full-system stable state (FINDINGS §13): what it is, where it sits
among the observed neighbourhoods, what flows into it, and where it is in the tissue.

    python run/portrait_stable_state_figure.py [--power 8] [--starts 400]

Count-based state (the only one with a stable point), OSDR's bound at power p
(8 by default; p = 4's stable point is drawn beside it in figure 1). Reads
outputs/portrait/equilibria/counts_osdr_p{p}.csv (run/portrait_full_equilibria.py).

1. stable_state_composition.png
   (a) each type's density at the stable point, against the observed spread among
       the 2% of neighbourhoods closest to it and the bound's x_max (the "wall");
   (b) the excess cycling pi_k - pi_bar_k at the point. At equilibrium the bound's
       correction cancels it exactly, so this is how much of each type's balance
       the bound supplies: zero means the fitted rates balance on their own.
2. stable_state_flow.png -- observed neighbourhood states in their first two
   principal components (log2(1 + X)), with the equilibria, and trajectories of
   the ODE dX_k/dt = X_k r_k(X) integrated from a random sample of observed states,
   coloured by where they end (t = 3000): the stable state, collapse (n < 1; the\n   only low-density state is CAFs alone at n = 0.16, far below any observed density,\n   where the density response is extrapolated), runaway, or other. Unbounded\n   (left) and under the bound (right).
3. stable_state_tissue.png -- the section, with the cells whose neighbourhood is
   among the 2% closest to the stable state highlighted.

Also writes stable_state_summary.json (trajectory outcomes, the nearest
neighbourhoods' composition and types).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import context                              # noqa: E402

import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import build_archetypes, fit_osdr_bound, net_rate  # noqa: E402

OUT = "outputs/portrait/equilibria"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
C_STABLE, C_FACE, C_RUN, C_OTHER = "#2563eb", "#c2410c", "#7c3aed", "#9a9893"
NEAR = 0.02


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def integrate(X0, model, arch, ox, bound, T=3000.0, dt=0.5, save_every=10):
    """RK4 in log X for present types; absent types stay at 0 (faces are invariant)."""
    present = X0 > 0
    u = np.where(present, np.log(np.where(present, X0, 1.0)), 0.0)

    def f(u):
        X = np.where(present, np.exp(np.clip(u, -30, 12)), 0.0)
        r, _ = net_rate(X, model, arch, ox, bound)
        return np.where(present, r, 0.0)

    path = [X0.copy()]
    for s in range(int(T / dt)):
        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (s + 1) % save_every == 0:
            path.append(np.where(present, np.exp(np.clip(u, -30, 12)), 0.0))
    Xend = path[-1]
    speed = np.abs(f(u)).max(1)
    return np.array(path), Xend, speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=int, default=8)
    ap.add_argument("--starts", type=int, default=400)
    a = ap.parse_args()
    ctx = context(load_config())
    arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state="counts",
                                radius_um=ctx["radius"])
    model, ox, names, K = ctx["model"], ctx["ox"], arch.names, arch.K
    bound = fit_osdr_bound(model, arch, ox, cell_archetype=ctx["arche"], power=a.power)
    Xo = np.clip(arch.X_obs, 0, None)
    L = np.log2(1 + Xo)
    xcols = [f"X_{n}" for n in names]

    eq = pd.read_csv(f"{OUT}/counts_osdr_p{a.power}.csv")
    st = eq[eq["class"] == "stable"]
    if len(st) != 1:
        raise SystemExit(f"expected one stable point at p = {a.power}, found {len(st)}")
    Xs = st[xcols].to_numpy(float)[0]
    eq4 = pd.read_csv(f"{OUT}/counts_osdr_p4.csv")
    X4 = eq4.loc[eq4["class"] == "stable", xcols].to_numpy(float)
    X4 = X4[0] if len(X4) else None

    # nearest observed neighbourhoods
    d = np.sqrt(((L - np.log2(1 + Xs)) ** 2).sum(1))
    near = d <= np.quantile(d, NEAR)
    r, P = net_rate(Xs[None], model, arch, ox, bound)
    excess = P[0] - arch.pi_bar                       # = -correction at equilibrium
    summary = {"power": a.power, "stable point": dict(zip(names, map(float, Xs))), "n": float(Xs.sum()),
               "excess cycling pi - pi_bar": dict(zip(names, map(float, excess))),
               "density / x_max": dict(zip(names, map(float, Xs / bound.x_max))),
               "nearest 2%": {"cells": int(near.sum()),
                              "median density": float(np.median(ctx["dens"][near])),
                              "median X": dict(zip(names, map(float, np.median(Xo[near], 0)))),
                              "own type of those cells": pd.Series(ctx["arche"][near]).value_counts(
                                  normalize=True).round(3).to_dict()}}

    # ---- figure 1: composition --------------------------------------------------------
    order = np.argsort(Xs)
    y = np.arange(K)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]},
                                 layout="constrained")
    q1, q2, q3 = np.percentile(Xo[near], [25, 50, 75], axis=0)
    ax.hlines(y, q1[order], q3[order], color="#c9c8c3", lw=6, label="nearest 2% of observed neighbourhoods (IQR)")
    ax.plot(q2[order], y, "|", color=MUTED, ms=12, mew=2)
    ax.plot(bound.x_max[order], y, marker="|", ls="none", color=INK, ms=16, mew=2.2,
            label="the bound's maximal density x_max (95th pct)")
    if X4 is not None:
        ax.plot(X4[order], y, "o", mfc="white", mec=C_STABLE, mew=1.6, ms=7, label="stable point, p = 4")
    ax.plot(Xs[order], y, "o", color=C_STABLE, ms=8, label=f"stable point, p = {a.power}")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("density X_k (type-k cells per (100 µm)²)")
    ax.set_title(f"(a) The stable state: n = {Xs.sum():.0f} cells per (100 µm)² "
                 f"({100 * np.mean(ctx['dens'] < Xs.sum()):.0f}th percentile)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    held = excess[order] > 0.005
    bx.barh(y, excess[order], color=np.where(held, C_FACE, C_STABLE), height=0.6)
    for yi, e, h in zip(y, excess[order], held):
        bx.text(e + 0.0006, yi, f"{e:+.3f}", va="center", fontsize=8, color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels([])
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlabel("excess cycling π_k − π̄_k at the stable point")
    bx.set_title("(b) What the bound must cancel", fontsize=10.5, loc="left")
    bx.text(0.98, 0.04, "orange: held by the bound\nblue: balanced by the fit", transform=bx.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    style(bx)
    fig.savefig(f"{OUT}/stable_state_composition.png", dpi=170)
    plt.close(fig)

    # ---- figure 2: flow ---------------------------------------------------------------
    rng = np.random.default_rng(0)
    fit_rows = rng.choice(len(L), size=min(60_000, len(L)), replace=False)
    mu = L[fit_rows].mean(0)
    _, S, Vt = np.linalg.svd(L[fit_rows] - mu, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    proj = lambda X: (np.log2(1 + np.clip(X, 0, None)) - mu) @ Vt[:2].T   # noqa: E731
    PO = (L - mu) @ Vt[:2].T
    starts = rng.choice(len(Xo), size=a.starts, replace=False)
    X0 = Xo[starts]
    eqX = eq[xcols].to_numpy(float)
    eqL = np.log2(1 + eqX)
    nmax = float(np.percentile(ctx["dens"], 99.9))
    outcomes, fates = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), layout="constrained", sharex=True, sharey=True)
    for ax, (lab, bd) in zip(axes, (("unbounded", None), (f"OSDR bound, p = {a.power}", bound))):
        path, Xend, speed = integrate(X0, model, arch, ox, bd)
        nend, nprev = Xend.sum(1), path[-20].sum(1)
        runaway = ~np.isfinite(Xend).all(1) | ((nend > nmax) & (nend > nprev))
        collapse = (~runaway) & (nend < 1.0)
        dist = np.sqrt(((np.log2(1 + np.clip(Xend, 0, 1e6))[:, None, :] - eqL[None]) ** 2).sum(2))
        j = dist.argmin(1)
        at_stable = ~(runaway | collapse) & (eq["class"].to_numpy()[j] == "stable") & (dist.min(1) < 0.05)
        other = ~(runaway | collapse | at_stable)
        cls = np.select([at_stable, collapse, runaway], ["stable", "collapse", "runaway"], "other")
        fates[lab] = cls
        outcomes[lab] = {"starts": int(len(X0)), "reach the stable state": float(at_stable.mean()),
                         "collapse (n < 1 cell per (100 um)^2)": float(collapse.mean()),
                         "run away (n above the 99.9th pct and rising)": float(runaway.mean()),
                         "other (another state, or still moving at t = 3000)": float(other.mean()),
                         "median n at the end of 'other'": (float(np.median(nend[other])) if other.any()
                                                            else None),
                         "start with every type present": float((X0 > 0).all(1).mean())}
        ax.hexbin(PO[:, 0], PO[:, 1], gridsize=80, bins="log", cmap="Greys", mincnt=1, linewidths=0)
        col = {"stable": C_STABLE, "collapse": C_FACE, "runaway": C_RUN, "other": C_OTHER}
        for i in rng.permutation(len(X0))[:160]:
            pp = proj(np.clip(path[:, i], 0, 1e6))
            ax.plot(pp[:, 0], pp[:, 1], color=col[cls[i]], lw=0.7, alpha=0.55)
            ax.plot(pp[0, 0], pp[0, 1], "o", color=col[cls[i]], ms=2, alpha=0.7)
        if bd is not None:
            for _, e in eq[eq["in support"]].iterrows():
                pe = proj(e[xcols].to_numpy(float)[None])[0]
                if e["class"] == "stable":
                    ax.plot(*pe, "*", color=C_STABLE, ms=19, mec="white", mew=1.2, zorder=5)
                else:
                    ax.plot(*pe, "X", color=INK, ms=10, mec="white", mew=1, zorder=5)
        o = outcomes[lab]
        txt = (f"{100 * o['reach the stable state']:.0f}% reach the stable state\n"
               f"{100 * o['collapse (n < 1 cell per (100 um)^2)']:.0f}% collapse\n"
               f"{100 * o['run away (n above the 99.9th pct and rising)']:.0f}% run away\n"
               f"{100 * o['other (another state, or still moving at t = 3000)']:.0f}% other")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_title(lab, fontsize=11, loc="left")
        ax.set_xlabel(f"PC1 of log2(1 + X) ({100 * ev[0]:.0f}% of variance)")
        style(ax)
    lo, hi = np.percentile(PO, [0.2, 99.8], axis=0)
    pad = 0.15 * (hi - lo)
    axes[0].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    axes[0].set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    axes[0].set_ylabel(f"PC2 ({100 * ev[1]:.0f}% of variance)")
    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in
               ((C_STABLE, "ends at the stable state"), (C_FACE, "collapses (n < 1)"),
                (C_RUN, "runs away"), (C_OTHER, "other"))]
    handles += [plt.Line2D([], [], marker="*", ls="none", color=C_STABLE, ms=14, label="stable point"),
                plt.Line2D([], [], marker="X", ls="none", color=INK, ms=9, label="saddle (in support)")]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Trajectories from {a.starts} observed neighbourhoods (160 drawn); grey: every observed "
                 "neighbourhood state", fontsize=11)
    fig.savefig(f"{OUT}/stable_state_flow.png", dpi=170)
    plt.close(fig)
    summary["trajectories"] = outcomes
    labs = list(fates)
    summary["fate crosstab (rows unbounded, columns bounded)"] = pd.crosstab(
        fates[labs[0]], fates[labs[1]]).to_dict()

    # ---- figure 3: tissue ---------------------------------------------------------------
    xy = ctx["df"][["x", "y"]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 9 * np.ptp(xy[:, 1]) / np.ptp(xy[:, 0])), layout="constrained")
    ax.scatter(xy[~near, 0], xy[~near, 1], s=0.05, color="#d9d8d3", rasterized=True, linewidths=0)
    ax.scatter(xy[near, 0], xy[near, 1], s=0.4, color=C_STABLE, rasterized=True, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Cells whose neighbourhood is among the {100 * NEAR:.0f}% closest to the stable state "
                 f"({near.sum():,} cells)", fontsize=11, loc="left")
    fig.savefig(f"{OUT}/stable_state_tissue.png", dpi=170)
    plt.close(fig)

    json.dump(summary, open(f"{OUT}/stable_state_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
