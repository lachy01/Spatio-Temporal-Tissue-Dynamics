"""Pairwise slices with the other seven densities held where the data put them.

    python run/portrait_conditional_slices.py [--bound none|osdr] [--min-cells 5] [--bw 0.25]

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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import root

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import context                              # noqa: E402

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import build_archetypes, fit_osdr_bound, net_rate  # noqa: E402
from tissue_dynamics.portrait.summary import PAIRS, support_mask      # noqa: E402

OUT = "outputs/portrait/conditional_slices"
NGRID = 100
AREA = np.pi * 80.0 ** 2 / 1e4                 # the disc in (100 um)^2: count = X * AREA
MODES = ("q50", "co-present", "conditional")
INK, MUTED = "#1f1f1e", "#8a8985"
C_A, C_B, C_OBS = "#2a78d6", "#e34948", "#1baf7a"


class Reference:
    """Full state X(X_a, X_b) for one pair under one reference mode."""

    def __init__(self, mode, Xo, ka, kb, min_x, bw, lo, hi):
        self.mode, self.ka, self.kb = mode, ka, kb
        both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
        if mode == "q50":
            self.ref = np.median(Xo, axis=0)
        elif mode == "co-present":
            sel = (Xo[:, ka] >= min_x) & (Xo[:, kb] >= min_x)
            self.ref = np.median(Xo[sel], axis=0)
            self.n_sel = int(sel.sum())
        else:
            # binned sums of every density on a fine log2 grid, Gaussian-smoothed: a
            # Nadaraya-Watson mean of X_others given (log2 X_a, log2 X_b)
            nb = 160
            ea = np.linspace(np.log2(lo[0]) - 1, np.log2(hi[0]) + 1, nb + 1)
            eb = np.linspace(np.log2(lo[1]) - 1, np.log2(hi[1]) + 1, nb + 1)
            La, Lb = np.log2(Xo[both, ka]), np.log2(Xo[both, kb])
            ia = np.clip(np.digitize(La, ea) - 1, 0, nb - 1)
            ib = np.clip(np.digitize(Lb, eb) - 1, 0, nb - 1)
            sig = bw / (ea[1] - ea[0]), bw / (eb[1] - eb[0])
            cnt = gaussian_filter(np.bincount(ia * nb + ib, minlength=nb * nb).reshape(nb, nb).astype(float),
                                  sig, mode="nearest")
            ca, cb = 0.5 * (ea[1:] + ea[:-1]), 0.5 * (eb[1:] + eb[:-1])
            self.interp = []
            for k in range(Xo.shape[1]):
                s = np.bincount(ia * nb + ib, weights=Xo[both, k], minlength=nb * nb).reshape(nb, nb)
                m = gaussian_filter(s, sig, mode="nearest") / np.clip(cnt, 1e-9, None)
                self.interp.append(RegularGridInterpolator((ca, cb), m, bounds_error=False, fill_value=None))

    def state(self, xa, xb):
        xa, xb = np.atleast_1d(xa).astype(float), np.atleast_1d(xb).astype(float)
        if self.mode == "conditional":
            pts = np.column_stack([np.log2(xa), np.log2(xb)])
            X = np.column_stack([np.clip(f(pts), 0, None) for f in self.interp])
        else:
            X = np.repeat(self.ref[None], len(xa), axis=0)
        X[:, self.ka], X[:, self.kb] = xa, xb
        return X


def rates(R, xa, xb, model, arch, ox, bound):
    r, _ = net_rate(R.state(xa, xb), model, arch, ox, bound)
    return r[:, R.ka], r[:, R.kb]


def fixed_points(R, U, V, GA, GB, lo, hi, model, arch, ox, bound):
    lb, ub = np.log([lo[0], lo[1]]) - 2.0, np.log([hi[0], hi[1]]) + 2.0

    def resid(u):
        u = np.clip(u, lb, ub)
        ra, rb = rates(R, np.exp(u[0]), np.exp(u[1]), model, arch, ox, bound)
        out = np.array([ra[0], rb[0]])
        return np.where(np.isfinite(out), out, 1e3)

    nrm = np.hypot(U, V)
    thr = np.nanpercentile(nrm, 5)
    sy, sx = np.where(nrm <= thr)
    found = []
    for i, j in list(zip(sy, sx))[::max(1, len(sy) // 60)]:
        s = root(resid, np.log([GA[i, j], GB[i, j]]), method="hybr")
        if not s.success or np.max(np.abs(resid(s.x))) > 1e-8:
            continue
        xf = np.exp(np.clip(s.x, lb, ub))
        if np.any(s.x <= lb + 1e-6) or np.any(s.x >= ub - 1e-6):
            continue
        if not any(np.allclose(xf, f, rtol=1e-3) for f in found):
            found.append(xf)
    out = []
    for xf in found:
        u, h = np.log(xf), 1e-4
        J = np.zeros((2, 2))
        for c in range(2):
            e = np.zeros(2)
            e[c] = h
            J[:, c] = (resid(u + e) - resid(u - e)) / (2 * h)
        ev = np.linalg.eigvals(J)
        cls = "stable" if np.all(ev.real < 0) else "unstable" if np.all(ev.real > 0) else "saddle"
        out.append((xf, cls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound", choices=("none", "osdr"), default="none")
    ap.add_argument("--min-cells", type=float, default=5)
    ap.add_argument("--bw", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx = context(load_config())
    arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state="counts", radius_um=ctx["radius"])
    model, ox, names = ctx["model"], ctx["ox"], arch.names
    bound = (fit_osdr_bound(model, arch, ox, cell_archetype=ctx["arche"], power=8)
             if a.bound == "osdr" else None)
    Xo = np.clip(arch.X_obs, 0, None)
    min_x = a.min_cells / AREA
    tag = "unbounded" if bound is None else "osdr_p8"

    fig, axes = plt.subplots(len(PAIRS), len(MODES), figsize=(5.2 * len(MODES), 4.9 * len(PAIRS)))
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
        for c, mode in enumerate(MODES):
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
            ax.set_title(f"{na} x {nb}: {mode}\n{note}\n{len(fps)} fixed points, {n_in} in support",
                         fontsize=9, loc="left")
            ax.set_xlabel(f"log2 {na} density")
            ax.set_ylabel(f"log2 {nb} density")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
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
