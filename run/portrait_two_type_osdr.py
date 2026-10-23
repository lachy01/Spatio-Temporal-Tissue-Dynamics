"""OSDR-style two-type phase portraits, fitted directly on our data.

    python run/portrait_two_type_osdr.py

OSDR's main portraits (Somer et al., fm_analysis, github.com/JonathanSomer/osdr @ 7d04970)
are not slices of a many-type model: the model is fitted on the two types alone. For each
target type, a logistic regression of the division indicator on a degree-2 polynomial
of the two raw neighbour counts within 80 um (the cell counts itself), unpenalised
(statsmodels Logit); death = the type's mean division rate; the division rate is capped
at its largest in-sample prediction; dX_k/dt = X_k (p_div - p_death + corr_k), with the
maximal-density correction corr_k = max(c_k X_k^8, -0.5) (see portrait.core.DensityBound).
Every other cell type is left out of the model, so it is averaged over as it occurs.

Here, as there, except:
  - response: our posterior P(cycling | y) from the production mixture (a quasi-binomial
    fit to the soft label), not a binary division call;
  - no edge extrapolation of partially observed neighbourhoods;
  - OSDR also drops cells with any neighbour outside an allowed set; on Xenium that keeps
    0.2% of cells (lymphoid cells are almost everywhere), so the main fit uses all cells
    of the two types, and CAFs x Myeloid is repeated on the cells whose disc has at most
    5 cells outside OSDR's allowed set {CAFs, Myeloid, Cancer Epithelial, Endothelial}.

Writes outputs/portrait/two_type_osdr/{portraits.png, restricted_CAFs_Myeloid.png, fits.csv,
fixed_points.csv}.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import root
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import context                              # noqa: E402

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.summary import PAIRS                    # noqa: E402

OUT = "outputs/portrait/two_type_osdr"
RADIUS, POWER, FLOOR, NGRID = 80.0, 8, -0.5, 120
ALLOWED_FM = ["CAFs", "Myeloid", "Cancer Epithelial", "Endothelial"]
INK = "#1f1f1e"
C_A, C_B, C_OBS = "#2a78d6", "#e34948", "#1baf7a"
log2_1p = lambda x: np.log2(1 + x)                                    # noqa: E731
inv = lambda v: 2.0 ** v - 1                                          # noqa: E731


def poly(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.column_stack([np.ones_like(a), a, b, a * a, a * b, b * b])


class TwoType:
    """Fitted two-type OSDR model for the pair (A, B)."""

    def __init__(self, counts, lab, pi_hat, na, nb, keep, bound=True, rng_seed=0):
        self.names = (na, nb)
        self.fit, self.death, self.cap, self.c, self.xmax = {}, {}, {}, {}, {}
        ab = counts[:, [0, 1]]
        for t, nm in enumerate((na, nb)):
            sel = keep & (lab == nm)
            X = poly(ab[sel, 0], ab[sel, 1])
            res = sm.GLM(pi_hat[sel], X, family=sm.families.Binomial()).fit()
            self.fit[nm] = res.params
            self.death[nm] = float(pi_hat[sel].mean())
            self.cap[nm] = float(self._p(nm, ab[sel, 0], ab[sel, 1]).max())
            self.n = getattr(self, "n", {})
            self.n[nm] = int(sel.sum())
        both = keep & np.isin(lab, [na, nb])
        self.xmax = np.percentile(ab[both], 95, axis=0)
        self.c = np.zeros(2)
        if bound:                       # OSDR: all densities uniform on log2(1+x) up to x_max
            rng = np.random.default_rng(rng_seed)
            S = inv(rng.uniform(0, log2_1p(self.xmax)[None, :], size=(10_000, 2)))
            r = self.rates(S[:, 0], S[:, 1], corr=False)
            mf = np.maximum(0.0, r.max(0))
            self.c = -mf / self.xmax ** POWER

    def _p(self, nm, a, b):
        return 1 / (1 + np.exp(-np.clip(poly(a, b) @ self.fit[nm], -30, 30)))

    def rates(self, a, b, corr=True):
        a, b = np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)
        out = []
        for t, nm in enumerate(self.names):
            r = np.minimum(self._p(nm, a, b), self.cap[nm]) - self.death[nm]
            if corr:
                x = a if t == 0 else b
                r = r + np.maximum(self.c[t] * x ** POWER, FLOOR)
            out.append(r)
        return np.column_stack(out)

    def F(self, a, b):
        r = self.rates(a, b)
        return np.column_stack([np.atleast_1d(a) * r[:, 0], np.atleast_1d(b) * r[:, 1]])


def fixed_points(m, hi):
    """Roots of F in (a, b) >= 0, interior and on the axes, classified by the 2x2 Jacobian."""
    found = []
    for a0 in np.linspace(0, hi[0], 9):
        for b0 in np.linspace(0, hi[1], 9):
            for free in ((True, True), (True, False), (False, True)):
                def g(u, free=free):
                    a = u[0] if free[0] else 0.0
                    b = (u[-1] if free[1] else 0.0)
                    r = m.rates(a, b)[0]
                    return np.array([r[i] for i in (0, 1) if free[i]])
                s0 = np.array([v for v, f in zip((a0, b0), free) if f]) + 0.5
                s = root(g, s0, method="hybr")
                if not s.success or np.max(np.abs(g(s.x))) > 1e-9 or np.any(s.x < 0):
                    continue
                a = s.x[0] if free[0] else 0.0
                b = s.x[-1] if free[1] else 0.0
                if a > 3 * hi[0] or b > 3 * hi[1]:
                    continue
                if not any(abs(a - p[0]) < 1e-3 * (1 + p[0]) and abs(b - p[1]) < 1e-3 * (1 + p[1]) for p in found):
                    found.append((a, b))
    out = []
    for a, b in found:
        h = 1e-4
        J = np.zeros((2, 2))
        for c, (da, db) in enumerate(((h, 0), (0, h))):
            J[:, c] = (m.F(a + da, b + db)[0] - m.F(max(a - da, 0), max(b - db, 0))[0]) / \
                      ((a + da - max(a - da, 0)) + (b + db - max(b - db, 0)))
        ev = np.linalg.eigvals(J).real
        cls = "stable" if np.all(ev < 0) else "unstable" if np.all(ev > 0) else "saddle"
        out.append((a, b, cls))
    return out


def panel(ax, m, obs, title, xlab, ylab):
    hi = np.maximum(np.percentile(obs, 99, axis=0), m.xmax * 1.3)
    la = np.linspace(0, log2_1p(hi[0]), NGRID)
    lb = np.linspace(0, log2_1p(hi[1]), NGRID)
    LA, LB = np.meshgrid(la, lb)
    A, B = inv(LA.ravel()), inv(LB.ravel())
    F = m.F(A, B)
    U = (F[:, 0] / ((1 + A) * np.log(2))).reshape(LA.shape)       # d log2(1+x) / dt
    V = (F[:, 1] / ((1 + B) * np.log(2))).reshape(LA.shape)
    r = m.rates(A, B)
    sub = obs[np.random.default_rng(0).choice(len(obs), min(4000, len(obs)), replace=False)]
    ax.plot(log2_1p(sub[:, 0]), log2_1p(sub[:, 1]), ".", ms=1.4, color=C_OBS, alpha=.25, zorder=1)
    ax.streamplot(la, lb, U, V, density=1.0, color="0.35", linewidth=.7, arrowsize=.85, zorder=2)
    ax.contour(la, lb, r[:, 0].reshape(LA.shape), [0], colors=C_A, linewidths=2.1, zorder=3)
    ax.contour(la, lb, r[:, 1].reshape(LA.shape), [0], colors=C_B, linewidths=2.1, zorder=3)
    fps = fixed_points(m, hi)
    for a, b, cls in fps:
        if a <= hi[0] and b <= hi[1]:
            ax.plot(log2_1p(a), log2_1p(b), {"stable": "o", "saddle": "s", "unstable": "^"}[cls], ms=10,
                    mfc=INK if cls == "stable" else "white", mec=INK, mew=1.6, zorder=6, clip_on=False)
    ax.set_xlim(0, la[-1])
    ax.set_ylim(0, lb[-1])
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel(f"log2(1 + {xlab} cells within 80 µm)")
    ax.set_ylabel(f"log2(1 + {ylab} cells within 80 µm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fps


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = context(load_config())
    df, names = ctx["df"], list(ctx["basis"].phi_names)
    lab = ctx["arche"]
    pi_hat = np.asarray(ctx["w"], float)
    xy = df[["x", "y"]].to_numpy(float)
    C = np.zeros((len(df), len(names)))
    for k, nm in enumerate(names):                     # counts within 80 um, the cell itself included
        sel = lab == nm
        C[:, k] = cKDTree(xy[sel]).query_ball_point(xy, r=RADIUS, return_length=True)
    allowed = np.isin(names, ALLOWED_FM)
    restricted = C[:, ~allowed].sum(1) <= 5
    print(f"restricted set (<= 5 cells outside {ALLOWED_FM}): {restricted.mean():.4f} of cells")

    fits, fprows = [], []
    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(11, 4.9 * len(PAIRS)))
    for p, (na, nb) in enumerate(PAIRS):
        cnt = C[:, [names.index(na), names.index(nb)]]
        obs = cnt[np.isin(lab, [na, nb])]
        keep = np.ones(len(df), bool)
        for c, bd in enumerate((False, True)):
            m = TwoType(cnt, lab, pi_hat, na, nb, keep, bound=bd)
            tag = "OSDR bound (p = 8)" if bd else "unbounded"
            fps = panel(axes[p, c], m, obs, f"{na} x {nb}, {tag}\nfit on all {m.n[na]:,} {na} and "
                        f"{m.n[nb]:,} {nb} cells", na, nb)
            for a, b, cls in fps:
                fprows.append({"pair": f"{na} x {nb}", "cells": "all", "bound": tag, "X_a": a, "X_b": b,
                               "classification": cls})
            if not bd:
                for nm in (na, nb):
                    fits.append({"pair": f"{na} x {nb}", "target": nm, "cells": "all", "n": m.n[nm],
                                 "death (mean pi_hat)": m.death[nm], "cap": m.cap[nm],
                                 **dict(zip(["b0", "b_a", "b_b", "b_aa", "b_ab", "b_bb"], m.fit[nm]))})
    fig.suptitle("OSDR-style two-type portraits fitted on our data (posterior P(cycling) as the response); "
                 "blue = x-nullcline, red = y-nullcline;\nfilled circle stable, square saddle, triangle "
                 "unstable; green = observed neighbourhoods of the two types' cells", x=0.01, ha="left",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(f"{OUT}/portraits.png", dpi=130)
    plt.close(fig)

    # OSDR's restriction, as far as Xenium allows: CAFs x Myeloid on cells with <= 5 others outside the set
    na, nb = "CAFs", "Myeloid"
    cnt = C[:, [names.index(na), names.index(nb)]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for c, bd in enumerate((False, True)):
        m = TwoType(cnt, lab, pi_hat, na, nb, restricted, bound=bd)
        obs = cnt[restricted & np.isin(lab, [na, nb])]
        tag = "OSDR bound (p = 8)" if bd else "unbounded"
        fps = panel(axes[c], m, obs, f"{na} x {nb}, {tag}, restricted\n{m.n[na]:,} CAFs and {m.n[nb]:,} Myeloid cells with <= 5\n"
                    "neighbours outside {CAFs, Myeloid, Tumour, Endothelial}", na, nb)
        for a, b, cls in fps:
            fprows.append({"pair": f"{na} x {nb}", "cells": "restricted", "bound": tag, "X_a": a, "X_b": b,
                           "classification": cls})
        if not bd:
            for nm in (na, nb):
                fits.append({"pair": f"{na} x {nb}", "target": nm, "cells": "restricted", "n": m.n[nm],
                             "death (mean pi_hat)": m.death[nm], "cap": m.cap[nm],
                             **dict(zip(["b0", "b_a", "b_b", "b_aa", "b_ab", "b_bb"], m.fit[nm]))})
    fig.tight_layout()
    fig.savefig(f"{OUT}/restricted_CAFs_Myeloid.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(fits).to_csv(f"{OUT}/fits.csv", index=False)
    fp = pd.DataFrame(fprows)
    fp.to_csv(f"{OUT}/fixed_points.csv", index=False)
    with pd.option_context("display.width", 200):
        print(fp.round(2).to_string(index=False))
    json.dump({"restricted_share": float(restricted.mean())}, open(f"{OUT}/meta.json", "w"))


if __name__ == "__main__":
    main()
