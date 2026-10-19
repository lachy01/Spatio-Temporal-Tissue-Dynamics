"""Count-model diagnostics shared by run/plot_model_fit.py and the notebooks.

A model's predictive distribution for each cell is a list of components
`(weight, family, mean, alpha)` with per-cell arrays: one `(1, "nb", mu, alpha)`
for an NB regression, two for a mixture (`family` "pois" or "nb"). From that:
expected count frequencies (rootogram), randomised PIT, the predictive mean,
the log-likelihood, and Moran's I of residuals averaged on a square grid.
`mixture_components` builds the list from a mixture fit.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

KMAX = 30


def nb_parts(mu, alpha):
    r = 1.0 / np.broadcast_to(alpha, mu.shape)
    return r, r / (r + mu)


def dist_summaries(y, comps, rng_u, chunk=100_000):
    """comps: list of (weight, family, mean, alpha) with per-cell arrays; the
    predictive pmf is sum_k weight_k f_k. Returns expected frequencies on
    0..KMAX-1 plus the pooled tail, randomised PIT, and the mean."""
    n = len(y)
    expected = np.zeros(KMAX)
    cdf_y, cdf_ym1 = np.zeros(n), np.zeros(n)
    ks = np.arange(KMAX)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        ys = y[sl]
        for w, fam, m, a in comps:
            ws, ms = w[sl], m[sl]
            if fam == "pois":
                pmf = stats.poisson.pmf(ks[None, :], ms[:, None])
                c_y, c_m = stats.poisson.cdf(ys, ms), stats.poisson.cdf(ys - 1, ms)
            else:
                r, p = nb_parts(ms, a[sl] if np.ndim(a) else a)
                pmf = stats.nbinom.pmf(ks[None, :], r[:, None], p[:, None])
                c_y, c_m = stats.nbinom.cdf(ys, r, p), stats.nbinom.cdf(ys - 1, r, p)
            expected += (ws[:, None] * pmf).sum(0)
            cdf_y[sl] += ws * c_y
            cdf_ym1[sl] += ws * np.where(ys > 0, c_m, 0.0)
    expected = np.append(expected, max(n - expected.sum(), 0.0))
    u = cdf_ym1 + rng_u * (cdf_y - cdf_ym1)
    mean = sum(w * m for w, _, m, _ in comps)
    return expected, np.clip(u, 1e-12, 1 - 1e-12), mean


def log_lik(y, comps):
    lp = None
    for w, fam, m, a in comps:
        if fam == "pois":
            l_ = stats.poisson.logpmf(y, m)
        else:
            r, p = nb_parts(m, a)
            l_ = stats.nbinom.logpmf(y, r, p)
        t = np.log(np.clip(w, 1e-300, None)) + l_
        lp = t if lp is None else np.logaddexp(lp, t)
    return float(lp.sum())


def grid_mean(v, xb, yb, shape, min_cells=3):
    flat = yb * shape[1] + xb
    cnt = np.bincount(flat, minlength=shape[0] * shape[1])
    s = np.bincount(flat, weights=v, minlength=shape[0] * shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(cnt >= min_cells, s / cnt, np.nan)
    return g.reshape(shape)


def morans_i(g):
    """Moran's I of a grid with NaN holes, rook neighbours, row-standardised off."""
    v = g - np.nanmean(g)
    ok = np.isfinite(v)
    num, wsum = 0.0, 0.0
    for dy, dx in ((0, 1), (1, 0)):
        a = v[: v.shape[0] - dy, : v.shape[1] - dx]
        b = v[dy:, dx:]
        m = np.isfinite(a) & np.isfinite(b)
        num += 2 * np.sum(a[m] * b[m])
        wsum += 2 * m.sum()
    return float(ok.sum() / wsum * num / np.nansum(v[ok] ** 2))




def mixture_components(params, D, spec, cycling_nb=False, noncycling_nb=False):
    """Components of a fitted mixture (production Poisson, Poisson + NB or NB + NB):
    returns (components, pi)."""
    import dataclasses

    import jax.numpy as jnp

    from .mixture.model import component_means, pi_hat

    P = {k: jnp.asarray(np.asarray(v)) for k, v in params.items()}
    s2 = dataclasses.replace(spec, cycling_nb=cycling_nb, noncycling_nb=noncycling_nb)
    pi = np.asarray(pi_hat(P, D, s2))
    m0, m1 = (np.asarray(v) for v in component_means(P, D, s2))
    a1 = float(np.exp(np.asarray(P["la1"]))) if cycling_nb else None
    a0 = float(np.exp(np.asarray(P["la0"]))) if noncycling_nb else None
    return [(1 - pi, "nb" if noncycling_nb else "pois", m0, a0),
            (pi, "nb" if cycling_nb else "pois", m1, a1)], pi


def summary_row(y, comps, k, u_draw, grid=None):
    """Log-likelihood, AIC/BIC on the nominal parameter count, rootogram misfit,
    zeros, PIT KS; Moran's I of mean quantile residuals when `grid` =
    (xb, yb, shape) is given. Returns (row, expected, pit, mean)."""
    n = len(y)
    exp_, u, mean = dist_summaries(y, comps, u_draw)
    obs = np.array([np.sum(y == j) for j in range(KMAX)] + [np.sum(y >= KMAX)], float)
    ll = log_lik(y, comps)
    row = {"loglik": ll, "parameters": k, "AIC": -2 * ll + 2 * k, "BIC": -2 * ll + k * np.log(n),
           "rootogram misfit": float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2)),
           "zeros observed": float(obs[0] / n), "zeros expected": float(exp_[0] / n),
           "PIT KS": float(stats.kstest(u, "uniform").statistic)}
    if grid is not None:
        row["residual Moran's I"] = morans_i(grid_mean(stats.norm.ppf(u), *grid))
    return row, exp_, u, mean


def cluster_mean_se(values, groups, blocks):
    """Mean of `values` per group, with a block-cluster SE: deviations from the
    group mean summed within each spatial block, blocks independent. Returns a
    DataFrame indexed by group with columns `mean`, `se`, and `se`."""
    import pandas as pd

    v = np.asarray(values, float)
    g = pd.Series(groups).astype("category")
    gi = g.cat.codes.to_numpy().astype(np.int64)
    _, b = np.unique(np.asarray(blocks), return_inverse=True)
    G, B = len(g.cat.categories), int(b.max()) + 1
    n_g = np.bincount(gi, minlength=G).astype(float)
    m = np.bincount(gi, weights=v, minlength=G) / n_g
    s = np.bincount(gi * B + b, weights=v - m[gi], minlength=G * B).reshape(G, B)
    se = np.sqrt((s ** 2).sum(1) * B / max(B - 1, 1)) / n_g
    return pd.DataFrame({"mean": m, "se": se}, index=g.cat.categories)
