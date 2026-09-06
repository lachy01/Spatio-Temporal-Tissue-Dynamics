"""Diagnostics that must accompany every reported fit (spec 8.7, 8.8, 10).

Spec 10 is explicit that a result reported without the entropy and the confound
R-squared "is not usable", so these are computed alongside the estimates rather
than left to the reader to request.
"""

from __future__ import annotations

import numpy as np


def _r2(X, y):
    """R^2 of an OLS fit of `y` on `[1, X]`."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    resid = np.asarray(y, dtype=float) - X @ beta
    return float(1.0 - np.var(resid) / max(np.var(y), 1e-300))


def confound_diagnostic(w, df, basis):
    """Spec 8.8: is `w` a cell-state readout, or a size-and-detection classifier?

    Regresses the responsibilities on the nuisance covariates and, separately,
    on the cell state, and returns both R-squared values. If the nuisance R^2
    substantially exceeds the state R^2, the model is classifying cells on how
    well they were captured rather than on what they are -- and the spec
    requires that number be reported prominently whatever it shows.
    """
    w = np.asarray(w, dtype=float)
    nuis = np.column_stack([
        np.log(df["area"].to_numpy(float)), np.log(df["N"].to_numpy(float)),
        np.log(df["density"].to_numpy(float)), df["oxygen"].to_numpy(float)])
    r_nuis, r_state = _r2(nuis, w), _r2(basis.own(df), w)
    per = {nm: float(np.corrcoef(v, w)[0, 1]) for nm, v in
           (("log_area", nuis[:, 0]), ("log_N", nuis[:, 1]),
            ("log_density", nuis[:, 2]), ("oxygen", nuis[:, 3]))}
    return {
        "confound_r2_nuisance": r_nuis,
        "confound_r2_state": r_state,
        "confound_ratio_state_over_nuisance": float(r_state / max(r_nuis, 1e-12)),
        "is_size_detection_classifier": bool(r_nuis > r_state),
        "confound_corr": per,
    }


def mixture_rootogram(y, p, d, spec, max_count=20):
    """Hanging rootogram against the MIXTURE marginal PMF (spec 6.9).

    Expected counts are `sum_i P(Y_i = k)` under
    `(1-pi_i) Pois(rho_i + lambda_0i) + pi_i Pois(rho_i + lambda_1i)`, not under
    a single fitted distribution -- the whole point is to score the two-component
    predictive distribution the model actually implies.
    """
    from .model import marginal_pmf

    y = np.asarray(y, dtype=float)
    pmf = np.asarray(marginal_pmf(p, d, spec, int(max_count)))     # (k+1, n)
    ks = np.arange(int(max_count) + 1)
    expected = pmf.sum(axis=1)
    observed = np.array([(y == k).sum() for k in ks], dtype=float)
    disc = np.sqrt(observed) - np.sqrt(expected)
    return ks, observed, expected, disc


def randomised_pit(y, p, d, spec, n_bins=10, seed=0, max_count=400):
    """Randomised PIT (spec 6.9 requires the randomised form).

    For a discrete outcome the non-randomised PIT is conservative -- it will look
    uniform even for a poor fit -- so the spec asks for
    `u_i = F(y_i - 1) + v_i * P(Y_i = y_i)`, `v_i ~ U(0,1)`, which is exactly
    uniform when the predictive distribution is correct.
    """
    from .model import marginal_pmf

    y = np.asarray(y, dtype=float).astype(int)
    kmax = int(min(max_count, max(y.max() + 1, n_bins)))
    pmf = np.asarray(marginal_pmf(p, d, spec, kmax))               # (kmax+1, n)
    cdf = np.cumsum(pmf, axis=0)
    idx = np.clip(y, 0, kmax)
    n = len(y)
    at = pmf[idx, np.arange(n)]
    below = np.where(idx > 0, cdf[np.clip(idx - 1, 0, kmax), np.arange(n)], 0.0)
    v = np.random.default_rng(seed).uniform(size=n)
    u = np.clip(below + v * at, 0.0, 1.0)
    counts, edges = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    return u, counts / (n / n_bins), edges


def single_component_reference(y, d, spec):
    """The one-component Poisson fit with the same rate design (spec 6.9).

    The rootogram comparison is against this, so the question is specifically
    what the *second component* buys, not what the covariates buy.
    """
    from ..count_glm import _poisson_irls

    cols = [np.ones(len(y))]
    if spec.theta_free:
        cols.append(np.asarray(d.logN, dtype=float))
    cols.append(np.asarray(d.area, dtype=float))
    X = np.column_stack(cols)
    off = None if spec.theta_free else np.asarray(d.logN, dtype=float)
    _beta, mu = _poisson_irls(np.asarray(y, dtype=float), X, off)
    return np.asarray(mu, dtype=float)


def poisson_rootogram(y, mu, max_count=20):
    """Rootogram for a single-component Poisson, for the same comparison."""
    from scipy.stats import poisson

    y = np.asarray(y, dtype=float)
    ks = np.arange(int(max_count) + 1)
    expected = np.array([poisson.pmf(k, mu).sum() for k in ks], dtype=float)
    observed = np.array([(y == k).sum() for k in ks], dtype=float)
    return ks, observed, expected, np.sqrt(observed) - np.sqrt(expected)


def pearson_residuals(y, p, d, spec):
    """Pearson residuals about the MIXTURE marginal mean (spec 6.14).

    `E[y_i] = rho_i + (1-pi_i) lambda_0i + pi_i lambda_1i`, and the variance is
    the mixture's, which exceeds the within-component Poisson variance by the
    between-component term `pi(1-pi)(lambda_1 - lambda_0)^2`. Dividing by the
    Poisson variance alone would manufacture apparent overdispersion.
    """
    from .model import marginal_mean, pi_hat, rates

    y = np.asarray(y, dtype=float)
    lam0, lam1, rho = (np.asarray(v) for v in rates(p, d, spec))
    pi = np.asarray(pi_hat(p, d, spec))
    mu = np.asarray(marginal_mean(p, d, spec))
    var = mu + pi * (1.0 - pi) * (lam1 - lam0) ** 2
    return (y - mu) / np.sqrt(np.clip(var, 1e-12, None))


def correlogram(values, xy, max_dist=500.0, n_bands=10, subsample=60000, seed=0):
    """Moran's I in distance bands out to `max_dist` (spec 6.14).

    The bandwidth for Conley standard errors is read off this, and it shows
    whether residual structure is local or extends across the section.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(values, dtype=float)
    xy = np.asarray(xy, dtype=float)
    rng = np.random.default_rng(seed)
    if len(v) > subsample:
        keep = rng.choice(len(v), int(subsample), replace=False)
        v, xy = v[keep], xy[keep]
    v = v - v.mean()
    denom = np.sum(v ** 2)
    tree = cKDTree(xy)
    edges = np.linspace(0.0, float(max_dist), int(n_bands) + 1)
    out = []
    prev = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        pairs = tree.query_pairs(hi, output_type="ndarray")
        cur = set(map(tuple, pairs))
        band = np.array(sorted(cur - prev), dtype=int) if prev else pairs
        prev = cur
        if len(band) == 0:
            out.append((0.5 * (lo + hi), np.nan, 0))
            continue
        num = np.sum(v[band[:, 0]] * v[band[:, 1]]) * 2.0
        out.append((0.5 * (lo + hi), float(len(v) * num / (2 * len(band) * denom)),
                    int(len(band))))
    return out
