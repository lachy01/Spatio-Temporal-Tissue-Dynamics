"""Survivor functions for labels and push-through (spec 2.2, 3; checked by V3).

Everything downstream needs `P(Y >= c)` for integer `c`, vectorised over cells.
These are written in closed form through regularised incomplete beta and gamma
functions rather than called through `scipy.stats`, so the parameterisation is
explicit and V3 can check it against scipy independently.

NB2 parameterisation, used throughout the project: mean `mu`, variance
`mu + alpha * mu**2`. In scipy's `nbinom(n, p)` that is `n = 1/alpha`,
`p = n / (n + mu)`.
"""

from __future__ import annotations

import numpy as np
from scipy import special


def nb_sf(k, mu, alpha):
    """`P(Y > k)` for `Y ~ NB2(mu, alpha)`; `k < 0` gives 1.

    With `r = 1/alpha` and `p = r/(r + mu)`, `P(Y <= k) = I_p(r, k + 1)`, so
    `P(Y > k) = I_{1-p}(k + 1, r)`.
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    r = 1.0 / np.asarray(alpha, dtype=float)
    q = mu / (r + mu)                                   # 1 - p
    kk = np.maximum(k, 0.0)
    out = special.betainc(kk + 1.0, r, q)
    return np.where(k < 0, 1.0, out)


def pois_sf(k, mu):
    """`P(Y > k)` for `Y ~ Poisson(mu)`; `k < 0` gives 1.

    `P(Y <= k) = Q(k + 1, mu)` (regularised upper gamma), so
    `P(Y > k) = P(k + 1, mu)`, the regularised lower gamma.
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    kk = np.maximum(k, 0.0)
    out = special.gammainc(kk + 1.0, mu)
    return np.where(k < 0, 1.0, out)


def nb_p_ge(c, mu, alpha):
    """`P(Y >= c) = P(Y > c - 1)` -- the push-through for an NB arm (spec 3)."""
    return nb_sf(np.asarray(c, dtype=float) - 1.0, mu, alpha)


def pois_p_ge(c, mu):
    return pois_sf(np.asarray(c, dtype=float) - 1.0, mu)


def mix_p_ge(c, pi, mu0, mu1):
    """Mixture push-through: `pi F1(c) + (1 - pi) F0(c)`, with the prior `pi`.

    Spec 3: never the posterior `w`, which conditions on the very count the
    label is built from.
    """
    pi = np.asarray(pi, dtype=float)
    return pi * pois_p_ge(c, mu1) + (1.0 - pi) * pois_p_ge(c, mu0)


def nb_scipy_params(mu, alpha):
    """`(n, p)` for `scipy.stats.nbinom` matching NB2(mu, alpha)."""
    n = 1.0 / np.asarray(alpha, dtype=float)
    return n, n / (n + np.asarray(mu, dtype=float))


def smallest_c_below(mu, alpha, level, family="nb", c_max=100000):
    """Per cell, the smallest integer `c >= 1` with `P(Y >= c) < level`.

    Starts from scipy's inverse survival function and then corrects in both
    directions with the closed-form survivor, so ties and floating edge cases
    resolve the same way the labels are computed.
    """
    from scipy import stats

    mu = np.asarray(mu, dtype=float)
    if family == "nb":
        n, p = nb_scipy_params(mu, alpha)
        k = stats.nbinom.isf(level, n, p)
        sf = lambda c: nb_p_ge(c, mu, alpha)              # noqa: E731
    else:
        k = stats.poisson.isf(level, mu)
        sf = lambda c: pois_p_ge(c, mu)                   # noqa: E731
    c = np.clip(np.nan_to_num(k, nan=0.0) + 1.0, 1.0, c_max)
    for _ in range(64):                                   # too high: step down
        down = (c > 1) & (sf(c - 1) < level)
        if not down.any():
            break
        c = np.where(down, c - 1, c)
    for _ in range(64):                                   # too low: step up
        up = sf(c) >= level
        if not up.any():
            break
        c = np.where(up, c + 1, c)
    return c.astype(np.int64)
