"""Addendum 1, section 1: the right-truncated NB likelihood behind BASE-T-trunc.

The likelihood against scipy, its forward-mode gradient against finite
differences, its reduction to the plain NB likelihood when nothing is truncated,
and recovery of a known null on simulated data (V11 in miniature). The formal
V11 runs on all cells in run/division_addendum_a1.py.
"""
import numpy as np
from mixture_report import ROOT  # noqa: F401  (puts src/ on the path)
from scipy import stats

from tissue_dynamics.division import labels as L


def _data(n, seed, trim):
    rng = np.random.default_rng(seed)
    off = rng.normal(3.0, 1.0, n)
    m = rng.gamma(2.0, 0.05, n)
    X = np.ones((n, 1))
    th = np.array([-2.5, np.log(1.5), np.log(0.8)])
    mu = np.exp(th[0] + off) + np.exp(th[1]) * m
    y = L.sim_nb(mu, np.exp(th[2]), rng)
    c = L.smallest_c_below(mu, np.exp(th[2]), trim)
    return y, X, off, m, c, th, mu


def _nll(th, y, X, off, m, c, w, k_max):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    return float(L.nb_trunc_nll(jnp.asarray(th), *[jnp.asarray(a, float) for a in (y, X, off, m, c, w)], k_max))


def test_truncated_likelihood_matches_scipy():
    y, X, off, m, c, th, mu = _data(4000, 1, 0.05)
    w = y < c
    r = 1.0 / np.exp(th[2])
    p = r / (r + mu)
    ref = -np.sum((stats.nbinom.logpmf(y, r, p) - stats.nbinom.logcdf(c - 1, r, p))[w])
    got = _nll(th, y, X, off, m, c, w, int(c[w].max()))
    assert abs(got - ref) / abs(ref) < 1e-12


def test_no_truncation_is_the_plain_likelihood():
    y, X, off, m, _c, th, mu = _data(3000, 2, 0.01)
    r = 1.0 / np.exp(th[2])
    ref = -np.sum(stats.nbinom.logpmf(y, r, r / (r + mu)))
    got = _nll(th, y, X, off, m, np.full(len(y), 4000.0), np.ones(len(y)), 4000)
    assert abs(got - ref) / abs(ref) < 1e-12


def test_forward_gradient_matches_finite_differences():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    y, X, off, m, c, th, _mu = _data(3000, 0, 0.01)
    w = (y < c).astype(float)
    k_max = int(c[w > 0].max())
    args = [jnp.asarray(a, float) for a in (y, X, off, m, c, w)]
    g = np.asarray(jax.jacfwd(lambda t: L.nb_trunc_nll(t, *args, k_max))(jnp.asarray(th)))
    fd = np.array([(_nll(th + e, y, X, off, m, c, w, k_max) - _nll(th - e, y, X, off, m, c, w, k_max)) / 2e-6
                   for e in np.eye(3) * 1e-6])
    assert np.max(np.abs(g - fd) / np.abs(fd)) < 1e-5


def test_truncated_null_recovers_a_known_null_where_naive_trimming_does_not():
    """At alpha_trim 0.05 naive trimming throws away the null's own upper tail
    and shrinks alpha0; the truncated fit does not."""
    y, X, off, m, _c, th, _mu = _data(40000, 3, 0.05)
    start = L.fit_nb_additive_null(y, X, off, m)
    tr = L.truncated_null(y, X, off, m, 0.05, start, max_iter=10)
    nv = L.robust_null(y, X, off, trim_p=0.05, m=m)
    a_true = np.exp(th[2])
    assert abs(np.log(tr["alpha"]) - th[2]) < 4 * tr["se"]["log_alpha"]
    assert abs(tr["eta"] - th[1]) < 4 * tr["se"]["eta"]
    assert nv["alpha"] < 0.7 * a_true          # the bias the truncation removes
