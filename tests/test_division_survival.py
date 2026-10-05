"""V3 (division classifier spec, section 8): survivor functions against scipy.

Max absolute difference < 1e-12 on 1,000-point grids, for NB2 with the project's
parameterisation and for Poisson. Also checks that the scipy conversion really is
NB2 -- mean mu, variance mu + alpha mu^2 -- since a wrong conversion would agree
with itself.
"""
import numpy as np
from mixture_report import ROOT  # noqa: F401  (puts src/ on the path)
from scipy import stats

from tissue_dynamics.division.labels import base_o
from tissue_dynamics.division.survival import (mix_p_ge, nb_p_ge, nb_scipy_params,
                                               nb_sf, pois_sf, smallest_c_below)


def _grid(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(-1, 60, n).astype(float)
    mu = np.exp(rng.uniform(np.log(1e-3), np.log(80.0), n))
    alpha = np.exp(rng.uniform(np.log(1e-3), np.log(20.0), n))
    return k, mu, alpha


def max_abs_diff_nb():
    k, mu, alpha = _grid()
    n, p = nb_scipy_params(mu, alpha)
    ref = np.where(k < 0, 1.0, stats.nbinom.sf(k, n, p))
    return float(np.max(np.abs(nb_sf(k, mu, alpha) - ref)))


def max_abs_diff_pois():
    k, mu, _ = _grid(seed=1)
    ref = np.where(k < 0, 1.0, stats.poisson.sf(k, mu))
    return float(np.max(np.abs(pois_sf(k, mu) - ref)))


def test_v3_nb_survivor_matches_scipy():
    assert max_abs_diff_nb() < 1e-12


def test_v3_poisson_survivor_matches_scipy():
    assert max_abs_diff_pois() < 1e-12


def test_v3_scipy_conversion_is_nb2():
    _, mu, alpha = _grid(seed=2)
    n, p = nb_scipy_params(mu, alpha)
    m, v = stats.nbinom.stats(n, p, moments="mv")
    assert np.allclose(m, mu, rtol=1e-12)
    assert np.allclose(v, mu + alpha * mu ** 2, rtol=1e-10)


def test_threshold_is_smallest_count_below_level():
    _, mu, alpha = _grid(seed=3)
    for level in (0.05, 0.001, 1e-4):
        c = smallest_c_below(mu, alpha, level)
        assert np.all(nb_p_ge(c, mu, alpha) < level)
        assert np.all((c == 1) | (nb_p_ge(c - 1, mu, alpha) >= level))


def test_mixture_push_through_uses_prior_weights():
    c = np.array([0, 1, 3, 10])
    p = mix_p_ge(c, pi=np.full(4, 0.25), mu0=np.full(4, 0.5), mu1=np.full(4, 5.0))
    ref = 0.25 * stats.poisson.sf(c - 1, 5.0) + 0.75 * stats.poisson.sf(c - 1, 0.5)
    assert np.max(np.abs(p - ref)) < 1e-12


def test_v10_base_o_discrete_equivalence_synthetic():
    rng = np.random.default_rng(4)
    b = np.exp(rng.uniform(-3, 2, 5000))
    sd = np.sqrt(b + 0.7 * b ** 2)
    y = rng.poisson(b * 2).astype(float)
    for td in (0.0, 0.25, 0.5, 0.75, 1.0):
        L, c, a = base_o(y, b, sd, 1.3, td)
        assert np.array_equal(L, a > td)
        assert np.array_equal(L, y >= c)
