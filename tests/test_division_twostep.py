"""Addendum 1, sections 3 and 4: the two-step arm and BASE-O-state's scale.

Step 1's constant-pi Poisson mixture against MIX's own likelihood at stage A,
its posterior against Bayes' rule, the NB cycling component against scipy,
and V14 in miniature (one-hot types reduce s+(phi) to OSDR's per-type SD).
The formal checks run on all cells in run/division_addendum_a2_twostep.py and
run/division_addendum_a4_state.py.
"""
import numpy as np
from mixture_report import ROOT  # noqa: F401  (puts src/ on the path)
from scipy import stats

from tissue_dynamics.division import labels as L
from tissue_dynamics.division import twostep as T


def _design(n, seed):
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    logN = rng.normal(3.5, 0.8, n)
    m = rng.gamma(2.0, 0.05, n)
    p = {"b0": jnp.asarray(-3.2), "delta": jnp.asarray(np.log(1.9)), "eta_m": jnp.asarray(0.4),
         "a0": jnp.asarray(-2.0), "la1": jnp.asarray(np.log(0.4))}
    lam0 = np.exp(-3.2 + logN)
    lam1 = lam0 * np.exp(1.9)
    z = rng.uniform(size=n) < 1 / (1 + np.exp(2.0))
    y = rng.poisson(np.where(z, lam1, lam0) + np.exp(0.4) * m).astype(float)
    D = {"y": jnp.asarray(y), "logN": jnp.asarray(logN), "m": jnp.asarray(m),
         "area": jnp.zeros((n, 0))}
    return D, p, lam0, lam1, m


def test_step1_matches_mix_stage_a():
    import jax.numpy as jnp
    from tissue_dynamics.mixture.model import ModelSpec, loglik_terms
    D, p, *_ = _design(3000, 0)
    spec = ModelSpec(stage="A", theta_free=False, df_area=0, df_weight=0, mistic=True,
                     technical="mistic", oxygen=False)
    mine = np.asarray(T.const_pi_terms(p, D, "pp"))
    ref = np.asarray(loglik_terms({k: p[k] for k in ("b0", "delta", "eta_m", "a0")}, D, spec))
    assert np.max(np.abs(mine - ref)) < 1e-10
    assert float(jnp.abs(jnp.sum(T.const_pi_terms(p, D, "pp")) - ref.sum())) < 1e-8


def test_posterior_and_nb_component():
    D, p, lam0, lam1, m = _design(2000, 1)
    y = np.asarray(D["y"])
    mu0, mu1 = lam0 + np.exp(0.4) * m, lam1 + np.exp(0.4) * m
    pi = 1 / (1 + np.exp(2.0))
    a1 = 0.4
    r = 1 / a1
    f1 = stats.nbinom.pmf(y, r, r / (r + mu1))
    f0 = stats.poisson.pmf(y, mu0)
    w_ref = pi * f1 / (pi * f1 + (1 - pi) * f0)
    assert np.max(np.abs(np.asarray(T.posterior(p, D, "pnb")) - w_ref)) < 1e-10
    ll_ref = np.log(pi * f1 + (1 - pi) * f0)
    assert np.max(np.abs(np.asarray(T.const_pi_terms(p, D, "pnb")) - ll_ref)) < 1e-9


def test_one_hot_state_scale_is_per_type_sd():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 9, 40000)
    r = rng.normal(0.3 * g - 1.0, 1.0 + 0.15 * g)
    X = (g[:, None] == np.arange(9)).astype(float)
    s, info = L.base_o_state_scale(r, X)
    assert np.max(np.abs(s - L.base_o_scale(r, g))) < 1e-10
    assert info["n_pos"] == int((r > 0).sum())
