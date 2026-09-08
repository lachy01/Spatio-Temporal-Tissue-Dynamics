"""Test 04 -- gradient check (spec 6.4).

Analytic JAX gradient against central finite differences at three random
parameter vectors. Pass: max relative error < 1e-5 on every component.

The comparison is made on a modestly sized problem whose log-likelihood is
O(1e3-1e4). That is not a convenience: a central difference of a function of
magnitude F has an unavoidable cancellation floor of about `eps_machine * F /
step`, so evaluating at a badly scaled parameter vector (one where the rates are
far from the data, making |F| enormous while the gradients stay small) makes the
finite difference itself inaccurate to ~1e-3 and the test measures nothing.
"""
import jax
import numpy as np
from mixture_report import write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.model import ModelSpec, flatten, init_params, loglik
from tissue_dynamics.mixture.simulate import simulate_stage_b


def test_04_gradient():
    cfg = load_config()
    data, _truth, sb = simulate_stage_b(n=4000, seed=0, d_own=6, d_nbr=8)
    data.gbasis = data.hbasis.copy()                 # Stage C background basis
    # column counts come from the bases, which carry df-1 columns each
    spec = ModelSpec(stage="C", rank=2, theta_free=True, d_own=6, d_nbr=8,
                     df_area=data.area.shape[1], df_weight=data.hbasis.shape[1],
                     df_bg=data.gbasis.shape[1], ridge_tau=sb.ridge_tau)
    D = data.as_jax()

    worst, rows = 0.0, []
    for trial in range(3):
        rng = np.random.default_rng(50 + trial)
        p = init_params(spec, cfg, data=data, rng=rng)
        flat, unflat = flatten(p)
        g = np.asarray(flatten(jax.grad(loglik)(p, D, spec))[0])

        def f(v):
            return float(loglik(unflat(v), D, spec))

        errs = []
        for j in range(len(flat)):
            e = np.inf
            for eps in (1e-5, 1e-6, 1e-4):
                v1, v2 = np.array(flat), np.array(flat)
                v1[j] += eps
                v2[j] -= eps
                fd = (f(v1) - f(v2)) / (2 * eps)
                e = min(e, abs(fd - g[j]) / max(abs(fd), abs(g[j]), 1e-8))
                if e < 1e-8:
                    break
            errs.append(e)
        worst = max(worst, max(errs))
        rows.append((trial, f(flat), len(flat), max(errs)))

    lines = ["| trial | loglik | n params | max relative error |", "|---|---:|---:|---:|"]
    lines += [f"| {t} | {ll:.2f} | {k} | {e:.2e} |" for t, ll, k, e in rows]
    lines += ["", f"- threshold (spec 6.4): **1e-5**; worst observed **{worst:.2e}**"]
    write("Test 04 -- gradient check (spec 6.4)", lines, passed=bool(worst < 1e-5))
    assert worst < 1e-5, f"max relative gradient error {worst:.2e} exceeds 1e-5"


def test_04_gradient_with_density_interaction():
    """The per-cell-type density slope must not break the analytic gradient."""
    cfg = load_config()
    data, _t, sb = simulate_stage_b(n=4000, seed=0, d_own=6, d_nbr=8)
    data.logdens = np.random.default_rng(0).normal(0, 0.4, len(data.y))
    spec = ModelSpec(stage="B", rank=2, theta_free=True, d_own=6, d_nbr=8,
                     df_area=data.area.shape[1], df_weight=data.hbasis.shape[1],
                     ridge_tau=sb.ridge_tau, density_interaction=True)
    D = data.as_jax()
    worst = 0.0
    for trial in range(2):
        p = init_params(spec, cfg, data=data, rng=np.random.default_rng(70 + trial))
        assert "k" in p and np.shape(p["k"]) == (6,)      # the new block exists
        flat, unflat = flatten(p)
        g = np.asarray(flatten(jax.grad(loglik)(p, D, spec))[0])
        f = lambda v: float(loglik(unflat(v), D, spec))
        for j in range(len(flat)):
            e = np.inf
            for eps in (1e-5, 1e-6, 1e-4):
                v1, v2 = np.array(flat), np.array(flat)
                v1[j] += eps; v2[j] -= eps
                e = min(e, abs((f(v1) - f(v2)) / (2 * eps) - g[j])
                        / max(abs(g[j]), 1e-8))
                if e < 1e-8:
                    break
            worst = max(worst, e)
    assert worst < 1e-5, f"max relative gradient error {worst:.2e} with the interaction"
