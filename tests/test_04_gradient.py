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


def test_04_gradient_with_density_spline_interaction():
    """The per-type density CURVE must not break the analytic gradient either."""
    cfg = load_config()
    data, _t, sb = simulate_stage_b(n=4000, seed=0, d_own=6, d_nbr=8)
    rng = np.random.default_rng(0)
    data.logdens = rng.normal(0, 0.4, len(data.y))
    spec = ModelSpec(stage="B", rank=2, theta_free=True, d_own=6, d_nbr=8,
                     df_area=data.area.shape[1], df_weight=data.hbasis.shape[1],
                     ridge_tau=sb.ridge_tau, density_interaction=True,
                     density_interaction_spline=True)
    D = data.as_jax()
    worst = 0.0
    for trial in range(2):
        p = init_params(spec, cfg, data=data, rng=np.random.default_rng(90 + trial))
        assert np.shape(p["K"]) == (6, data.hbasis.shape[1])
        flat, unflat = flatten(p)
        g = np.asarray(flatten(jax.grad(loglik)(p, D, spec))[0])
        f = lambda v: float(loglik(unflat(v), D, spec))
        for j in range(len(flat)):
            e = np.inf
            for eps in (1e-5, 1e-6, 1e-4):
                v1, v2 = np.array(flat), np.array(flat)
                v1[j] += eps; v2[j] -= eps
                e = min(e, abs((f(v1) - f(v2)) / (2 * eps) - g[j]) / max(abs(g[j]), 1e-8))
                if e < 1e-8:
                    break
            worst = max(worst, e)
    assert worst < 1e-5, f"max relative gradient error {worst:.2e} with the spline interaction"


def test_04_spline_interaction_can_bend_where_a_tilt_cannot():
    """The point of the upgrade: a per-type response that turns DOWN at high
    density, which the linear tilt provably cannot represent."""
    import dataclasses
    from tissue_dynamics.mixture.basis import _Spline
    x = np.random.default_rng(0).normal(0.0, 0.6, 20000)
    S = _Spline(x, 3, knot_quantiles=[25, 50, 75, 90])
    H = S(x)
    # a curve that rises then falls -- contact inhibition's shape
    target = np.where(x < np.percentile(x, 75), 0.4 * x, 0.4 * np.percentile(x, 75)
                      - 0.9 * (x - np.percentile(x, 75)))
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), H]), target, rcond=None)
    fit_spline = np.column_stack([np.ones(len(x)), H]) @ coef
    coef_l, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), target, rcond=None)
    fit_lin = np.column_stack([np.ones(len(x)), x]) @ coef_l
    r2 = lambda f: 1 - np.var(target - f) / np.var(target)
    assert r2(fit_spline) > 0.95, f"spline basis cannot represent the shape: R2 {r2(fit_spline):.3f}"
    assert r2(fit_lin) < 0.6, f"a linear tilt should NOT capture it: R2 {r2(fit_lin):.3f}"


def test_04_dropping_h_also_drops_the_density_interaction():
    """Test 13's h-dropped arm must remove density from the weight ENTIRELY.

    The per-type interaction is built on the h basis, so keeping it while
    dropping h() would leave nine per-type density curves in the model and
    measure nothing -- besides failing on shape.
    """
    import dataclasses
    cfg = load_config()
    data, _t, sb = simulate_stage_b(n=3000, seed=0, d_own=6, d_nbr=8)
    data.logdens = np.random.default_rng(0).normal(0, 0.4, len(data.y))
    full = ModelSpec(stage="B", rank=2, theta_free=True, d_own=6, d_nbr=8,
                     df_area=data.area.shape[1], df_weight=data.hbasis.shape[1],
                     ridge_tau=sb.ridge_tau, density_interaction=True,
                     density_interaction_spline=True)
    dropped = dataclasses.replace(full, df_weight=0)
    p_full = init_params(full, cfg, data=data, rng=np.random.default_rng(1))
    p_drop = init_params(dropped, cfg, data=data, rng=np.random.default_rng(1))
    assert "K" in p_full and "h" in p_full
    assert "K" not in p_drop and "h" not in p_drop, "density survived the drop"
    # and it must actually evaluate
    assert np.isfinite(float(loglik(p_drop, data.as_jax(), dropped)))
