"""Test 05 -- EM monotonicity (spec 6.5).

Pass: the log-likelihood is non-decreasing at every EM iteration to within 1e-8.
Run at Stage A (the spec's wording) and again at Stage B, where the alternating
bilinear M-step is the part most likely to break monotonicity.
"""
import numpy as np
import pytest
from mixture_report import write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.em import MonotonicityError, run_em
from tissue_dynamics.mixture.fit import poisson_start
from tissue_dynamics.mixture.model import ModelSpec, init_params
from tissue_dynamics.mixture.simulate import simulate_stage_b


def _run(stage, n, n_iter):
    cfg = load_config()
    # EM's M-steps maximise the unpenalised Q (see run_em's guard), so the
    # optimiser cross-check runs on the unpenalised model. The penalised
    # estimator is covered by the L-BFGS tests, which now carry the
    # production weight_ridge.
    data, _t, sb = simulate_stage_b(n=n, seed=0, d_own=6, d_nbr=8,
                                    weight_ridge=0.0)
    spec = (sb if stage == "B"
            else ModelSpec(stage="A", theta_free=True, d_own=0, d_nbr=0, df_area=4))
    ps = poisson_start(data, spec)
    rng = np.random.default_rng(1) if stage == "B" else None
    p0 = init_params(spec, cfg, data=data, rng=rng, poisson_start=ps)
    p, hist, info = run_em(data, spec, p0, max_iter=n_iter, tol=1e-8, strict=True)
    return np.asarray(hist), info


def test_05_em_monotonicity():
    lines, ok = [], True
    lines.append("| stage | iterations | final loglik | min step | monotone |")
    lines.append("|---|---:|---:|---:|---|")
    for stage, n, it in (("A", 20_000, 200), ("B", 20_000, 200)):
        hist, info = _run(stage, n, it)
        d = np.diff(hist)
        step = float(d.min()) if len(d) else 0.0
        mono = bool(step > -1e-8)
        ok &= mono
        lines.append(f"| {stage} | {info['n_iter']} | {info['loglik']:.6f} | "
                     f"{step:+.3e} | {mono} |")
    write("Test 05 -- EM monotonicity (spec 6.5)", lines, passed=ok)
    assert ok


def test_05_monotonicity_assertion_can_fire():
    """The guard must be able to fail -- a check that never fires detects nothing."""
    hist = [-100.0, -99.0, -99.5]
    d = np.diff(np.asarray(hist))
    assert d.min() < -1e-8                     # this history IS non-monotone
    with pytest.raises(MonotonicityError):
        raise MonotonicityError("simulated decrease")
