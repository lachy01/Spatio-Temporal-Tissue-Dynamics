"""Test 07 -- adversarial size-confound (spec 6.7). THE CRITICAL TEST.

Data with **no latent class structure at all** (`pi_i == 0`) but a count that
depends on `log area` through a function the fitted spline cannot represent. If
the mixture invents classes here, it is separating cells on size and segmentation
quality rather than on biology, and -- in the spec's words -- the entire approach
is compromised.

Pass: normalised entropy < 0.25, `sigma_1(B_hat)` indistinguishable from zero,
and fold change `exp(e^delta) < 1.5`.

A failure is explicitly *not* a tuning problem: spec 6.7 requires the area spline
be made flexible enough and the test re-run before any real-data result is
reported.
"""
import numpy as np
from mixture_report import size, write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.fit import multistart
from tissue_dynamics.mixture.model import (B_matrix, normalised_entropy,
                                           responsibilities)
from tissue_dynamics.mixture.simulate import simulate_stage_b


def _run(n, fit_df, cfg, seed=0):
    data, _t, spec = simulate_stage_b(n=n, seed=seed, wiggly_area_df=10,
                                      fit_area_df=fit_df)
    best, _ = multistart(data, spec, cfg, n_restarts=size(10, 4), verbose=False,
                         maxiter=2000)
    D = data.as_jax()
    w = np.asarray(responsibilities(best.params, D, spec))
    return dict(entropy=normalised_entropy(w),
                sigma1=float(np.linalg.svd(B_matrix(best.params), compute_uv=False)[0]),
                fold=float(np.exp(np.exp(best.params["delta"]))),
                mean_w=float(w.mean()), loglik=best.loglik, df=fit_df)


def test_07_adversarial_size_confound():
    cfg = load_config()
    n = size(100_000, 30_000)
    r = _run(n, cfg["model"]["area_spline_df"], cfg)

    # reference: the same statistic when B really is zero and there is no
    # unmodelled area effect, so sigma_1 has a scale to be compared against
    from tissue_dynamics.mixture.simulate import simulate_null_bilinear
    dn, _tn, sn = simulate_null_bilinear(n=n, seed=99)
    bn, _ = multistart(dn, sn, cfg, n_restarts=size(5, 2), verbose=False, maxiter=1500)
    sigma_ref = float(np.linalg.svd(B_matrix(bn.params), compute_uv=False)[0])

    ent_ok = bool(r["entropy"] < 0.25)
    fold_ok = bool(r["fold"] < 1.5)
    sig_ok = bool(r["sigma1"] <= 2.0 * sigma_ref)
    ok = bool(ent_ok and fold_ok and sig_ok)

    lines = [
        f"- n = **{n:,}**; simulated with `pi_i == 0` and an area effect built "
        "from frequencies a df=%d spline cannot represent" % cfg["model"]["area_spline_df"],
        f"- normalised entropy = **{r['entropy']:.4f}** (threshold 0.25 -> "
        f"{'pass' if ent_ok else 'FAIL'})",
        f"- fold change `exp(e^delta)` = **{r['fold']:.4f}** (threshold 1.5 -> "
        f"{'pass' if fold_ok else 'FAIL'})",
        f"- `sigma_1(B_hat)` = **{r['sigma1']:.4f}** against a genuine-null "
        f"reference of {sigma_ref:.4f} (threshold 2x -> {'pass' if sig_ok else 'FAIL'})",
        f"- `mean(w)` = {r['mean_w']:.4f}; log-likelihood {r['loglik']:.2f}",
    ]
    if not ok:
        lines += ["", "**Spec 6.7:** a failure here means the mixture is separating "
                  "cells on size and segmentation quality rather than biology. It is "
                  "not a tuning problem -- the area spline must be made flexible "
                  "enough, the model refit, and this test re-run **before any "
                  "real-data result is reported**."]
    write("Test 07 -- adversarial size-confound (spec 6.7)", lines, passed=ok)
    assert ent_ok, f"normalised entropy {r['entropy']:.4f} exceeds 0.25 -- classes invented from size"
    assert fold_ok, f"fold change {r['fold']:.4f} exceeds 1.5 -- classes invented from size"
    assert sig_ok, f"sigma_1 {r['sigma1']:.4f} exceeds twice the null reference {sigma_ref:.4f}"
