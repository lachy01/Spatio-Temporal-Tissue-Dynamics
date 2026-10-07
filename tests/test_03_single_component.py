"""Test 03 -- single-component null (spec 6.3).

Simulate with `pi_i == 0` (all cells one Poisson rate) and fit the two-component
model. Pass if the model reports no structure, in either of two ways:
- it collapses to one component: e^delta < 0.05, or mean(pi_hat) within 0.01
  of a boundary (the spec's own branch); or
- its two components barely differ and cells are not split between them:
  rate ratio exp(e^delta) < 1.5 (the threshold Test 07 uses; the real data's
  is 5.6) and normalised entropy < 0.10.
Entropy alone cannot judge a collapse: with pi ~ 0 for every cell it is ~1,
the same reading as sharply separated classes.

This verifies the model *reports* absence of structure rather than manufacturing
it -- so a pass here is the model failing to find something, which is the point.

Criterion changed 2026-10-07, decided by the user. The spec's rule asked for
`e^delta < 0.05` or `mean(pi_hat)` at a boundary, plus the spec-3.3 degeneracy
flags. No lambda can deliver that: on null data every fit settles on two nearly
identical components (e^delta ~0.15, mean pi ~0.5, entropy ~0.01 at lambda =
300), because the rates are unpenalised and the likelihood is flat in e^delta
there. That is the model reporting no structure, so the rule now accepts it alongside
the collapse branch. The degeneracy flags are still reported, not required.

The formal result is the Spartan job `sim_single_component`, 20 replicates at
the real n (sim_test03_n); this local version uses one replicate.
"""
import numpy as np
from mixture_report import size, write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.fit import multistart
from tissue_dynamics.mixture.model import (degeneracy_flags, normalised_entropy,
                                           responsibilities)
from tissue_dynamics.mixture.simulate import simulate_stage_b


def test_03_single_component_null():
    cfg = load_config()
    n = size(int(cfg.get("sim_test03_n", 100_000)), 30_000)
    data, _truth, spec = simulate_stage_b(n=n, seed=0, single_component=True)
    best, _ = multistart(data, spec, cfg, n_restarts=size(10, 4), verbose=False,
                         maxiter=2000)
    D = data.as_jax()
    w = np.asarray(responsibilities(best.params, D, spec))
    ent = normalised_entropy(w)
    flags = degeneracy_flags(best.params, w, spec)

    e_delta = flags["e_delta"]
    mean_w = flags["mean_w"]
    ratio = float(np.exp(e_delta))
    ratio_ok = bool(ratio < 1.5)
    ent_ok = bool(ent < 0.10)
    collapse = bool(e_delta < 0.05 or min(mean_w, 1 - mean_w) < 1e-2)
    flags_ok = bool(flags["any"])
    ok = bool(collapse or (ratio_ok and ent_ok))

    lines = [
        f"- n = **{n:,}**, simulated with `pi_i == 0` (no latent class structure)",
        f"- component rate ratio exp(e^delta) = **{ratio:.4f}** (threshold 1.5 -> "
        f"{'pass' if ratio_ok else 'FAIL'}; real data 5.6)",
        f"- normalised entropy = **{ent:.4f}** (threshold 0.10 -> "
        f"{'pass' if ent_ok else 'FAIL'})",
        f"- `mean(pi_hat)` = {mean_w:.4f}; collapsed to one component: **{collapse}**",
        f"- spec 3.3 degeneracy flags fired (reported, not required): {flags_ok} "
        f"(fold_collapsed={flags['fold_collapsed']}, "
        f"weight_degenerate={flags['weight_degenerate']})",
        f"- log-likelihood {best.loglik:.4f}, {best.n_iter} iterations",
    ]
    write("Test 03 -- single-component null (spec 6.3)", lines, passed=ok)
    assert ok, (f"structure reported on null data: rate ratio {ratio:.3f}, entropy "
                f"{ent:.4f}, mean pi {mean_w:.4f}, e^delta {e_delta:.4f}")
