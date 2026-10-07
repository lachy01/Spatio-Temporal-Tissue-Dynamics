"""Test 03 -- single-component null (spec 6.3).

Simulate with `pi_i == 0` (all cells one Poisson rate) and fit the two-component
model. Pass: normalised entropy < 0.10 (cells are not split between components)
and a component rate ratio exp(e^delta) < 1.5 (the threshold Test 07 uses for
the same quantity; the real data's is 5.6).

This verifies the model *reports* absence of structure rather than manufacturing
it -- so a pass here is the model failing to find something, which is the point.

Criterion changed 2026-10-07, decided by the user. The spec's rule asked for
`e^delta < 0.05` or `mean(pi_hat)` at a boundary, plus the spec-3.3 degeneracy
flags. No lambda can deliver that: on null data every fit settles on two nearly
identical components (e^delta ~0.15, mean pi ~0.5, entropy ~0.01 at lambda =
300), because the rates are unpenalised and the likelihood is flat in e^delta
there. That is the model reporting no structure, so the rule now tests that
directly. The degeneracy flags are still reported, not required.

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
    flags_ok = bool(flags["any"])
    ok = bool(ratio_ok and ent_ok)

    lines = [
        f"- n = **{n:,}**, simulated with `pi_i == 0` (no latent class structure)",
        f"- component rate ratio exp(e^delta) = **{ratio:.4f}** (threshold 1.5 -> "
        f"{'pass' if ratio_ok else 'FAIL'}; real data 5.6)",
        f"- normalised entropy = **{ent:.4f}** (threshold 0.10 -> "
        f"{'pass' if ent_ok else 'FAIL'})",
        f"- `mean(pi_hat)` = {mean_w:.4f}",
        f"- spec 3.3 degeneracy flags fired (reported, not required): {flags_ok} "
        f"(fold_collapsed={flags['fold_collapsed']}, "
        f"weight_degenerate={flags['weight_degenerate']})",
        f"- log-likelihood {best.loglik:.4f}, {best.n_iter} iterations",
    ]
    write("Test 03 -- single-component null (spec 6.3)", lines, passed=ok)
    assert ratio_ok, f"component rate ratio {ratio:.3f} is not below 1.5"
    assert ent_ok, f"normalised entropy {ent:.4f} exceeds 0.10"
