"""Test 03 -- single-component null (spec 6.3).

Simulate with `pi_i == 0` (all cells one Poisson rate) and fit the two-component
model. Pass: `e^delta < 0.05` **or** `mean(pi_hat)` within 1e-2 of a boundary;
normalised entropy < 0.10; and the spec-3.3 degeneracy flags fire.

This verifies the model *reports* absence of structure rather than manufacturing
it -- so a pass here is the model failing to find something, which is the point.
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
    n = size(100_000, 30_000)
    data, _truth, spec = simulate_stage_b(n=n, seed=0, single_component=True)
    best, _ = multistart(data, spec, cfg, n_restarts=size(10, 4), verbose=False,
                         maxiter=2000)
    D = data.as_jax()
    w = np.asarray(responsibilities(best.params, D, spec))
    ent = normalised_entropy(w)
    flags = degeneracy_flags(best.params, w, spec)

    e_delta = flags["e_delta"]
    mean_w = flags["mean_w"]
    collapse = bool(e_delta < 0.05 or min(mean_w, 1 - mean_w) < 1e-2)
    ent_ok = bool(ent < 0.10)
    flags_ok = bool(flags["any"])
    ok = bool(collapse and ent_ok and flags_ok)

    lines = [
        f"- n = **{n:,}**, simulated with `pi_i == 0` (no latent class structure)",
        f"- `e^delta` = **{e_delta:.5f}** (collapse threshold 0.05)",
        f"- `mean(pi_hat)` = **{mean_w:.6f}** (boundary threshold 1e-2)",
        f"- component collapse detected: **{collapse}**",
        f"- normalised entropy = **{ent:.4f}** (threshold 0.10 -> "
        f"{'pass' if ent_ok else 'FAIL'})",
        f"- spec 3.3 degeneracy flags fired: **{flags_ok}** "
        f"(fold_collapsed={flags['fold_collapsed']}, "
        f"weight_degenerate={flags['weight_degenerate']})",
        f"- log-likelihood {best.loglik:.4f}, {best.n_iter} iterations",
    ]
    write("Test 03 -- single-component null (spec 6.3)", lines, passed=ok)
    assert collapse, f"components did not collapse: e^delta={e_delta:.4f}, mean w={mean_w:.4f}"
    assert ent_ok, f"normalised entropy {ent:.4f} exceeds 0.10"
    assert flags_ok, "degeneracy flags did not fire on a single-component null"
