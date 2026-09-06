"""Test 08 -- held-out gene validation (spec 6.8).

Unit-level checks on the machinery. The real-data run is the `heldout_genes`
SLURM job; these verify the pieces behave before 5 full-n fits depend on them.
"""
import numpy as np
from mixture_report import write

from tissue_dynamics.gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES
from tissue_dynamics.mixture.heldout import (block_permute, heldout_gene_test,
                                             poisson_deviance, split_gene_set)


def test_08_split_is_phase_stratified_and_disjoint():
    present = set(TIROSH_S_GENES) | set(TIROSH_G2M_GENES)
    A, B = split_gene_set(TIROSH_S_GENES, TIROSH_G2M_GENES, present, seed=0)
    assert not (set(A) & set(B))                       # disjoint
    assert set(A) | set(B) == present                  # exhaustive
    for half in (A, B):                                # both phases in both halves
        assert set(half) & set(TIROSH_S_GENES)
        assert set(half) & set(TIROSH_G2M_GENES)


def test_08_block_permute_preserves_marginal_and_moves_cells():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(20), 50)
    w = rng.uniform(size=1000)
    wp = block_permute(w, labels, rng)
    np.testing.assert_allclose(np.sort(wp), np.sort(w))   # same values
    assert (wp != w).mean() > 0.5                          # most cells reassigned


def test_08_detects_real_signal_and_rejects_a_null():
    """A `w` that drives y_B must beat its permutations; an unrelated one must not."""
    rng = np.random.default_rng(0)
    n = 20000
    labels = np.repeat(np.arange(100), n // 100)
    log_n = rng.normal(0.0, 0.5, n)
    area = rng.normal(size=(n, 3))
    dens = rng.normal(size=(n, 2))

    # w varies by spatial block, so the permutation null is a fair comparison
    w_true = np.clip(rng.uniform(size=100)[labels] + rng.normal(0, .05, n), 0, 1)
    y_b = rng.poisson(np.exp(-1.0 + log_n + 1.4 * w_true)).astype(float)
    real = heldout_gene_test(y_b, y_b, w_true, log_n, area, dens, labels,
                             n_perm=30, seed=0)
    w_null = np.clip(rng.uniform(size=100)[labels] + rng.normal(0, .05, n), 0, 1)
    null = heldout_gene_test(y_b, y_b, w_null, log_n, area, dens, labels,
                             n_perm=30, seed=1)

    lines = [
        f"- real `w`: coef **{real['coef_w']:+.4f}**, delta-deviance "
        f"**{real['delta_deviance']:.1f}** vs permuted mean {real['null_mean']:.1f} "
        f"(ratio **{real['ratio_to_null_mean']:.1f}x**, p = {real['perm_p_value']:.3f})",
        f"- unrelated `w`: delta-deviance {null['delta_deviance']:.1f} vs permuted "
        f"mean {null['null_mean']:.1f} (ratio {null['ratio_to_null_mean']:.1f}x, "
        f"p = {null['perm_p_value']:.3f})",
        "", "Verifies the statistic separates a `w` that drives the held-out "
        "counts from one that does not, using the same spatially-blocked "
        "permutation null the real-data job uses.",
    ]
    ok = bool(real["ratio_to_null_mean"] > 10 and real["coef_w"] > 0
              and null["ratio_to_null_mean"] < real["ratio_to_null_mean"] / 5)
    write("Test 08 -- held-out gene machinery (spec 6.8)", lines, passed=ok)
    assert real["coef_w"] > 0
    assert real["ratio_to_null_mean"] > 10, "failed to detect a w that truly drives y_B"
    assert null["ratio_to_null_mean"] < real["ratio_to_null_mean"] / 5, \
        "an unrelated w scored comparably to the real one"


def test_08_poisson_deviance_zero_at_perfect_fit():
    y = np.array([0.0, 1.0, 5.0, 12.0])
    assert poisson_deviance(y, y) < 1e-9        # y log(y/mu) term handles y=0
    assert poisson_deviance(y, y * 2) > 0
