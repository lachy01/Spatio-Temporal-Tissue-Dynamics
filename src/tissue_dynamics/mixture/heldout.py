"""Held-out gene validation (spec 6.8) -- the only external check on `w`.

Everything else in the suite establishes that the model is *internally* sound:
it recovers known parameters, does not invent interactions from noise, and its
class separation survives the size covariate that could fake it. None of that
shows the latent class is **proliferation** rather than something correlated
with it.

This test asks the question directly. Split the Tirosh set into halves A and B,
fit the mixture using only half A, and ask whether the responsibilities `w`
predict the counts of half B *beyond* what depth, area and density already
explain. Dispersion the rate model failed to absorb has no reason to predict
held-out cell-cycle genes; real proliferation does.

The permutation control is the point of the test, not a formality. `w` will
predict `y_B` trivially through library size, so what matters is the improvement
**relative to a `w` that has been spatially block-permuted** -- same marginal
distribution, same within-block structure, alignment to the tissue destroyed.
"""

from __future__ import annotations

import numpy as np


def split_gene_set(s_genes, g2m_genes, present, seed=0):
    """Split the signature into halves A and B, stratified by phase (spec 6.8).

    Stratifying matters: an unstratified split can put most S-phase genes in one
    half, so the two halves would measure different things and a weak result
    could be phase mismatch rather than absence of signal.
    """
    rng = np.random.default_rng(seed)
    present = set(present)
    A, B = [], []
    for block in (list(s_genes), list(g2m_genes)):
        g = [x for x in dict.fromkeys(block) if x in present]
        rng.shuffle(g)
        A += g[: len(g) // 2]
        B += g[len(g) // 2:]
    return sorted(set(A)), sorted(set(B) - set(A))


def poisson_deviance(y, mu):
    """Residual deviance of a Poisson fit; the `y log(y/mu)` term is 0 at y=0."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(t - (y - mu)))


def _fit_poisson(y, X, offset, m=None):
    """Poisson GLM; with `m`, the mean is exp(X b + offset) + theta m (MisTIC)."""
    if m is not None:
        from ..additive_glm import fit_poisson_additive

        r = fit_poisson_additive(y, X, offset, m, se=False)
        return r["beta"], r["mu"]
    from ..count_glm import _poisson_irls

    beta, mu = _poisson_irls(np.asarray(y, dtype=float), X, offset)
    return beta, mu


def block_permute(values, labels, rng):
    """Permute `values` between spatial blocks, preserving within-block order.

    Whole blocks are reassigned to other blocks' cells, so the permuted vector
    keeps the marginal distribution and the local structure of `w` but loses its
    alignment with the tissue. A plain element-wise shuffle would destroy the
    spatial autocorrelation too, making the null far too easy to beat.
    """
    values = np.asarray(values)
    blocks = np.unique(labels)
    idx = [np.where(labels == b)[0] for b in blocks]
    src = np.concatenate([idx[i] for i in rng.permutation(len(blocks))])
    dst = np.concatenate(idx)
    out = np.empty_like(values)
    out[dst] = values[src]
    return out


def heldout_gene_test(y_a, y_b, w, log_n, area_basis, density_basis, labels,
                      n_perm=100, seed=0, verbose=False, m_b=None):
    """Spec 6.8. Returns the observed and permuted improvements in deviance.

    `w` must come from a model fitted on `y_a` only.  The nuisance model is
    `y_b ~ offset(log N) + s(log area) + s(log density)`; the full model adds
    `w`. Both are Poisson GLMs, so twice the log-likelihood difference is the
    drop in residual deviance.

    Under MisTIC (`m_b`, half B's expected misassigned count) the area basis
    has no columns and both models' means gain theta * m_b, fitted directly;
    the deviance difference keeps its meaning.
    """
    n = len(y_b)
    nuis = np.column_stack([np.ones(n), area_basis, density_basis])
    off = np.asarray(log_n, dtype=float)

    _b0, mu0 = _fit_poisson(y_b, nuis, off, m_b)
    dev0 = poisson_deviance(y_b, mu0)

    full = np.column_stack([nuis, np.asarray(w, dtype=float)])
    beta1, mu1 = _fit_poisson(y_b, full, off, m_b)
    dev1 = poisson_deviance(y_b, mu1)
    obs = dev0 - dev1                       # deviance explained by w

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm))
    for i in range(int(n_perm)):
        wp = block_permute(w, labels, rng)
        Xp = np.column_stack([nuis, wp])
        _bp, mup = _fit_poisson(y_b, Xp, off, m_b)
        null[i] = dev0 - poisson_deviance(y_b, mup)
        if verbose and (i + 1) % 25 == 0:
            print(f"    permutation {i + 1}/{n_perm}", flush=True)

    return {
        "coef_w": float(beta1[-1]),
        "delta_deviance": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_max": float(null.max()),
        "null_q95": float(np.percentile(null, 95)),
        # spec 6.8's criterion: the real improvement must be >= 10x the
        # improvement a spatially-permuted w achieves
        "ratio_to_null_mean": float(obs / max(null.mean(), 1e-12)),
        "exceeds_all_permutations": bool(obs > null.max()),
        "perm_p_value": float((1.0 + np.sum(null >= obs)) / (1.0 + len(null))),
        "deviance_nuisance": float(dev0),
        "deviance_full": float(dev1),
        "n_perm": int(n_perm),
        "null": null.tolist(),
    }
