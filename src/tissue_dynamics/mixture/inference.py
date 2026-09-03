"""Spatial blocking, penalty selection and the block bootstrap (spec 6.11, 7).

Everything here rests on one fact the spec states repeatedly: cells are spatially
dependent, so the effective sample size is far below n. That rules out
inverse-Hessian standard errors (spec 7 forbids reporting them) and rules out
AIC/BIC for any selection problem (spec 6.11). Contiguous spatial blocks are the
unit of both resampling and cross-validation.
"""

from __future__ import annotations

import numpy as np

from .fit import fit_lbfgs, multistart


def spatial_blocks(xy, n_blocks=250, seed=0):
    """Partition cells into contiguous blocks by k-means on `(x, y)` (spec 7.1)."""
    from sklearn.cluster import MiniBatchKMeans

    xy = np.asarray(xy, dtype=float)
    km = MiniBatchKMeans(n_clusters=int(n_blocks), random_state=int(seed),
                         n_init=3, batch_size=4096)
    return km.fit_predict(xy)


def block_side_length(xy, labels):
    """Median block diameter, to check it against the correlation range (spec 7.1)."""
    xy = np.asarray(xy, dtype=float)
    out = []
    for b in np.unique(labels):
        m = labels == b
        if m.sum() > 1:
            out.append(float(np.sqrt(((xy[m] - xy[m].mean(0)) ** 2).sum(1).mean()) * 2))
    return float(np.median(out)) if out else float("nan")


def block_folds(labels, k=5, seed=0):
    """Assign whole blocks to `k` folds, so no fold splits a block."""
    rng = np.random.default_rng(seed)
    blocks = np.unique(labels)
    rng.shuffle(blocks)
    fold_of = {b: i % k for i, b in enumerate(blocks)}
    return np.array([fold_of[b] for b in labels])


def select_weight_ridge(data, spec, cfg, folds, grid=None, verbose=True,
                        maxiter=1500, n_restarts=2, seed=0):
    """Choose `weight_ridge` by held-out log-likelihood on spatial blocks.

    The held-out score is the **unpenalised** marginal log-likelihood on the left
    out fold, evaluated per cell so folds of different sizes are comparable. The
    penalty only shapes the fit; it must not appear in the score used to compare
    penalties, or the comparison would reward shrinkage for its own sake.
    """
    import dataclasses

    import jax.numpy as jnp

    from .model import loglik_terms

    grid = list(grid if grid is not None else
                cfg["model"].get("weight_ridge_grid", [0.0, 0.1, 1.0, 10.0, 100.0]))
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    k = int(folds.max()) + 1
    rows = []

    for lam in grid:
        sp = dataclasses.replace(spec, weight_ridge=float(lam))
        scores = []
        for f in range(k):
            tr, te = folds != f, folds == f
            dtr, dte = data.subset(np.where(tr)[0]), data.subset(np.where(te)[0])
            best, _ = multistart(dtr, sp, cfg, n_restarts=n_restarts, seed=seed + f,
                                 verbose=False, maxiter=maxiter)
            Dte = dte.as_jax()
            scores.append(float(jnp.sum(loglik_terms(best.params, Dte, sp))) / int(te.sum()))
        rows.append((float(lam), float(np.mean(scores)), float(np.std(scores)), scores))
        say(f"  weight_ridge {lam:>8g}: held-out loglik/cell "
            f"{np.mean(scores):.6f} +/- {np.std(scores):.6f}")

    best_lam = max(rows, key=lambda r: r[1])[0]
    say(f"  selected weight_ridge = {best_lam:g}")
    return best_lam, rows


def block_bootstrap(data, spec, cfg, params_hat, labels, n_boot=None, seed=0,
                    verbose=True, statistic=None):
    """Spatial block bootstrap (spec 7.1): resample blocks with replacement.

    Warm-starts each replicate from the full-data estimate and caps iterations, as
    spec 7.1 directs. Records the non-convergence rate rather than silently
    dropping replicates (spec 9.13).
    """
    n_boot = int(n_boot if n_boot is not None else cfg["inference"]["n_bootstrap"])
    cap = int(cfg["inference"]["bootstrap_maxiter"])
    rng = np.random.default_rng(seed)
    blocks = np.unique(labels)
    idx_of = {b: np.where(labels == b)[0] for b in blocks}
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    out, n_fail = [], 0
    for r in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([idx_of[b] for b in pick])
        res = fit_lbfgs(data.subset(idx), spec, params_hat, maxiter=cap)
        if not np.isfinite(res.loglik):
            n_fail += 1
            continue
        if not res.converged:
            n_fail += 1                       # counted, not dropped (spec 9.13)
        out.append(res.params if statistic is None else statistic(res.params))
        if verbose and (r + 1) % 25 == 0:
            say(f"  bootstrap {r + 1}/{n_boot} ({100 * n_fail / (r + 1):.0f}% not converged)")

    rate = n_fail / max(n_boot, 1)
    if rate > float(cfg["inference"]["max_failure_rate"]):
        say(f"  ** {100 * rate:.1f}% of replicates failed to converge -- above the "
            f"{100 * cfg['inference']['max_failure_rate']:.0f}% threshold; spec 7.1 "
            "requires investigating rather than proceeding **")
    return out, {"n_boot": n_boot, "n_failed": n_fail, "failure_rate": rate}
