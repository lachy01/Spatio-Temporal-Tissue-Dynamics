"""Parameter variance and Wald hypothesis tests for the fitted fate model.

`SpatialMultinomialFateModel` is exactly a multinomial logit that is *linear*
in an expanded per-cell feature vector

    x_i = [1, phi_i, phi_bar_i, vec(phi_i outer phi_bar_i)]

(the intercept, own-state, neighbor-state, and the bilinear interaction whose
coefficients are `B`), with the last class fixed as the reference. That lets
us compute the model's observed Fisher information analytically and invert it
for the asymptotic parameter covariance — the standard maximum-likelihood
route to standard errors, Wald z-statistics, and p-values (the same machinery
statsmodels' `MNLogit` uses under the hood).

IMPORTANT — these standard errors are only valid inference if the estimates
they annotate are the *unpenalized maximum-likelihood* estimates:

  * Fit with `weight_decay=0`. L2 (weight decay) shrinks coefficients toward
    zero, so the Wald test on a penalized fit is biased toward not rejecting.
  * Fit with `class_weight=None`. Class weighting changes the objective away
    from the log-likelihood, so the inverse-information is no longer the
    covariance of what was optimized.

So: fit a dedicated `weight_decay=0, class_weight=None` model for inference,
even if you use a penalized/weighted model for prediction. Two further
cautions the returned table can't enforce for you: with hundreds of thousands
of cells, p-values are almost all ~0 (n is enormous) so read effect sizes and
z-magnitudes, not just significance; and the bilinear `B` block uses D*D
strongly-collinear outer-product features, so its information sub-block is
often near-singular — check the reported condition number and treat B's
individual SEs skeptically when it's large.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats


def _build_design(phi, phi_bar, covariates=None, full_interaction=True):
    """Expanded per-cell design matrix matching the model's linear structure.

    Columns, in order: [1, phi (D), phi_bar (D), covariates (C), outer(phi,
    phi_bar) (D*D)], where outer[d, e] = phi[d] * phi_bar[e] flattened
    row-major — the same ordering as the model's flat coefficient vector
    [b0, a, c, d, vec(B)], so column j lines up with parameter j of each free
    class. The covariate block is omitted when `covariates` is None; the outer
    block is omitted when `full_interaction` is False. Only "full" and "none"
    models are linear in their parameters and thus expressible as a fixed
    design matrix — a low-rank model is not.
    """
    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    n, d = phi.shape
    columns = [np.ones((n, 1)), phi, phi_bar]
    if covariates is not None:
        columns.append(np.asarray(covariates, dtype=np.float64))
    if full_interaction:
        columns.append((phi[:, :, None] * phi_bar[:, None, :]).reshape(n, d * d))
    return np.concatenate(columns, axis=1)


def _mnl_observed_information(design, proba_free, chunk_size=20000):
    """Observed Fisher information of a multinomial logit, class-major blocks.

    Parameters
    ----------
    design : (n, P) array
        Per-observation feature matrix (including the intercept column).
    proba_free : (n, K) array
        Fitted softmax probabilities of the K free (non-reference) classes.

    Returns
    -------
    (K*P, K*P) information matrix. Block (a, b) is
    `sum_i p_ia (delta_ab - p_ib) x_i x_i^T`, the standard MNL Hessian of the
    negative log-likelihood (which, MNL being a canonical GLM, equals the
    expected information and does not depend on the observed labels). Blocks
    are laid out class-major: rows/cols `a*P : (a+1)*P` belong to free class a.
    """
    n, p = design.shape
    k = proba_free.shape[1]
    info = np.zeros((k * p, k * p))

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x = design[start:stop]
        pr = proba_free[start:stop]
        for a in range(k):
            for b in range(a, k):
                w = pr[:, a] * ((1.0 if a == b else 0.0) - pr[:, b])
                block = x.T @ (x * w[:, None])
                info[a * p:(a + 1) * p, b * p:(b + 1) * p] += block
                if b != a:
                    info[b * p:(b + 1) * p, a * p:(a + 1) * p] += block.T
    return info


def _flat_params(model):
    """Flatten model coefficients per free class as [b0, a, c, (d), (vec(B))]."""
    b0 = model.b0.detach().cpu().numpy()
    a = model.a.detach().cpu().numpy()
    c = model.c.detach().cpu().numpy()
    k, d = a.shape
    cov = model.d.detach().cpu().numpy() if model.n_covariates > 0 else None
    B = model.B.detach().cpu().numpy() if model.interaction_mode == "full" else None

    rows = []
    for i in range(k):
        parts = [[b0[i]], a[i], c[i]]
        if cov is not None:
            parts.append(cov[i])
        if B is not None:
            parts.append(B[i].reshape(d * d))
        rows.append(np.concatenate(parts))
    return np.concatenate(rows), k, d


def _param_terms(feature_names, covariate_names=None, full_interaction=True):
    """Per-class parameter term labels, in the flat-coefficient order."""
    terms = ["b0"]
    terms += [f"a[{f}]" for f in feature_names]           # own state
    terms += [f"c[{f}]" for f in feature_names]           # neighbor state
    if covariate_names:
        terms += [f"d[{c}]" for c in covariate_names]     # technical covariates
    if full_interaction:
        terms += [f"B[{fd} x {fe}]" for fd in feature_names for fe in feature_names]
    return terms


def fate_parameter_stats(model, phi, phi_bar, covariates=None, feature_names=None,
                          covariate_names=None, chunk_size=20000, rcond=1e-12, cond_warn=1e10):
    """Standard errors, Wald z-statistics and p-values for a fitted fate model.

    See this module's docstring for the (important) conditions under which
    these are valid inference — in short, fit the model with `weight_decay=0`
    and `class_weight=None` first.

    Parameters
    ----------
    model : SpatialMultinomialFateModel
        A fitted model.
    phi, phi_bar : arrays (n, D)
        The same own-state / neighbor-state embeddings the model was fit on.
    covariates : array (n, C), optional
        The technical-covariate matrix the model was fit with (e.g.
        `adata.obsm["covariates"]`); required iff `model.n_covariates > 0`.
    feature_names : sequence of str, optional
        Names for the D embedding dimensions (e.g. `adata.uns["phi_names"]`);
        defaults to `phi_0..phi_{D-1}`. Used only to label rows.
    covariate_names : sequence of str, optional
        Names for the C covariates (e.g. `adata.uns["covariates_names"]`);
        defaults to `cov_0..cov_{C-1}`.

    Only "full" (`interaction_rank=None`) and "none" (`interaction_rank=0`)
    models are supported: these are linear in their parameters, so the Wald
    machinery applies. A low-rank model is not linear in its parameters (and
    its U/V factors are non-identifiable), so this raises — use
    `likelihood_ratio_test` to test whether its interaction improves fit, and
    `interaction_modes` to interpret it.

    Returns
    -------
    pandas.DataFrame with one row per free-class parameter and columns:
    `class`, `term`, `estimate`, `variance`, `std_error`, `z`, `p_value`.
    `df.attrs["condition_number"]` records the information matrix's condition
    number (large => the covariance, especially for the bilinear `B` block,
    is numerically fragile).
    """
    import pandas as pd

    if model.interaction_mode == "lowrank":
        raise ValueError(
            "fate_parameter_stats needs a model that is linear in its parameters "
            "(interaction_rank=None for full B, or 0 for no interaction), but this model is "
            "low-rank (interaction_rank="
            f"{model.interaction_rank}). Low-rank interactions are non-identifiable in U/V, so "
            "per-coefficient Wald tests are invalid — use likelihood_ratio_test(...) to test "
            "whether the interaction helps overall, and interaction_modes(...) to interpret it."
        )

    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    d = phi.shape[1]
    if feature_names is None:
        feature_names = [f"phi_{i}" for i in range(d)]
    elif len(feature_names) != d:
        raise ValueError(f"feature_names has length {len(feature_names)}, expected {d}")

    if model.n_covariates > 0:
        if covariates is None:
            raise ValueError(f"model has n_covariates={model.n_covariates}; pass the covariates matrix")
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.shape[1] != model.n_covariates:
            raise ValueError(f"covariates has {covariates.shape[1]} columns, expected {model.n_covariates}")
        if covariate_names is None:
            covariate_names = [f"cov_{i}" for i in range(model.n_covariates)]
        elif len(covariate_names) != model.n_covariates:
            raise ValueError(f"covariate_names has length {len(covariate_names)}, expected {model.n_covariates}")
    else:
        covariates, covariate_names = None, None

    full = model.interaction_mode == "full"
    estimate, k, _ = _flat_params(model)

    with torch.no_grad():
        cov_t = None if covariates is None else torch.as_tensor(covariates, dtype=torch.float32)
        logits = model(
            torch.as_tensor(phi, dtype=torch.float32),
            torch.as_tensor(phi_bar, dtype=torch.float32),
            covariates=cov_t,
        )
        proba_free = F.softmax(logits, dim=1).numpy()[:, :k]

    design = _build_design(phi, phi_bar, covariates=covariates, full_interaction=full)
    info = _mnl_observed_information(design, proba_free, chunk_size=chunk_size)

    condition_number = float(np.linalg.cond(info))
    if condition_number > cond_warn:
        warnings.warn(
            f"Information matrix is ill-conditioned (condition number {condition_number:.2e}); "
            "standard errors — especially for the bilinear B terms built from collinear "
            "outer-product features — may be unreliable. Consider inference on a model "
            "without the interaction term, or on a lower-dimensional embedding.",
            stacklevel=2,
        )
    covariance = np.linalg.pinv(info, rcond=rcond)

    variance = np.clip(np.diag(covariance), 0.0, None)
    std_error = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std_error > 0, estimate / std_error, np.nan)
    p_value = 2.0 * scipy_stats.norm.sf(np.abs(z))

    terms = _param_terms(feature_names, covariate_names=covariate_names, full_interaction=full)
    free_class_names = model.class_names[:-1]
    class_col = np.repeat(free_class_names, len(terms))
    term_col = terms * k

    df = pd.DataFrame({
        "class": class_col,
        "term": term_col,
        "estimate": estimate,
        "variance": variance,
        "std_error": std_error,
        "z": z,
        "p_value": p_value,
    })
    df.attrs["condition_number"] = condition_number
    return df


def _total_log_likelihood(model, phi, phi_bar, target, covariates=None):
    """Total (summed) multinomial log-likelihood of `target` under `model`."""
    with torch.no_grad():
        cov_t = None if covariates is None else torch.as_tensor(np.asarray(covariates), dtype=torch.float32)
        logits = model(
            torch.as_tensor(np.asarray(phi), dtype=torch.float32),
            torch.as_tensor(np.asarray(phi_bar), dtype=torch.float32),
            covariates=cov_t,
        )
        y = torch.as_tensor(np.asarray(target), dtype=torch.long)
        # F.cross_entropy is mean negative log-likelihood in nats; scale to a total.
        return -F.cross_entropy(logits, y).item() * len(y)


def _interaction_dof(model):
    """Effective free-parameter count of a model's interaction term.

    none -> 0; full -> n_free * D^2; low-rank r -> n_free * (2*D*r - r^2),
    the dimension of the rank-r bilinear manifold (the naive 2*D*r parameter
    count minus the r^2-dimensional U/V rotation redundancy).
    """
    n_free = len(model.class_names) - 1
    d = model.phi_dim
    mode = model.interaction_mode
    if mode == "none":
        return 0
    if mode == "full":
        return n_free * d * d
    r = model.interaction_rank
    return n_free * (2 * d * r - r * r)


def likelihood_ratio_test(null_model, alt_model, phi, phi_bar, target, covariates=None):
    """Likelihood-ratio test that `alt_model`'s interaction improves fit.

    Compares two *fitted* nested models (the null nested in the alternative —
    e.g. null=no-interaction, alt=low-rank or full) via
    `LR = 2 (loglik_alt - loglik_null)`, referred to a chi-square with degrees
    of freedom equal to the difference in effective interaction parameters
    (see `_interaction_dof`).

    Both models must already be fit on the SAME `(phi, phi_bar, target)` you
    pass here, and — as with `fate_parameter_stats` — should be fit as the
    unpenalized, unweighted MLE (`weight_decay=0`, `class_weight=None`) for
    the test to be exact.

    Caveat: when the alternative is low-rank, the null (zero interaction) sits
    on a singular boundary of the low-rank parameter manifold, so the
    chi-square approximation is only rough — treat its p-value as indicative
    and prefer a held-out log-loss comparison as the robust check. The test
    is regular (and the p-value trustworthy) for null=none vs alt=full.

    Returns
    -------
    dict with `lr_statistic`, `df`, `p_value`, `loglik_null`, `loglik_alt`.
    """
    if _interaction_dof(alt_model) <= _interaction_dof(null_model):
        raise ValueError(
            "alt_model must have more interaction degrees of freedom than null_model "
            "(the null must be nested in the alternative)."
        )

    ll_null = _total_log_likelihood(null_model, phi, phi_bar, target, covariates=covariates)
    ll_alt = _total_log_likelihood(alt_model, phi, phi_bar, target, covariates=covariates)
    lr = 2.0 * (ll_alt - ll_null)
    df = _interaction_dof(alt_model) - _interaction_dof(null_model)
    p_value = float(scipy_stats.chi2.sf(lr, df)) if lr > 0 else 1.0
    return {
        "lr_statistic": float(lr),
        "df": int(df),
        "p_value": p_value,
        "loglik_null": float(ll_null),
        "loglik_alt": float(ll_alt),
    }


def interaction_modes(model, feature_names=None, n_modes=None):
    """Decompose each class's interaction matrix into interpretable modes.

    For every free class, SVD the effective interaction matrix
    `B^(k) = U^(k) V^(k)^T` (or the dense `B^(k)` for a full model) into
    `sum_j sigma_j p_j q_j^T`. Each mode j is a rank-1 "cross-talk" channel:
    a cell's fate shifts when its own state aligns with own-loading `p_j`
    *and* its neighborhood aligns with neighbor-loading `q_j`, scaled by
    `sigma_j`. Unlike the raw (rotation-ambiguous) U/V factors, these SVD
    modes are identified and ordered by strength, so they're what to inspect
    and plot. Signs are fixed so each own-loading's largest-magnitude entry
    is positive.

    Returns
    -------
    Long-form pandas.DataFrame with columns `class`, `mode`, `singular_value`,
    `feature`, `own_loading`, `neighbor_loading` (one row per class x mode x
    feature).
    """
    import pandas as pd

    B = model.interaction_matrix().cpu().numpy()  # (n_free, D, D); raises if no interaction
    n_free, d, _ = B.shape
    if feature_names is None:
        feature_names = [f"phi_{i}" for i in range(d)]
    elif len(feature_names) != d:
        raise ValueError(f"feature_names has length {len(feature_names)}, expected {d}")
    max_modes = d if n_modes is None else min(n_modes, d)

    rows = []
    for k in range(n_free):
        p_mat, sigma, qt = np.linalg.svd(B[k])
        for j in range(max_modes):
            own = p_mat[:, j]
            neighbor = qt[j, :]
            sign = np.sign(own[np.argmax(np.abs(own))]) or 1.0  # stable, interpretable sign
            own, neighbor = own * sign, neighbor * sign
            for f, o, ne in zip(feature_names, own, neighbor):
                rows.append({
                    "class": model.class_names[k],
                    "mode": j,
                    "singular_value": float(sigma[j]),
                    "feature": f,
                    "own_loading": float(o),
                    "neighbor_loading": float(ne),
                })
    return pd.DataFrame(rows)
