"""Negative-binomial GLM of a marker gene's counts on the spatial neighborhood.

This is the count-based analogue of the fate model that sidesteps hard-label
thresholding entirely: instead of calling each cell "dividing" from a noisy,
near-null cell-cycle score, regress a direct proliferation marker's *counts*
(MKI67) on the neighborhood embedding, with `log(total_counts)` as an offset
so the model explains detection *rate* rather than raw depth. It's the closest
transcriptomic replica of a Ki67-index readout, and — modeled as counts — it
neither invents nor destroys signal the way a 50%-precision binary label does.

MKI67 is typically very sparse on a shallow targeted panel (order ~1% of cells
positive), so power is limited per cell; the same model applied at the tile
level (summed counts per tile, see tiles.aggregate_tiles) is usually where the
neighborhood signal, if any, shows up.

Requires statsmodels (an optional dependency: `pip install statsmodels`).
"""

import numpy as np


def _require_statsmodels():
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "count_glm needs statsmodels; install it with `pip install statsmodels` "
            "(or `pip install -e '.[stats]')."
        ) from exc
    return sm


def get_gene_counts(adata, gene, counts_layer="counts"):
    """Raw counts of `gene` and each cell's total library size.

    Reads raw counts from `layers[counts_layer]` if present (where
    `ensure_lognormalized` stashes them), else from `X`. Returns
    `(gene_counts, total_counts)` as 1-D float arrays.
    """
    import scipy.sparse as sp

    counts = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    total = np.asarray(counts.sum(1)).ravel().astype(float)
    col = adata.var_names.get_loc(gene)
    g = counts[:, col]
    g = g.toarray().ravel() if sp.issparse(g) else np.asarray(g).ravel()
    return g.astype(float), total


def get_signature_counts(adata, genes, counts_layer="counts"):
    """Summed raw counts over a gene set (a "division count") and library size.

    `genes` is a list of gene symbols; only those present in the panel are
    used (case-insensitive, via `gene_sets.match_gene_set`). Reads raw counts
    from `layers[counts_layer]` if present (where `ensure_lognormalized`
    stashes them), else from `X`. Returns `(signature_counts, total_counts,
    matched_genes)` — the first two as 1-D float arrays.

    This is the count-model response for division: rather than threshold a
    proliferation *score* into a label, sum the underlying cell-cycle
    transcript counts (e.g. the Tirosh S+G2M genes) so the outcome is a
    genuine over-dispersed count for negative-binomial regression.
    """
    import scipy.sparse as sp

    from .gene_sets import match_gene_set

    matched = match_gene_set(genes, adata.var_names)
    if not matched:
        raise ValueError(
            f"none of the {len(list(genes))} requested genes are present in this panel; "
            "pass a gene set curated for it."
        )
    counts = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    total = np.asarray(counts.sum(1)).ravel().astype(float)
    cols = [adata.var_names.get_loc(g) for g in matched]
    sub = counts[:, cols]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    y = sub.sum(1).ravel().astype(float)
    return y, total, matched


def _reduce(block, n_pcs, fitted=None):
    """Optionally PC-reduce a predictor block; reuse `fitted` PCA if given."""
    if n_pcs is None:
        return block, None
    from sklearn.decomposition import PCA

    pca = fitted or PCA(n_components=min(n_pcs, block.shape[1]), random_state=0).fit(block)
    return pca.transform(block), pca


def build_predictors(adata, predictors="neighbor", n_pcs=None, phi_key="phi",
                      phi_bar_key="phi_bar", extra_covariates=None):
    """Design matrix (no intercept) for a predictor spec.

    predictors : "none" | "own" | "neighbor" | "own+neighbor"
        Which embedding blocks to include — `phi` (own state), `phi_bar`
        (neighbor state), both, or neither. Each block is optionally
        PC-reduced to `n_pcs` components (recommended: the raw 29-dim blocks
        are strongly collinear).
    extra_covariates : sequence of str, optional
        `obs` columns to append as columns (e.g. `"log_cell_area"`); you must
        have created them. Standardize them yourself if needed.

    Returns
    -------
    (X, names) with X of shape (n, p) and matching column names. `p` may be 0
    for `predictors="none"` with no covariates (intercept-only model).
    """
    blocks, names = [], []
    if predictors in ("own", "own+neighbor"):
        b, _ = _reduce(np.asarray(adata.obsm[phi_key]), n_pcs)
        blocks.append(b)
        names += [f"own_pc{j}" if n_pcs else f"own_{j}" for j in range(b.shape[1])]
    if predictors in ("neighbor", "own+neighbor"):
        b, _ = _reduce(np.asarray(adata.obsm[phi_bar_key]), n_pcs)
        blocks.append(b)
        names += [f"nbr_pc{j}" if n_pcs else f"nbr_{j}" for j in range(b.shape[1])]
    if predictors not in ("none", "own", "neighbor", "own+neighbor"):
        raise ValueError(f"Unknown predictors {predictors!r}")

    if extra_covariates:
        cov = np.column_stack([adata.obs[c].to_numpy(dtype=float) for c in extra_covariates])
        blocks.append(cov)
        names += list(extra_covariates)

    X = np.column_stack(blocks) if blocks else np.empty((adata.n_obs, 0))
    return X, names


def _build_interaction(own, nbr, own_names, nbr_names):
    """Full-rank own x neighbour outer-product design block and its names.

    Column (d, e) is `own[:, d] * nbr[:, e]` — the coefficient a linear GLM
    fits for it is the (d, e) entry of the dense interaction matrix `B` in the
    fate model's `phi^T B phi_bar` term. Produces `D_own * D_nbr` columns.
    These outer products are strongly collinear (which is exactly why the torch
    model offers a low-rank `B = U V^T` alternative), so expect inflated
    standard errors on this block at full rank — judge it by the block
    likelihood-ratio test, not per-coefficient Wald p-values.
    """
    n, d = own.shape
    e = nbr.shape[1]
    block = (own[:, :, None] * nbr[:, None, :]).reshape(n, d * e)
    names = [f"int[{own_names[i]} x {nbr_names[j]}]" for i in range(d) for j in range(e)]
    return block, names


def _spline_basis(x, df=5):
    """Natural cubic (restricted) spline basis for a 1-D covariate, no intercept.

    Uses patsy's `cr` (natural cubic regression spline) — well-behaved at the
    boundaries, with `df` basis columns placed at data quantiles — so a
    covariate like local density can enter the model as a smooth nonlinear term
    instead of a single linear slope. Requires patsy (a statsmodels dependency).
    """
    import patsy

    x = np.asarray(x, dtype=float)
    basis = patsy.dmatrix(f"cr(x, df={int(df)}) - 1", {"x": x}, return_type="matrix")
    return np.asarray(basis)


def _estimate_nb_alpha(y, mu):
    """NB2 dispersion via the Cameron-Trivedi auxiliary OLS regression.

    Regresses ((y-mu)^2 - y)/mu on mu through the origin; the slope estimates
    alpha in Var = mu + alpha*mu^2. Clipped to a small positive floor so a
    (rare) underdispersed estimate still yields a usable NB family.
    """
    w = ((y - mu) ** 2 - y) / np.where(mu > 0, mu, np.nan)
    ok = np.isfinite(w)
    alpha = float(np.sum(w[ok] * mu[ok]) / np.sum(mu[ok] ** 2))
    return max(alpha, 1e-6)


def _irls_offset(offset, n):
    return np.zeros(n) if offset is None else np.asarray(offset, dtype=float)


def _poisson_irls(y, X, offset=None, max_iter=100, tol=1e-9):
    """Poisson (log-link) GLM via IRLS on the p x p normal equations.

    `X` includes the intercept column; `offset` (if given) is added to the
    linear predictor with a fixed coefficient of 1. Returns `(beta, mu)` — used
    to seed the NB dispersion estimate memory-efficiently on a tall design.
    """
    y = np.asarray(y, dtype=float)
    off = _irls_offset(offset, len(y))
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))
    ll = -np.inf
    for _ in range(max_iter):
        mu = np.exp(np.clip(X @ beta + off, -30, 30))
        z_work = X @ beta + (y - mu) / mu                 # offset held out of the working response
        beta = np.linalg.pinv(X.T @ (X * mu[:, None])) @ (X.T @ (mu * z_work))
        ll_new = float(np.sum(y * np.log(np.clip(np.exp(np.clip(X @ beta + off, -30, 30)), 1e-300, None))
                              - np.exp(np.clip(X @ beta + off, -30, 30))))
        if np.abs(ll_new - ll) < tol * (abs(ll) + 1e-8):
            break
        ll = ll_new
    mu = np.exp(np.clip(X @ beta + off, -30, 30))
    return beta, mu


def _nb2_irls(y, X, alpha, offset=None, max_iter=100, tol=1e-9):
    """Fixed-dispersion NB2 (log-link) GLM via IRLS on the p x p normal equations.

    `X` must already include an intercept column; `offset` (if given) is added
    to the linear predictor with a fixed coefficient of 1. Returns
    `(beta, z, se, loglik)` with Wald `z`/`se` from `(X^T W X)^{-1}`. Much more
    memory-efficient than statsmodels on a tall design (it never factorizes the
    full n x p matrix — it forms `X^T W X`, p x p, and pseudo-inverts that),
    which both makes repeated refits fast (the stepwise path) and lets the big
    full-interaction model fit without the multi-GB SVD statsmodels would need.
    Validated against `fit_count_glm` in the tests. NB2 with log link has IRLS
    weight `w = mu / (1 + alpha*mu)` and working response `eta + (y - mu)/mu`;
    the pseudo-inverse mirrors statsmodels' handling of a collinear design.
    """
    from scipy.special import gammaln

    y = np.asarray(y, dtype=float)
    off = _irls_offset(offset, len(y))
    r = 1.0 / alpha
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))                 # intercept at log mean

    def loglik(mu):
        return float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
                            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))

    ll = -np.inf
    for _ in range(max_iter):
        mu = np.exp(np.clip(X @ beta + off, -30, 30))
        w = mu / (1.0 + alpha * mu)
        z_work = X @ beta + (y - mu) / mu                 # offset held out of the working response
        XtWX = X.T @ (X * w[:, None])
        beta = np.linalg.pinv(XtWX) @ (X.T @ (w * z_work))
        ll_new = loglik(np.exp(np.clip(X @ beta + off, -30, 30)))
        if np.abs(ll_new - ll) < tol * (abs(ll) + 1e-8):
            ll = ll_new
            break
        ll = ll_new

    mu = np.exp(np.clip(X @ beta + off, -30, 30))
    w = mu / (1.0 + alpha * mu)
    cov = np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    return beta, z, se, ll


def _nb_dispersion_fit(y, X, Wd, offset=None, max_outer=60, tol=1e-9, verbose=False):
    """Joint MLE of an NB2 whose dispersion depends on the cell's own state.

        log mu_i     = offset_i + X_i . beta        (mean, as before)
        log alpha_i  = Wd_i . gamma                 (dispersion)

    The **log link on alpha** is a deliberate departure from modelling
    `alpha_i = phi_i . gamma` directly: a dispersion must stay strictly positive,
    and an unconstrained linear predictor would wander negative and make the
    likelihood undefined. `gamma` is still a free coefficient vector on the state;
    `exp(gamma_j)` is the multiplicative effect of state dimension `j` on the
    dispersion.

    This is no longer a GLM (the dispersion is itself modelled), so it is fit by
    block coordinate ascent: a beta step (IRLS at the current per-cell alpha)
    alternating with a gamma step (L-BFGS on the dispersion score, analytic
    gradient). Both blocks maximise the same joint log-likelihood, so the ascent
    is monotone.

    Returns `(beta, gamma, alpha_i, loglik, gamma_se, n_outer)`.
    """
    from scipy.optimize import minimize
    from scipy.special import digamma, gammaln

    y = np.asarray(y, dtype=float)
    off = _irls_offset(offset, len(y))
    Wd = np.asarray(Wd, dtype=float)

    _b, mu0 = _poisson_irls(y, X, off)              # start at the constant-dispersion solution
    gamma = np.zeros(Wd.shape[1]); gamma[0] = np.log(_estimate_nb_alpha(y, mu0))
    beta, ll_prev, it = None, -np.inf, 0

    def _nll_grad(g, mu):
        a = np.exp(np.clip(Wd @ g, -20.0, 20.0))
        r = 1.0 / a
        rm = r + mu
        ll = float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
                          + r * np.log(r / rm) + y * np.log(mu / rm)))
        dl_dr = digamma(y + r) - digamma(r) + np.log(r / rm) + 1.0 - (y + r) / rm
        return -ll, -(Wd.T @ (dl_dr * (-r)))         # dr/dgamma = -r * Wd

    for it in range(max_outer):
        alpha = np.exp(np.clip(Wd @ gamma, -20.0, 20.0))
        beta, _z, _se, _ll = _nb2_irls(y, X, alpha, offset=off)
        mu = np.exp(np.clip(X @ beta + off, -30.0, 30.0))
        res = minimize(_nll_grad, gamma, args=(mu,), jac=True, method="L-BFGS-B")
        gamma, ll = res.x, -res.fun
        if verbose:
            print("  outer %2d  loglik %.3f" % (it, ll), flush=True)
        if np.abs(ll - ll_prev) < tol * (abs(ll_prev) + 1e-8):
            ll_prev = ll
            break
        ll_prev = ll

    alpha = np.exp(np.clip(Wd @ gamma, -20.0, 20.0))
    beta, _z, _se, ll = _nb2_irls(y, X, alpha, offset=off)
    mu = np.exp(np.clip(X @ beta + off, -30.0, 30.0))

    q = len(gamma); H = np.zeros((q, q)); eps = 1e-5   # observed information for gamma
    _f0, g0 = _nll_grad(gamma, mu)
    for j in range(q):
        gp = gamma.copy(); gp[j] += eps
        _f1, g1 = _nll_grad(gp, mu)
        H[:, j] = (g1 - g0) / eps
    H = 0.5 * (H + H.T)
    gamma_se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    return beta, gamma, alpha, ll, gamma_se, it + 1


def _fit_glm_irls(y, X, names, offset=None, family="nb", alpha=None):
    """Memory-efficient analogue of `fit_count_glm` (normal-equations IRLS).

    Returns the same dict shape as `fit_count_glm` — `loglik`, `n_params`,
    `alpha`, and a `coef` table — minus the statsmodels `result` object, plus
    `beta`/`offset` so the caller can form fitted means. For NB with
    `alpha=None` the dispersion is estimated from a Poisson pre-fit
    (Cameron-Trivedi). Used by `fit_division_nb` for designs too tall-and-wide
    for statsmodels' SVD-based fit to hold in memory (e.g. the full interaction).
    """
    import pandas as pd
    from scipy import stats as sps

    Xc = np.column_stack([np.ones(len(y)), X])
    all_names = ["const"] + list(names)
    off = None if offset is None else np.asarray(offset, dtype=float)

    if family == "poisson":
        beta, mu = _poisson_irls(y, Xc, off)
        cov = np.linalg.pinv(Xc.T @ (Xc * mu[:, None]))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        loglik = float(np.sum(np.asarray(y, float) * np.log(np.clip(mu, 1e-300, None)) - mu))
        alpha_out = None
    elif family == "nb":
        if alpha is None:
            _b, mu = _poisson_irls(y, Xc, off)
            alpha = _estimate_nb_alpha(np.asarray(y, dtype=float), mu)
        beta, z, se, loglik = _nb2_irls(y, Xc, alpha, offset=off)
        alpha_out = alpha
    else:
        raise ValueError(f"Unknown family {family!r}; expected 'nb' or 'poisson'")

    coef = pd.DataFrame({
        "term": all_names, "estimate": beta, "std_error": se, "z": z,
        "p_value": 2.0 * sps.norm.sf(np.abs(z)),
    })
    return {"loglik": float(loglik), "n_params": Xc.shape[1], "alpha": alpha_out,
            "coef": coef, "beta": beta, "offset": off}


def fit_count_glm(y, X, names, offset=None, family="nb", alpha=None, exog_is_design=True):
    """Fit a Poisson / negative-binomial GLM with an optional offset.

    Low-level fitter on arrays (see `fit_neighbourhood_count_glm` for the
    AnnData-driven entry point). `offset` is added to the linear predictor on
    the log scale (pass `log(total_counts)` there, not the raw counts).

    `alpha` (NB only): the negative-binomial dispersion. If None (default) it
    is estimated from a Poisson pre-fit via the Cameron-Trivedi auxiliary
    regression; pass an explicit value to hold it fixed — needed to make a
    likelihood-ratio test between nested NB models valid (both must share the
    same dispersion for the difference in log-likelihoods to be a clean LR
    statistic).

    Returns a dict with `result` (the fitted statsmodels object),
    `loglik`, `n_params`, `alpha` (NB dispersion, or None for Poisson), and a
    coefficient table `coef` (DataFrame: term, estimate, std_error, z, p_value).
    """
    import pandas as pd

    sm = _require_statsmodels()
    Xc = sm.add_constant(X, has_constant="add")
    all_names = ["const"] + list(names)

    if family == "poisson":
        res = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset).fit()
        alpha = None
    elif family == "nb":
        if alpha is None:
            pois = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset).fit()
            alpha = _estimate_nb_alpha(np.asarray(y, dtype=float), pois.mu)
        res = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    else:
        raise ValueError(f"Unknown family {family!r}; expected 'nb' or 'poisson'")

    coef = pd.DataFrame({
        "term": all_names,
        "estimate": np.asarray(res.params),
        "std_error": np.asarray(res.bse),
        "z": np.asarray(res.tvalues),
        "p_value": np.asarray(res.pvalues),
    })
    return {"result": res, "loglik": float(res.llf), "n_params": Xc.shape[1],
            "alpha": alpha, "coef": coef}


def fit_neighbourhood_count_glm(adata, gene="MKI67", predictors="neighbor", n_pcs=6,
                                 family="nb", extra_covariates=None, use_offset=True,
                                 counts_layer="counts"):
    """Regress `gene` counts on the neighborhood, with a log-library offset.

    Convenience wrapper: pulls raw `gene` counts and library size, builds the
    predictor design (see `build_predictors`), sets `offset=log(total_counts)`
    (so coefficients describe detection *rate*), drops cells with zero library
    size, and fits the GLM. Also computes McFadden's pseudo-R^2 and a
    likelihood-ratio test against the offset-only null.

    Returns the `fit_count_glm` dict, extended with `mcfadden_r2`, `lr_vs_null`
    (dict: statistic/df/p_value), and `n` (cells used).
    """
    from scipy import stats as scipy_stats

    y, total = get_gene_counts(adata, gene, counts_layer=counts_layer)
    keep = total > 0
    X, names = build_predictors(adata, predictors=predictors, n_pcs=n_pcs,
                                 extra_covariates=extra_covariates)
    y, X = y[keep], X[keep]
    offset = np.log(total[keep]) if use_offset else None

    fit = fit_count_glm(y, X, names, offset=offset, family=family)

    null = fit_count_glm(y, np.empty((len(y), 0)), [], offset=offset, family=family)
    ll_full, ll_null = fit["loglik"], null["loglik"]
    fit["mcfadden_r2"] = 1.0 - ll_full / ll_null if ll_null != 0 else np.nan
    lr = 2.0 * (ll_full - ll_null)
    df = fit["n_params"] - null["n_params"]
    fit["lr_vs_null"] = {
        "statistic": float(lr),
        "df": int(df),
        "p_value": float(scipy_stats.chi2.sf(lr, df)) if df > 0 and lr > 0 else 1.0,
    }
    fit["n"] = int(len(y))
    return fit


def compare_count_glms(adata, gene="MKI67", specs=("none", "own", "neighbor", "own+neighbor"),
                        n_pcs=6, family="nb", extra_covariates=None, counts_layer="counts"):
    """Fit a nested ladder of count models and tabulate their fit.

    Fits `gene`'s count GLM under each predictor spec (all sharing the same
    log-library offset and PCA reduction) and returns a comparison DataFrame:
    `spec`, `n_params`, `loglik`, `mcfadden_r2`, `lr_vs_null_p`. Reading down
    the rows answers the OSDR question directly — does adding the neighborhood
    block improve fit beyond own-state / the null?
    """
    import pandas as pd

    rows = []
    for spec in specs:
        fit = fit_neighbourhood_count_glm(
            adata, gene=gene, predictors=spec, n_pcs=n_pcs, family=family,
            extra_covariates=extra_covariates, counts_layer=counts_layer,
        )
        rows.append({
            "spec": spec,
            "n_params": fit["n_params"],
            "loglik": fit["loglik"],
            "mcfadden_r2": fit["mcfadden_r2"],
            "lr_vs_null_p": fit["lr_vs_null"]["p_value"],
        })
    return pd.DataFrame(rows)


def _division_design(adata, predictors, interaction_rank, covariate_keys,
                     standardize_covariates, phi_key="phi", phi_bar_key="phi_bar",
                     spline_keys=None, spline_df=5, neighbour_contrast=False):
    """Assemble the division-model design matrix (no intercept) and term metadata.

    Shared by `fit_division_nb` and `stepwise_division_nb`. Returns
    `(X, names, meta, nuisance, nuisance_names)`: `X` stacks the requested state
    blocks (own, neighbour, full interaction) then the nuisance block; `meta` is
    a per-column list of `{"kind", "own", "nbr"}` dicts (kind in
    own/nbr/int/spline/cov) so callers can reason about term structure without
    parsing names; `nuisance` is the standalone nuisance block — the density
    spline basis (`spline_keys`, one natural-cubic basis of `spline_df` columns
    each) followed by the standardized free covariates (`covariate_keys`) — used
    as the covariate-only null. See `fit_division_nb` for argument semantics.

    With `neighbour_contrast`, the neighbour block (and the interaction's
    neighbour factor) is `phi_bar - phi` (the neighbourhood's *deviation* from
    the cell) instead of `phi_bar`, which decorrelates it from the own block;
    terms are labelled `nbr_dev[...]` / `int[X x dev:Y]`.
    """
    if predictors not in ("none", "own", "neighbor", "own+neighbor"):
        raise ValueError(f"Unknown predictors {predictors!r}")
    if interaction_rank is not None and (not isinstance(interaction_rank, int) or interaction_rank < 0):
        raise ValueError(
            f"interaction_rank must be None (full B), 0 (no interaction), or a positive "
            f"int (low-rank); got {interaction_rank!r}")

    phi_names = list(adata.uns.get(
        f"{phi_key}_names", [str(j) for j in range(np.asarray(adata.obsm[phi_key]).shape[1])]))
    bar_names = list(adata.uns.get(f"{phi_bar_key}_names", phi_names))

    def _nbr_values():
        v = np.asarray(adata.obsm[phi_bar_key], dtype=float)
        return v - np.asarray(adata.obsm[phi_key], dtype=float) if neighbour_contrast else v

    nbr_label = "nbr_dev" if neighbour_contrast else "nbr"

    blocks, names, meta = [], [], []
    if predictors in ("own", "own+neighbor"):
        blocks.append(np.asarray(adata.obsm[phi_key], dtype=float))
        for nm in phi_names:
            names.append(f"own[{nm}]"); meta.append({"kind": "own", "own": nm, "nbr": None})
    if predictors in ("neighbor", "own+neighbor"):
        blocks.append(_nbr_values())
        for nm in bar_names:
            names.append(f"{nbr_label}[{nm}]"); meta.append({"kind": "nbr", "own": None, "nbr": nm})

    if interaction_rank is None:                     # full dense B: outer-product columns
        if predictors != "own+neighbor":
            raise ValueError(
                "interaction_rank=None (full B) needs predictors='own+neighbor' — the "
                "bilinear term multiplies own and neighbour state, so both must be present")
        own = np.asarray(adata.obsm[phi_key], dtype=float)
        nbr = _nbr_values()
        int_bar_names = [f"dev:{nm}" for nm in bar_names] if neighbour_contrast else bar_names
        inter, inter_names = _build_interaction(own, nbr, phi_names, int_bar_names)
        blocks.append(inter)
        names += inter_names
        for i in range(own.shape[1]):
            for j in range(nbr.shape[1]):
                meta.append({"kind": "int", "own": phi_names[i], "nbr": bar_names[j]})
    elif interaction_rank != 0:                       # positive int -> low-rank, deferred
        raise NotImplementedError(
            f"low-rank interaction (interaction_rank={interaction_rank}) is bilinear in U, V "
            "and can't be fit as a linear GLM; use interaction_rank=None for full rank, or "
            "the torch SpatialMultinomialFateModel for a low-rank B.")

    # nuisance block: density spline basis (if any) then free linear covariates.
    # Both are non-state confounder controls (kept out of the state block, and
    # out of stepwise selection); they form the covariate-only null.
    nuis_blocks, nuis_names = [], []
    for key in (list(spline_keys) if spline_keys else []):
        if key not in adata.obs.columns:
            raise ValueError(f"spline column {key!r} not found in obs; compute it first "
                             "(e.g. compute_local_density for 'local_density')")
        basis = _spline_basis(adata.obs[key].to_numpy(dtype=float), df=spline_df)
        nuis_blocks.append(basis)
        for j in range(basis.shape[1]):
            nuis_names.append(f"spline[{key}]_{j + 1}")
            meta.append({"kind": "spline", "own": None, "nbr": None})

    cov_names = list(covariate_keys) if covariate_keys else []
    if cov_names:
        missing = [c for c in cov_names if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"covariate columns {missing} not found in obs; compute them first "
                             "(compute_library_size / compute_local_density)")
        cov = np.column_stack([adata.obs[c].to_numpy(dtype=float) for c in cov_names])
        if standardize_covariates:
            mu, sd = cov.mean(0), cov.std(0)
            sd[sd == 0] = 1.0
            cov = (cov - mu) / sd
        nuis_blocks.append(cov)
        for _ in cov_names:
            meta.append({"kind": "cov", "own": None, "nbr": None})

    nuisance = np.column_stack(nuis_blocks) if nuis_blocks else np.empty((adata.n_obs, 0))
    nuis_all_names = nuis_names + cov_names

    X = np.column_stack(blocks + nuis_blocks) if (blocks or nuis_blocks) else np.empty((adata.n_obs, 0))
    return X, names + nuis_all_names, meta, nuisance, nuis_all_names


def _drop_candidates(active, protected, meta_of, respect_hierarchy):
    """Terms eligible for removal in a backward step.

    Excludes `protected` terms (the technical covariates). With
    `respect_hierarchy` (marginality), an own/neighbour main effect is not
    removable while any interaction term involving that cell type remains.
    """
    cand = [nm for nm in active if nm not in protected and meta_of[nm]["kind"] != "cov"]
    if respect_hierarchy:
        active_ints = [meta_of[nm] for nm in active if meta_of[nm]["kind"] == "int"]
        owns_needed = {m["own"] for m in active_ints}
        nbrs_needed = {m["nbr"] for m in active_ints}

        def blocked(nm):
            m = meta_of[nm]
            return ((m["kind"] == "own" and m["own"] in owns_needed)
                    or (m["kind"] == "nbr" and m["nbr"] in nbrs_needed))
        cand = [nm for nm in cand if not blocked(nm)]
    return cand


def fit_division_nb(adata, genes=None, predictors="own+neighbor", interaction_rank=0,
                     covariate_keys=("log_total_counts", "local_density"),
                     offset_key=None, spline_keys=None, spline_df=5, neighbour_contrast=False,
                     standardize_covariates=True, family="nb", fitter="auto", dispersion_predictors=None,
                     phi_key="phi", phi_bar_key="phi_bar", counts_layer="counts"):
    """Alternative to the multinomial fate model: NB regression of the *division
    count* on own + neighbour cell state and the technical covariates.

    Death is not modelled at all. Instead of thresholding a proliferation
    score into a "dividing" class, the response is the summed raw counts over
    a cell-cycle gene set (default: the Tirosh S+G2M genes present in the
    panel — the same genes `score_cell_cycle` uses), which is strongly
    over-dispersed (variance >> mean) and so modelled as negative binomial.

    Predictors are the named own-state (`obsm[phi_key]`) and neighbour-state
    (`obsm[phi_bar_key]`) embeddings. The technical nuisance terms can be
    entered three ways, reflecting their roles: as a fixed **offset**
    (`offset_key`, on the log scale — so the model explains the division *rate*
    per transcript, coefficient fixed at 1, the natural treatment for library
    size); as a **spline** (`spline_keys`, a smooth natural-cubic term rather
    than a single slope, for effects that need not be linear, e.g. local
    density); and as **free covariates** (`covariate_keys`, standardized linear
    terms, e.g. cell area). The defaults keep the earlier behaviour (log library
    size + local density as free covariates).

    **Recommended division specification** — `offset_key=None,
    spline_keys=("local_density",), covariate_keys=("log_total_counts",
    "cell_area")`, i.e. log library size **free, not an offset**. Division counts
    scale sub-proportionally with depth on the breast sample (implied exponent
    `beta / sd(log T) = 0.70`, against the offset's fixed 1.0), and pinning it at
    1 costs on every diagnostic: it drops the log-likelihood by ~880, worsens the
    PIT (0.040 -> 0.049) and inflates the rootogram misfit by about 20%. It also
    pushes the mis-specification into the *dispersion*, raising alpha from 0.70 to
    0.77 — which matters when using `dispersion_predictors`. State coefficients
    are stable across the two (Pearson r = 0.995), but they then read as effects
    on the count at fixed depth rather than strictly "per transcript".
    `fit_death_nb` is the opposite case: death counts scale near-proportionally
    (exponent 0.93), and freeing the term *worsens* its rootogram, so keep
    `offset_key="log_total_counts"` there.

    Parameters
    ----------
    genes : sequence of str, optional
        The division gene set to sum for the response. Defaults to the union
        of the Tirosh S- and G2M-phase genes; only those present in the panel
        are used.
    covariate_keys : sequence of str or None
        `obs` columns entered as free, standardized linear covariates (default
        the log library size + local density). Pass `None`/`()` for none.
    offset_key : str or None
        `obs` column added to the linear predictor as an offset (fixed
        coefficient 1). Set to `"log_total_counts"` to model the cell-cycle
        transcript *rate* (and drop library size from `covariate_keys`). `None`
        (default) disables it.
    spline_keys : sequence of str or None
        `obs` columns expanded as natural-cubic splines of `spline_df` columns
        each (e.g. `("local_density",)`). `None` (default) means none — those
        columns then only enter if listed in `covariate_keys` (linear).
    spline_df : int
        Degrees of freedom (basis columns) per spline covariate.
    dispersion_predictors : None | "own" | "neighbor" | "own+neighbor"
        Let the NB dispersion vary with cell state instead of being one number
        shared by every cell: `log alpha_i = gamma . [1, phi_i]`, fit jointly
        with the mean by `_nb_dispersion_fit`. `None` (default) keeps the
        single constant dispersion. The log link keeps `alpha_i` positive;
        `exp(gamma_j)` reads as the multiplicative effect of state dimension
        `j` on the dispersion. Setting this forces `fitter="irls"` (statsmodels
        cannot take a per-observation dispersion) and adds a `dispersion` entry
        to the result carrying the `gamma` table and a likelihood-ratio test
        against the constant-dispersion MLE of the same mean model.
    neighbour_contrast : bool
        If True, represent the neighbour state as `phi_bar - phi` (the
        neighbourhood's deviation from the cell) rather than `phi_bar`, which
        decorrelates it from the own-state block. For the main-effects model
        this is an exact reparameterisation (identical fit), but far better
        conditioned; with the interaction it spans a different, less collinear
        space. Neighbour terms are labelled `nbr_dev[...]`.
    predictors : "none" | "own" | "neighbor" | "own+neighbor"
        Which state blocks to include (nuisance terms are always included when
        given). Column names are taken from
        `uns[f"{phi_key}_names"]` when present, so coefficients read as named
        cell types (e.g. `own[T-cells]`, `nbr[Cancer Epithelial]`).
    interaction_rank : None | 0 | positive int
        The own x neighbour bilinear interaction `phi^T B phi_bar` — the same
        `B` term as the fate model.
          * `0` (default) — no interaction (own + neighbour main effects only).
          * `None` — full dense `B`: every `own[d] * nbr[e]` outer product is
            added as a design column (`int[own x nbr]` terms). Because the GLM
            is linear in these, Wald/LR inference stays valid — but the outer
            products are highly collinear, so read the block via its
            likelihood-ratio test, not per-term p-values. Requires
            `predictors="own+neighbor"`.
          * a positive int `r` — low-rank `B = U V^T`. This is bilinear in
            `U, V` (not linear in the parameters), so it cannot be fit as a
            plain GLM; not yet implemented here (raises `NotImplementedError`).
            The API accepts it so a future custom fitter can slot in.
    fitter : "auto" | "statsmodels" | "irls"
        Backend for the GLM fit. `"auto"` (default) uses statsmodels for modest
        designs and the memory-efficient normal-equations IRLS for wide ones
        (>= 30 predictors, e.g. the full interaction), where statsmodels' SVD
        over the tall design would exhaust memory. `"irls"` forces the efficient
        path (no statsmodels `result` object; coefficients validated to match).

    Returns
    -------
    The `fit_count_glm` dict, extended with `genes` (the matched gene set),
    `n` (cells used), `pseudo_r2_vs_covariates` (McFadden-style improvement of
    the full model over a covariate-only null), and `lr_vs_covariate_only`
    (dict statistic/df/p_value) — the likelihood-ratio test isolating what the
    included state block (own + neighbour, and the interaction if any) adds for
    division *after* depth and density are controlled for. With
    `covariate_keys=None` the null is intercept-only, so the test then measures
    the state block against a bare intercept.
    """
    import pandas as pd
    from scipy import stats as scipy_stats

    from .gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES

    if genes is None:
        genes = list(dict.fromkeys(list(TIROSH_S_GENES) + list(TIROSH_G2M_GENES)))
    y, total, matched = get_signature_counts(adata, genes, counts_layer=counts_layer)

    X_full, names_full, _meta, nuis, nuis_names = _division_design(
        adata, predictors, interaction_rank, covariate_keys, standardize_covariates,
        phi_key, phi_bar_key, spline_keys=spline_keys, spline_df=spline_df,
        neighbour_contrast=neighbour_contrast)

    if offset_key is not None and offset_key not in adata.obs.columns:
        raise ValueError(f"offset_key {offset_key!r} not found in obs; compute it first "
                         "(e.g. compute_library_size for 'log_total_counts')")
    offset = adata.obs[offset_key].to_numpy(dtype=float) if offset_key is not None else None

    keep = total > 0
    off_k = offset[keep] if offset is not None else None

    # `fitter="auto"` uses statsmodels for modest designs but the memory-efficient
    # normal-equations IRLS for wide ones (e.g. the full interaction block), where
    # statsmodels' SVD over the 680k-row design would exhaust memory.
    if fitter not in ("auto", "statsmodels", "irls"):
        raise ValueError(f"fitter must be 'auto', 'statsmodels', or 'irls'; got {fitter!r}")
    use_irls = fitter == "irls" or (fitter == "auto" and X_full.shape[1] >= 30)
    if dispersion_predictors not in (None, "none", "own", "neighbor", "own+neighbor"):
        raise ValueError("dispersion_predictors must be None, 'own', 'neighbor' or "
                         f"'own+neighbor'; got {dispersion_predictors!r}")
    if dispersion_predictors in (None, "none"):
        dispersion_predictors = None
    elif family != "nb":
        raise ValueError("dispersion_predictors requires family='nb'")
    else:
        use_irls = True                # a per-cell alpha rules statsmodels out
    _fit = _fit_glm_irls if use_irls else fit_count_glm

    disp = None
    if dispersion_predictors is None:
        fit = _fit(y[keep], X_full[keep], names_full, offset=off_k, family=family)
    else:
        # log alpha_i = gamma . [1, state_i], fit jointly with the mean, and the
        # same mean model at a constant dispersion as the LR-test baseline.
        Wd_state, _wn = build_predictors(adata, dispersion_predictors, n_pcs=None,
                                         phi_key=phi_key, phi_bar_key=phi_bar_key)
        # name the dispersion terms the way the mean model names its state block,
        # so gamma reads as cell types rather than own_0, own_1, ...
        _pn = list(adata.uns.get(f"{phi_key}_names",
                                 [str(j) for j in range(np.asarray(adata.obsm[phi_key]).shape[1])]))
        _bn = list(adata.uns.get(f"{phi_bar_key}_names", _pn))
        wd_names = []
        if dispersion_predictors in ("own", "own+neighbor"):
            wd_names += [f"own[{nm}]" for nm in _pn]
        if dispersion_predictors in ("neighbor", "own+neighbor"):
            wd_names += [f"nbr[{nm}]" for nm in _bn]
        if len(wd_names) != Wd_state.shape[1]:       # e.g. an unexpected block width
            wd_names = list(_wn)
        Wd = np.column_stack([np.ones(int(keep.sum())), Wd_state[keep]])
        Xc = np.column_stack([np.ones(int(keep.sum())), X_full[keep]])
        _b, gamma, alpha_i, ll_v, gamma_se, n_outer = _nb_dispersion_fit(
            y[keep], Xc, Wd, offset=off_k)
        beta_c, _g0, alpha_c, ll_c, _s0, _i0 = _nb_dispersion_fit(
            y[keep], Xc, np.ones((int(keep.sum()), 1)), offset=off_k)
        mu_c = np.full(adata.n_obs, np.nan)           # the like-for-like baseline mean
        mu_c[keep] = np.exp(np.clip(Xc @ beta_c + (off_k if off_k is not None else 0.0),
                                    -30, 30))
        # re-fit at the converged per-cell alpha to get the usual coef table shape
        fit = _fit(y[keep], X_full[keep], names_full, offset=off_k,
                   family=family, alpha=alpha_i)
        lr_d = 2.0 * (ll_v - ll_c)
        df_d = Wd.shape[1] - 1
        disp = {
            "terms": ["const"] + list(wd_names),
            "gamma": gamma, "gamma_se": gamma_se, "alpha": alpha_i,
            "coef": pd.DataFrame({
                "term": ["const"] + list(wd_names), "estimate": gamma,
                "std_error": gamma_se,
                "z": np.divide(gamma, gamma_se, out=np.zeros_like(gamma),
                               where=gamma_se > 0),
                "exp_estimate": np.exp(gamma),
            }),
            "predictors": dispersion_predictors, "n_outer": int(n_outer),
            "loglik": float(ll_v), "loglik_constant_alpha": float(ll_c),
            "alpha_constant_alpha": float(alpha_c[0]),
            "fitted_mu_constant_alpha": mu_c,
            "lr_vs_constant_alpha": {
                "statistic": float(lr_d), "df": int(df_d),
                "p_value": float(scipy_stats.chi2.sf(lr_d, df_d))
                if df_d > 0 and lr_d > 0 else 1.0,
            },
        }

    # null = nuisance only (offset + splines + covariates, no cell-state block),
    # at the full model's NB dispersion so the log-likelihood difference is a
    # valid likelihood-ratio statistic.
    null = _fit(y[keep], nuis[keep], nuis_names, offset=off_k, family=family, alpha=fit["alpha"])

    ll_full, ll_null = fit["loglik"], null["loglik"]
    lr = 2.0 * (ll_full - ll_null)
    df = fit["n_params"] - null["n_params"]
    fit["genes"] = matched
    fit["n"] = int(keep.sum())
    fit["offset_key"] = offset_key
    fit["fitter"] = "irls" if use_irls else "statsmodels"
    fit["dispersion"] = disp
    fit["pseudo_r2_vs_covariates"] = 1.0 - ll_full / ll_null if ll_null != 0 else np.nan
    fit["lr_vs_covariate_only"] = {
        "statistic": float(lr),
        "df": int(df),
        "p_value": float(scipy_stats.chi2.sf(lr, df)) if df > 0 and lr > 0 else 1.0,
    }

    # per-cell fitted mean, model P(division) = P(>=1 cell-cycle transcript), and
    # NB2 Pearson residuals, aligned to adata (nan for dropped zero-count cells)
    alpha = fit["alpha"]
    mu = np.full(adata.n_obs, np.nan)
    if use_irls:
        eta = np.column_stack([np.ones(int(keep.sum())), X_full[keep]]) @ fit["beta"]
        mu[keep] = np.exp(np.clip(eta + (off_k if off_k is not None else 0.0), -30, 30))
    else:
        mu[keep] = np.asarray(fit["result"].fittedvalues)
    fit["keep"] = keep
    fit["fitted_mu"] = mu
    with np.errstate(invalid="ignore", divide="ignore"):
        if alpha is None:                                 # Poisson
            fit["prob_division"] = 1.0 - np.exp(-mu)
            fit["pearson_resid"] = (y - mu) / np.sqrt(mu)
        else:                                             # NB2
            fit["prob_division"] = 1.0 - (1.0 + alpha * mu) ** (-1.0 / alpha)
            fit["pearson_resid"] = (y - mu) / np.sqrt(mu + alpha * mu ** 2)
    return fit


def fit_death_nb(adata, genes=None, predictors="own+neighbor", interaction_rank=0,
                  covariate_keys=("log_total_counts", "local_density"),
                  offset_key=None, spline_keys=None, spline_df=5, neighbour_contrast=False,
                  standardize_covariates=True, family="nb", fitter="auto",
                  phi_key="phi", phi_bar_key="phi_bar", counts_layer="counts"):
    """Death/apoptosis analogue of `fit_division_nb`, same architecture.

    NB regression of an apoptosis marker-*count* response on own + neighbour cell
    state (optionally with the `phi^T B phi_bar` interaction, `interaction_rank`),
    a library-size offset (`offset_key`), a density spline (`spline_keys`), and
    free covariates (`covariate_keys`) — identical in every way to the division
    model except the response is the summed counts over an apoptosis signature
    instead of the cell-cycle genes. `genes` defaults to
    `gene_sets.APOPTOSIS_SIGNATURE` (the pro-apoptotic machinery in the dying
    cell; only those present in the panel are used). Every parameter behaves
    exactly as in `fit_division_nb` — see it for the full description.

    Returns the same dict, with the per-cell death probability P(>=1 apoptosis
    transcript) under the fitted NB stored as `prob_death` (rather than
    `prob_division`).
    """
    from .gene_sets import APOPTOSIS_SIGNATURE

    if genes is None:
        genes = list(APOPTOSIS_SIGNATURE)
    fit = fit_division_nb(
        adata, genes=genes, predictors=predictors, interaction_rank=interaction_rank,
        covariate_keys=covariate_keys, offset_key=offset_key, spline_keys=spline_keys,
        spline_df=spline_df, neighbour_contrast=neighbour_contrast,
        standardize_covariates=standardize_covariates, family=family, fitter=fitter,
        phi_key=phi_key, phi_bar_key=phi_bar_key, counts_layer=counts_layer)
    fit["prob_death"] = fit.pop("prob_division")
    return fit


def stepwise_division_nb(adata, genes=None, criterion="bic",
                          covariate_keys=("log_total_counts", "local_density"),
                          standardize_covariates=True, respect_hierarchy=True,
                          phi_key="phi", phi_bar_key="phi_bar", counts_layer="counts",
                          verbose=False):
    """Backward stepwise selection from the full-interaction NB division model.

    Starts from the full model — own state + neighbour state + the full dense
    own x neighbour interaction (`interaction_rank=None`) + the technical
    covariates — and greedily removes one term at a time to minimise an
    information criterion (`criterion`: "bic", default, or "aic").

    At this n every term is "significant" by p-value, so selection is by BIC
    (penalty `ln(n)` per parameter — ~13 here, far stricter than AIC's 2, and
    the sensible default at this sample size). Each step drops the eligible term
    with the smallest Wald |z| and refits *once*; a term is removed only while
    its `z^2` is below the criterion's per-parameter penalty (i.e. removing it
    lowers the criterion, by the Wald approximation `Δ(-2logL) ≈ z^2`), so the
    search reaches a local optimum in O(p) refits instead of the O(p^2) of
    refitting for every candidate at every step. The NB dispersion is estimated
    once on the full model and held fixed along the path — this both speeds the
    refits and keeps the log-likelihoods (hence BIC) comparable across the
    nested models. The returned model is the lowest-criterion one actually
    visited (guarding against a Wald step that overshoots).

    The technical covariates are protected (never dropped — they are confounder
    controls, not candidates), as is the intercept. With `respect_hierarchy`
    (default) an own/neighbour main effect is not removed while an interaction
    involving that cell type survives (marginality).

    Returns the `fit_count_glm` dict for the selected model, extended with
    `selected_terms`, `dropped_terms`, `criterion`, `alpha`, `genes`, `n`, and
    `path` — a DataFrame (columns step/dropped/n_terms/n_params/loglik/aic/bic)
    tracing the elimination from the full model to the stopping point.
    """
    import gc

    import pandas as pd

    from .gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES

    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic'; got {criterion!r}")

    if genes is None:
        genes = list(dict.fromkeys(list(TIROSH_S_GENES) + list(TIROSH_G2M_GENES)))
    y, total, matched = get_signature_counts(adata, genes, counts_layer=counts_layer)

    X, names, meta, _cov, cov_names = _division_design(
        adata, "own+neighbor", None, covariate_keys, standardize_covariates,
        phi_key, phi_bar_key)

    keep = total > 0
    y, X = y[keep], X[keep]
    n = int(len(y))
    penalty = np.log(n) if criterion == "bic" else 2.0

    col_of = {nm: i for i, nm in enumerate(names)}
    meta_of = {nm: meta[i] for i, nm in enumerate(names)}
    protected = set(cov_names)

    def ic(loglik, n_params, k_pen):
        return -2.0 * loglik + k_pen * n_params

    def light_fit(active_terms, alpha):
        """Fast fixed-dispersion refit on the active columns (path only).

        Uses the internal p x p IRLS (`_nb2_irls`) rather than statsmodels —
        which would re-factorize the whole 680k-row design on every iteration of
        every refit — and returns just the coef z-values (with an intercept),
        the log-likelihood and the parameter count. The big temporaries are
        freed each call so nothing accumulates across the ~p refits.
        """
        Xc = np.column_stack([np.ones(n), X[:, [col_of[nm] for nm in active_terms]]])
        _beta, zvals, _se, ll = _nb2_irls(y, Xc, alpha)
        zmap = dict(zip(["const"] + list(active_terms), zvals))
        k = Xc.shape[1]
        del Xc
        gc.collect()
        return zmap, ll, k

    # one statsmodels fit of the full model to estimate the NB dispersion; then
    # the whole path runs at that fixed alpha via the fast internal fitter.
    full_sm = fit_count_glm(y, X, names, family="nb")
    alpha = full_sm["alpha"]
    del full_sm
    gc.collect()

    zmap, ll, k = light_fit(names, alpha)                 # full-model row (fixed alpha)
    active = list(names)
    best = {"crit": ic(ll, k, penalty), "active": list(active)}

    def row(step, dropped, ll, k):
        return {"step": step, "dropped": dropped, "n_terms": len(active), "n_params": k,
                "loglik": ll, "aic": ic(ll, k, 2.0), "bic": ic(ll, k, np.log(n))}

    path = [row(0, None, ll, k)]
    step = 0
    while True:
        cand = _drop_candidates(active, protected, meta_of, respect_hierarchy)
        if not cand:
            break
        drop = min(cand, key=lambda nm: abs(zmap.get(nm, np.inf)))
        if abs(zmap.get(drop, np.inf)) ** 2 >= penalty:
            break                                        # no single removal lowers the criterion
        active.remove(drop)
        zmap, ll, k = light_fit(active, alpha)
        step += 1
        path.append(row(step, drop, ll, k))
        if verbose:
            print("step %3d  drop %-34s n_params=%3d  bic=%.1f" % (step, drop, k, ic(ll, k, np.log(n))),
                  flush=True)
        if ic(ll, k, penalty) < best["crit"]:
            best = {"crit": ic(ll, k, penalty), "active": list(active)}

    # refit the selected model once, keeping the full statsmodels result this time
    final = fit_count_glm(y, X[:, [col_of[nm] for nm in best["active"]]], list(best["active"]),
                          family="nb", alpha=alpha)
    result = dict(final)
    result["selected_terms"] = list(best["active"])
    result["dropped_terms"] = [nm for nm in names if nm not in set(best["active"])]
    result["criterion"] = criterion
    result["alpha"] = alpha
    result["genes"] = matched
    result["n"] = n
    result["path"] = pd.DataFrame(path)
    return result
