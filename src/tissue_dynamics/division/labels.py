"""Label constructions (spec section 2): technical null, BASE-T, BASE-O, SCORE.

Labels are TARGET DEFINITIONS. They are fitted once on all analysis cells and
frozen before any model is fitted (spec 2.1); nothing here sees a model.
"""

from __future__ import annotations

import numpy as np

from .survival import nb_p_ge, smallest_c_below

ALPHA_B_GRID = (0.05, 0.01, 0.001, 1e-4)
ALPHA_B_PRIMARY = 0.05     # decided by the user after Stage 1 (spec default 0.001)
TD_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TD_PRIMARY = 0.5
TRIM_P = 0.001
SCORE_P = 0.001


def ab_tag(a):
    return {0.05: "a05", 0.01: "a01", 0.001: "a001", 1e-4: "a0001"}[float(a)]


def td_tag(t):
    return f"td{int(round(100 * t)):03d}"


# --------------------------------------------------------------------------
# Cell type
# --------------------------------------------------------------------------
def cell_type(df, phi_names):
    """Argmax of column-standardised PhiSpace scores (spec 4.3).

    Never the raw argmax: the nine columns differ ~18-fold in scale, and a raw
    argmax matches the stored label for only 36.6% of cells.
    """
    cols = sorted(c for c in df.columns if c.startswith("phi_"))
    P = df[cols].to_numpy(float)
    Z = (P - P.mean(0)) / np.where(P.std(0) > 0, P.std(0), 1.0)
    return np.asarray(phi_names)[Z.argmax(1)]


# --------------------------------------------------------------------------
# 2.1 Technical null: NB(mu0, alpha0). By model.technical:
#     "area"    log mu0 = b0 + log N + s(log A)
#     "mistic"  mu0 = Nt e^b0 + theta m    (MisTIC spec 3.2; no area spline)
# --------------------------------------------------------------------------
def null_design(df, basis, m=None):
    """`(X, offset, m)` of the technical null for the table rows in `df`.

    "area": X = [1, s(log A)], offset = raw log N, m = None (as before).
    "mistic": X = [1], offset = log of the MisTIC-weighted N, and m -- the
    table's m_cyc unless another gene set's m is passed (the negative-control
    and spillover tests).
    """
    n = len(df)
    X = np.column_stack([np.ones(n), basis.log_area_basis(df)])
    if getattr(basis, "technical", "area") == "mistic":
        mm = basis.mistic_m(df) if m is None else np.asarray(m, float)
        return X, np.log(df[basis.logN_column].to_numpy(float)), mm
    return X, np.log(df["N"].to_numpy(float)), None


def fit_nb_offset(y, X, offset, weight=None, beta0=None, log_alpha0=0.0,
                  maxiter=10000, grad_tol=1e-9, rel_tol=1e-14):
    """NB2 regression with a fixed offset, by L-BFGS on the per-cell NLL.

    `weight` (0/1) selects the cells that enter the fit while keeping array
    shapes fixed, so the trimming loop compiles once.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from jax.scipy.special import gammaln

    jax.config.update("jax_enable_x64", True)
    Y, Xj, O = jnp.asarray(y, float), jnp.asarray(X, float), jnp.asarray(offset, float)
    W = jnp.ones(len(y)) if weight is None else jnp.asarray(weight, float)
    sw = jnp.sum(W)

    def nll(q):
        mu = jnp.exp(jnp.clip(Xj @ q["b"] + O, -30.0, 30.0))
        r = jnp.exp(-q["la"])
        ll = (gammaln(Y + r) - gammaln(r) - gammaln(Y + 1.0)
              + r * (jnp.log(r) - jnp.log(r + mu)) + Y * (jnp.log(mu) - jnp.log(r + mu)))
        return -jnp.sum(W * ll) / sw

    if beta0 is None:
        beta0 = np.zeros(X.shape[1])
        m = np.asarray(W) > 0
        beta0[0] = np.log(np.mean(np.asarray(y)[m]) + 1e-9) - np.mean(np.asarray(offset)[m])
    q = {"b": jnp.asarray(beta0, float), "la": jnp.asarray(float(log_alpha0))}
    vg = jax.value_and_grad(nll)
    opt = optax.lbfgs()
    st = opt.init(q)

    @jax.jit
    def step(q, st):
        v, g = vg(q)
        up, st = opt.update(g, st, q, value=v, grad=g, value_fn=nll)
        return optax.apply_updates(q, up), st, v, g

    prev, conv = np.inf, False
    for it in range(maxiter):
        q, st, v, g = step(q, st)
        v = float(v)
        gi = float(jnp.max(jnp.abs(jax.flatten_util.ravel_pytree(g)[0])))
        if np.isfinite(prev) and abs(prev - v) < rel_tol * abs(prev) and gi < grad_tol:
            conv = True
            break
        prev = v
    b = np.asarray(q["b"])
    return dict(beta=b, alpha=float(np.exp(q["la"])), converged=conv, iters=it + 1,
                grad_inf=gi, nll_per_cell=v,
                mu=np.exp(np.clip(np.asarray(X) @ b + np.asarray(offset), -30, 30)))


def fit_nb_additive_null(y, X, offset, m, weight=None, beta0=None, eta0=None,
                         log_alpha0=0.0):
    """The MisTIC technical null (MisTIC spec 3.2): NB2 with mean

        mu_i = exp(x_i'b + offset_i) + theta m_i,   theta = exp(eta)

    -- with X = 1 and offset = log Nt, `Nt_i e^b0 + theta m_i`. The additive
    term is not a log-link GLM, so the NB negative log-likelihood is written out
    and minimised with scipy's L-BFGS-B on analytic gradients
    (`additive_glm.fit_nb_additive`). Returns `fit_nb_offset`'s keys plus
    `eta`, `theta_m` and their standard errors.
    """
    from ..additive_glm import fit_nb_additive

    r = fit_nb_additive(y, X, offset, m, weight=weight, beta0=beta0, eta0=eta0,
                        log_alpha0=log_alpha0)
    se = dict(zip(r["names"], r["se"]))
    return dict(beta=r["beta"], alpha=r["alpha"], eta=r["eta"], theta_m=r["theta"],
                theta_m_at_floor=r["theta_at_floor"], se=se,
                converged=r["converged"], iters=r["n_iter"], grad_inf=r["grad_inf_per_cell"],
                nll_per_cell=r["nll_per_cell"], mu=r["mu"])


def robust_null(y, X, offset, trim_p=TRIM_P, tol_frac=0.001, max_iter=20, say=None, m=None):
    """Iteratively trimmed technical null (spec 2.1).

    Fit on all cells; flag `P(Y >= y_i) < trim_p`; refit on the unflagged; stop
    when the flagged set changes by fewer than `tol_frac` of cells, or after
    `max_iter` fits. Returns the final fit and the trace.

    `m` given: the MisTIC null, mean `exp(Xb + offset) + theta m`
    (`fit_nb_additive_null`), trimmed the same way.
    """
    say = say or (lambda *a: None)
    n = len(y)
    keep = np.ones(n, bool)
    prev, trace, fit = None, [], None
    stopped = "max_iter"
    for it in range(max_iter):
        if m is None:
            fit = fit_nb_offset(y, X, offset, weight=keep.astype(float),
                                beta0=None if fit is None else fit["beta"],
                                log_alpha0=0.0 if fit is None else np.log(fit["alpha"]))
        else:
            fit = fit_nb_additive_null(y, X, offset, m, weight=keep.astype(float),
                                       beta0=None if fit is None else fit["beta"],
                                       eta0=None if fit is None else fit["eta"],
                                       log_alpha0=0.0 if fit is None else np.log(fit["alpha"]))
        p = nb_p_ge(y, fit["mu"], fit["alpha"])
        flagged = p < trim_p
        changed = int(np.sum(flagged != prev)) if prev is not None else int(flagged.sum())
        trace.append(dict(iteration=it + 1, n_fit=int(keep.sum()), alpha0=fit["alpha"],
                          intercept=float(fit["beta"][0]), flagged=int(flagged.sum()),
                          flagged_frac=float(flagged.mean()), changed=changed,
                          converged=fit["converged"], grad_inf=fit["grad_inf"],
                          **({"theta_m": fit["theta_m"]} if m is not None else {})))
        say(f"    trim {it + 1:2d}: fit on {keep.sum():,} cells, alpha0 {fit['alpha']:.4f}"
            + (f", theta_m {fit['theta_m']:.4f}" if m is not None else "")
            + f", flagged {flagged.sum():,} ({100 * flagged.mean():.3f}%), changed {changed:,}")
        if prev is not None and changed < tol_frac * n:
            stopped = "tolerance"
            break
        prev = flagged
        keep = ~flagged
    fit["trace"] = trace
    fit["stopped_on"] = stopped
    return fit


# --------------------------------------------------------------------------
# 2.2 BASE-T
# --------------------------------------------------------------------------
def base_t(y, mu0, alpha0, alpha_b):
    """Tail-probability label. Returns (L, c, alpha_i, p)."""
    p = nb_p_ge(y, mu0, alpha0)
    c = smallest_c_below(mu0, alpha0, alpha_b, family="nb")
    alpha_i = nb_p_ge(c, mu0, alpha0)
    L = p < alpha_b
    return L, c, alpha_i, p


# --------------------------------------------------------------------------
# 2.3 BASE-O
# --------------------------------------------------------------------------
def base_o_scale(r, groups=None):
    """`s+`: SD of the residual over cells above the floor (r > 0).

    `groups` = cell type gives the primary BASE-O, matching OSDR's code:
    `tdm.preprocess.ki67.is_dividing` (github.com/JonathanSomer/osdr, commit
    7d04970) sets one cutoff per cell type, `Td * sigma_type + Tn`, with
    `sigma_type = np.std(x[x > Tn] - Tn)` -- population SD (ddof 0), computed
    within the type. `groups=None` is the global variant (BASE-O-global).
    Returns a per-cell array.
    """
    pos = r > 0
    if groups is None:
        return np.full(len(r), float(np.std(r[pos])))
    out = np.empty(len(r))
    for g in np.unique(groups):
        m = groups == g
        mp = m & pos
        out[m] = float(np.std(r[mp])) if mp.sum() > 1 else np.nan
    return out


def base_o(y, b, sd, s_plus, td):
    """OSDR-faithful label at threshold `td`. Returns (L, c, a).

    `a > td` is `(y - b) / sd > td * s+`, i.e. `y > b + td * s+ * sd`, so the
    discrete equivalent is `c = floor(b + td * s+ * sd) + 1` (V10). The label is
    computed from the same threshold so the two cannot disagree by rounding.
    """
    r = (y - b) / sd
    a = r / s_plus
    thr = b + td * s_plus * sd
    L = y > thr
    c = (np.floor(thr) + 1).astype(np.int64)
    return L, c, a


# --------------------------------------------------------------------------
# 2.4 SCORE
# --------------------------------------------------------------------------
def module_score(counts, genes, N, tirosh, blocklist, seed=0):
    """Seurat/Tirosh module score via scanpy `score_genes` (spec 2.4).

    Normalisation `log1p(X * s / N)`, `s = median N`, with `N` the library size
    without the blocklist (which contains the Tirosh genes), as the mixture
    uses. Control genes come from the panel minus the blocklist.

    scanpy 1.12 bins genes by indexing the pool WITH the scored genes, so a pool
    that excludes them raises a KeyError. The scored genes are therefore kept in
    the pool for binning only, and `ctrl_as_ref=False` removes them from the
    control draw before sampling -- so no Tirosh or blocklist gene can be a
    control, and each bin still gets a full 50 controls.
    """
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp

    X = counts.tocsr().astype(np.float32) if sp.issparse(counts) else sp.csr_matrix(counts)
    s = float(np.median(N))
    scale = (s / np.asarray(N, float)).astype(np.float32)
    X = sp.diags(scale) @ X
    X.data = np.log1p(X.data)
    a = ad.AnnData(X=X.tocsr())
    a.var_names = list(genes)
    block = set(blocklist) | set(tirosh)
    pool = [g for g in genes if g not in block] + [g for g in tirosh if g in set(genes)]
    sc.tl.score_genes(a, gene_list=list(tirosh), gene_pool=pool, n_bins=25, ctrl_size=50,
                      ctrl_as_ref=False, random_state=seed, score_name="score")
    return a.obs["score"].to_numpy(float), dict(n_pool=len(pool) - len(tirosh),
                                                normaliser_s=s)


def score_threshold(score, p=SCORE_P):
    """Robust half-normal null: KDE mode, left-half RMS spread, upper tail p."""
    from scipy import stats
    from statsmodels.nonparametric.kde import KDEUnivariate

    kde = KDEUnivariate(np.asarray(score, float))
    kde.fit(kernel="gau", bw="scott", fft=True, gridsize=8192)
    mode = float(kde.support[np.argmax(kde.density)])
    left = score[score < mode]
    sigma = float(np.sqrt(np.mean((left - mode) ** 2)))
    thr = mode + sigma * float(stats.norm.isf(p))
    return dict(mode=mode, sigma=sigma, threshold=thr,
                p_upper=stats.norm.sf((score - mode) / sigma))


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


# --------------------------------------------------------------------------
# Null simulation (V2)
# --------------------------------------------------------------------------
def sim_nb(mu, alpha, rng):
    """Gamma-Poisson draws from NB2(mu, alpha)."""
    lam = rng.gamma(shape=1.0 / alpha, scale=alpha * np.asarray(mu, float))
    return rng.poisson(lam).astype(float)
