"""Label constructions (spec section 2): technical null, BASE-T, BASE-O, SCORE.

Labels are TARGET DEFINITIONS. They are fitted once on all analysis cells and
frozen before any model is fitted (spec 2.1); nothing here sees a model.
"""

from __future__ import annotations

import os

import numpy as np

from .survival import nb_p_ge, smallest_c_below

ALPHA_B_GRID = (0.05, 0.01, 0.001, 1e-4)
ALPHA_B_PRIMARY = 0.05     # decided by the user after Stage 1 (spec default 0.001)
# addendum 1, section 2.1: the grid every BASE-T null is built on from now on
ALPHA_B_GRID_EXT = (0.001, 0.01, 0.05, 0.1, 0.2)
TRUNC_GRID = (0.001, 0.01, 0.05)                  # addendum 1, section 1.3: alpha_trim

# Label variants (DIV_VARIANT; outputs in outputs/division_classifier_<variant>, see
# paths.out_dir). Unset = the labels above, as Stages 1-7 were first run.
#   t10  user, 2026-10-17: BASE-T from the alpha_trim 0.2 truncated null
#        (truncated_null from the untrimmed fit, param_tol 1e-3, as the calibration
#        fitted it) at the alpha_B that labels 10% of the breast section's cells.
#        Both calibrations support alpha_trim ~0.2; 0.1085 is the 10% level (report.md,
#        "BASE-T-trunc calibrated ..."). BASE-O and SCORE are unchanged.
VARIANTS = {"t10": dict(trunc=0.2, alpha_b=0.1085, grid=(0.1085, 0.05, 0.01, 0.001),
                        dataset=None)}
VARIANT = os.environ.get("DIV_VARIANT") or None
BASE_T_TRUNC = None            # alpha_trim of BASE-T's null; None = Stage 1's naive null
TRUNC_PARAM_TOL = 1e-3
if VARIANT is not None:
    if VARIANT not in VARIANTS:
        raise ValueError(f"unknown DIV_VARIANT {VARIANT!r}; known: {sorted(VARIANTS)}")
    if (os.environ.get("MIX_DATASET") or None) != VARIANTS[VARIANT]["dataset"]:
        raise ValueError(f"DIV_VARIANT {VARIANT} is calibrated on the "
                         f"{VARIANTS[VARIANT]['dataset'] or 'breast'} section only")
    ALPHA_B_PRIMARY = VARIANTS[VARIANT]["alpha_b"]
    ALPHA_B_GRID = VARIANTS[VARIANT]["grid"]
    BASE_T_TRUNC = VARIANTS[VARIANT]["trunc"]
TD_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TD_PRIMARY = 0.5
TRIM_P = 0.001
SCORE_P = 0.001


def ab_tag(a):
    return {0.2: "a2", 0.1085: "a1085", 0.1: "a1", 0.05: "a05", 0.01: "a01", 0.001: "a001",
            1e-4: "a0001"}[float(a)]


def trunc_tag(a):
    """`LTtr{trunc_tag}_{ab_tag}`, e.g. LTtr05_a1 (addendum 1, section 1.3)."""
    return {0.05: "05", 0.01: "01", 0.001: "001"}[float(a)]


def td_tag(t):
    return f"td{int(round(100 * t)):03d}"


# --------------------------------------------------------------------------
# Cell type
# --------------------------------------------------------------------------
def cell_type(df, phi_names, data_cfg=None):
    """Cell type from the table's PhiSpace scores, by the section's typing rule
    (`mixture.data.phi_cell_types`; spec 4.3).

    On breast that is the argmax of column-standardised scores -- never the raw
    argmax: the nine columns differ ~18-fold in scale, and a raw argmax matches
    the stored label for only 36.6% of cells. `data_cfg` defaults to the
    current configuration's `data` section.
    """
    from ..mixture.data import load_config, phi_cell_types

    cols = sorted(c for c in df.columns if c.startswith("phi_") and c[4:].isdigit())
    dc = load_config()["data"] if data_cfg is None else data_cfg
    return phi_cell_types(df[cols].to_numpy(float), phi_names, dc)


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
# Addendum 1, section 1: the truncated-likelihood null (BASE-T-trunc)
# --------------------------------------------------------------------------
def nb_trunc_nll(theta, y, X, offset, m, c, w, k_max):
    """Negative right-truncated NB2 log-likelihood (summed, weighted by `w`).

    Mean `exp(Xb + offset) + exp(eta) m`, dispersion `alpha = exp(la)`;
    `theta = [b..., eta, la]`. Each cell with `w = 1` contributes
    `log f(y) - log F(c - 1)`: its count is known to lie below `c`. F is
    accumulated by the pmf recursion f(k+1) = f(k) (k + r) p / (k + 1), up to
    `k_max` = the largest `c` among the weighted cells, so memory stays O(n).
    A JAX function; gradients by forward mode (`trunc_value_and_grad`).
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.special import gammaln

    k = X.shape[1]
    b, eta, la = theta[:k], theta[k], theta[k + 1]
    mu = jnp.exp(jnp.clip(X @ b + offset, -30.0, 30.0)) + jnp.exp(eta) * m
    r = jnp.exp(-la)
    lrm = jnp.log(r + mu)
    ll_y = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * (jnp.log(r) - lrm) + y * (jnp.log(mu) - lrm))
    logp = jnp.log(mu) - lrm

    def body(j, carry):
        logf, logF = carry
        logF = jnp.where(j < c, jnp.logaddexp(logF, logf), logF)
        return logf + jnp.log(j + r) - jnp.log(j + 1.0) + logp, logF

    _, logF = lax.fori_loop(0, k_max, body, (r * (jnp.log(r) - lrm), jnp.full_like(mu, -jnp.inf)))
    return -jnp.sum(w * (ll_y - logF))


def fit_nb_truncated(y, X, offset, m, c, keep, x0, maxiter=2000):
    """Maximise the right-truncated NB likelihood over the cells in `keep`
    (all with y < c), holding `c` fixed. `x0 = [b..., eta, log alpha]`."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from ..additive_glm import ETA_MAX, ETA_MIN

    jax.config.update("jax_enable_x64", True)
    keep = np.asarray(keep, bool)
    cc = np.where(keep, np.asarray(c, float), 1.0)
    k_max = int(cc[keep].max())
    args = [jnp.asarray(a, float) for a in (y, X, offset, m, cc, keep.astype(float))]
    sw = float(keep.sum())

    def f(t):
        v = nb_trunc_nll(t, *args, k_max) / sw
        return v, v

    jf = jax.jit(jax.jacfwd(f, has_aux=True))     # forward mode: 3 parameters, O(n) memory

    def fg(t):
        g, v = jf(jnp.asarray(t))
        return float(v), np.asarray(g, float)

    kx = X.shape[1]
    bounds = [(None, None)] * kx + [(ETA_MIN, ETA_MAX), (-25.0, 10.0)]
    res = minimize(fg, np.asarray(x0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-16, "gtol": 1e-11,
                            "maxls": 50})
    x = res.x
    v, g = fg(x)
    free = np.ones(len(x), bool)
    free[kx] = x[kx] > ETA_MIN + 1e-9
    mu = np.exp(np.clip(np.asarray(X) @ x[:kx] + offset, -30, 30)) + np.exp(x[kx]) * m
    # standard errors on (b, eta, log alpha) from the exact Hessian (forward over forward)
    H = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(
        lambda t: nb_trunc_nll(t, *args, k_max))))(jnp.asarray(x)), float)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(len(x), np.nan)
    return dict(beta=x[:kx], eta=float(x[kx]), theta_m=float(np.exp(x[kx])),
                alpha=float(np.exp(x[kx + 1])), nll_per_cell=v,
                se={"beta0": float(se[0]), "eta": float(se[kx]), "log_alpha": float(se[kx + 1])},
                grad_inf=float(np.max(np.abs(g[free]))), converged=bool(res.success),
                n_iter=int(res.nit), mu=mu, k_max=k_max, x=x)


def truncated_null(y, X, offset, m, alpha_trim, start, max_iter=20, say=None, param_tol=None):
    """Addendum 1, section 1.2: the iteratively truncated technical null.

    From `start` (a null fit: beta, eta, alpha), repeat: c_i = smallest count
    with P(Y >= c | previous fit) < alpha_trim; retain y_i < c_i; maximise the
    right-truncated likelihood over the retained cells with c fixed. Stop when
    no c_i changes, or after `max_iter` iterations. Returns the final fit,
    its `c`, and the trace.

    `param_tol` given: also stop once no parameter (b, log theta_m, log alpha)
    moves by more than `param_tol` between iterations. At large alpha_trim a
    few dozen boundary cells can flip back and forth indefinitely after the
    parameters have settled (A1: alpha_trim 0.05 ran to the cap). The default
    (None) keeps the A1 rule.
    """
    say = say or (lambda *a: None)
    y = np.asarray(y, float)
    x = np.concatenate([np.asarray(start["beta"], float),
                        [float(start["eta"]), np.log(float(start["alpha"]))]])
    kx = X.shape[1]
    mu = np.exp(np.clip(X @ x[:kx] + offset, -30, 30)) + np.exp(x[kx]) * m
    alpha = float(np.exp(x[kx + 1]))
    c_prev, trace, fit, stopped = None, [], None, "max_iter"
    x_prev = x.copy()
    for it in range(1, max_iter + 1):
        c = smallest_c_below(mu, alpha, alpha_trim, family="nb")
        if c_prev is not None and np.array_equal(c, c_prev):
            stopped = "thresholds unchanged"
            break
        keep = y < c
        fit = fit_nb_truncated(y, X, offset, m, c, keep, x)
        x, mu, alpha = fit["x"], fit["mu"], fit["alpha"]
        changed = int(np.sum(c != c_prev)) if c_prev is not None else int(len(c))
        trace.append(dict(iteration=it, beta0=float(fit["beta"][0]), theta_m=fit["theta_m"],
                          alpha0=fit["alpha"], n_retained=int(keep.sum()),
                          trimmed_frac=float(1 - keep.mean()), c_changed=changed,
                          converged=fit["converged"], grad_inf=fit["grad_inf"],
                          nll_per_cell=fit["nll_per_cell"]))
        say(f"    trunc {alpha_trim:g} it {it:2d}: retained {keep.sum():,} "
            f"({100 * (1 - keep.mean()):.2f}% trimmed), alpha0 {fit['alpha']:.4f}, "
            f"theta_m {fit['theta_m']:.4f}, b0 {fit['beta'][0]:+.4f}, c changed {changed:,}")
        c_prev = c
        if param_tol is not None and len(trace) > 1 and np.max(np.abs(x - x_prev)) < param_tol:
            stopped = "parameters settled"
            break
        x_prev = x.copy()
    fit["c_trim"] = c_prev
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


def gamma_glm_log(y, X, max_iter=200, tol=1e-13):
    """Gamma GLM with log link by Fisher scoring. For the gamma family with a
    log link the working weights are constant, so each step is an ordinary
    least-squares fit of the working response eta + (y - mu) / mu on X.
    Returns the coefficients. `y` must be positive."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta = np.linalg.lstsq(X, np.full(len(y), np.log(y.mean())), rcond=None)[0]
    dev_prev = np.inf
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        beta = np.linalg.lstsq(X, eta + (y - mu) / mu, rcond=None)[0]
        mu = np.exp(X @ beta)
        dev = 2 * np.sum((y - mu) / mu - np.log(y / mu))
        if abs(dev_prev - dev) <= tol * max(abs(dev), 1.0):
            break
        dev_prev = dev
    return beta


def base_o_state_scale(r, X):
    """BASE-O-state's s+(phi) (addendum 1, section 4.2), per cell.

    On the cells with r > 0: a gamma GLM with log link for the mean,
    E[r | r > 0, phi] = m1(phi), then a second for the squared residuals,
    E[(r - m1(phi))^2 | r > 0, phi] = v(phi); s+ = sqrt(v(phi)) at every cell.
    `X` is the design (an intercept plus the projected own state phi; or
    one-hot type indicators, with which it reduces to OSDR's per-type ddof-0
    SD of the above-noise values -- V14). Returns (s+, info).
    """
    r = np.asarray(r, float)
    X = np.asarray(X, float)
    pos = r > 0
    b1 = gamma_glm_log(r[pos], X[pos])
    res2 = (r[pos] - np.exp(X[pos] @ b1)) ** 2
    b2 = gamma_glm_log(res2, X[pos])
    return np.sqrt(np.exp(X @ b2)), {"beta_mean": b1, "beta_var": b2, "n_pos": int(pos.sum())}


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
