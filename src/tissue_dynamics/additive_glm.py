"""Count regressions with an additive term in the mean (MisTIC spec, section 3).

    mu_i = exp(x_i'b + offset_i) + theta * m_i,      theta = exp(eta)

`m_i` is the expected number of misassigned outcome transcripts in cell i
(`tissue_dynamics.mistic`). It adds to the count's mean on the natural scale,
so it cannot enter a log-link GLM. These fits write the negative
log-likelihood directly and minimise it with scipy's L-BFGS-B on analytic
gradients, as spec 3.2 asks. Poisson and NB2 (variance mu + alpha mu^2) share
the mean function.

theta = exp(eta) keeps the misassignment term non-negative. eta is bounded
below at ETA_MIN: if the data want no misassignment term at all, theta settles
at ~1e-13 rather than drifting off to -infinity, and the fit reports it.

Standard errors come from the observed information: the Hessian of the summed
negative log-likelihood, by central differences of the analytic gradient.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

ETA_MIN, ETA_MAX = -30.0, 10.0
LIN_CLIP = 30.0


def _prep(y, X, offset, m, weight):
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64).reshape(len(y), -1)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, np.float64)
    m = np.asarray(m, np.float64)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, np.float64)
    if np.any(m < 0) or not np.all(np.isfinite(m)):
        raise ValueError("m must be finite and non-negative")
    return y, X, off, m, w


def _mean(beta, eta, X, off, m):
    lam = np.exp(np.clip(X @ beta + off, -LIN_CLIP, LIN_CLIP))
    t = np.exp(eta) * m
    return lam, t, lam + t


def _start(y, X, off, w, beta0, eta0):
    if beta0 is None:
        beta0 = np.zeros(X.shape[1])
        sel = w > 0
        # the intercept (a constant, non-zero column) matched to the mean count
        const = np.flatnonzero((np.ptp(X[sel], axis=0) == 0) & (X[sel][0] != 0))
        if len(const):
            j = const[0]
            beta0[j] = (np.log(np.average(y[sel], weights=w[sel]) + 1e-9)
                        - np.average(off[sel], weights=w[sel])) / X[sel][0, j]
    return np.asarray(beta0, np.float64), float(0.0 if eta0 is None else eta0)


# --------------------------------------------------------------------------
# Poisson
# --------------------------------------------------------------------------
def poisson_nll_grad(theta_vec, y, X, off, m, w):
    """Weighted Poisson NLL (summed) and its gradient in (b, eta)."""
    k = X.shape[1]
    beta, eta = theta_vec[:k], theta_vec[k]
    lam, t, mu = _mean(beta, eta, X, off, m)
    nll = -np.sum(w * (y * np.log(mu) - mu - gammaln(y + 1.0)))
    d_mu = w * (y / mu - 1.0)                       # dl/dmu
    g = np.concatenate([X.T @ (d_mu * lam), [np.sum(d_mu * t)]])
    return nll, -g


def fit_poisson_additive(y, X, offset, m, weight=None, beta0=None, eta0=None,
                         maxiter=5000, se=True):
    """Poisson with mean exp(Xb + offset) + exp(eta) m. Returns a dict."""
    y, X, off, m, w = _prep(y, X, offset, m, weight)
    b0, e0 = _start(y, X, off, w, beta0, eta0)
    return _fit(poisson_nll_grad, np.concatenate([b0, [e0]]), (y, X, off, m, w),
                X.shape[1], has_alpha=False, maxiter=maxiter, se=se)


# --------------------------------------------------------------------------
# Negative binomial (NB2)
# --------------------------------------------------------------------------
def nb_nll_grad(theta_vec, y, X, off, m, w):
    """Weighted NB2 NLL (summed) and its gradient in (b, eta, log alpha)."""
    k = X.shape[1]
    beta, eta, la = theta_vec[:k], theta_vec[k], theta_vec[k + 1]
    lam, t, mu = _mean(beta, eta, X, off, m)
    r = np.exp(-la)
    lrm = np.log(r + mu)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + r * (np.log(r) - lrm) + y * (np.log(mu) - lrm))
    nll = -np.sum(w * ll)
    d_mu = w * (y / mu - (y + r) / (r + mu))
    d_r = w * (digamma(y + r) - digamma(r) + np.log(r) - lrm + 1.0 - (y + r) / (r + mu))
    g = np.concatenate([X.T @ (d_mu * lam), [np.sum(d_mu * t)], [np.sum(d_r) * (-r)]])
    return nll, -g


def fit_nb_additive(y, X, offset, m, weight=None, beta0=None, eta0=None, log_alpha0=0.0,
                    maxiter=5000, se=True):
    """NB2 with mean exp(Xb + offset) + exp(eta) m. Returns a dict."""
    y, X, off, m, w = _prep(y, X, offset, m, weight)
    b0, e0 = _start(y, X, off, w, beta0, eta0)
    return _fit(nb_nll_grad, np.concatenate([b0, [e0, float(log_alpha0)]]), (y, X, off, m, w),
                X.shape[1], has_alpha=True, maxiter=maxiter, se=se)


# --------------------------------------------------------------------------
def _fit(fg, x0, args, k, has_alpha, maxiter, se):
    w = args[-1]
    sw = float(np.sum(w))

    def f(x):                                       # per-cell scale for L-BFGS
        v, g = fg(x, *args)
        return v / sw, g / sw

    bounds = [(None, None)] * k + [(ETA_MIN, ETA_MAX)] + ([(-25.0, 10.0)] if has_alpha else [])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": int(maxiter), "maxcor": 30, "ftol": 1e-16,
                            "gtol": 1e-11, "maxls": 50})
    x = res.x
    nll, g = fg(x, *args)
    # the eta bound is the only one that can bind; its gradient is excluded there
    free = np.ones(len(x), bool)
    at_floor = x[k] <= ETA_MIN + 1e-9
    if at_floor:
        free[k] = False
    out = {"beta": x[:k], "eta": float(x[k]), "theta": float(np.exp(x[k])),
           "theta_at_floor": bool(at_floor),
           "alpha": float(np.exp(x[k + 1])) if has_alpha else 0.0,
           "nll": float(nll), "nll_per_cell": float(nll / sw),
           "grad_inf": float(np.max(np.abs(g[free]))) if free.any() else 0.0,
           "grad_inf_per_cell": float(np.max(np.abs(g[free])) / sw) if free.any() else 0.0,
           "converged": bool(res.success), "message": str(res.message), "n_iter": int(res.nit)}
    y, X, off, m, _w = args
    out["lam"], out["t"], out["mu"] = _mean(x[:k], x[k], X, off, m)
    if se:
        H = hessian_fd(lambda z: fg(z, *args)[1], x)
        cov = np.full_like(H, np.nan)
        try:
            cov[np.ix_(free, free)] = np.linalg.inv(H[np.ix_(free, free)])
        except np.linalg.LinAlgError:
            pass
        out["cov"] = cov
        out["se"] = np.sqrt(np.clip(np.diag(cov), 0, None))
        names = [f"b{j}" for j in range(k)] + ["eta"] + (["log_alpha"] if has_alpha else [])
        out["names"] = names
    return out


def hessian_fd(grad, x, rel=1e-5):
    """Hessian by central differences of an analytic gradient, symmetrised."""
    x = np.asarray(x, np.float64)
    H = np.empty((len(x), len(x)))
    for j in range(len(x)):
        h = rel * max(1.0, abs(x[j]))
        e = np.zeros(len(x))
        e[j] = h
        H[:, j] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)


def sim_nb(mu, alpha, rng):
    """Gamma-Poisson draws from NB2(mu, alpha); alpha = 0 is Poisson."""
    mu = np.asarray(mu, float)
    if alpha <= 0:
        return rng.poisson(mu).astype(float)
    return rng.poisson(rng.gamma(shape=1.0 / alpha, scale=alpha * mu)).astype(float)
