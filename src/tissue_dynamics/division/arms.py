"""Model arms for the division classifier comparison (spec sections 1 and 3).

Every arm shares one weight block, `eta_state` (spec 1, "shared weight block").
It is written once here and checked against the production mixture's
`_logit_pi` to machine precision (V6), so the arms cannot drift apart.

    eta(x) = phi'a + phibar'c + sum_r (phi'U_r)(phibar'V_r) [+ psi ox] + H'h + phi'K H

The oxygen term is present only when `psi` is in the parameters, matching the
mixture's `model.oxygen` switch (off in production since 2026-10-06).

The intercept is NOT part of `eta_state`: in the NB it is not separately
identified from the technical intercept, and each arm adds exactly one.

Penalty parity (spec 1): the arms carry the production mixture's penalty term
for term -- `weight_ridge * n / n_reference` on every eta coefficient (B through
`model.b_penalty_terms`), plus its conditioning term weighted by `ridge_tau`.
"""

from __future__ import annotations

import time

import numpy as np

ETA_BLOCKS = ("a", "c", "U", "V", "psi", "h", "K")


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


# --------------------------------------------------------------------------
# Shared weight block
# --------------------------------------------------------------------------
def eta_state(p, D):
    import jax.numpy as jnp

    eta = (D["own"] @ p["a"] + D["nbr"] @ p["c"]
           + jnp.sum((D["own"] @ p["U"]) * (D["nbr"] @ p["V"]), axis=-1)
           + D["hbasis"] @ p["h"]
           + jnp.einsum("ij,jm,im->i", D["own"], p["K"], D["hbasis"]))
    return eta + p["psi"] * D["oxygen"] if "psi" in p else eta


def init_eta(d_own, d_nbr, n_h, rank, rng, oxygen=False):
    """Zero start, except U and V: `U = V = 0` is a saddle (each block's
    gradient vanishes there), so they start at small random values."""
    import jax.numpy as jnp

    p = {"a": jnp.zeros(d_own), "c": jnp.zeros(d_nbr),
         "U": jnp.asarray(rng.normal(0, 0.01, (d_own, rank))),
         "V": jnp.asarray(rng.normal(0, 0.01, (d_nbr, rank))),
         "h": jnp.zeros(n_h), "K": jnp.zeros((d_own, n_h))}
    if oxygen:
        p["psi"] = jnp.asarray(0.0)
    return p


def eta_column_names(p):
    """Ordered names of every eta coefficient, for the V6 parity audit."""
    out = []
    for b in ETA_BLOCKS:
        if b not in p:
            continue
        shape = np.shape(np.asarray(p[b]))
        if shape == ():
            out.append(b)
        else:
            out += [f"{b}[{','.join(map(str, ix))}]" for ix in np.ndindex(*shape)]
    return out


def penalty(p, n, weight_ridge, n_reference, ridge_tau, b_penalty="frobenius"):
    """The production mixture's penalty, term for term (model.loglik)."""
    import jax.numpy as jnp
    from types import SimpleNamespace

    from ..mixture.model import b_penalty_terms

    b_pen, cond = b_penalty_terms(p, SimpleNamespace(b_penalty=b_penalty))
    pen = (jnp.sum(p["a"] ** 2) + jnp.sum(p["c"] ** 2) + b_pen
           + jnp.sum(p["h"] ** 2) + jnp.sum(p["K"] ** 2))
    if "psi" in p:
        pen = pen + p["psi"] ** 2
    lam = weight_ridge * (n / n_reference) if n_reference > 0 else weight_ridge
    return ridge_tau * cond + lam * pen


# --------------------------------------------------------------------------
# NB arms
# --------------------------------------------------------------------------
def nb_log_mu(p, D):
    import jax.numpy as jnp

    theta = p["theta"] if "theta" in p else 1.0          # absent = offset (F)
    eta = p["b0"] + theta * D["logN"] + D["area"] @ p["s"] + eta_state(p, D)
    return jnp.clip(eta, -30.0, 30.0)


def nb_log_alpha(p, D):
    """NB-c: one constant. NB-d: `gamma0 + phi' gamma`, the earlier dispersion
    submodel, on the 8-dim projected own state (decided by the user)."""
    if "gamma" in p:
        return p["la"] + D["own"] @ p["gamma"]
    return p["la"]


def nb_logpmf(y, log_mu, log_alpha):
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    mu = jnp.exp(log_mu)
    r = jnp.exp(-log_alpha)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * (jnp.log(r) - jnp.log(r + mu)) + y * (log_mu - jnp.log(r + mu)))


def init_nb(D, dispersion, rank, seed, oxygen=None, theta_free=True):
    """Starting values. `oxygen=None` includes psi only if `D` carries an
    oxygen column; `theta_free=False` leaves theta out (log N an offset)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    y = np.asarray(D["y"])
    logN = np.asarray(D["logN"])
    oxygen = ("oxygen" in D) if oxygen is None else bool(oxygen)
    p = init_eta(D["own"].shape[1], D["nbr"].shape[1], D["hbasis"].shape[1], rank, rng,
                 oxygen=oxygen)
    p.update({"b0": jnp.asarray(float(np.log(y.mean() + 1e-9) - logN.mean())),
              "s": jnp.zeros(D["area"].shape[1]), "la": jnp.asarray(0.0)})
    if theta_free:
        p["theta"] = jnp.asarray(1.0)
    if dispersion == "state":
        p["gamma"] = jnp.zeros(D["own"].shape[1])
    return p


def nb_objective(D, n, weight_ridge, n_reference, ridge_tau, b_penalty="frobenius"):
    """Penalised negative log-likelihood, summed -- the scale the mixture's
    convergence criteria are stated on."""
    import jax.numpy as jnp

    def f(p):
        ll = jnp.sum(nb_logpmf(D["y"], nb_log_mu(p, D), nb_log_alpha(p, D)))
        return -ll + penalty(p, n, weight_ridge, n_reference, ridge_tau, b_penalty)
    return f


# --------------------------------------------------------------------------
# LOG arms: logit P(L = 1) = b0 + theta log N + s(log A) + eta_state (spec 1)
# --------------------------------------------------------------------------
def logistic_eta(p, D):
    """The LOG arms' linear predictor: the technical block g(log N, log A) with
    a free linear log N term, plus the shared weight block."""
    return p["b0"] + p["theta"] * D["logN"] + D["area"] @ p["s"] + eta_state(p, D)


def logistic_objective(D, lab, n, weight_ridge, n_reference, ridge_tau, b_penalty="frobenius"):
    """Penalised negative log-likelihood of a LOG arm, the mixture's penalty term
    for term (as run/division_stage3_oof.py fits them)."""
    import jax
    import jax.numpy as jnp

    def f(p):
        eta = logistic_eta(p, D)
        ll = jnp.sum(lab * eta - jax.nn.softplus(eta))
        return -ll + penalty(p, n, weight_ridge, n_reference, ridge_tau, b_penalty)
    return f


def init_logistic(D, lab, rank, seed):
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    p = init_eta(D["own"].shape[1], D["nbr"].shape[1], D["hbasis"].shape[1], rank, rng)
    m = float(np.clip(np.mean(np.asarray(lab)), 1e-6, 1 - 1e-6))
    p.update({"b0": jnp.asarray(np.log(m / (1 - m))), "theta": jnp.asarray(0.0),
              "s": jnp.zeros(D["area"].shape[1])})
    return p


# --------------------------------------------------------------------------
# Optimiser: L-BFGS, then Newton on the exact Hessian
# --------------------------------------------------------------------------
def minimise(f, p0, rel_tol=1e-9, grad_tol=1e-5, lbfgs_maxiter=6000, newton_maxiter=40,
             hess_chunk=8, say=None):
    """Minimise `f` over the pytree `p0` to V1's tolerances.

    L-BFGS does the global work. It stalls short of grad_tol on these problems,
    in part because `B = U V'` is invariant to rotating U and V together, which
    leaves an exactly flat direction. The finish is therefore Newton on the
    exact Hessian (built in chunks of Hessian-vector products, to bound
    memory), with eigenvalues replaced by their absolute values floored at
    1e-12 x the largest -- so flat and negative-curvature directions cannot
    produce wild steps -- and a backtracking line search.

    Converged means BOTH `|f_t - f_{t-1}| < rel_tol |f_{t-1}|` and
    `||grad||_inf < grad_tol`, as V1 and the mixture's fits require.
    """
    jax = _jax()
    import jax.numpy as jnp
    import optax
    from jax.flatten_util import ravel_pytree

    say = say or (lambda *a: None)
    t0 = time.time()
    x0, unravel = ravel_pytree(p0)
    fx = jax.jit(lambda x: f(unravel(x)))
    vg = jax.jit(jax.value_and_grad(lambda x: f(unravel(x))))

    # ---- L-BFGS ---------------------------------------------------------
    opt = optax.lbfgs()
    st = opt.init(x0)
    fun = lambda x: f(unravel(x))                          # noqa: E731

    @jax.jit
    def step(x, st):
        v, g = jax.value_and_grad(fun)(x)
        up, st = opt.update(g, st, x, value=v, grad=g, value_fn=fun)
        return optax.apply_updates(x, up), st, v, g

    x, prev, stall, it = x0, np.inf, 0, 0
    for it in range(lbfgs_maxiter):
        x, st, v, g = step(x, st)
        v = float(v)
        gi = float(jnp.max(jnp.abs(g)))
        if not np.isfinite(v):
            break
        rel = abs(prev - v) / max(abs(prev), 1e-300) if np.isfinite(prev) else np.inf
        if rel < rel_tol and gi < grad_tol:
            break
        stall = stall + 1 if rel < 1e-12 else 0
        if stall >= 50:
            break
        prev = v
    n_lbfgs = it + 1
    say(f"      L-BFGS: {n_lbfgs} iterations, f {v:,.6f}, |g|inf {gi:.2e} "
        f"({time.time() - t0:.0f}s)")

    # ---- Newton polish --------------------------------------------------
    gfun = jax.grad(lambda x: f(unravel(x)))

    @jax.jit
    def hvp_rows(x, V):
        return jax.vmap(lambda v: jax.jvp(gfun, (x,), (v,))[1])(V)

    d = x.shape[0]
    eye = np.eye(d)
    hist, converged, min_eig, n_neg = [], False, np.nan, 0
    f_prev = float(fx(x))
    for k in range(newton_maxiter):
        fv, g = vg(x)
        fv, g = float(fv), np.asarray(g)
        gi = float(np.max(np.abs(g)))
        rel = abs(f_prev - fv) / abs(f_prev) if k > 0 else np.inf
        hist.append((k, fv, gi))
        if k > 0 and rel < rel_tol and gi < grad_tol:
            converged = True
            break
        H = np.vstack([np.asarray(hvp_rows(x, jnp.asarray(eye[i:i + hess_chunk])))
                       for i in range(0, d, hess_chunk)])
        H = 0.5 * (H + H.T)
        w, Q = np.linalg.eigh(H)
        min_eig, n_neg = float(w.min()), int(np.sum(w < -1e-8 * np.abs(w).max()))
        wf = np.maximum(np.abs(w), 1e-12 * np.abs(w).max())
        stp = -(Q @ ((Q.T @ g) / wf))
        slope = float(g @ stp)
        t, accepted = 1.0, False
        for _ in range(40):
            xn = x + t * jnp.asarray(stp)
            fn = float(fx(xn))
            if np.isfinite(fn) and fn <= fv + 1e-4 * t * slope:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        f_prev = fv
        x = xn
    fv, g = vg(x)
    gi = float(jnp.max(jnp.abs(g)))
    say(f"      Newton: {len(hist)} iterations, f {float(fv):,.6f}, |g|inf {gi:.2e}, "
        f"converged {converged}, min Hessian eigenvalue {min_eig:.3e} "
        f"({time.time() - t0:.0f}s total)")
    return unravel(x), dict(converged=bool(converged), f=float(fv), grad_inf=gi,
                            lbfgs_iters=n_lbfgs, newton_iters=len(hist),
                            newton_trace=hist, min_hessian_eig=min_eig,
                            n_negative_eigs=n_neg, wall_time=time.time() - t0)


# --------------------------------------------------------------------------
# Diagnostics (spec 1: report for every fit; flag sd(eta) > 20)
# --------------------------------------------------------------------------
def eta_diagnostics(p, D):
    U, V = np.asarray(p["U"]), np.asarray(p["V"])
    sig = np.linalg.svd(U @ V.T, compute_uv=False)
    eta = np.asarray(eta_state(p, D))
    return dict(norm_a=float(np.linalg.norm(np.asarray(p["a"]))),
                norm_c=float(np.linalg.norm(np.asarray(p["c"]))),
                sigma1_B=float(sig[0]), sd_eta=float(eta.std()),
                flag_sd_eta=bool(eta.std() > 20))
