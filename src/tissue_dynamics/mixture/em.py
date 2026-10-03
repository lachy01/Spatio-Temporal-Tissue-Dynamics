"""EM for Stages A and B (spec 4.3).

EM is kept as an *independent* check on the L-BFGS optimiser, not as the primary
estimator: it is monotone by construction and its M-steps are ordinary weighted
GLMs, so an agreement between the two is strong evidence that neither has a bug.
It applies only when `rho_i == 0` -- with the additive background the M-step is
no longer a standard GLM, which is why spec 5 restricts EM to Stages A and B.

Spec 4.3's closing instruction is treated as a hard assertion: if the
log-likelihood ever decreases by more than `tol`, `run_em` raises. That check is
described in the spec as the single most valuable bug detector in the build, and
it has to be able to fail loudly.
"""

from __future__ import annotations

import numpy as np

from .model import loglik, responsibilities


class MonotonicityError(RuntimeError):
    """The EM log-likelihood decreased -- an implementation bug by construction."""


def _weighted_irls(y, X, w, family, offset=None, beta0=None, n_iter=60, tol=1e-10):
    """Weighted IRLS for Poisson (log link) or binomial (logit link).

    `w` are observation weights (EM responsibilities); `y` may be fractional for
    the binomial case, which is exactly what the weight-model M-step needs.
    """
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    for _ in range(n_iter):
        eta = X @ beta + off
        if family == "poisson":
            mu = np.exp(np.clip(eta, -30, 30))
            W, z = w * mu, eta - off + (y - mu) / np.maximum(mu, 1e-12)
        else:
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            v = np.maximum(mu * (1 - mu), 1e-10)
            W, z = w * v, eta - off + (y - mu) / v
        A = X.T @ (X * W[:, None])
        b = X.T @ (W * z)
        A.flat[:: p + 1] += 1e-10                      # tiny ridge for conditioning
        new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def _component_mstep(data, spec, w, p):
    """M-step 1: one Poisson GLM on the data stacked twice (spec 4.3 step 1).

    Each cell appears with weight `1 - w_i` and class indicator 0, and again with
    weight `w_i` and indicator 1. The indicator's coefficient is `e^delta`.
    """
    y, logN, area = data.y, data.logN, data.area
    n = len(y)
    cols = [np.ones(n)]
    if spec.theta_free:
        cols.append(logN)
    cols.append(area)
    Xb = np.column_stack(cols)
    X = np.vstack([np.column_stack([Xb, np.zeros(n)]),
                   np.column_stack([Xb, np.ones(n)])])
    ys = np.concatenate([y, y])
    ws = np.concatenate([1.0 - w, w])
    off = None if spec.theta_free else np.concatenate([logN, logN])

    b0 = [float(p["b0"])]
    if spec.theta_free:
        b0.append(float(p["theta"]))
    b0 = np.concatenate([b0, np.asarray(p["s"], dtype=float),
                         [float(np.exp(p["delta"]))]])
    beta = _weighted_irls(ys, X, ws, "poisson", offset=off, beta0=b0)

    e_delta = float(beta[-1])
    out = dict(p)
    out["b0"] = np.asarray(beta[0])
    k = 1
    if spec.theta_free:
        out["theta"] = np.asarray(beta[1])
        k = 2
    out["s"] = np.asarray(beta[k:-1])
    # spec 4.3: e^delta must stay positive to pin component identity
    out["delta"] = np.asarray(np.log(max(e_delta, 1e-8)))
    return out, e_delta


def _weight_design(data, spec, p, block):
    """Weight-model design for the alternating bilinear M-step (spec 4.3 step 2).

    `block="U"` holds `V` fixed, so the bilinear term is linear in `U`; `block="V"`
    is the mirror. Returns `(X, names, unpack)`.
    """
    n = len(data.y)
    cols = [np.ones((n, 1))]
    sizes = [("a0", 1)]
    if spec.has_state:
        cols += [data.own, data.nbr]
        sizes += [("a", spec.d_own), ("c", spec.d_nbr)]
        if block == "U":
            NV = data.nbr @ np.asarray(p["V"])                       # (n, r)
            cols.append(np.concatenate(
                [data.own * NV[:, [k]] for k in range(spec.rank)], axis=1))
            sizes.append(("U", spec.d_own * spec.rank))
        else:
            OU = data.own @ np.asarray(p["U"])                       # (n, r)
            cols.append(np.concatenate(
                [data.nbr * OU[:, [k]] for k in range(spec.rank)], axis=1))
            sizes.append(("V", spec.d_nbr * spec.rank))
        cols += [data.hbasis, data.oxygen[:, None]]
        sizes += [("h", spec.df_weight), ("psi", 1)]
        if spec.density_interaction and spec.df_weight > 0:
            # still linear in the coefficients given the design, so it slots
            # straight into the logistic M-step alongside a and c
            if spec.density_interaction_spline:
                cols.append(np.concatenate(
                    [data.own * data.hbasis[:, [m]] for m in range(spec.df_weight)],
                    axis=1))
                sizes.append(("K", spec.d_own * spec.df_weight))
            else:
                cols.append(data.own * data.logdens[:, None])
                sizes.append(("k", spec.d_own))
    X = np.concatenate(cols, axis=1)

    def unpack(beta, p_in):
        out, i = dict(p_in), 0
        for name, k in sizes:
            v = beta[i:i + k]
            i += k
            if name == "a0":
                out["a0"] = np.asarray(v[0])
            elif name == "psi":
                out["psi"] = np.asarray(v[0])
            elif name == "U":
                out["U"] = np.asarray(v).reshape(spec.rank, spec.d_own).T
            elif name == "V":
                out["V"] = np.asarray(v).reshape(spec.rank, spec.d_nbr).T
            elif name == "K":
                out["K"] = np.asarray(v).reshape(spec.df_weight, spec.d_own).T
            else:
                out[name] = np.asarray(v)
        return out

    start = [np.atleast_1d(np.asarray(p["a0"], dtype=float))]
    if spec.has_state:
        start += [np.asarray(p["a"], float), np.asarray(p["c"], float)]
        start.append(np.asarray(p[block], float).T.ravel())
        start += [np.asarray(p["h"], float),
                  np.atleast_1d(np.asarray(p["psi"], dtype=float))]
        if spec.density_interaction and spec.df_weight > 0:
            start.append(np.asarray(
                p["K"], float).T.ravel() if spec.density_interaction_spline
                else np.asarray(p["k"], float))
    return X, np.concatenate(start), unpack


def _weight_mstep(data, spec, w, p, n_alt=2):
    """M-step 2: logistic regression of the fractional response `w` on the weight
    design, alternating `U` and `V` (spec 4.3 says two inner alternations suffice)."""
    out = dict(p)
    blocks = ("U", "V") if spec.has_state else ("U",)
    for _ in range(n_alt):
        for block in blocks:
            if not spec.has_state and block == "V":
                continue
            X, beta0, unpack = _weight_design(data, spec, out, block)
            beta = _weighted_irls(w, X, np.ones(len(w)), "binomial",
                                  beta0=beta0, n_iter=25)
            out = unpack(beta, out)
            if not spec.has_state:
                break
    return out


def run_em(data, spec, params0, max_iter=500, tol=1e-8, ll_tol=1e-13,
           verbose=False, strict=True, record=True):
    """Run EM to convergence. Returns `(params, history, info)`.

    Raises `MonotonicityError` if the log-likelihood ever falls by more than
    `tol` (spec 4.3). Set `strict=False` only to *observe* a violation in a test.
    `ll_tol` is the relative log-likelihood change at which EM stops.
    """
    if spec.has_background:
        raise ValueError("EM does not apply with the additive background "
                         "(spec 4.3); use fit_lbfgs for Stage C")
    if getattr(spec, "weight_ridge", 0.0) > 0:
        # The M-steps maximise the UNPENALISED Q: `_weighted_irls` carries only a
        # 1e-10 diagonal for conditioning, and nothing adds `weight_ridge` to the
        # normal equations. `run_em` then checks monotonicity of the PENALISED
        # log-likelihood, so a penalised spec would make the guard fire on an
        # objective mismatch rather than on a bug -- exactly the signal it exists
        # to protect. Refuse instead of reporting a false violation.
        raise ValueError(
            f"EM is implemented for the unpenalised objective only "
            f"(spec.weight_ridge = {spec.weight_ridge}). Its M-steps do not "
            f"include the ridge, while the monotonicity check reads the penalised "
            f"log-likelihood, so the two would disagree. Pass a spec with "
            f"weight_ridge=0.0 to cross-check the optimiser, or use fit_lbfgs.")
    import jax.numpy as jnp

    D = data.as_jax() if hasattr(data, "as_jax") else data
    p = {k: np.asarray(v, dtype=float) for k, v in params0.items()}
    hist, prev, violations = [], -np.inf, []

    for it in range(int(max_iter)):
        pj = {k: jnp.asarray(v) for k, v in p.items()}
        ll = float(loglik(pj, D, spec))
        if record:
            hist.append(ll)
        if ll < prev - tol:
            msg = (f"EM log-likelihood decreased at iteration {it}: "
                   f"{prev:.10f} -> {ll:.10f} (drop {prev - ll:.3e})")
            violations.append(msg)
            if strict:
                raise MonotonicityError(msg)
        if verbose and it % 10 == 0:
            print(f"    EM {it:4d}  loglik {ll:.8f}", flush=True)
        # The likelihood is very flat near the optimum, so a loose stopping rule
        # leaves the weakly-identified directions (notably b0) short of the
        # stationary point even though the log-likelihood already matches
        # L-BFGS to ~1e-10. Test 06 compares parameters to 4 s.f., so EM has to
        # be run to a tolerance that actually settles them.
        if np.isfinite(prev) and abs(ll - prev) < ll_tol * (abs(prev) + 1e-12):
            prev = ll
            break
        prev = ll

        w = np.asarray(responsibilities(pj, D, spec), dtype=float)
        w = np.clip(w, 1e-12, 1 - 1e-12)
        p, _e_delta = _component_mstep(data, spec, w, p)
        p = _weight_mstep(data, spec, w, p)

    info = {"n_iter": it + 1, "loglik": prev, "violations": violations,
            "monotone": not violations}
    return {k: jnp.asarray(v) for k, v in p.items()}, hist, info
