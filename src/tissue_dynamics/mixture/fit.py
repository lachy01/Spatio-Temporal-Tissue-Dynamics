"""Direct maximisation of the marginal likelihood (spec 4.2, 4.4).

L-BFGS with a zoom line search on the full batch, plus the multi-start protocol
of spec 4.4. Convergence requires **both** a relative log-likelihood change below
`rel_tol_loglik` and `||grad||_inf` below `grad_inf_tol` -- spec 4.2 is explicit
that either alone is insufficient.

`multistart` reports the fraction of restarts reaching the best optimum. Spec 4.4
requires that below 60% agreement the surface is declared multimodal and the
best-of-N reported with an explicit caveat rather than silently taken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import init_params, loglik, neg_loglik


@dataclass
class FitResult:
    params: dict
    loglik: float
    grad_inf: float
    n_iter: int
    converged: bool
    wall_time: float
    spec: object = None
    history: list = field(default_factory=list)
    restart_agreement: float = float("nan")
    n_restarts: int = 1
    multimodal: bool = False
    diverged: bool = False

    def summary(self):
        return (f"loglik {self.loglik:.4f} | ||grad||inf {self.grad_inf:.2e} | "
                f"{self.n_iter} iters | converged {self.converged} | "
                f"{self.wall_time:.1f}s")


def poisson_start(data, spec):
    """Spec 4.4 step 1: single-component Poisson GLM with the component design."""
    from ..count_glm import _poisson_irls

    y = np.asarray(data.y, dtype=float)
    cols = [np.ones(len(y))]
    if spec.theta_free:
        cols.append(np.asarray(data.logN, dtype=float))
    cols.append(np.asarray(data.area, dtype=float))
    X = np.column_stack(cols)
    off = None if spec.theta_free else np.asarray(data.logN, dtype=float)
    beta, _mu = _poisson_irls(y, X, off)
    out = {"b0": float(beta[0])}
    if spec.theta_free:
        out["theta"] = float(beta[1])
        out["s"] = np.asarray(beta[2:], dtype=float)
    else:
        out["s"] = np.asarray(beta[1:], dtype=float)
    return out


def fit_lbfgs(data, spec, params0, maxiter=2000, rel_tol=1e-9, grad_tol=1e-5,
              verbose=False, record_history=False):
    """Maximise the penalised marginal log-likelihood from `params0`."""
    D = data.as_jax() if hasattr(data, "as_jax") else data
    fun = lambda p: neg_loglik(p, D, spec)
    vg = jax.value_and_grad(fun)

    opt = optax.lbfgs()
    p = params0
    state = opt.init(p)
    prev, hist, t0 = np.inf, [], time.time()
    n_done, converged = 0, False

    # The whole step -- gradient, zoom line search, parameter update -- is jitted
    # as one unit. optax's line search calls `fun` several times per iteration,
    # and leaving those calls outside jit dominates the runtime (~30x slower).
    @jax.jit
    def step(p, state):
        v, g = vg(p)
        updates, state = opt.update(g, state, p, value=v, grad=g, value_fn=fun)
        return optax.apply_updates(p, updates), state, v, g

    for it in range(int(maxiter)):
        p, state, v, g = step(p, state)
        v = float(v)
        gi = float(jnp.max(jnp.abs(jax.flatten_util.ravel_pytree(g)[0])))
        n_done = it + 1
        if record_history:
            hist.append(-v)
        if verbose and it % 25 == 0:
            print(f"    iter {it:4d}  loglik {-v:.6f}  ||g||inf {gi:.3e}", flush=True)
        # spec 4.2: BOTH criteria, not either
        if np.isfinite(prev) and abs(prev - v) < rel_tol * (abs(prev) + 1e-12) and gi < grad_tol:
            converged = True
            break
        prev = v

    ll = float(loglik(p, D, spec))
    gi = float(jnp.max(jnp.abs(jax.flatten_util.ravel_pytree(
        jax.grad(lambda q: neg_loglik(q, D, spec))(p))[0])))
    # An unpenalised two-component Poisson mixture can have an unbounded
    # likelihood (a component collapsing onto a subset of cells), and L-BFGS
    # will happily run off to non-finite parameters. Report that as a failed
    # fit rather than handing back values that crash the caller's SVD.
    finite = bool(np.isfinite(ll) and np.isfinite(gi) and all(
        np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(p)))
    if not finite:
        ll, converged = -np.inf, False
    return FitResult(params=p, loglik=ll, grad_inf=gi, n_iter=n_done,
                     converged=bool(converged) and finite, wall_time=time.time() - t0,
                     spec=spec, history=hist, diverged=not finite)


def multistart(data, spec, cfg, n_restarts=None, seed=0, verbose=True,
               maxiter=None):
    """Spec 4.4: one deterministic start plus `n_restarts-1` perturbed ones.

    Step 1 (the Poisson GLM component block) is held fixed across restarts, as
    the spec requires; only steps 2-5 are perturbed.
    """
    oc = cfg["optim"]
    n_restarts = int(cfg["init"]["n_restarts"] if n_restarts is None else n_restarts)
    maxiter = int(oc["maxiter"] if maxiter is None else maxiter)
    ps = poisson_start(data, spec)
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    say(f"  Poisson start: b0={ps['b0']:.4f}"
        + (f", theta={ps['theta']:.4f}" if spec.theta_free else " (theta fixed at 1)"))

    results = []
    for r in range(n_restarts):
        rng = None if r == 0 else np.random.default_rng(seed + r)
        p0 = init_params(spec, cfg, data=data, rng=rng, poisson_start=ps)
        res = fit_lbfgs(data, spec, p0, maxiter=maxiter,
                        rel_tol=float(oc["rel_tol_loglik"]),
                        grad_tol=float(oc["grad_inf_tol"]))
        results.append(res)
        say(f"  restart {r:2d}: {res.summary()}")

    lls = np.array([r.loglik for r in results])
    n_div = int(sum(r.diverged for r in results))
    if n_div:
        say(f"  {n_div}/{n_restarts} restarts diverged to non-finite parameters")
    if not np.isfinite(lls).any():
        raise RuntimeError(
            "every restart diverged to non-finite parameters for this "
            "specification -- a weight_ridge > 0 is needed. (The likelihood "
            "itself is bounded above by 0, since a Poisson pmf is <= 1; this is "
            "numerical divergence of the optimiser, not an unbounded objective.)")
    best = results[int(np.nanargmax(np.where(np.isfinite(lls), lls, -np.inf)))]
    agree = float(np.mean(lls > lls.max() - 1e-6))
    best.restart_agreement = agree
    best.n_restarts = n_restarts
    best.multimodal = bool(agree < 0.60)
    say(f"  best loglik {lls.max():.6f}; {100 * agree:.0f}% of restarts within 1e-6"
        + ("  ** MULTIMODAL: spec 4.4 requires reporting best-of-N with a caveat **"
           if best.multimodal else ""))
    return best, results
