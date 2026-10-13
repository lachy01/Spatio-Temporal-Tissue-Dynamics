"""The two-step arm, TWO (addendum 1, section 3).

Step 1 is a constant-pi two-component mixture on MIX's production rate block:

    log lambda_0i = b0 + log N_i            (log N coefficient fixed at 1)
    lambda_1i     = exp(e^delta) lambda_0i  (proportional rates, as in MIX)
    mu_ki         = theta_m m_i + lambda_ki (theta_m = e^eta_m, outside the ratio)
    y_i ~ pi C1(mu_1i) + (1 - pi) Pois(mu_0i),     pi = sigmoid(a0), one scalar

with C1 Poisson (TWO-P, the primary, whose components are MIX's) or NB2 with
one shared dispersion alpha1 = e^la1 (TWO-PNB, the sensitivity). The posterior
w_i = P(cycling | y_i, N_i) is step 1's output.

Step 2 regresses w on MIX's production weight block -- a0 plus the shared
`arms.eta_state` -- by penalised fractional (quasi-binomial) logistic
regression, with MIX's penalty term for term. No technical term enters step 2.

Everything here works on the dict-of-arrays design `MixtureData.as_jax()`.
"""

from __future__ import annotations

import numpy as np

from . import arms

RATE_KEYS = ("b0", "delta", "eta_m")


def _jnp():
    arms._jax()
    import jax.numpy as jnp
    return jnp


def log_means(p, D):
    """`(log mu_0, log mu_1)` on the log scale, MisTIC term added exactly as in
    `mixture.model._log_mu` (log1p, so cells with m = 0 are exact)."""
    jnp = _jnp()
    l0 = p["b0"] + D["logN"]
    l1 = l0 + jnp.exp(p["delta"])
    t = jnp.exp(p["eta_m"])
    return (l0 + jnp.log1p(t * jnp.exp(-l0) * D["m"]),
            l1 + jnp.log1p(t * jnp.exp(-l1) * D["m"]))


def _log_pois(y, lmu):
    jnp = _jnp()
    from jax.scipy.special import gammaln
    return y * lmu - jnp.exp(lmu) - gammaln(y + 1.0)


def component_logpmf(p, D, family):
    """`(log f0(y), log f1(y))` per cell."""
    lm0, lm1 = log_means(p, D)
    y = D["y"]
    f0 = _log_pois(y, lm0)
    f1 = _log_pois(y, lm1) if family == "pp" else arms.nb_logpmf(y, lm1, p["la1"])
    return f0, f1


def const_pi_terms(p, D, family):
    """Per-cell marginal log-likelihood of the constant-pi mixture."""
    import jax
    jnp = _jnp()
    f0, f1 = component_logpmf(p, D, family)
    return jnp.logaddexp(jax.nn.log_sigmoid(-p["a0"]) + f0, jax.nn.log_sigmoid(p["a0"]) + f1)


def const_pi_objective(D, family):
    """Negative log-likelihood, summed (no penalty: nothing in step 1 is a
    weight-block coefficient)."""
    jnp = _jnp()
    return lambda p: -jnp.sum(const_pi_terms(p, D, family))


def posterior(p, D, family):
    """`w_i = P(cycling | y_i, N_i)` under step 1."""
    import jax
    jnp = _jnp()
    f0, f1 = component_logpmf(p, D, family)
    a0 = jax.nn.log_sigmoid(-p["a0"]) + f0
    a1 = jax.nn.log_sigmoid(p["a0"]) + f1
    return jnp.exp(a1 - jnp.logaddexp(a0, a1))


def means(p, D):
    """`(mu_0, mu_1)` on the natural scale."""
    jnp = _jnp()
    lm0, lm1 = log_means(p, D)
    return jnp.exp(lm0), jnp.exp(lm1)


def start_from_mix(pe_params, family, la1=None):
    """Step 1 start from a MIX parameter set: its rate block and a0."""
    jnp = _jnp()
    p = {k: jnp.asarray(float(np.asarray(pe_params[k]))) for k in (*RATE_KEYS, "a0")}
    if family == "pnb":
        p["la1"] = jnp.asarray(float(np.log(0.3) if la1 is None else la1))
    return p


def random_start(base, family, rng):
    """A perturbed step 1 start, for TWO-PNB's multi-start (V13)."""
    jnp = _jnp()
    frac = rng.uniform(0.02, 0.30)
    p = {"b0": jnp.asarray(float(base["b0"]) + rng.normal(0, 0.3)),
         "delta": jnp.asarray(float(base["delta"]) + rng.normal(0, 0.5)),
         "eta_m": jnp.asarray(float(base["eta_m"]) + rng.normal(0, 0.3)),
         "a0": jnp.asarray(float(np.log(frac / (1 - frac))))}
    if family == "pnb":
        p["la1"] = jnp.asarray(float(rng.normal(np.log(0.3), 1.0)))
    return p


def fit_step1(D, family, p0, say=None):
    """Step 1 to the mixture's convergence criteria (relative change < 1e-9 and
    |grad|inf < 1e-5), via `arms.minimise` (L-BFGS then exact Newton)."""
    return arms.minimise(const_pi_objective(D, family), p0, say=say)


# --------------------------------------------------------------------------
# Step 2: penalised fractional logistic regression on the weight block
# --------------------------------------------------------------------------
def weight_eta(p, D):
    """logit pi(x) = a0 + eta_state(x): MIX's `_logit_pi`, nothing technical."""
    return p["a0"] + arms.eta_state(p, D)


def step2_objective(D, w, n, spec):
    """-sum[w eta - softplus(eta)] + MIX's penalty (weight_ridge scaled to n,
    ||UV'||_F^2, balance term at ridge_tau)."""
    import jax
    jnp = _jnp()

    def f(p):
        eta = weight_eta(p, D)
        return (-jnp.sum(w * eta - jax.nn.softplus(eta))
                + arms.penalty(p, n, spec.weight_ridge, spec.n_reference,
                               spec.ridge_tau, spec.b_penalty))
    return f


def init_step2(D, w, rank, seed):
    jnp = _jnp()
    rng = np.random.default_rng(seed)
    p = arms.init_eta(D["own"].shape[1], D["nbr"].shape[1], D["hbasis"].shape[1], rank, rng,
                      oxygen="oxygen" in D)
    m = float(np.clip(np.mean(np.asarray(w)), 1e-6, 1 - 1e-6))
    p["a0"] = jnp.asarray(np.log(m / (1 - m)))
    return p


def fit_step2(D, w, n, spec, seed=0, say=None, p0=None):
    f = step2_objective(D, w, n, spec)
    for attempt in range(3):
        p, info = arms.minimise(f, p0 if (p0 is not None and attempt == 0)
                                else init_step2(D, w, spec.rank, seed + 100 * attempt), say=say)
        if info["converged"]:
            break
    info["seed"] = seed + 100 * attempt
    return p, info


def pi_two(p, D):
    import jax
    return jax.nn.sigmoid(weight_eta(p, D))
