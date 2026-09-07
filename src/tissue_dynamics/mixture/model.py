"""Marginal log-likelihood of the concomitant-variable Poisson mixture (spec 2, 4.1).

    P(Z_i = 1) = pi_i
    y_i | Z_i = k  ~  Poisson(rho_i + lambda_ki)

    log lambda_0i = b0        + theta * logN_i + s(log area_i)
    log lambda_1i = b0 + e^d  + theta * logN_i + s(log area_i)
    log rho_i     = r0 + g(log density_i)                      [Stage C onward]
    logit(pi_i)   = a0 + phi_i.a + nbr_i.c + phi_i' B nbr_i
                       + h(log density_i) + psi * oxygen_i

with `B = U V^T`. The structural rules of spec 0 are enforced by construction
here, not by convention: the state blocks are only ever read by `_logit_pi`, and
`logN`/the area spline are only ever read by `_log_lambda`. There is no code path
that puts state into the rates or detection into the weight.

Three points of numerical care, all mandated by spec 4.1:

* the two-component mixture is combined with `logsumexp`, never by summing
  densities -- at n = 700,000 some cells have a component probability that
  underflows float64;
* `log pi` and `log(1 - pi)` come from `log_sigmoid(+/- eta)`, not from
  exponentiating the logit and taking a log;
* the additive background is combined as `logaddexp(log rho, log lambda)`, so
  the Poisson mean is never formed by adding two exponentials.

**Deviation from spec 4.2:** the design is carried in float64 rather than
float32. Test 04 requires the analytic gradient to match central finite
differences to a max relative error of 1e-5, which float32 cannot deliver. At
682,748 cells x ~70 columns this costs ~380 MB, which fits comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                     # noqa: E402
import numpy as np                                          # noqa: E402
from jax.scipy.special import gammaln, logsumexp            # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    """Which terms are switched on. `stage` follows the spec's staged build."""

    stage: str = "B"                # "A" (weight = intercept), "B" (+state), "C" (+background)
    rank: int = 2                   # rank of B = U V^T
    theta_free: bool = True         # Configuration V; False is Configuration F (theta == 1)
    d_own: int = 0
    d_nbr: int = 0
    df_area: int = 4
    df_weight: int = 3
    df_bg: int = 3
    ridge_tau: float = 1e-4
    weight_ridge: float = 0.0     # ridge on a, c, h, psi -- see loglik()

    @property
    def has_state(self):
        return self.stage in ("B", "C")

    @property
    def has_background(self):
        return self.stage == "C"

    def describe(self):
        bits = [f"Stage {self.stage}",
                f"theta {'free (V)' if self.theta_free else 'fixed at 1 (F)'}"]
        if self.has_state:
            bits.append(f"state {self.d_own}x{self.d_nbr}, rank {self.rank}")
        if self.has_background:
            bits.append(f"additive background, df {self.df_bg}")
        return "; ".join(bits)


@dataclass
class MixtureData:
    """Design arrays, all float64, all built from a FrozenBasis."""

    y: np.ndarray
    logN: np.ndarray
    area: np.ndarray                      # s(log area) basis
    own: np.ndarray = field(default=None)
    nbr: np.ndarray = field(default=None)
    hbasis: np.ndarray = field(default=None)
    oxygen: np.ndarray = field(default=None)
    gbasis: np.ndarray = field(default=None)

    @property
    def n(self):
        return len(self.y)

    def as_jax(self):
        out = {}
        for k, v in self.__dict__.items():
            if v is not None:
                out[k] = jnp.asarray(np.asarray(v, dtype=np.float64))
        return out

    def subset(self, idx):
        idx = np.asarray(idx)
        kw = {k: (None if v is None else np.asarray(v)[idx])
              for k, v in self.__dict__.items()}
        return MixtureData(**kw)


def build_data(df, basis, cfg, spec):
    """Assemble the design arrays for `df` from the frozen basis."""
    d = MixtureData(
        y=df["y_count"].to_numpy(np.float64),
        logN=np.asarray(basis.log_N(df), dtype=np.float64),
        area=np.asarray(basis.log_area_basis(df), dtype=np.float64),
    )
    if spec.has_state:
        d.own = np.asarray(basis.own(df), dtype=np.float64)
        d.nbr = np.asarray(basis.nbr(df), dtype=np.float64)
        d.hbasis = np.asarray(basis.density_weight_basis(df), dtype=np.float64)
        d.oxygen = df["oxygen"].to_numpy(np.float64)
    if spec.has_background:
        d.gbasis = np.asarray(basis.density_bg_basis(df), dtype=np.float64)
    return d


def init_params(spec, cfg, data=None, rng=None, poisson_start=None):
    """Starting values per spec 4.4. `rng` perturbs steps 2-5 for multi-start."""
    ic = cfg["init"]
    p = {}
    if poisson_start is not None:                # step 1: single-component GLM fit
        p["b0"] = jnp.asarray(float(poisson_start["b0"]))
        p["s"] = jnp.asarray(np.asarray(poisson_start["s"], dtype=np.float64))
        if spec.theta_free:
            p["theta"] = jnp.asarray(float(poisson_start.get("theta", 1.0)))
    else:
        p["b0"] = jnp.asarray(-1.0)
        p["s"] = jnp.zeros(spec.df_area)
        if spec.theta_free:
            p["theta"] = jnp.asarray(1.0)

    scale = float(ic["random_scale"])
    draw = (lambda *sh: jnp.asarray(rng.normal(scale=scale, size=sh))) if rng is not None \
        else (lambda *sh: jnp.zeros(sh))

    p["delta"] = jnp.asarray(float(ic["delta"])) + (draw() if rng is not None else 0.0)
    frac = float(ic["a0_cycling_fraction"])
    p["a0"] = jnp.asarray(np.log(frac / (1.0 - frac))) + (draw() if rng is not None else 0.0)

    if spec.has_state:
        p["a"] = draw(spec.d_own)
        p["c"] = draw(spec.d_nbr)
        p["U"] = draw(spec.d_own, spec.rank)
        p["V"] = draw(spec.d_nbr, spec.rank)
        if spec.df_weight > 0:
            p["h"] = jnp.zeros(spec.df_weight)
        p["psi"] = jnp.asarray(0.0)
    if spec.has_background:
        p["r0"] = jnp.asarray(float(np.log(max(
            1e-6, float(np.mean(data.y)) * 0.05 if data is not None else 1e-3))))
        p["g"] = jnp.zeros(spec.df_bg)
    return p


# --- linear predictors -------------------------------------------------------

def _log_lambda(p, d, spec):
    """`(log lambda_0, log lambda_1)`. Reads detection covariates only."""
    theta = p["theta"] if spec.theta_free else 1.0
    base = p["b0"] + theta * d["logN"] + d["area"] @ p["s"]
    return base, base + jnp.exp(p["delta"])      # e^delta > 0 pins lambda_1 > lambda_0


def _logit_pi(p, d, spec):
    """`logit(pi)`. Reads state, density and oxygen only -- never logN or area."""
    eta = p["a0"]
    if spec.has_state:
        eta = (eta + d["own"] @ p["a"] + d["nbr"] @ p["c"]
               + jnp.sum((d["own"] @ p["U"]) * (d["nbr"] @ p["V"]), axis=-1)
               + p["psi"] * d["oxygen"])
        if spec.df_weight > 0:                  # h() dropped when df_weight == 0
            eta = eta + d["hbasis"] @ p["h"]
    return eta


def _log_rho(p, d, spec):
    return p["r0"] + d["gbasis"] @ p["g"] if spec.has_background else None


def _log_mu(log_lam, log_rho):
    """Poisson mean on the log scale; additive background via logaddexp."""
    return log_lam if log_rho is None else jnp.logaddexp(log_rho, log_lam)


def _log_pois(y, log_mu):
    return y * log_mu - jnp.exp(log_mu) - gammaln(y + 1.0)


# --- likelihood --------------------------------------------------------------

def loglik_terms(p, d, spec):
    """Per-cell marginal log-likelihood, `(n,)`."""
    l0, l1 = _log_lambda(p, d, spec)
    lr = _log_rho(p, d, spec)
    eta = _logit_pi(p, d, spec)
    y = d["y"]
    comp = jnp.stack([
        jax.nn.log_sigmoid(-eta) + _log_pois(y, _log_mu(l0, lr)),
        jax.nn.log_sigmoid(eta) + _log_pois(y, _log_mu(l1, lr)),
    ], axis=0)
    return logsumexp(comp, axis=0)


def loglik(p, d, spec):
    """Penalised marginal log-likelihood (spec 4.1), a scalar to maximise.

    `weight_ridge` penalises **every** weight-model coefficient -- `a`, `c`, `h`,
    `psi`, `U` and `V` -- a deliberate addition to spec 3.2, which rides only a
    fixed `ridge_tau = 1e-4` on `U` and `V`. Without it the concomitant logistic
    model separates: on data simulated with no latent class structure at all the
    fit reaches `||a|| = 57.5`, `||c|| = 81.8` on standardised covariates, driving
    `sd(eta) = 203` and pushing `pi` to 0 or 1 for 98.6% of cells, so the mixture
    "explains" the counts by near-deterministic prior assignment rather than by
    any rate difference. That is the Test 03 / Test 07 failure mode.

    `U` and `V` are included deliberately. Penalising only the linear block moves
    the problem rather than fixing it: at a linear ridge of 10 the fitted `||a||`
    falls to 0.02 but `sigma_1(B_hat)` rises to 1731 and `sd(eta)` to 2372, the
    divergence having migrated into the bilinear term, which `ridge_tau` is four
    orders of magnitude too weak to restrain.

    The intercepts `a0`, `r0` and the whole component block stay unpenalised, as
    is conventional. `weight_ridge` is selected by held-out log-likelihood on
    contiguous spatial blocks (`inference.select_weight_ridge`), for the reason
    spec 6.11 gives for rank selection: spatial dependence puts the effective
    sample size far below n, so AIC/BIC would systematically under-penalise.
    """
    ll = jnp.sum(loglik_terms(p, d, spec))
    if spec.has_state:
        uv = jnp.sum(p["U"] ** 2) + jnp.sum(p["V"] ** 2)
        if spec.ridge_tau > 0:                      # spec 3.2 conditioning term
            ll = ll - spec.ridge_tau * uv
        if spec.weight_ridge > 0:
            pen = jnp.sum(p["a"] ** 2) + jnp.sum(p["c"] ** 2) + p["psi"] ** 2 + uv
            if spec.df_weight > 0:
                pen = pen + jnp.sum(p["h"] ** 2)
            ll = ll - spec.weight_ridge * pen
    return ll


def neg_loglik(p, d, spec):
    return -loglik(p, d, spec)


def responsibilities(p, d, spec):
    """`w_i = P(Z_i = 1 | y_i)` -- the E-step quantity (spec 4.3)."""
    l0, l1 = _log_lambda(p, d, spec)
    lr = _log_rho(p, d, spec)
    eta = _logit_pi(p, d, spec)
    y = d["y"]
    a0 = jax.nn.log_sigmoid(-eta) + _log_pois(y, _log_mu(l0, lr))
    a1 = jax.nn.log_sigmoid(eta) + _log_pois(y, _log_mu(l1, lr))
    return jnp.exp(a1 - jnp.logaddexp(a0, a1))


def pi_hat(p, d, spec):
    return jax.nn.sigmoid(_logit_pi(p, d, spec))


def rates(p, d, spec):
    """`(lambda_0, lambda_1, rho)` on the natural scale."""
    l0, l1 = _log_lambda(p, d, spec)
    lr = _log_rho(p, d, spec)
    rho = jnp.zeros_like(l0) if lr is None else jnp.exp(lr)
    return jnp.exp(l0), jnp.exp(l1), rho


def marginal_mean(p, d, spec):
    """`E[y_i] = rho_i + (1-pi_i) lambda_0i + pi_i lambda_1i` (spec 6.14)."""
    lam0, lam1, rho = rates(p, d, spec)
    pi = pi_hat(p, d, spec)
    return rho + (1.0 - pi) * lam0 + pi * lam1


def marginal_pmf(p, d, spec, kmax):
    """`P(y = k)` for `k = 0..kmax` under the mixture, per cell (spec 6.9)."""
    lam0, lam1, rho = rates(p, d, spec)
    pi = pi_hat(p, d, spec)
    m0, m1 = rho + lam0, rho + lam1
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)[:, None]
    lp0 = ks * jnp.log(m0)[None, :] - m0[None, :] - gammaln(ks + 1.0)
    lp1 = ks * jnp.log(m1)[None, :] - m1[None, :] - gammaln(ks + 1.0)
    return (1.0 - pi)[None, :] * jnp.exp(lp0) + pi[None, :] * jnp.exp(lp1)


def B_matrix(p):
    """The identified object `B = U V^T` (spec 3.2: never report U or V)."""
    return np.asarray(p["U"]) @ np.asarray(p["V"]).T


def singular_values(p):
    """Singular values of `B = U V^T`, or NaN if the fit diverged (spec 3.2)."""
    B = B_matrix(p)
    if not np.all(np.isfinite(B)):
        return np.full(min(B.shape), np.nan)
    try:
        return np.linalg.svd(B, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.full(min(B.shape), np.nan)


def normalised_entropy(w):
    """Spec 8.7. `E < 0.30` means the components are not separated."""
    w = np.clip(np.asarray(w, dtype=float), 1e-12, 1 - 1e-12)
    h = -(w * np.log(w) + (1 - w) * np.log1p(-w))
    return float(1.0 - h.mean() / np.log(2.0))


def degeneracy_flags(p, w, spec):
    """Spec 3.3: report absence of structure rather than manufacturing it."""
    fold = float(np.exp(np.asarray(p["delta"])))
    mean_pi = float(np.mean(np.asarray(w)))
    flags = {}
    flags["fold_collapsed"] = bool(fold < 1e-3)
    flags["weight_degenerate"] = bool(min(mean_pi, 1 - mean_pi) < 1e-3)
    flags["e_delta"] = fold
    flags["rate_ratio"] = float(np.exp(fold))
    flags["mean_w"] = mean_pi
    flags["any"] = bool(flags["fold_collapsed"] or flags["weight_degenerate"])
    return flags


def flatten(p):
    """Flat float64 vector plus an unravel function (for Hessians and FD checks)."""
    from jax.flatten_util import ravel_pytree

    return ravel_pytree(p)
