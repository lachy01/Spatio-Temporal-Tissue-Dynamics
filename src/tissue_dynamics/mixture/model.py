"""Marginal log-likelihood of the concomitant-variable Poisson mixture (spec 2, 4.1).

    P(Z_i = 1) = pi_i
    y_i | Z_i = k  ~  Poisson(rho_i + theta_m m_i + lambda_ki)

    log lambda_0i = b0        + theta * logN_i [+ s(log area_i)]
    log lambda_1i = b0 + e^d  + theta * logN_i [+ s(log area_i)]
    log rho_i     = r0 + g(log density_i)                      [Stage C onward]
    logit(pi_i)   = a0 + phi_i.a + nbr_i.c + phi_i' B nbr_i
                       + h(log density_i) [+ psi * oxygen_i]

The technical covariates follow `model.technical` (2026-10-12). Under "mistic"
there is no area spline, logN is the MisTIC-weighted library size, and
theta_m m_i -- the expected number of outcome transcripts misassigned into
the cell from its neighbours, theta_m = exp(eta_m) -- adds to both components'
means alike, next to the Stage C background. Under "area" the area spline is
in and the m term is out, as before.

The oxygen term is off in production (`model.oxygen: false`): on the log1p
table the field explained nothing beyond cell state and never beat its rotated
or mirrored decoys (notebooks/nb_baseline.ipynb, section 3).

with `B = U V^T`. The structural rules of spec 0 are enforced by construction
here, not by convention: the state blocks are only ever read by `_logit_pi`, and
`logN`/the area spline are only ever read by `_log_lambda` (and `m` only by the
mean, `_log_mu` / `_background`). There is no code path that puts state into the
rates or detection into the weight.

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
    weight_ridge: float = 0.0     # ridge on a, c, k, h, psi, U, V -- see loglik()
    density_interaction: bool = False   # per-cell-type density response
    density_interaction_spline: bool = False   # curve per type, not just a tilt
    n_reference: int = 0          # sample size weight_ridge was calibrated at
    oxygen: bool = True           # psi * oxygen in the weight; off in production
    # How B = U V' is penalised (see loglik): "frobenius" (production since
    # 2026-10-06) puts weight_ridge * ||U V'||_F^2 in the shrinkage and uses
    # ridge_tau as the weight of the balancing term ||U'U - V'V||_F^2;
    # "nuclear" is the earlier form, (weight_ridge + ridge_tau)(||U||^2 + ||V||^2).
    b_penalty: str = "frobenius"
    # theta_m * m_i in every component's mean (model.technical: mistic); the
    # area spline is then absent (df_area = 0)
    mistic: bool = False
    # model.technical, for the record ("mistic", "plain" or "area"); the terms
    # themselves follow `mistic` and `df_area`
    technical: str = "area"

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
        bits.append("MisTIC theta_m m, no area" if self.mistic else
                    f"area spline, df {self.df_area}" if self.df_area > 0 else "no area, no MisTIC")
        return "; ".join(bits)


@dataclass
class MixtureData:
    """Design arrays, all float64, all built from a FrozenBasis."""

    y: np.ndarray
    logN: np.ndarray
    area: np.ndarray                      # s(log area) basis; 0 columns under MisTIC
    m: np.ndarray = field(default=None)   # expected misassigned outcome count (MisTIC)
    own: np.ndarray = field(default=None)
    nbr: np.ndarray = field(default=None)
    hbasis: np.ndarray = field(default=None)
    logdens: np.ndarray = field(default=None)      # centred log density
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
    if spec.mistic:
        d.m = np.asarray(basis.mistic_m(df), dtype=np.float64)
    if spec.has_state:
        d.own = np.asarray(basis.own(df), dtype=np.float64)
        d.nbr = np.asarray(basis.nbr(df), dtype=np.float64)
        d.hbasis = np.asarray(basis.density_weight_basis(df), dtype=np.float64)
        # centred, so `a` keeps its meaning as the own-state effect AT MEAN
        # DENSITY; uncentred, `a` and `k` would be entangled through a baseline
        # (log density = 0) that no cell is anywhere near.
        ld = np.log(df["density"].to_numpy(np.float64))
        d.logdens = ld - basis.mean_log_density
        if spec.oxygen:
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
        if spec.df_area > 0:
            p["s"] = jnp.asarray(np.asarray(poisson_start["s"], dtype=np.float64))
        if spec.theta_free:
            p["theta"] = jnp.asarray(float(poisson_start.get("theta", 1.0)))
        if spec.mistic:
            p["eta_m"] = jnp.asarray(float(poisson_start.get("eta_m", 0.0)))
    else:
        p["b0"] = jnp.asarray(-1.0)
        if spec.df_area > 0:
            p["s"] = jnp.zeros(spec.df_area)
        if spec.theta_free:
            p["theta"] = jnp.asarray(1.0)
        if spec.mistic:
            p["eta_m"] = jnp.asarray(0.0)

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
        if spec.density_interaction and spec.df_weight > 0:
            if spec.density_interaction_spline:
                p["K"] = draw(spec.d_own, spec.df_weight)
            else:
                p["k"] = draw(spec.d_own)
        if spec.oxygen:
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
    base = p["b0"] + theta * d["logN"]
    if spec.df_area > 0:
        base = base + d["area"] @ p["s"]
    return base, base + jnp.exp(p["delta"])      # e^delta > 0 pins lambda_1 > lambda_0


def _logit_pi(p, d, spec):
    """`logit(pi)`. Reads state, density and oxygen only -- never logN or area."""
    eta = p["a0"]
    if spec.has_state:
        eta = (eta + d["own"] @ p["a"] + d["nbr"] @ p["c"]
               + jnp.sum((d["own"] @ p["U"]) * (d["nbr"] @ p["V"]), axis=-1))
        if spec.oxygen:
            eta = eta + p["psi"] * d["oxygen"]
        if spec.df_weight > 0:                  # h() dropped when df_weight == 0
            eta = eta + d["hbasis"] @ p["h"]
        # `df_weight == 0` means density is removed from the weight altogether
        # (Test 13's comparison). The per-type interaction is ALSO a density
        # term -- it is built on the same basis -- so it must go with h(), or the
        # test would drop the shared curve while keeping nine per-type ones and
        # measure nothing.
        if spec.density_interaction and spec.df_weight > 0:
            # A single shared h() averages opposite mechanisms: adjusted for
            # detection and neighbourhood, log density raises the cycling odds
            # for Myeloid/Endothelial/Normal-Epithelial/PVL/CAFs (+0.15 to
            # +0.24) but LOWERS them for Cancer Epithelial (-0.031, z = -4.1).
            if spec.density_interaction_spline:
                # Each type gets its own density CURVE. The linear version below
                # gives every type the same curve shape with a constant tilt,
                # which cannot express a response that turns downward only at
                # high density -- i.e. contact inhibition, the one mechanism the
                # phase portrait's confinement depends on.
                eta = eta + jnp.einsum("ij,jm,im->i", d["own"], p["K"], d["hbasis"])
            else:
                eta = eta + (d["own"] @ p["k"]) * d["logdens"]
    return eta


def _log_rho(p, d, spec):
    return p["r0"] + d["gbasis"] @ p["g"] if spec.has_background else None


def _log_mu(log_lam, log_rho, p=None, d=None, spec=None):
    """Poisson mean on the log scale: lambda, plus the Stage C background via
    logaddexp, plus theta_m m_i under MisTIC.

    The MisTIC term is added as log(mu + t) = log mu + log1p(t / mu), with
    t = exp(eta_m) m_i. That is exact where m_i = 0 (t = 0, nothing added) and
    never forms log(0), which logaddexp(eta_m + log m_i, log mu) would carry
    as -inf through every cell with m_i = 0.
    """
    out = log_lam if log_rho is None else jnp.logaddexp(log_rho, log_lam)
    if spec is not None and spec.mistic:
        out = out + jnp.log1p(jnp.exp(p["eta_m"] - out) * d["m"])
    return out


def _background(p, d, spec, like):
    """The additive part of each component's mean, natural scale: the Stage C
    background rho_i plus theta_m m_i under MisTIC (zeros if neither)."""
    lr = _log_rho(p, d, spec)
    bg = jnp.zeros_like(like) if lr is None else jnp.exp(lr)
    if spec.mistic:
        bg = bg + jnp.exp(p["eta_m"]) * d["m"]
    return bg


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
        jax.nn.log_sigmoid(-eta) + _log_pois(y, _log_mu(l0, lr, p, d, spec)),
        jax.nn.log_sigmoid(eta) + _log_pois(y, _log_mu(l1, lr, p, d, spec)),
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
        b_pen, cond = b_penalty_terms(p, spec)
        if spec.ridge_tau > 0:                      # conditioning term
            ll = ll - spec.ridge_tau * cond
        if spec.weight_ridge > 0:
            pen = jnp.sum(p["a"] ** 2) + jnp.sum(p["c"] ** 2) + b_pen
            if spec.oxygen:
                pen = pen + p["psi"] ** 2
            if spec.df_weight > 0:
                pen = pen + jnp.sum(p["h"] ** 2)
            if spec.density_interaction and spec.df_weight > 0:
                pen = pen + jnp.sum(
                    p["K"] ** 2 if spec.density_interaction_spline else p["k"] ** 2)
            # Scale the penalty with the sample size it is applied to. The
            # log-likelihood grows with n while an absolute ridge does not, so a
            # fixed lambda shrinks a subsample far harder: at n = 50,000 it is
            # 13.6x stronger per observation than at n = 678,885, which made
            # mean(pi) drift monotonically with n (0.0865 -> 0.1024 -> 0.1160,
            # correlation +0.98) and Test 10 fail. Scaled, the drift disappears
            # (0.1141 -> 0.1094 -> 0.1160). At n == n_reference this is exactly
            # `weight_ridge`, so full-sample results are unchanged.
            lam = spec.weight_ridge
            if spec.n_reference > 0:
                lam = lam * (d["y"].shape[0] / spec.n_reference)
            ll = ll - lam * pen
    return ll


def b_penalty_terms(p, spec):
    """`(shrinkage term for B, conditioning term)` for the configured form.

    "frobenius" (production since 2026-10-06):
        shrinkage    ||U V'||_F^2 = sum(U'U * V'V), a ridge on the entries of B,
                     the same squared-L2 penalty as a, c, h and K;
        conditioning ||U'U - V'V||_F^2, the balancing term. Every B has a
                     factorisation with U'U = V'V, so this is exactly zero at
                     any optimum: it removes the U -> U A, V -> V A^-T scaling
                     freedom without penalising B, and the estimate of B does
                     not depend on ridge_tau at all. Only an orthogonal rotation
                     of U and V together stays flat.
    "nuclear" (the earlier form): both terms are ||U||^2 + ||V||^2, which at
        the balanced optimum equals 2 ||B||_*. It soft-thresholds the singular
        values of B, and ridge_tau then shrinks B a little too.
    """
    if spec.b_penalty == "frobenius":
        UtU, VtV = p["U"].T @ p["U"], p["V"].T @ p["V"]
        return jnp.sum(UtU * VtV), jnp.sum((UtU - VtV) ** 2)
    if spec.b_penalty == "nuclear":
        uv = jnp.sum(p["U"] ** 2) + jnp.sum(p["V"] ** 2)
        return uv, uv
    raise ValueError(f"unknown b_penalty {spec.b_penalty!r}")


def neg_loglik(p, d, spec):
    return -loglik(p, d, spec)


def responsibilities(p, d, spec):
    """`w_i = P(Z_i = 1 | y_i)` -- the E-step quantity (spec 4.3)."""
    l0, l1 = _log_lambda(p, d, spec)
    lr = _log_rho(p, d, spec)
    eta = _logit_pi(p, d, spec)
    y = d["y"]
    a0 = jax.nn.log_sigmoid(-eta) + _log_pois(y, _log_mu(l0, lr, p, d, spec))
    a1 = jax.nn.log_sigmoid(eta) + _log_pois(y, _log_mu(l1, lr, p, d, spec))
    return jnp.exp(a1 - jnp.logaddexp(a0, a1))


def pi_hat(p, d, spec):
    return jax.nn.sigmoid(_logit_pi(p, d, spec))


def rates(p, d, spec):
    """`(lambda_0, lambda_1, rho)` on the natural scale.

    `rho` is the whole additive part of the mean: the Stage C background plus
    theta_m m_i under MisTIC. A component's Poisson mean is `rho + lambda_k`
    (`component_means`); the rate ratio lambda_1 / lambda_0 = exp(e^delta)
    does not involve it.
    """
    l0, l1 = _log_lambda(p, d, spec)
    return jnp.exp(l0), jnp.exp(l1), _background(p, d, spec, l0)


def component_means(p, d, spec):
    """`(mu_0, mu_1)`: each component's Poisson mean, `rho + lambda_k`, the
    quantities any count probability under a component must use."""
    lam0, lam1, bg = rates(p, d, spec)
    return bg + lam0, bg + lam1


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
