"""Phase portrait over archetype-density state space (portrait spec 1-6).

Converts an already-fitted mixture into a deterministic vector field, locates and
classifies fixed points, and renders pairwise slices. **No model fitting happens
here**, and no simulation: the portrait is the right-hand side of the ODE
evaluated on a grid, not a trajectory through it.

Only the mixing-weight submodel is used. The component rates -- library offset,
area spline, the proportional-rates constant -- never enter, because the state
variable is P(cycling), not expected counts.

Two adaptations to the fit we actually have. The spec assumes 29 raw PhiSpace
components; ours is `celltype_major`, 9 compositional scores projected to the
8-dimensional sum-to-zero subspace, so every coefficient lives in that basis and
archetypes must be expressed there too. And the spec's `n` is a count in the
neighbourhood disc while the model's density covariate is cells per (100um)^2 in
that same disc -- now the same 80um disc, so `n = sum(X)` feeds the density
spline directly with no conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PortraitModel:
    """The mixing-weight submodel, lifted out of the fit object (spec 1.1)."""

    a0: float
    a: np.ndarray                 # (d,)   own-state
    c: np.ndarray                 # (d,)   neighbour-state
    B: np.ndarray                 # (d, d) = U V^T, the identified object
    h: np.ndarray                 # (n_h,) shared density spline
    K: np.ndarray | None          # (d, n_h) per-archetype density curves, or None
    psi: float                    # oxygen
    density_basis: object         # exposes __call__(log density) -> (m, n_h)
    mean_log_density: float

    @classmethod
    def from_fit(cls, params, basis):
        p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
        return cls(a0=float(p["a0"]), a=p["a"], c=p["c"],
                   B=p["U"] @ p["V"].T, h=p.get("h", np.zeros(0)),
                   K=p.get("K"), psi=float(p["psi"]),
                   density_basis=basis.density_weight_spline,
                   mean_log_density=float(basis.mean_log_density))

    def eta(self, phi, phibar, n, oxygen):
        """Linear predictor of the mixing weight. `phi`,`phibar` are (M, d)."""
        phi = np.atleast_2d(phi)
        phibar = np.atleast_2d(phibar)
        n = np.atleast_1d(np.asarray(n, dtype=float))
        H = self.density_basis(np.log(np.clip(n, 1e-9, None)))
        out = (self.a0 + phi @ self.a + phibar @ self.c
               + np.einsum("ij,jk,ik->i", phi, self.B, phibar)
               + self.psi * np.asarray(oxygen, dtype=float))
        if self.h.size:
            out = out + H @ self.h
        if self.K is not None:
            out = out + np.einsum("ij,jm,im->i", phi, self.K, H)
        return out

    def pi(self, phi, phibar, n, oxygen):
        return 1.0 / (1.0 + np.exp(-np.clip(self.eta(phi, phibar, n, oxygen), -30, 30)))

    def deta_dphibar(self, phi):
        """`g_k = c + B^T phi` -- gradient of eta wrt phibar (spec 4)."""
        return self.c + np.atleast_2d(phi) @ self.B

    def dbasis_dlogn(self, log_n, eps=1e-3):
        """`dH/d(log n)` by a five-point stencil.

        The density basis is a patsy spline with no exposed analytic derivative,
        so this is the one numerical step in an otherwise analytic Jacobian. A
        two-point difference leaves an O(eps^2) error around 1e-5, which is
        larger than spec 11.2's 1e-6 tolerance on the Jacobian; the five-point
        form is O(eps^4) and lands near machine precision.
        """
        log_n = np.atleast_1d(np.asarray(log_n, dtype=float))
        f = self.density_basis
        return (-f(log_n + 2 * eps) + 8 * f(log_n + eps)
                - 8 * f(log_n - eps) + f(log_n - 2 * eps)) / (12 * eps)

    def dh_dlogn(self, phi, n):
        """Total density slope for a cell at own state `phi`.

        Includes the per-archetype term: `K` is part of the density response,
        not a separate effect, so omitting it would understate the slope for
        every archetype whose curve differs from the shared one.
        """
        n = np.atleast_1d(np.asarray(n, dtype=float))
        dH = self.dbasis_dlogn(np.log(np.clip(n, 1e-9, None)))
        d = np.zeros(len(n))
        if self.h.size:
            d = d + dH @ self.h
        if self.K is not None:
            phi = np.atleast_2d(phi)
            if len(phi) == 1:
                phi = np.repeat(phi, len(n), axis=0)
            d = d + np.einsum("ij,jm,im->i", phi, self.K, dH)
        return d


@dataclass
class Archetypes:
    """Named archetype states and the loadings that place cells among them."""

    names: list
    A: np.ndarray                 # (K, d) in the projected basis
    Q: np.ndarray                 # (d0, d) projection, for reporting in cell types
    pi_bar: np.ndarray = field(default=None)   # (K,) death anchor, spec 3.2
    X_obs: np.ndarray = field(default=None)    # (N, K) observed state
    r2_recon: float = float("nan")
    pi_bar_clipped: np.ndarray = field(default=None)   # sensitivity variant

    @property
    def K(self):
        return len(self.names)


def simplex_loadings(phibar, A):
    """Barycentric coordinates of `phibar` in the simplex spanned by `A`.

    With K = 9 archetypes in an 8-dimensional space plus the sum-to-one
    constraint the system is square, so the loadings are the unique exact
    solution of a 9x9 linear system rather than a non-negative least squares
    fit -- and `phibar - W A` is exactly zero, so nothing is discarded.
    Loadings may fall outside [0, 1] for cells beyond the simplex; that is
    reported rather than clipped, since clipping would silently move cells.
    """
    A = np.asarray(A, dtype=float)
    M = np.vstack([A.T, np.ones(len(A))])                 # (d+1, K)
    rhs = np.vstack([np.asarray(phibar, dtype=float).T,
                     np.ones(len(phibar))])               # (d+1, N)
    W, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return W.T


def build_archetypes(df, basis, model=None, pi_hat=None):
    """Named centroids (spec 1.3's `"centroid"` provider), plus loadings."""
    names = list(basis.phi_names)
    own = basis.own(df)
    arche = df["archetype"].astype(str).to_numpy()
    A = np.vstack([own[arche == nm].mean(0) for nm in names])
    nbr = basis.nbr(df)
    W = simplex_loadings(nbr, A)
    recon = W @ A
    ss = np.sum((nbr - nbr.mean(0)) ** 2)
    r2 = float(1.0 - np.sum((nbr - recon) ** 2) / ss)
    n_local = df["density"].to_numpy(float)
    X = n_local[:, None] * W
    pb = None
    if pi_hat is not None:
        # Spec 3.2 exactly: signed loadings. 28% of loadings are negative here
        # (cells sit outside the archetype simplex), but every column's weight
        # sum is large and positive (36k-129k), so the ratio is well conditioned.
        # Clipping the negatives moves every anchor by less than 0.017 and is
        # kept as `pi_bar_clipped` for the sensitivity sweep rather than used.
        pb = np.array([np.sum(W[:, k] * pi_hat) / np.sum(W[:, k])
                       for k in range(len(names))])
    pbc = None
    if pi_hat is not None:
        wp = np.clip(W, 0, None)
        pbc = np.array([np.sum(wp[:, k] * pi_hat) / np.sum(wp[:, k])
                        for k in range(len(names))])
    return Archetypes(names=names, A=A, Q=basis.Q_own, pi_bar=pb, X_obs=X,
                      r2_recon=r2, pi_bar_clipped=pbc), W


def covariates_from_state(X, A):
    """`X -> (phibar, n)`. Inverse of `X_k = n * wbar_k` (spec 2)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.sum(1)
    wbar = X / np.clip(n, 1e-12, None)[:, None]
    return wbar @ A, n


def velocity(X, model, arch, oxygen, tau=None):
    """`F_k(X) = X_k (pi_k(X) - pi_bar_k) / tau_k` (spec 3.3).

    `n = sum(X)` is recomputed at every point. Freezing it would delete the
    density channel, which is the only thing that can bound the flow.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tau = np.ones(arch.K) if tau is None else np.asarray(tau, dtype=float)
    phibar, n = covariates_from_state(X, arch.A)
    P = np.empty((len(X), arch.K))
    for k in range(arch.K):
        P[:, k] = model.pi(np.repeat(arch.A[k][None], len(X), axis=0), phibar, n, oxygen)
    return X * (P - arch.pi_bar[None, :]) / tau[None, :], P


def jacobian(X, model, arch, oxygen, tau=None):
    """Analytic Jacobian of `F` (spec 4), one state at a time."""
    X = np.asarray(X, dtype=float).ravel()
    tau = np.ones(arch.K) if tau is None else np.asarray(tau, dtype=float)
    phibar, n = covariates_from_state(X[None], arch.A)
    phibar, n = phibar[0], float(n[0])
    F, P = velocity(X[None], model, arch, oxygen, tau)
    P = P[0]
    J = np.zeros((arch.K, arch.K))
    for k in range(arch.K):
        g_k = model.deta_dphibar(arch.A[k])[0]
        sp = float(model.dh_dlogn(arch.A[k], np.array([n]))[0]) / n   # d/dn, not d/dlog n
        for j in range(arch.K):
            deta = g_k @ (arch.A[j] - phibar) / n + sp
            J[k, j] = (1.0 if k == j else 0.0) * (P[k] - arch.pi_bar[k]) / tau[k] \
                + X[k] * P[k] * (1 - P[k]) / tau[k] * deta
    return J


def interaction_matrix(model, arch):
    """`Atilde = A B A^T` (spec 4) -- the generalised Lotka-Volterra matrix the
    portrait is a geometric rendering of."""
    return arch.A @ model.B @ arch.A.T
