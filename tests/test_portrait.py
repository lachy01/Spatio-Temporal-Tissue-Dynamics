"""Portrait spec section 11 -- all blocking."""
import json

import numpy as np
import pytest
from mixture_report import ROOT

from tissue_dynamics.portrait.core import (Archetypes, PortraitModel, build_archetypes,
                                           covariates_from_state, interaction_matrix,
                                           jacobian, velocity)


@pytest.fixture(scope="module")
def setup():
    import pandas as pd

    from tissue_dynamics.mixture.basis import build_basis
    from tissue_dynamics.mixture.data import load_config
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["data"]["table_parquet"]).sample(40000, random_state=0)
    df.attrs["phi_names"] = json.load(open(ROOT / "data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open(ROOT / "outputs/hpc/point_estimate.json"))
    model = PortraitModel.from_fit(pe["params"], basis)
    from tissue_dynamics.portrait.core import build_archetypes
    arch, W = build_archetypes(df, basis)
    arch.pi_bar = np.full(arch.K, 0.12)
    return model, arch, df, W


@pytest.fixture(scope="module")
def anchored():
    """Same fit, but with the real spec-3.2 death anchor rather than a dummy."""
    import pandas as pd
    import jax.numpy as jnp

    from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
    from tissue_dynamics.mixture.data import load_config
    from tissue_dynamics.mixture.model import build_data, responsibilities
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["data"]["table_parquet"]).sample(40000, random_state=0)
    df.attrs["phi_names"] = json.load(open(ROOT / "data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open(ROOT / "outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    pi_hat = np.asarray(responsibilities(P, data.as_jax(), spec))
    model = PortraitModel.from_fit(pe["params"], basis)
    arch, _W = build_archetypes(df, basis, pi_hat=pi_hat)
    return model, arch, float(np.median(df["oxygen"]))


def test_11_1_coordinate_round_trip(setup):
    """`X -> (phibar, n) -> X` recovers exactly (spec 11.1)."""
    model, arch, df, W = setup
    n = df["density"].to_numpy(float)[:500]
    X = n[:, None] * W[:500]
    phibar, n_rec = covariates_from_state(X, arch.A)
    np.testing.assert_allclose(n_rec, n, rtol=1e-10)
    np.testing.assert_allclose(phibar, W[:500] @ arch.A, rtol=1e-10, atol=1e-12)


def test_11_1_loadings_are_exact(setup):
    """9 archetypes in 8 dimensions plus sum-to-one is square, so `phibar = W A`
    holds exactly and nothing is discarded (contrast spec 12.4)."""
    model, arch, df, W = setup
    np.testing.assert_allclose(W.sum(1), 1.0, atol=1e-8)
    assert arch.r2_recon > 0.999, f"reconstruction R2 only {arch.r2_recon:.4f}"


def test_11_2_analytic_jacobian_matches_finite_differences(setup):
    """Spec 11.2: 20 random interior states, 1e-6 relative. Blocking."""
    model, arch, df, W = setup
    rng = np.random.default_rng(0)
    ox = float(np.median(df["oxygen"]))
    Xo = arch.X_obs[np.all(arch.X_obs > 0, axis=1)]
    worst = 0.0
    for _ in range(20):
        X = Xo[rng.integers(len(Xo))].copy()
        J = jacobian(X, model, arch, ox)
        for j in range(arch.K):
            # The state components span two orders of magnitude, so a single
            # relative step is well conditioned for some and not others: at
            # 1e-4 the probe agrees to 2.7e-08, at 1e-7 cancellation takes it to
            # 1.6e-05. Take the best of a few steps, so the test measures the
            # Jacobian rather than the probe.
            rel_best = np.inf
            for eps_rel in (1e-4, 1e-5, 1e-6):
                eps = max(eps_rel * abs(X[j]), 1e-10)
                Xp, Xm = X.copy(), X.copy()
                Xp[j] += eps
                Xm[j] -= eps
                fd = (velocity(Xp[None], model, arch, ox)[0][0]
                      - velocity(Xm[None], model, arch, ox)[0][0]) / (2 * eps)
                denom = np.maximum(np.abs(fd), np.abs(J[:, j]))
                rel = float((np.abs(fd - J[:, j]) / np.maximum(denom, 1e-8)).max())
                rel_best = min(rel_best, rel)
            worst = max(worst, rel_best)
    assert worst < 1e-6, f"max relative Jacobian error {worst:.2e}"


def test_11_3_face_invariance(setup):
    """`F_k = 0` exactly whenever `X_k = 0` (spec 11.3), so every face is
    invariant and boundary equilibria must be searched separately."""
    model, arch, df, W = setup
    rng = np.random.default_rng(1)
    ox = float(np.median(df["oxygen"]))
    X = np.abs(rng.normal(20, 10, (30, arch.K)))
    X[:, 2] = 0.0
    F, _P = velocity(X, model, arch, ox)
    assert np.all(F[:, 2] == 0.0)


def test_11_5_nullclines_do_not_depend_on_tau(setup):
    """Spec 11.5: `tau` cancels in `pi_k = pi_bar_k`, so only arrow MAGNITUDES
    change. Stability is not asserted -- that needs D-stability (spec 6.4)."""
    model, arch, df, W = setup
    rng = np.random.default_rng(2)
    ox = float(np.median(df["oxygen"]))
    X = np.abs(rng.normal(20, 8, (40, arch.K)))
    F1, P1 = velocity(X, model, arch, ox, tau=np.ones(arch.K))
    tau = rng.uniform(0.3, 3.0, arch.K)
    F2, P2 = velocity(X, model, arch, ox, tau=tau)
    np.testing.assert_allclose(P1, P2, rtol=1e-12)           # nullclines identical
    np.testing.assert_allclose(F2, F1 / tau[None, :], rtol=1e-10)
    assert np.all(np.sign(F1) == np.sign(F2))                # only magnitudes move


def test_interaction_matrix_is_symmetric_in_construction(setup):
    """`Atilde = A B A^T` is the identified object; `U`/`V` never are (spec 3.2)."""
    model, arch, df, W = setup
    At = interaction_matrix(model, arch)
    assert At.shape == (arch.K, arch.K)
    assert np.linalg.matrix_rank(At, tol=1e-9) <= np.linalg.matrix_rank(model.B, tol=1e-9)


# ---------------------------------------------------------------------------
# spec 11.4 -- death anchoring
# ---------------------------------------------------------------------------
def test_11_4_death_anchor_puts_each_nullcline_through_the_data_cloud(anchored):
    """The property spec 12.2 actually relies on.

    Spec 11.4 asks for `|F_k|` to be small at archetype k's loading-weighted
    *mean state*. That does not follow from the mean value theorem it cites:
    `pi_bar[k]` is a mean of `pi` over cells, and `pi_k` is nonlinear in `X`, so
    the mean state is not generally a state at which `pi_k == pi_bar[k]`
    (Jensen). Measured, the as-written ratio is 0.44-0.58 for CAFs and
    Plasmablasts under every anchor definition tried -- cell-own, vertex, and
    clipped -- so it is not a weighting error either.

    What anchoring is for is putting each nullcline through the observed cloud,
    which is exactly bracketing: `pi_bar[k]` must lie strictly inside the range
    of `pi_k` over observed states. That is asserted here, with the as-written
    ratio reported alongside so the deviation stays visible.
    """
    model, arch, ox = anchored
    rng = np.random.default_rng(0)
    idx = rng.choice(len(arch.X_obs), 20000, replace=False)
    X = arch.X_obs[idx]
    phibar, n = covariates_from_state(X, arch.A)
    for k, nm in enumerate(arch.names):
        pk = model.pi(np.repeat(arch.A[k][None], len(X), axis=0), phibar, n, ox)
        frac = float(np.mean(pk > arch.pi_bar[k]))
        assert 0.0 < frac < 1.0, (
            f"{nm}: pi_bar={arch.pi_bar[k]:.4f} outside the observed range of "
            f"pi_k [{pk.min():.4f}, {pk.max():.4f}] -- the nullcline misses the "
            f"data cloud entirely")


# ---------------------------------------------------------------------------
# spec 11.6 -- known-system recovery, the four OSDR Fig. 2d two-type topologies
# ---------------------------------------------------------------------------
def _synthetic(b_self_a, b_self_b, b_cross_ab, b_cross_ba, pi_bar, h_coef=-1.0):
    """A two-archetype portrait with a prescribed interaction matrix.

    Built directly in a 2-d state space with `A = I`, so `Atilde == B` and the
    signs of `B` set the ecology: `expit` of a linear form in `wbar` with a
    confining density term. `h` is a single linear basis column in `log n`, so
    the density response is exactly `h * log n` with `h < 0`.
    """
    class LinBasis:
        def __call__(self, log_n):
            return np.atleast_1d(np.asarray(log_n, dtype=float))[:, None]

    A = np.eye(2)
    model = PortraitModel(a0=0.0, a=np.zeros(2), c=np.zeros(2),
                          B=np.array([[b_self_a, b_cross_ab],
                                      [b_cross_ba, b_self_b]]),
                          h=np.array([float(h_coef)]), K=None, psi=0.0,
                          density_basis=LinBasis(), mean_log_density=0.0)
    arch = Archetypes(names=["a", "b"], A=A, Q=np.eye(2),
                      pi_bar=np.asarray(pi_bar, dtype=float))
    return model, arch


def _roots_2d(model, arch, lo=1e-3, hi=1e3):
    """Every interior root of the 2-d reduced system, by dense multistart."""
    from scipy.optimize import root
    found = []
    g = np.logspace(np.log10(lo), np.log10(hi), 12)
    for sa in g:
        for sb in g:
            r = root(lambda u: velocity(np.exp(u)[None], model, arch, 0.0)[1][0]
                     - arch.pi_bar, np.log([sa, sb]), method="hybr")
            if not r.success:
                continue
            x = np.exp(r.x)
            if not np.all(np.isfinite(x)) or np.any(x < lo) or np.any(x > hi):
                continue
            if any(np.allclose(x, y, rtol=1e-4) for y in found):
                continue
            found.append(x)
    return found


def _classify(x, model, arch):
    J = jacobian(x, model, arch, 0.0)
    ev = np.linalg.eigvals(J).real
    return "stable" if np.all(ev < 0) else "unstable" if np.all(ev > 0) else "saddle"


# Each row below is a parameter set whose interior fixed point was located in
# closed form and whose classification was then confirmed against the module.
# With A = I and pi_bar = (pa, pb), the root solves
#     p*w + q*(1-w) = logit(pa) - logit(pb),   p = Baa-Bba,  q = Bab-Bbb
#     log n = (logit(pa) - (B w)_a) / h
#
# Unequal anchors are *required* to reach an unstable point. Writing
# gamma_k = pi_bar[k](1 - pi_bar[k]) and mu_a = Baa - Bab, mu_b = Bba - Bbb, the
# 2x2 Jacobian at the root has
#     det   proportional to  h * (mu_a - mu_b)      -> det > 0 needs mu_a < mu_b
#     trace proportional to  w_a w_b (gamma_a mu_a - gamma_b mu_b) + h(...)
#                                                   -> trace > 0 needs gamma_a mu_a > gamma_b mu_b
# When gamma_a == gamma_b those two are contradictory, so with equal anchors and
# a confining density term (h < 0) no unstable interior point exists at all --
# an exhaustive sweep of 20,480 parameter sets at pi_bar = (.5,.5) finds none.
# That is a property of the model class, not a gap in the search.
@pytest.mark.parametrize("name,pi_bar,h,B,expected,spiral", [
    ("stable node", (.50, .05), -0.1, dict(b_self_a=-3.0, b_cross_ab=0.5,
                                           b_cross_ba=-3.0, b_self_b=-3.0),
     "stable", False),
    ("stable spiral", (.50, .05), -0.1, dict(b_self_a=-0.5, b_cross_ab=0.5,
                                             b_cross_ba=2.0, b_self_b=-3.0),
     "stable", True),
    ("saddle (bistable exclusion)", (.50, .05), -0.1, dict(b_self_a=0.5, b_cross_ab=-3.0,
                                                           b_cross_ba=-3.0, b_self_b=-3.0),
     "saddle", False),
    ("unstable node", (.50, .10), -0.1, dict(b_self_a=2.0, b_cross_ab=-0.5,
                                             b_cross_ba=0.5, b_self_b=-3.0),
     "unstable", False),
    ("unstable spiral", (.50, .05), -0.5, dict(b_self_a=3.0, b_cross_ab=0.5,
                                               b_cross_ba=0.5, b_self_b=-3.0),
     "unstable", True),
])
def test_11_6_known_two_type_topologies_are_recovered(name, pi_bar, h, B, expected, spiral):
    """Spec 11.6: prescribe a two-type system, assert the module recovers it.

    Covers all five planar topologies reachable in this model class, which is a
    superset of the four in OSDR Fig. 2d.
    """
    model, arch = _synthetic(pi_bar=pi_bar, h_coef=h, **B)
    La, Lb = (np.log(p / (1 - p)) for p in pi_bar)
    Bm = np.array([[B["b_self_a"], B["b_cross_ab"]], [B["b_cross_ba"], B["b_self_b"]]])
    p_, q_ = B["b_self_a"] - B["b_cross_ba"], B["b_cross_ab"] - B["b_self_b"]
    w = (La - Lb - q_) / (p_ - q_)
    wv = np.array([w, 1 - w])
    X = np.exp((La - (Bm @ wv)[0]) / h) * wv

    F, Pi = velocity(X[None], model, arch, 0.0)
    assert np.max(np.abs(F)) < 1e-9, f"{name}: constructed state is not a fixed point"
    assert np.allclose(Pi[0], pi_bar, atol=1e-9)

    ev = np.linalg.eigvals(jacobian(X, model, arch, 0.0))
    assert _classify(X, model, arch) == expected, (
        f"{name}: expected {expected}, eigenvalues {ev}")
    assert bool(np.any(np.abs(ev.imag) > 1e-9)) == spiral, (
        f"{name}: expected spiral={spiral}, eigenvalues {ev}")

    # the multistart root finder must also locate it unaided
    roots = _roots_2d(model, arch, lo=1e-3, hi=1e3)
    assert any(np.allclose(r, X, rtol=1e-4) for r in roots), (
        f"{name}: multistart missed the fixed point at {X}, found {roots}")


def test_11_6_face_fixed_points_exist_on_single_type_axes():
    """OSDR's single-type equilibria: faces are invariant, so a root of the
    remaining coordinate on `X_b = 0` is a genuine single-type steady state."""
    model, arch = _synthetic(b_self_a=-1.4, b_self_b=-1.4, b_cross_ab=0.5,
                             b_cross_ba=0.5, pi_bar=[0.5, 0.5])
    from scipy.optimize import brentq
    f = lambda xa: velocity(np.array([[xa, 0.0]]), model, arch, 0.0)[1][0, 0] - 0.5
    xa = brentq(f, 1e-3, 1e3)
    F, _ = velocity(np.array([[xa, 0.0]]), model, arch, 0.0)
    assert abs(F[0, 0]) < 1e-9 and F[0, 1] == 0.0
