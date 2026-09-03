"""Generators for the simulation-based tests (spec 6.1, 6.2, 6.3, 6.7).

Every generator returns `(MixtureData, truth_dict)` so a test can fit the model
and compare against the parameters that produced the data. The design blocks are
drawn to resemble the real sample -- centred state scores, a log library size on
the observed scale, and a component rate near the observed mean count -- because
a generator on the wrong scale makes finite-difference and recovery checks
meaningless (a likelihood dominated by a huge constant offset has gradients so
small relative to it that a central difference cannot resolve them).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .model import MixtureData, ModelSpec


def _spline(x, df):
    """Natural cubic basis, standardised -- the same construction the model uses.

    Raw monomials were used here originally and are unusable past df=4: `la**5`
    and `la**6` are so collinear and so heavy-tailed that the fit diverges
    (log-likelihood -2.8e10 at df=6). A `cr` basis is what `basis.py` builds on
    real data, so the simulator should use it too.
    """
    from .basis import _Spline

    b = _Spline(x, df)(x)
    sd = b.std(0)
    return b / np.where(sd > 0, sd, 1.0)      # unit scale, natural shape preserved


def _design(n, d_own, d_nbr, df_area=4, df_weight=3, df_bg=3, rng=None):
    """Design blocks on the real sample's scale."""
    rng = np.random.default_rng(0) if rng is None else rng
    logN = rng.normal(0.0, 0.9, n)                 # centred; sd near the real 1.0
    la = rng.normal(0.0, 0.5, n)                   # standing in for log area
    area = _spline(la, df_area)
    dens = rng.normal(0.0, 1.0, n)
    hb = _spline(dens, df_weight)
    gb = _spline(dens, df_bg)
    own = rng.normal(0.0, 1.0, (n, d_own))
    nbr = rng.normal(0.0, 1.0, (n, d_nbr))
    own -= own.mean(0)
    nbr -= nbr.mean(0)
    ox = rng.uniform(0.0, 1.0, n)
    return dict(logN=logN, area=area, own=own, nbr=nbr, hbasis=hb, oxygen=ox,
                gbasis=gb)


def _draw(truth, blocks, spec, rng, wiggly_area=None):
    """Sample counts from the generative model of spec 2.1."""
    n = len(blocks["logN"])
    theta = truth.get("theta", 1.0)
    base = truth["b0"] + theta * blocks["logN"] + blocks["area"] @ truth["s"]
    if wiggly_area is not None:                    # spec 6.7: misspecified area effect
        base = base + wiggly_area
    lam0 = np.exp(base)
    lam1 = np.exp(base + np.exp(truth["delta"]))

    eta = np.full(n, truth["a0"])
    if spec.has_state:
        eta = (eta + blocks["own"] @ truth["a"] + blocks["nbr"] @ truth["c"]
               + np.einsum("ij,jk,ik->i", blocks["own"], truth["B"], blocks["nbr"])
               + blocks["hbasis"] @ truth["h"] + truth["psi"] * blocks["oxygen"])
    pi = 1.0 / (1.0 + np.exp(-eta))
    rho = (np.exp(truth["r0"] + blocks["gbasis"] @ truth["g"])
           if spec.has_background else np.zeros(n))

    z = rng.random(n) < pi
    y = rng.poisson(rho + np.where(z, lam1, lam0)).astype(float)
    return y, pi, z


def simulate_stage_b(n=100_000, d_own=6, d_nbr=8, rank=2, seed=0,
                     e_delta=np.log(4.0), mean_pi=0.15, b_scale=0.26,
                     theta_free=True, single_component=False,
                     wiggly_area_df=None, fit_area_df=4):
    """Spec 6.1/6.2/6.3/6.7 generator.

    Parameters
    ----------
    e_delta : float
        The value of `e^delta`, so the component rate ratio is `exp(e_delta)`.
        The spec's Test 01 asks for a rate ratio of 4, i.e. `e_delta = log 4`.
    b_scale : float
        Scale of `B`'s entries. The default gives an odds-ratio modification of
        roughly 1.3 across a one-sd move in the state, per spec 6.1.
    single_component : bool
        Spec 6.3: draw with `pi_i == 0`, i.e. no latent class structure at all.
    wiggly_area_df : int or None
        Spec 6.7: add an area effect built from a `wiggly_area_df` basis that a
        `fit_area_df` spline cannot represent, with no class structure. The
        adversarial size-confound test.
    """
    rng = np.random.default_rng(seed)
    blocks = _design(n, d_own, d_nbr, df_area=fit_area_df, rng=rng)
    # the spline bases carry df-1 columns (the constant is projected out), so the
    # spec's column counts are read off the blocks rather than assumed
    spec = ModelSpec(stage="B", rank=rank, theta_free=theta_free,
                     d_own=d_own, d_nbr=d_nbr,
                     df_area=blocks["area"].shape[1],
                     df_weight=blocks["hbasis"].shape[1],
                     df_bg=blocks["gbasis"].shape[1])

    a0 = float(np.log(mean_pi / (1 - mean_pi)))
    truth = {
        "b0": -0.55, "delta": float(np.log(e_delta)),
        "s": rng.normal(0, 0.15, spec.df_area),
        "a0": a0, "a": rng.normal(0, 0.30, d_own), "c": rng.normal(0, 0.30, d_nbr),
        "h": rng.normal(0, 0.10, spec.df_weight), "psi": 0.35,
        "r0": -3.0, "g": np.zeros(spec.df_bg),
    }
    if theta_free:
        truth["theta"] = 0.85
    U = rng.normal(0, b_scale, (d_own, rank))
    V = rng.normal(0, b_scale, (d_nbr, rank))
    truth["B"] = U @ V.T
    truth["U"], truth["V"] = U, V

    wig = None
    if wiggly_area_df is not None:
        # an area effect the fitted df=4 spline provably cannot absorb
        la = rng.normal(0.0, 0.5, n)
        W = np.column_stack([np.sin(3 * la), np.cos(5 * la), np.sin(8 * la),
                             np.cos(11 * la), np.sin(14 * la)])
        wig = W @ rng.normal(0, 0.45, W.shape[1])
        wig -= wig.mean()
        blocks["area"] = _spline(la, fit_area_df)
        if blocks["area"].shape[1] != spec.df_area:      # keep truth["s"] aligned
            spec = dataclasses.replace(spec, df_area=blocks["area"].shape[1])
            truth["s"] = rng.normal(0, 0.15, spec.df_area)

    if single_component or wiggly_area_df is not None:
        truth["a0"] = -50.0                        # pi ~ 0 for every cell
        truth["a"] = np.zeros(d_own)
        truth["c"] = np.zeros(d_nbr)
        truth["B"] = np.zeros((d_own, d_nbr))
        truth["h"] = np.zeros(spec.df_weight)
        truth["psi"] = 0.0

    y, pi, z = _draw(truth, blocks, spec, rng, wiggly_area=wig)
    data = MixtureData(y=y, logN=blocks["logN"], area=blocks["area"],
                       own=blocks["own"], nbr=blocks["nbr"],
                       hbasis=blocks["hbasis"], oxygen=blocks["oxygen"])
    truth["pi"], truth["z"] = pi, z
    truth["mean_y"] = float(y.mean())
    truth["zero_frac"] = float((y == 0).mean())
    return data, truth, spec


def simulate_null_bilinear(n=100_000, seed=0, **kw):
    """Spec 6.2: everything as Test 01 but `B = 0`."""
    return simulate_stage_b(n=n, seed=seed, b_scale=0.0, **kw)
