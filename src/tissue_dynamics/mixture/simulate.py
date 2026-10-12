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


def _design(n, d_own, d_nbr, df_area=4, df_weight=3, df_bg=3, rng=None, mistic=False,
            area=True):
    """Design blocks on the real sample's scale.

    `mistic`: no area basis, and `m`, the expected misassigned outcome count,
    drawn like the real m = f N s -- proportional to the library size times a
    cell-to-cell factor (mean 0.1 x N here, so theta_m m is ~20% of lambda_0).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    logN = rng.normal(0.0, 0.9, n)                 # centred; sd near the real 1.0
    la = rng.normal(0.0, 0.5, n)                   # standing in for log area
    area = _spline(la, df_area) if area and not mistic else np.zeros((n, 0))
    dens = rng.normal(0.0, 1.0, n)
    hb = _spline(dens, df_weight)
    gb = _spline(dens, df_bg)
    own = rng.normal(0.0, 1.0, (n, d_own))
    nbr = rng.normal(0.0, 1.0, (n, d_nbr))
    own -= own.mean(0)
    nbr -= nbr.mean(0)
    ox = rng.uniform(0.0, 1.0, n)
    m = 0.1 * np.exp(logN) * rng.gamma(2.0, 0.5, n) if mistic else None
    return dict(logN=logN, area=area, own=own, nbr=nbr, hbasis=hb, oxygen=ox,
                gbasis=gb, m=m)


def _draw(truth, blocks, spec, rng, wiggly_area=None):
    """Sample counts from the generative model of spec 2.1."""
    n = len(blocks["logN"])
    theta = truth.get("theta", 1.0)
    base = truth["b0"] + theta * blocks["logN"]
    if spec.df_area > 0:
        base = base + blocks["area"] @ truth["s"]
    if wiggly_area is not None:                    # spec 6.7: misspecified area effect
        base = base + wiggly_area
    lam0 = np.exp(base)
    lam1 = np.exp(base + np.exp(truth["delta"]))

    eta = np.full(n, truth["a0"])
    if spec.has_state:
        eta = (eta + blocks["own"] @ truth["a"] + blocks["nbr"] @ truth["c"]
               + np.einsum("ij,jk,ik->i", blocks["own"], truth["B"], blocks["nbr"])
               + blocks["hbasis"] @ truth["h"])
        if spec.oxygen:
            eta = eta + truth["psi"] * blocks["oxygen"]
    pi = 1.0 / (1.0 + np.exp(-eta))
    rho = (np.exp(truth["r0"] + blocks["gbasis"] @ truth["g"])
           if spec.has_background else np.zeros(n))
    if spec.mistic:
        rho = rho + np.exp(truth["eta_m"]) * blocks["m"]

    z = rng.random(n) < pi
    y = rng.poisson(rho + np.where(z, lam1, lam0)).astype(float)
    return y, pi, z


def simulate_stage_b(n=100_000, d_own=6, d_nbr=8, rank=2, seed=0,
                     e_delta=np.log(4.0), mean_pi=0.15, b_scale=0.26,
                     theta_free=True, single_component=False,
                     wiggly_area_df=None, fit_area_df=4, weight_ridge=None, oxygen=None,
                     technical=None, theta_m=1.0):
    """Spec 6.1/6.2/6.3/6.7 generator.

    Parameters
    ----------
    e_delta : float
        The value of `e^delta`, so the component rate ratio is `exp(e_delta)`.
        The spec's Test 01 asks for a rate ratio of 4, i.e. `e_delta = log 4`.
    b_scale : float
        Scale of `B`'s entries. The default gives an odds-ratio modification of
        roughly 1.3 across a one-sd move in the state, per spec 6.1.
    weight_ridge : float or None
        The weight-model penalty the returned spec carries. `None` (the default)
        reads `model.weight_ridge` from the config, so a simulation exercises the
        SAME estimator as the real-data fit. Pass `0.0` for the unpenalised model
        the spec itself describes.

        This matters more than it looks. The penalty is scaled by
        `n / weight_ridge_n_reference` and so is constant per observation: a
        simulation at n = 100,000 carries `lambda = 10 * 100000/678885 = 1.47`,
        which is the same `lambda / n` as the full-sample fit. Before this was
        wired through, every simulation test ran at `weight_ridge = 0` and so
        validated a model that produced none of the reported results.
    single_component : bool
        Spec 6.3: draw with `pi_i == 0`, i.e. no latent class structure at all.
    wiggly_area_df : int or None
        Spec 6.7: add an area effect built from a `wiggly_area_df` basis that a
        `fit_area_df` spline cannot represent, with no class structure. The
        adversarial size-confound test.
    oxygen : bool or None
        Whether the weight carries `psi * oxygen`. `None` reads `model.oxygen`
        from the config, so simulations follow the production weight model
        (which dropped the term on 2026-10-06).
    technical : "area", "mistic", "plain" or None
        The technical covariates (`model.technical`; None reads the config).
        "mistic" drops the area spline and adds `theta_m * m_i` to every
        component's mean, with `m` drawn by `_design` and truth
        `eta_m = log(theta_m)`; "plain" has neither term. The size-confound
        generator (`wiggly_area_df`) is an area construction and needs "area".
    """
    from .data import load_config as _lc

    if technical is None:
        technical = str(_lc()["model"].get("technical", "area"))
    mistic = technical == "mistic"
    if technical != "area" and wiggly_area_df is not None:
        raise ValueError("the size-confound generator is built on area; use technical='area'")
    rng = np.random.default_rng(seed)
    blocks = _design(n, d_own, d_nbr, df_area=fit_area_df, rng=rng, mistic=mistic,
                     area=technical == "area")
    # the spline bases carry df-1 columns (the constant is projected out), so the
    # spec's column counts are read off the blocks rather than assumed
    if weight_ridge is None:
        from .data import load_config
        _mc = load_config()["model"]
        weight_ridge = float(_mc["weight_ridge"])
        n_reference = int(_mc["weight_ridge_n_reference"])
    else:
        # an explicit value is taken as given at the simulated n, so that
        # `weight_ridge=0.0` really is unpenalised rather than 0 x something
        n_reference = n
    from .data import load_config
    _mcfg = load_config()["model"]
    if oxygen is None:
        oxygen = bool(_mcfg.get("oxygen", True))
    # the weight model's penalty form follows the production config, so the
    # simulations exercise the same estimator as the real-data fit
    b_pen = str(_mcfg.get("b_penalty", "nuclear"))
    tau = float(_mcfg["balance_tau"] if b_pen == "frobenius" else _mcfg["ridge_tau"])
    spec = ModelSpec(stage="B", rank=rank, theta_free=theta_free,
                     d_own=d_own, d_nbr=d_nbr, oxygen=bool(oxygen),
                     b_penalty=b_pen, ridge_tau=tau,
                     df_area=blocks["area"].shape[1],
                     df_weight=blocks["hbasis"].shape[1],
                     df_bg=blocks["gbasis"].shape[1],
                     weight_ridge=weight_ridge, n_reference=n_reference,
                     mistic=mistic, technical=technical)

    a0 = float(np.log(mean_pi / (1 - mean_pi)))
    truth = {
        "b0": -0.55, "delta": float(np.log(e_delta)),
        "s": rng.normal(0, 0.15, spec.df_area),
        "a0": a0, "a": rng.normal(0, 0.30, d_own), "c": rng.normal(0, 0.30, d_nbr),
        "h": rng.normal(0, 0.10, spec.df_weight),
        "r0": -3.0, "g": np.zeros(spec.df_bg),
    }
    if spec.oxygen:
        truth["psi"] = 0.35
    if theta_free:
        truth["theta"] = 0.85
    if mistic:
        truth["eta_m"] = float(np.log(theta_m))
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
        if spec.oxygen:
            truth["psi"] = 0.0

    y, pi, z = _draw(truth, blocks, spec, rng, wiggly_area=wig)
    data = MixtureData(y=y, logN=blocks["logN"], area=blocks["area"], m=blocks["m"],
                       own=blocks["own"], nbr=blocks["nbr"],
                       hbasis=blocks["hbasis"],
                       oxygen=blocks["oxygen"] if spec.oxygen else None)
    truth["pi"], truth["z"] = pi, z
    truth["mean_y"] = float(y.mean())
    truth["zero_frac"] = float((y == 0).mean())
    return data, truth, spec


def simulate_null_bilinear(n=100_000, seed=0, **kw):
    """Spec 6.2: everything as Test 01 but `B = 0`."""
    return simulate_stage_b(n=n, seed=seed, b_scale=0.0, **kw)
