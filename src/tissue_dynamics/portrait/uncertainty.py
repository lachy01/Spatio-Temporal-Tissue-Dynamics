"""Portrait uncertainty (portrait spec 8), as Spartan array jobs.

  portrait_param_boot  spec 8.1. Draws of the mixture's parameters from
                       MVN(theta_hat, V), V the block-cluster sandwich covariance
                       of the production fit (run/portrait_uncertainty_prepare.py
                       writes it to outputs/portrait/uncertainty/param_cov.npz).
                       The spec names a Conley HAC covariance; the block form is
                       the same idea over the 250 spatial blocks the bootstrap and
                       CV already use, with no bandwidth to choose. 50 tasks x 10
                       draws = 500.
  portrait_block_boot  spec 8.2. Contiguous spatial blocks resampled with
                       replacement, the mixture refitted (warm-started from the
                       point estimate, as the `bootstrap` job does), the portrait
                       recomputed. 50 replicates. The full parameters are kept,
                       which the 300-replicate `bootstrap` job did not (it lacks
                       a0 and K, both of which the portrait needs).

Each replicate's death anchor is recomputed from its own fit: the mean posterior
pi over the cells of each archetype (over the resampled cells for the block
bootstrap). The portrait geometry -- archetype centroids A, the observed states,
the slice grids and support masks -- is held at its full-data values, so every
replicate is read on the same axes (spec 3.1's frozen-basis rule). Both states
are summarised: barycentric and type counts (FINDINGS §11).
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

COV_PATH = "outputs/portrait/uncertainty/param_cov.npz"
N_DRAWS_PER_TASK = 10
STATES = ("barycentric", "counts")


def _setup(cfg):
    import jax.numpy as jnp

    from ..mixture.basis import spec_from_basis
    from ..mixture.model import build_data, responsibilities
    from ..mixture.tasks import _load
    from .core import build_archetypes
    from .summary import build_slices

    df, basis = _load(cfg)
    hpc = cfg.get("hpc_dir") or "outputs/hpc"
    pe = json.load(open(os.path.join(hpc, "point_estimate.json")))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    w0 = np.asarray(responsibilities(P0, data.as_jax(), spec))
    radius = float(cfg["data"]["neighbour_radius_um"])
    geo = {}
    for st in STATES:
        arch, _W = build_archetypes(df, basis, pi_hat=w0, state=st, radius_um=radius)
        geo[st] = (arch, build_slices(arch.X_obs, arch.names))
    ctx = dict(df=df, basis=basis, spec=spec, data=data, pe=pe, geo=geo,
               arche=df["archetype"].astype(str).to_numpy(),
               dens=df["density"].to_numpy(float), nbr_own=basis.disc_in_own(df),
               ox=float(np.median(df["oxygen"])))
    return ctx


def _summaries(ctx, params, pi_hat, cells=None):
    """Both states' summaries for one parameter set; anchors from `pi_hat` over
    `cells` (row indices into the table; all cells when None)."""
    from .core import PortraitModel
    from .summary import summarise

    model = PortraitModel.from_fit({k: np.asarray(v) for k, v in params.items()}, ctx["basis"])
    arche = ctx["arche"] if cells is None else ctx["arche"][cells]
    out = {}
    for st in STATES:
        arch, slices = ctx["geo"][st]
        pb = np.array([pi_hat[arche == n].mean() for n in arch.names])
        a2 = dataclasses.replace(arch, pi_bar=pb)
        out[st] = summarise(model, a2, ctx["ox"], slices, ctx["dens"], ctx["nbr_own"], ctx["arche"])
    return out


def run_param_chunk(task, cfg, say=None):
    """spec 8.1: `N_DRAWS_PER_TASK` draws from MVN(theta_hat, V)."""
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from ..mixture.model import responsibilities

    say = say or (lambda *a: None)
    ctx = _setup(cfg)
    z = np.load(COV_PATH)
    theta, evals, evecs = z["theta"], z["evals"], z["evecs"]
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in ctx["pe"]["params"].items()}
    flat0, unravel = ravel_pytree(P0)
    if not np.allclose(np.asarray(flat0), theta):
        raise RuntimeError("param_cov.npz was computed on a different point estimate")
    rng = np.random.default_rng(20261020 + int(task["chunk"]))
    D = ctx["data"].as_jax()
    draws = []
    for b in range(N_DRAWS_PER_TASK):
        zb = rng.standard_normal(len(evals))
        th = theta + evecs @ (np.sqrt(np.clip(evals, 0, None)) * zb)
        P = unravel(jnp.asarray(th))
        w = np.asarray(responsibilities(P, D, ctx["spec"]))
        draws.append({"draw": int(task["chunk"]) * N_DRAWS_PER_TASK + b,
                      "summary": _summaries(ctx, P, w)})
        say(f"  draw {b + 1}/{N_DRAWS_PER_TASK}")
    return {"chunk": int(task["chunk"]), "draws": draws}


def run_block_rep(task, cfg, say=None):
    """spec 8.2: one spatial-block bootstrap refit and its portrait."""
    import jax.numpy as jnp

    from ..mixture.fit import fit_lbfgs
    from ..mixture.inference import spatial_blocks
    from ..mixture.model import responsibilities

    say = say or (lambda *a: None)
    ctx = _setup(cfg)
    xy = ctx["df"][["x", "y"]].to_numpy(float)
    labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
    rng = np.random.default_rng(int(task["rep"]))            # the `bootstrap` job's draws
    blocks = np.unique(labels)
    idx_of = {b: np.where(labels == b)[0] for b in blocks}
    pick = rng.choice(blocks, size=len(blocks), replace=True)
    idx = np.concatenate([idx_of[b] for b in pick])
    sub = ctx["data"].subset(idx)
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in ctx["pe"]["params"].items()}
    res = fit_lbfgs(sub, ctx["spec"], P0, maxiter=int(cfg["inference"]["bootstrap_maxiter"]))
    say(f"  refit: converged {res.converged}, loglik {res.loglik:,.1f}")
    w = np.asarray(responsibilities(res.params, sub.as_jax(), ctx["spec"]))
    return {"rep": int(task["rep"]), "n_cells": int(len(idx)), "converged": bool(res.converged),
            "loglik": float(res.loglik),
            "params": {k: np.asarray(v).tolist() for k, v in res.params.items()},
            "summary": _summaries(ctx, res.params, w, cells=idx)}
