"""Work-unit enumeration for SLURM array jobs.

Every expensive part of this model is a set of *independent* fits: bootstrap
replicates, cross-validation folds, multi-start restarts, simulation replicates.
This module turns each into a flat list of self-describing tasks, so a SLURM
array index maps to exactly one unit of work with no coordination between them.

The contract is deliberately narrow: `enumerate_tasks(job, cfg)` returns a list
of plain dicts, `run_task(task, cfg, ...)` executes one and returns a plain dict
of results, and `hpc/worker.py` is the thin wrapper that writes it to disk. That
keeps the parallel layer testable locally -- you can run any task on this laptop
by index and get the same answer the cluster would.

Seeds are derived from the task index, so a replicate is reproducible whether it
runs as array element 173 or on its own.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def enumerate_tasks(job, cfg):
    """All work units for `job`, as a list of dicts. Index in the list is the
    SLURM array index."""
    mc, ic, nc = cfg["model"], cfg["init"], cfg["inference"]
    if job == "bootstrap":
        return [{"job": job, "rep": r} for r in range(int(nc["n_bootstrap"]))]
    if job in ("cv_penalty", "cv_penalty_unwhitened", "cv_penalty_buffered"):
        # cv_penalty_unwhitened: the same grid with the state blocks left
        # unwhitened -- the test of state.whiten. cv_penalty_buffered: training
        # cells within cv_buffer_um of the held-out fold dropped -- the check on
        # leakage across fold edges.
        extra = {"cv_penalty": {}, "cv_penalty_unwhitened": {"whiten": False},
                 "cv_penalty_buffered": {"buffer_um": float(mc.get("cv_buffer_um", 200.0))}}[job]
        return [{"job": job, "lam": float(lam), "fold": f, **extra}
                for lam in mc["weight_ridge_grid"]
                for f in range(int(mc.get("cv_folds", 5)))]
    if job == "cv_rank":
        return [{"job": job, "rank": int(r), "fold": f}
                for r in cfg.get("rank_grid", [1, 2, 3, 5])
                for f in range(int(mc.get("cv_folds", 5)))]
    if job == "multistart":
        return [{"job": job, "restart": r} for r in range(int(ic["n_restarts"]))]
    if job == "heldout_genes":
        return [{"job": job, "split": s} for s in range(int(cfg.get("n_gene_splits", 5)))]
    if job == "theta_bracket":
        # Spec 2.2 / 6.12: F and V are a BRACKET, reported side by side. They are
        # not competing models to be selected between, and spec 9.11 forbids
        # picking one by fit.
        return [{"job": job, "theta_free": tf, "restart": r}
                for tf in (False, True) for r in range(int(cfg.get("bracket_restarts", 5)))]
    if job == "weight_variants":
        # Isolate what the density interaction buys over simply giving h() more
        # knots. All three share the same explicit knots, so the only thing
        # varying is how the density response is allowed to differ by cell type.
        return [{"job": job, "variant": v, "restart": r}
                for v in ("knots_only", "linear_interaction", "spline_interaction")
                for r in range(int(cfg["init"]["n_restarts"]))]
    if job == "subsample_stability":
        # Spec 6.10: fit at 50K, 200K and the full set; rate ratios must be
        # stable within their bootstrap CIs and mean(pi) must not drift with n.
        return [{"job": job, "n_cells": n, "rep": r}
                for n in cfg.get("stability_sizes", [50000, 200000, 0])
                for r in range(int(cfg.get("stability_reps", 3)))]
    if job == "density_sensitivity":
        # Spec 6.13: the density spline in the WEIGHT competes with c and B --
        # density and phi_bar are both spatially smooth on similar scales, so a
        # flexible h() can absorb genuine neighbourhood signal. This is spatial
        # confounding acting on the primary estimand, not on a nuisance term.
        #
        # The sweep varies the KNOTS, not `df`. Once explicit knot quantiles are
        # configured they override `df` entirely, so a df sweep silently fits the
        # same model repeatedly -- which is exactly what happened in the previous
        # run, where the df=3 and df=5 arms returned identical log-likelihoods.
        return [{"job": job, "knots": kq, "rep": r}
                for kq in cfg.get("density_knot_grid",
                                  [None, [50], [25, 50, 75], [25, 50, 75, 90]])
                for r in range(int(cfg.get("stability_reps", 3)))]
    if job == "stage_c":
        # Additive background. EM does not apply (spec 4.3), so direct only.
        return [{"job": job, "restart": r}
                for r in range(int(cfg["init"]["n_restarts"]))]
    if job == "stage_d":
        return [{"job": job, "variant": v, "restart": r}
                for v in ("free_theta_per_component", "rank_sweep_1", "rank_sweep_3",
                          "rank_sweep_5", "kNN_density")
                for r in range(int(cfg.get("stage_d_restarts", 3)))]
    if job in ("sim_recovery", "sim_null_bilinear", "sim_single_component",
               "sim_size_confound"):
        # Test 03 runs at the real n (sim_test03_n): invented structure shrinks
        # with n, so a smaller null overstates what the real fit risks
        extra = ({"n": int(cfg["sim_test03_n"])}
                 if job == "sim_single_component" and cfg.get("sim_test03_n") else {})
        return [{"job": job, "rep": r, **extra} for r in range(int(cfg.get("sim_reps", 20)))]
    if job == "sim_single_component_path":
        # Test 03 along a lambda path: on data with no class structure, which
        # weight block grows as the penalty is relaxed.
        return [{"job": job, "lam": float(lam), "rep": r}
                for lam in cfg.get("sim_lambda_grid", [0.0, 1.0, 10.0, 100.0, 1000.0])
                for r in range(int(cfg.get("sim_path_reps", 8)))]
    if job == "sim_single_component_path_full":
        # The same at the real sample size: invented structure shrinks with n,
        # so the n = 100,000 path overstates what the full-table fit risks.
        return [{"job": job, "lam": float(lam), "rep": r, "n": int(cfg.get("sim_full_n", 561897))}
                for lam in cfg.get("sim_full_lambda_grid", [30.0, 100.0, 300.0, 1000.0, 3000.0])
                for r in range(int(cfg.get("sim_full_reps", 4)))]
    raise ValueError(f"unknown job {job!r}")


def _load(cfg):
    """Table + frozen basis. Both are read-only, so every task builds them the
    same way and no task can perturb another's parameterisation (spec 3.1)."""
    import pandas as pd

    from .basis import build_basis

    import json
    import os

    path = cfg["data"]["table_parquet"]
    df = pd.read_parquet(path)
    # Prefer the sidecar written by run/build_table.py; fall back to the source
    # CSV only if it is absent, so a cluster run needs the parquet alone.
    side = os.path.splitext(path)[0] + ".phi_names.json"
    if os.path.exists(side):
        with open(side) as fh:
            df.attrs["phi_names"] = json.load(fh)
    else:
        df.attrs["phi_names"] = list(
            pd.read_csv(cfg["data"]["phi_csv"], index_col=0, nrows=1).columns)
    return df, build_basis(df, cfg, verbose=False)


def run_task(task, cfg, verbose=False, out_dir=None):
    """Execute one work unit. Returns a JSON-serialisable dict.

    A specification that cannot be fit at all (every restart diverging, which
    happens on the real data whenever `weight_ridge` is 0) is **recorded as a
    failed task**, not raised. An array element that crashes looks identical to
    one that was never scheduled, and `gather.py` would then report it as
    missing rather than as the finding it is.
    """
    job = task["job"]
    try:
        return _run_sim(task, cfg) if job.startswith("sim_") else _run_real(
            task, cfg, verbose=verbose, out_dir=out_dir)
    except RuntimeError as exc:
        return {**task, "failed": str(exc), "converged": False, "diverged": True}


def _summarise(res, data, spec, extra=None, include_params=False):
    from .model import (B_matrix, normalised_entropy, responsibilities,
                        singular_values)

    D = data.as_jax()
    w = np.asarray(responsibilities(res.params, D, spec))
    sv = singular_values(res.params)
    out = {
        # provenance: the settings this fit actually used, so a collector never
        # has to infer them by re-reading a config that may have moved on
        "weight_ridge": float(spec.weight_ridge), "rank": int(spec.rank),
        "df_area": int(spec.df_area), "stage": spec.stage,
        "theta_free": bool(spec.theta_free),
        "loglik": float(res.loglik), "grad_inf": float(res.grad_inf),
        "n_iter": int(res.n_iter), "converged": bool(res.converged),
        "diverged": bool(res.diverged), "wall_time": float(res.wall_time),
        "e_delta": float(np.exp(np.asarray(res.params["delta"]))),
        "mean_w": float(w.mean()), "entropy": float(normalised_entropy(w)),
        "sigma": [float(v) for v in np.atleast_1d(sv)],
        # spec 3.2: B is the identified object, never U or V
        "B": B_matrix(res.params).tolist() if spec.has_state else None,
        "b_penalty": getattr(spec, "b_penalty", None),
    }
    if spec.has_state:
        out.update(_eta_blocks(res.params, data, spec))
    for k in ("a", "c", "psi", "b0", "theta", "h"):
        if k in res.params:
            out[k] = np.asarray(res.params[k]).tolist()
    if include_params:
        # the FULL pytree, including U, V, delta, a0, s -- the bootstrap
        # warm-starts from it, and the summary fields above are not enough to
        # reconstruct a starting point
        out["params"] = {k: np.asarray(v).tolist() for k, v in res.params.items()}
    out.update(extra or {})
    return out


def _eta_blocks(p, data, spec):
    """Each weight block's contribution to the spread of logit(pi) across cells.

    On the logit scale, so blocks are comparable whatever their covariates'
    scales -- unlike coefficient norms. Tracked along the lambda path, the block
    whose contribution keeps growing as lambda -> 0 is the one the penalty is
    restraining (and the honest basis for any differential penalty).
    """
    own, nbr = np.asarray(data.own), np.asarray(data.nbr)
    P = {k: np.asarray(v) for k, v in p.items()}
    parts = {"own": own @ P["a"], "nbr": nbr @ P["c"],
             "bilinear": np.sum((own @ P["U"]) * (nbr @ P["V"]), axis=1)}
    if spec.df_weight > 0:
        hb = np.asarray(data.hbasis)
        parts["density"] = hb @ P["h"]
        if "K" in P:
            parts["density_by_type"] = np.einsum("ij,jm,im->i", own, P["K"], hb)
    if "psi" in P:
        parts["oxygen"] = P["psi"] * np.asarray(data.oxygen)
    total = sum(parts.values())
    out = {f"sd_eta_{k}": float(np.std(v)) for k, v in parts.items()}
    out["sd_eta"] = float(np.std(total))
    UtU, VtV = P["U"].T @ P["U"], P["V"].T @ P["V"]
    out["norm_K"] = float(np.linalg.norm(P["K"])) if "K" in P else None
    out["balance_gap"] = float(np.linalg.norm(UtU - VtV))
    return out


def _run_real(task, cfg, verbose=False, out_dir=None):
    from .basis import build_basis, spec_from_basis
    from .fit import fit_lbfgs, multistart, poisson_start
    from .inference import block_folds, spatial_blocks
    from .model import build_data, init_params

    job = task["job"]
    if "whiten" in task:                       # cv_penalty_unwhitened
        cfg = {**cfg, "state": {**cfg["state"], "whiten": bool(task["whiten"])}}
    df, basis = _load(cfg)
    mc = cfg["model"]
    lam = task.get("lam", mc.get("weight_ridge") or 0.0)
    spec = spec_from_basis(basis, cfg, stage=cfg.get("stage", "B"),
                           weight_ridge=float(lam), rank=task.get("rank"))
    data = build_data(df, basis, cfg, spec)
    xy = df[["x", "y"]].to_numpy(float)

    if job in ("cv_penalty", "cv_penalty_unwhitened", "cv_penalty_buffered", "cv_rank"):
        labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
        folds = block_folds(labels, int(mc.get("cv_folds", 5)))
        f = int(task["fold"])
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        n_dropped = 0
        if float(task.get("buffer_um", 0.0)) > 0:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(xy[te]).query(xy[tr], k=1)
            keep = dist > float(task["buffer_um"])
            n_dropped = int((~keep).sum())
            tr = tr[keep]
        best, _ = multistart(data.subset(tr), spec, cfg, n_restarts=2,
                             seed=1000 * f, verbose=verbose)
        import jax.numpy as jnp

        from .model import loglik_terms
        ll_te = np.asarray(loglik_terms(best.params, data.subset(te).as_jax(), spec))
        held = float(ll_te.sum())
        # Held-out log-likelihood per spatial block: the blocks, not the five
        # folds, are the units the selection rule resamples (hpc/gather.py), so
        # its uncertainty reflects ~50 independent-ish regions per fold rather
        # than 4 degrees of freedom.
        bl = labels[te]
        ub = np.unique(bl)
        by_block = {int(b): [float(ll_te[bl == b].sum()), int((bl == b).sum())] for b in ub}
        return {**task, **_summarise(best, data.subset(tr), spec,
                                     {"heldout_loglik_per_cell": held / len(te),
                                      "heldout_by_block": by_block,
                                      "n_train": len(tr), "n_test": len(te),
                                      "n_buffer_dropped": n_dropped,
                                      "whiten": bool(cfg["state"].get("whiten", False))})}

    if job == "multistart":
        r = int(task["restart"])
        ps = poisson_start(data, spec)
        rng = None if r == 0 else np.random.default_rng(r)
        p0 = init_params(spec, cfg, data=data, rng=rng, poisson_start=ps)
        res = fit_lbfgs(data, spec, p0, maxiter=int(cfg["optim"]["maxiter"]),
                        rel_tol=float(cfg["optim"]["rel_tol_loglik"]),
                        grad_tol=float(cfg["optim"]["grad_inf_tol"]))
        from .diagnostics import confound_diagnostic
        from .model import responsibilities
        out = {**task, **_summarise(res, data, spec, include_params=True)}
        # spec 10: a fit reported without the entropy and the confound R^2 is
        # "not usable", so they travel with the estimate rather than on request
        out.update(confound_diagnostic(
            np.asarray(responsibilities(res.params, data.as_jax(), spec)), df, basis))
        return out

    if job in ("theta_bracket", "stage_c", "stage_d", "subsample_stability",
               "density_sensitivity", "weight_variants"):
        import dataclasses

        from .diagnostics import confound_diagnostic
        from .model import responsibilities

        sp_ = spec
        if job == "theta_bracket":
            sp_ = dataclasses.replace(spec, theta_free=bool(task["theta_free"]))
        elif job == "stage_c":
            sp_ = dataclasses.replace(spec, stage="C")
        elif job == "stage_d":
            v = task["variant"]
            if v.startswith("rank_sweep_"):
                sp_ = dataclasses.replace(spec, rank=int(v.rsplit("_", 1)[1]))
            elif v == "kNN_density":
                # Spec 1.3 prefers fixed-k kNN density to the radius definition
                # this project uses. Recompute the column here rather than
                # leaving the variant a silent no-op returning Stage B.
                from scipy.spatial import cKDTree
                k = int(cfg.get("knn_density_k", 30))
                dist, _ = cKDTree(xy).query(xy, k=k + 1)
                df = df.copy()
                df["density"] = k / (np.pi * np.maximum(dist[:, -1], 1e-9) ** 2) * 1e4
                basis = build_basis(df, cfg, verbose=False)
                sp_ = spec_from_basis(basis, cfg, stage=cfg.get("stage", "B"),
                                      weight_ridge=float(lam))
            elif v == "free_theta_per_component":
                raise NotImplementedError(
                    "relaxing the proportional-rates restriction needs a second "
                    "theta in the model, which model.py does not carry. Spec 2.2 "
                    "makes that restriction load-bearing for identifiability, so "
                    "it is a deliberate change, not a config flag.")
        elif job == "weight_variants":
            v = task["variant"]
            sp_ = dataclasses.replace(
                spec,
                density_interaction=(v != "knots_only"),
                density_interaction_spline=(v == "spline_interaction"))
        elif job == "subsample_stability":
            n_ = int(task["n_cells"])
            if n_ and n_ < len(df):
                rng_ = np.random.default_rng(1000 + int(task["rep"]))
                keep = np.sort(rng_.choice(len(df), n_, replace=False))
                df, xy = df.iloc[keep].reset_index(drop=True), xy[keep]
                df.attrs["phi_names"] = basis.phi_names
        elif job == "density_sensitivity":
            kq = task["knots"]
            if kq is None:
                # h() dropped entirely -- spec 6.13's mandatory comparison
                sp_ = dataclasses.replace(spec, df_weight=0)
            else:
                cfg = {**cfg, "model": {**cfg["model"],
                                        "weight_density_knot_quantiles": list(kq)}}
                basis = build_basis(df, cfg, verbose=False)
                sp_ = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(lam))
        data = build_data(df, basis, cfg, sp_)
        # subsample_stability and density_sensitivity index their repeats as
        # "rep"; the others use "restart". Reading only "restart" silently made
        # every repeat the SAME deterministic fit, so those jobs ran one start
        # three times and reported a bad local optimum as if it were stable.
        r_ = int(task.get("restart", task.get("rep", 0)))
        ps = poisson_start(data, sp_)
        p0 = init_params(sp_, cfg, data=data,
                         rng=None if r_ == 0 else np.random.default_rng(r_),
                         poisson_start=ps)
        res = fit_lbfgs(data, sp_, p0, maxiter=int(cfg["optim"]["maxiter"]),
                        rel_tol=float(cfg["optim"]["rel_tol_loglik"]),
                        grad_tol=float(cfg["optim"]["grad_inf_tol"]))
        out = {**task, **_summarise(res, data, sp_, include_params=True)}
        out.update(confound_diagnostic(
            np.asarray(responsibilities(res.params, data.as_jax(), sp_)), df, basis))
        return out

    if job == "heldout_genes":
        # Spec 6.8. Refit the mixture using only half A of the signature, then
        # ask whether the responsibilities predict half B beyond the nuisance
        # covariates -- against a spatially block-permuted null.
        from .heldout import heldout_gene_test
        from .model import responsibilities

        sp_ = int(task["split"])
        for col in (f"y_A{sp_}", f"y_B{sp_}"):
            if col not in df.columns:
                raise KeyError(
                    f"{col} missing from the table. Run run/add_gene_splits.py "
                    "locally and re-copy the parquet: the half-set counts are "
                    "not derivable from the summed y_count.")
        data.y = df[f"y_A{sp_}"].to_numpy(np.float64)     # fit on half A only
        best, _ = multistart(data, spec, cfg, n_restarts=3, seed=7 * sp_,
                             verbose=verbose)
        w = np.asarray(responsibilities(best.params, data.as_jax(), spec))
        labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
        res = heldout_gene_test(
            y_a=df[f"y_A{sp_}"].to_numpy(float), y_b=df[f"y_B{sp_}"].to_numpy(float),
            w=w, log_n=basis.log_N(df), area_basis=basis.log_area_basis(df),
            density_basis=basis.density_weight_basis(df), labels=labels,
            n_perm=int(cfg.get("n_permutations", 100)), seed=sp_, verbose=verbose)
        out = {**task, **_summarise(best, data, spec), **res}
        out["mean_y_A"] = float(df[f"y_A{sp_}"].mean())
        out["mean_y_B"] = float(df[f"y_B{sp_}"].mean())
        return out

    if job == "bootstrap":
        import json
        import os

        # The point estimate this replicate warm-starts from must already exist.
        # Fail with an instruction rather than a bare FileNotFoundError: a
        # missing prerequisite otherwise reappears identically in all 300 array
        # tasks with nothing in the message to say what to do about it.
        base = out_dir or cfg.get("hpc_dir") or "outputs/hpc"
        pt = os.path.join(base, "point_estimate.json")
        if not os.path.exists(pt):
            raise FileNotFoundError(
                f"{pt} not found. The bootstrap warm-starts from the point "
                "estimate, so it has to run after the multistart array:\n"
                "    ./hpc/submit.sh multistart\n"
                "    python run/collect_point_estimate.py\n"
                "    ./hpc/submit.sh bootstrap 50\n"
                "Chain them with --dependency=afterok:<jobid> to avoid the wait.")
        with open(pt) as fh:
            hat = json.load(fh)
        import jax.numpy as jnp
        params_hat = {k: jnp.asarray(np.asarray(v))
                      for k, v in hat["params"].items()}
        labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
        rng = np.random.default_rng(int(task["rep"]))
        blocks = np.unique(labels)
        idx_of = {b: np.where(labels == b)[0] for b in blocks}
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([idx_of[b] for b in pick])
        sub = data.subset(idx)
        res = fit_lbfgs(sub, spec, params_hat,
                        maxiter=int(cfg["inference"]["bootstrap_maxiter"]))
        return {**task, **_summarise(res, sub, spec, {"n_cells": len(idx)})}

    raise ValueError(f"unhandled job {job!r}")


def _run_sim(task, cfg):
    from .fit import multistart
    from .simulate import simulate_stage_b

    rep = int(task["rep"])
    n = int(task.get("n", cfg.get("sim_n", 100_000)))
    # the simulated neighbour block is as wide as the real one: 8 projected
    # coordinates for the disc, 16 for the contact + regional split
    d_nbr = 16 if cfg["state"].get("neighbourhood", "disc") == "split" else 8
    lam = float(task.get("lam", cfg["model"].get("weight_ridge") or 0.0))
    kw = {"sim_recovery": {},
          "sim_null_bilinear": {"b_scale": 0.0},
          "sim_single_component": {"single_component": True},
          "sim_single_component_path": {"single_component": True},
          "sim_single_component_path_full": {"single_component": True},
          "sim_size_confound": {"wiggly_area_df": 10}}[task["job"]]
    data, truth, spec = simulate_stage_b(n=n, seed=rep, d_own=8, d_nbr=d_nbr,
                                         fit_area_df=int(cfg["model"]["area_spline_df"]),
                                         **kw)
    spec = dataclasses.replace(spec, weight_ridge=lam)
    best, _ = multistart(data, spec, cfg, n_restarts=int(cfg.get("sim_restarts", 4)),
                         seed=100 * rep, verbose=False,
                         maxiter=int(cfg["optim"]["maxiter"]))
    out = _summarise(best, data, spec)
    out["truth"] = {k: (np.asarray(v).tolist() if np.ndim(v) else float(v))
                    for k, v in truth.items()
                    if k in ("b0", "theta", "delta", "a0", "psi", "a", "c", "B")}
    out["truth_sigma1"] = float(np.linalg.svd(np.asarray(truth["B"]),
                                              compute_uv=False)[0]) if np.any(truth["B"]) else 0.0
    return {**task, **out}
