"""Known-truth simulation for the division classifier comparison (spec 7).

One replicate = one generator and one draw:

* Cells: a random `division_sim_n` of the real analysis cells, with their real
  state, neighbourhood and technical covariates (depth N and area A, or under
  MisTIC the weighted depth and m -- `model.technical`). The draw of cells is shared by the
  three generators of a replicate, so they are paired. Fold `rep % 5` of the
  stored spatial partition is held out; every arm is fitted on the other four
  and scored on it.
* Truth. The cycling probability pi_true is the production mixture's prior pi
  at each cell; Z ~ Bernoulli(pi_true).
  - G1 (home advantage for MIX): y | Z ~ Poisson(mu_Z), the fitted mixture
    (mu_Z = lambda_Z, plus theta_m m under MisTIC).
  - G2 (misspecified for MIX): as G1, but the cycling component is NB with one
    dispersion alpha_1, chosen so the model's pooled variance-to-mean ratio of y
    in Cancer Epithelial equals the observed one.
  - G3 (no latent classes): y ~ NB from the fitted NB-c (Stage 2's full fit).
    There is no Z; only label-scale scores and implied fractions are reported.
* Labels, rebuilt end to end from the simulated y, as Stage 1 builds them: the
  robustly trimmed technical null, BASE-T (alpha_B 0.05), BASE-O (per-type s+,
  Td 0.5) and SCORE. SCORE splits each cell's count across the Tirosh genes by
  the adopted allocation (sim_inputs.npz: one Dirichlet-multinomial profile for
  cycling and one for non-cycling cells per type; the state is Z in G1 and G2,
  and a draw from the fitted mixture's posterior given the simulated count in
  G3), holds each cell's control-gene term fixed, and refits the robust
  half-normal threshold.
* Arms, fitted on the training folds with the production weight block and
  penalty: NB-c, MIX (multistart, `division_sim_restarts` starts), LOG-T,
  LOG-O, LOG-S.

State-recovery metrics on the held-out fold (spec 7.3; G1 and G2):
  1. Spearman correlation of each arm's p (or MIX's pi) with pi_true;
  2. the neighbourhood effect: per own type, the difference in mean logit
     between cells at or above the type's 90th percentile of neighbourhood
     tumour fraction (the 80 um disc's Cancer Epithelial score) and cells at
     or below its 10th, for each arm's prediction and for pi_true;
  3. per-type fractions: each arm's mean prediction against the realised mean Z.
Label-scale scores (log loss, AUPRC, mean p / rate) are recorded for every
generator.

Addendum 1 (job `division_sim_addendum`, `addendum=True`; sections 2.2, 3.5
and 6). The same cells, truth and Stage 6 labels and arms -- the draws are
made in the same order, so a replicate pairs with its Stage 6 namesake --
plus:
  * every BASE-T label on the extended grid (alpha_B 0.001-0.2) for the naive
    null and for the truncated null refitted on the simulated y at each
    alpha_trim (0.001, 0.01, 0.05), with LOG-T fitted on each, and NB-c, MIX
    and TWO pushed through each;
  * TWO-P and TWO-PNB (step 1 on the training folds, step 2 on the weight
    block; TWO-PNB from three starts, the best kept), as pi and pushed through
    BASE-T and BASE-O;
  * B direction recovery: cosine of vec(B-hat) with vec(B_true) for every
    fitted arm, in the shared whitened coordinates (B_true is the production
    mixture's U V' under G1 and G2, and the full-data NB-c's under G3, which
    has no latent classes).
  * only with `division_sim_state: true` in the config (addendum section 4,
    which runs only if Lachy approves after the A3 gate): BASE-O-state rebuilt
    on the simulated y (s+(phi) from the gamma GLMs on the projected own
    state) and LOG-O-state fitted on it.
The per-type ratio spread (section 6) and the alpha_B surface (section 2.3)
are computed from the stored contrasts by run/division_addendum_a5_gather.py.

Label variant (job `division_sim_t10`, `variant="t10"`; labels.VARIANTS):
BASE-T is rebuilt from the truncated-likelihood null at the variant's
alpha_trim, refitted on the simulated y from the untrimmed fit as Stage 1
fits it, at the variant's primary alpha_B; inputs are read from, and
compared with, outputs/division_classifier_<variant>. BASE-O and SCORE as
above.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

GENERATORS = ("G1", "G2", "G3")
SEED = 20261008
OUT = "outputs/division_classifier"


def sim_dm(y, q, a, rng):
    """Split each total y_i across genes, Dirichlet-multinomial DM(y_i, a q)."""
    y = np.asarray(y)
    out = np.zeros((len(y), len(q)), np.int64)
    m = y > 0
    if m.any():
        g = rng.gamma(shape=a * np.asarray(q), size=(int(m.sum()), len(q)))
        p = g / g.sum(1, keepdims=True)
        out[m] = rng.multinomial(y[m].astype(np.int64), p)
    return out


def g2_alpha(pi, lam0, lam1, vmr):
    """Cycling-class NB dispersion that makes the model's pooled
    variance-to-mean ratio over these cells equal `vmr` (spec 7.1, G2).

    Pooled moments over cells: E[y] = mean m_i, E[y^2] = mean s_i, with
    m_i = (1-pi) l0 + pi l1 and s_i = (1-pi)(l0 + l0^2) + pi(l1 + l1^2 (1 + alpha)).
    The variance is linear in alpha, so the solution is closed form; it is
    floored at 0 (the Poisson mixture already over-dispersed enough)."""
    m = (1 - pi) * lam0 + pi * lam1
    s0 = (1 - pi) * (lam0 + lam0 ** 2) + pi * (lam1 + lam1 ** 2)
    need = vmr * m.mean() + m.mean() ** 2 - s0.mean()
    return max(float(need / np.mean(pi * lam1 ** 2)), 0.0)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _scores(p, L):
    from sklearn.metrics import average_precision_score

    pc = np.clip(p, 1e-12, 1 - 1e-12)
    rate = float(L.mean())
    return {"log loss": float(-np.mean(L * np.log(pc) + (1 - L) * np.log1p(-pc))),
            "AUPRC": float(average_precision_score(L, p)) if 0 < rate < 1 else np.nan,
            "mean p / rate": float(p.mean() / rate) if rate > 0 else np.nan, "rate": rate}


def _cosine(P, Q):
    """Cosine of vec(U_P V_P') with vec(U_Q V_Q')."""
    a = (np.asarray(P["U"]) @ np.asarray(P["V"]).T).ravel()
    b = (np.asarray(Q["U"]) @ np.asarray(Q["V"]).T).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def run_replicate(task, cfg, say=None, addendum=False, variant=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import pandas as pd
    from scipy.stats import spearmanr

    from ..mixture.basis import build_basis, spec_from_basis
    from ..mixture.fit import multistart
    from ..mixture.model import build_data, component_means, pi_hat, responsibilities
    from . import arms
    from . import labels as L
    from .survival import mix_p_ge, nb_p_ge

    say = say or (lambda *a: None)
    t0 = time.time()
    out_dir = f"{OUT}_{variant}" if variant else OUT
    vset = L.VARIANTS[variant] if variant else None
    ab_primary = vset["alpha_b"] if vset else L.ALPHA_B_PRIMARY
    if variant and addendum:
        raise ValueError("the addendum job runs on the original labels only")
    gen, rep = task["generator"], int(task["rep"])
    gi = GENERATORS.index(gen)
    n_sim = int(task.get("n", cfg.get("division_sim_n", 100_000)))
    restarts = int(task.get("restarts", cfg.get("division_sim_restarts", 4)))

    df = pd.read_parquet(cfg["data"]["table_parquet"])
    types = json.load(open(cfg["data"]["table_parquet"].replace(".parquet", ".phi_names.json")))
    df.attrs["phi_names"] = types
    ids = df["cell_id"].astype(str).to_numpy()
    pe = json.load(open(cfg.get("hpc_dir", "outputs/hpc") + "/point_estimate.json"))
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    if int(pe["rank"]) != spec.rank:
        raise RuntimeError(f"point estimate rank {pe['rank']}, config rank {spec.rank}")
    data = build_data(df, basis, cfg, spec)
    inp = np.load(f"{out_dir}/sim_inputs.npz")
    if str(inp["cell_id_sha256"]) != hashlib.sha256("\n".join(ids).encode()).hexdigest():
        raise RuntimeError("sim_inputs.npz was built on a different cell list")
    fm = pd.read_parquet(f"{out_dir}/folds.parquet")
    if not np.array_equal(fm["cell_id"].astype(str).to_numpy(), ids):
        raise RuntimeError("folds.parquet lists cells in a different order from the table")
    nbc = {k: jnp.asarray(np.asarray(v))
           for k, v in json.load(open(f"{out_dir}/nb_params_nbc_full.json")).items()}
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    ctype_all = L.cell_type(df, types)
    tix = {t: i for i, t in enumerate(types)}

    # ---- cells: shared across generators within a replicate ---------------------
    n = len(df)
    idx = np.sort(np.random.default_rng(SEED + rep).choice(n, n_sim, replace=False))
    hold = rep % 5
    test = fm["fold"].to_numpy()[idx] == hold
    sub = data.subset(idx)
    Dsub = sub.as_jax()
    ct = ctype_all[idx]
    N = df["N"].to_numpy(float)[idx]
    tumour_nbr = df[f"nbr_phi_{types.index('Cancer Epithelial'):02d}"].to_numpy(float)[idx]

    # ---- truth -----------------------------------------------------------------------
    rng = np.random.default_rng(SEED + 1000 * (gi + 1) + rep)
    pi_true = np.asarray(pi_hat(P, Dsub, spec))
    lam0, lam1 = (np.asarray(v) for v in component_means(P, Dsub, spec))
    alpha1 = np.nan
    if gen in ("G1", "G2"):
        Z = rng.uniform(size=n_sim) < pi_true
        if gen == "G1":
            y = rng.poisson(np.where(Z, lam1, lam0)).astype(float)
        else:
            D = data.as_jax()
            ce = ctype_all == "Cancer Epithelial"
            l0f, l1f = (np.asarray(v)[ce] for v in component_means(P, D, spec))
            alpha1 = g2_alpha(np.asarray(pi_hat(P, D, spec))[ce], l0f, l1f, float(inp["vmr_ce"]))
            y0 = rng.poisson(lam0).astype(float)
            y1 = L.sim_nb(lam1, alpha1, rng) if alpha1 > 0 else rng.poisson(lam1).astype(float)
            y = np.where(Z, y1, y0)
    else:
        Z = None
        mu = np.exp(np.asarray(arms.nb_log_mu(nbc, Dsub)))
        al = float(np.exp(np.asarray(arms.nb_log_alpha(nbc, Dsub))))
        y = L.sim_nb(mu, al, rng)

    # ---- labels, rebuilt from the simulated counts ----------------------------------
    Xa, off_null, m_null = L.null_design(df.iloc[idx], basis)
    null = L.robust_null(y, Xa, off_null, m=m_null)
    mu0, a0 = null["mu"], null["alpha"]
    sd0 = np.sqrt(mu0 + a0 * mu0 ** 2)
    if vset:
        mz = m_null if m_null is not None else np.zeros(n_sim)
        ft = L.truncated_null(y, Xa, off_null, mz, vset["trunc"],
                              L.fit_nb_additive_null(y, Xa, off_null, mz), param_tol=L.TRUNC_PARAM_TOL)
        LT, cT, _, _ = L.base_t(y, ft["mu"], ft["alpha"], ab_primary)
        baset_alpha0 = float(ft["alpha"])
    else:
        baset_alpha0 = float(a0)
        LT, cT, _, _ = L.base_t(y, mu0, a0, ab_primary)
    s_plus = L.base_o_scale((y - mu0) / sd0, ct)
    LO, cO, _ = L.base_o(y, mu0, sd0, s_plus, L.TD_PRIMARY)
    if Z is None:
        sub.y = y
        w = np.asarray(responsibilities(P, sub.as_jax(), spec))
        state = rng.uniform(size=n_sim) < w
    else:
        state = Z
    Yg = np.zeros((n_sim, len(inp["tirosh"])), np.int64)
    for t in types:
        for k, (Q, A) in ((1, (inp["q1"], inp["a1"])), (0, (inp["q0"], inp["a0"]))):
            sel = (ct == t) & (state == bool(k))
            Yg[sel] = sim_dm(y[sel], Q[tix[t]], float(A[tix[t]]), rng)
    s = float(inp["normaliser_s"])
    score = np.log1p(Yg * (s / N)[:, None]).mean(1) - inp["ctrl"].astype(float)[idx]
    thr = L.score_threshold(score)
    LS = score > thr["threshold"]
    labels = {"BASE-T": LT.astype(float), "BASE-O": LO.astype(float), "SCORE": LS.astype(float)}
    say(f"  {gen} rep {rep}: labels built; rates T {LT.mean():.4f}, O {LO.mean():.4f}, "
        f"S {LS.mean():.4f} ({time.time() - t0:.0f}s)")
    grid, trunc_fits = {}, {}
    if addendum:
        # extended grid: the naive null as above, and the truncated null refitted on
        # the simulated y at each alpha_trim from the untrimmed fit (as A1 does)
        for a in L.ALPHA_B_GRID_EXT:
            Lg, cg, _, _ = L.base_t(y, mu0, a0, a)
            grid[f"naive|{a:g}"] = (Lg.astype(float), cg)
        full_null = L.fit_nb_additive_null(y, Xa, off_null, m_null)
        for at in L.TRUNC_GRID:
            ft = L.truncated_null(y, Xa, off_null, m_null, at, full_null)
            trunc_fits[f"{at:g}"] = {"alpha0": float(ft["alpha"]), "theta_m": float(ft["theta_m"]),
                                     "iterations": len(ft["trace"]), "stopped_on": ft["stopped_on"]}
            for a in L.ALPHA_B_GRID_EXT:
                Lg, cg, _, _ = L.base_t(y, ft["mu"], ft["alpha"], a)
                grid[f"trunc {at:g}|{a:g}"] = (Lg.astype(float), cg)
        for g, (Lg, _) in grid.items():
            labels[f"BASE-T [{g}]"] = Lg
        if cfg.get("division_sim_state", False):
            Xs = np.column_stack([np.ones(n_sim), np.asarray(Dsub["own"])])
            s_state, _ = L.base_o_state_scale((y - mu0) / sd0, Xs)
            LOs, _, _ = L.base_o(y, mu0, sd0, s_state, L.TD_PRIMARY)
            labels["BASE-O-state"] = LOs.astype(float)
        say(f"  extended grid: {len(grid)} labels ({time.time() - t0:.0f}s)")

    # ---- fits on the training folds --------------------------------------------------
    sub.y = y
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    d_tr, d_te = sub.subset(tr), sub.subset(te)
    D_tr, D_te = d_tr.as_jax(), d_te.as_jax()
    fits = {}

    def minimise(f, inits):
        for p0 in inits:
            p, info = arms.minimise(f, p0)
            if info["converged"]:
                break
        return p, info

    f_nb = arms.nb_objective(D_tr, len(tr), spec.weight_ridge, spec.n_reference, spec.ridge_tau,
                             spec.b_penalty)
    p_nb, info = minimise(f_nb, (arms.init_nb(D_tr, "const", spec.rank, s_, theta_free=spec.theta_free)
                                 for s_ in (0, 100, 200)))
    fits["NB-c"] = {"converged": info["converged"], "sd_eta": arms.eta_diagnostics(p_nb, D_tr)["sd_eta"]}
    say(f"  NB-c converged {info['converged']} ({time.time() - t0:.0f}s)")
    best, _ = multistart(d_tr, spec, cfg, n_restarts=restarts, seed=100 * rep + gi, verbose=False,
                         maxiter=int(cfg["optim"]["maxiter"]))
    fits["MIX"] = {"converged": bool(best.converged), "e_delta": float(np.exp(np.asarray(best.params["delta"]))),
                   "mean_pi_train": float(np.asarray(pi_hat(best.params, D_tr, spec)).mean()),
                   "loglik": float(best.loglik)}
    say(f"  MIX converged {best.converged}, e^delta {fits['MIX']['e_delta']:.3f} ({time.time() - t0:.0f}s)")
    p_log = {}
    for arm, lname in (("LOG-T", "BASE-T"), ("LOG-O", "BASE-O"), ("LOG-S", "SCORE")):
        yl = jnp.asarray(labels[lname][tr])
        f = arms.logistic_objective(D_tr, yl, len(tr), spec.weight_ridge, spec.n_reference,
                                    spec.ridge_tau, spec.b_penalty)
        p, info = minimise(f, (arms.init_logistic(D_tr, yl, spec.rank, s_) for s_ in (0, 100, 200)))
        p_log[arm] = p
        fits[arm] = {"converged": info["converged"], "sd_eta": arms.eta_diagnostics(p, D_tr)["sd_eta"]}
    say(f"  LOG arms fitted ({time.time() - t0:.0f}s)")
    two = {}
    if addendum:
        from . import twostep as T
        primary_g = f"naive|{L.ALPHA_B_PRIMARY:g}"
        if "BASE-O-state" in labels:
            yl = jnp.asarray(labels["BASE-O-state"][tr])
            f = arms.logistic_objective(D_tr, yl, len(tr), spec.weight_ridge, spec.n_reference,
                                        spec.ridge_tau, spec.b_penalty)
            p, info = minimise(f, (arms.init_logistic(D_tr, yl, spec.rank, s_) for s_ in (0, 100, 200)))
            p_log["LOG-O-state"] = p
            fits["LOG-O-state"] = {"converged": info["converged"],
                                   "sd_eta": arms.eta_diagnostics(p, D_tr)["sd_eta"]}
        for g in grid:
            if g == primary_g:
                p_log[f"LOG-T [{g}]"] = p_log["LOG-T"]
                fits[f"LOG-T [{g}]"] = fits["LOG-T"]
                continue
            yl = jnp.asarray(labels[f"BASE-T [{g}]"][tr])
            f = arms.logistic_objective(D_tr, yl, len(tr), spec.weight_ridge, spec.n_reference,
                                        spec.ridge_tau, spec.b_penalty)
            p, info = minimise(f, (arms.init_logistic(D_tr, yl, spec.rank, s_) for s_ in (0, 100, 200)))
            p_log[f"LOG-T [{g}]"] = p
            fits[f"LOG-T [{g}]"] = {"converged": info["converged"],
                                    "sd_eta": arms.eta_diagnostics(p, D_tr)["sd_eta"]}
        say(f"  LOG-T on the grid fitted ({time.time() - t0:.0f}s)")
        w0 = {k: best.params[k] for k in ("a0", "a", "c", "U", "V", "h", "K")}
        srng = np.random.default_rng(SEED + 7000 + 10 * rep + gi)
        if float(np.abs(np.asarray(w0["U"]) @ np.asarray(w0["V"]).T).max()) < 1e-8:
            # U = V = 0 is a saddle of the bilinear term; never start step 2 there
            w0["U"] = jnp.asarray(srng.normal(0, 0.01, np.shape(w0["U"])))
            w0["V"] = jnp.asarray(srng.normal(0, 0.01, np.shape(w0["V"])))
        for arm, family in (("TWO-P", "pp"), ("TWO-PNB", "pnb")):
            starts = [T.start_from_mix(P, family)]
            if family == "pnb":
                starts += [T.random_start(P, family, srng) for _ in range(2)]
            res = [T.fit_step1(D_tr, family, s0) for s0 in starts]
            ok = [r for r in res if np.isfinite(r[1]["f"])]
            q1, i1 = min(ok, key=lambda r: r[1]["f"])
            w = T.posterior(q1, D_tr, family)
            q2, i2 = T.fit_step2(D_tr, w, len(tr), spec, seed=0, p0=w0)
            two[arm] = (family, q1, q2)
            fits[arm] = {"converged": bool(i1["converged"] and i2["converged"]),
                         "step1_objectives": [float(r[1]["f"]) for r in res],
                         "pi_const": float(jax.nn.sigmoid(q1["a0"])),
                         "e_delta": float(jnp.exp(q1["delta"])),
                         "alpha1": float(jnp.exp(q1["la1"])) if family == "pnb" else None,
                         "sd_eta": arms.eta_diagnostics(q2, D_tr)["sd_eta"]}
        say(f"  TWO-P and TWO-PNB fitted ({time.time() - t0:.0f}s)")

    # ---- predictions on the held-out fold ----------------------------------------------
    mu_nb = np.exp(np.asarray(arms.nb_log_mu(p_nb, D_te)))
    al_nb = np.exp(np.asarray(arms.nb_log_alpha(p_nb, D_te))) * np.ones(len(te))
    pi_mix = np.asarray(pi_hat(best.params, D_te, spec))
    l0m, l1m = (np.asarray(v) for v in component_means(best.params, D_te, spec))
    preds = {"NB-c, BASE-T": (nb_p_ge(cT[te], mu_nb, al_nb), "BASE-T"),
             "NB-c, BASE-O": (nb_p_ge(cO[te], mu_nb, al_nb), "BASE-O"),
             "MIX pi": (pi_mix, None),
             "MIX, BASE-T": (mix_p_ge(cT[te], pi_mix, l0m, l1m), "BASE-T"),
             "MIX, BASE-O": (mix_p_ge(cO[te], pi_mix, l0m, l1m), "BASE-O")}
    for arm in p_log:
        lname = {"LOG-T": "BASE-T", "LOG-O": "BASE-O", "LOG-S": "SCORE",
                 "LOG-O-state": "BASE-O-state"}.get(arm)
        if lname is None:                                  # LOG-T on a grid label
            lname = arm.replace("LOG-T", "BASE-T")
        preds[arm] = (np.asarray(jax.nn.sigmoid(arms.logistic_eta(p_log[arm], D_te))), lname)
    if addendum:
        from . import twostep as T
        from .survival import pois_p_ge
        comp = {}
        for arm, (family, q1, q2) in two.items():
            pi2 = np.asarray(T.pi_two(q2, D_te))
            m0, m1 = (np.asarray(v) for v in T.means(q1, D_te))
            a1 = float(jnp.exp(q1["la1"])) if family == "pnb" else None
            push = (lambda c, pi2=pi2, m0=m0, m1=m1, a1=a1, family=family:
                    pi2 * (pois_p_ge(c, m1) if family == "pp" else nb_p_ge(c, m1, a1))
                    + (1 - pi2) * pois_p_ge(c, m0))
            comp[arm] = push
            preds[f"{arm} pi"] = (pi2, None)
            preds[f"{arm}, BASE-T"] = (push(cT[te]), "BASE-T")
            preds[f"{arm}, BASE-O"] = (push(cO[te]), "BASE-O")
        for g, (_, cg) in grid.items():
            lname = f"BASE-T [{g}]"
            preds[f"NB-c, {lname}"] = (nb_p_ge(cg[te], mu_nb, al_nb), lname)
            preds[f"MIX, {lname}"] = (mix_p_ge(cg[te], pi_mix, l0m, l1m), lname)
            for arm, push in comp.items():
                preds[f"{arm}, {lname}"] = (push(cg[te]), lname)

    # ---- metrics -----------------------------------------------------------------------------
    ct_te, tum_te = ct[te], tumour_nbr[te]
    label_scores = {k: _scores(p, labels[lname][te]) for k, (p, lname) in preds.items() if lname}
    fractions = {k: {t: float(p[ct_te == t].mean()) for t in types} for k, (p, _) in preds.items()}
    out = {"generator": gen, "rep": rep, "fold": hold, "n_train": int(len(tr)), "n_test": int(len(te)),
           "label_rates": {k: float(v.mean()) for k, v in labels.items()},
           "null_alpha0": float(a0), "baset_null_alpha0": baset_alpha0, "variant": variant,
           "score_threshold": float(thr["threshold"]),
           "g2_alpha1": alpha1, "fits": fits, "label_scores": label_scores, "fractions": fractions,
           "rank": int(spec.rank), "weight_ridge": float(spec.weight_ridge), "b_penalty": spec.b_penalty}
    if addendum:
        B_true = nbc if gen == "G3" else P
        models = {"NB-c": p_nb, "MIX": best.params, **p_log,
                  **{arm: q2 for arm, (_, _, q2) in two.items()}}
        out["b_cosine"] = {k: _cosine(v, B_true) for k, v in models.items()}
        out["b_truth"] = "NB-c (full data)" if gen == "G3" else "production MIX"
        out["trunc_nulls"] = trunc_fits
        out["addendum"] = True
    if Z is not None:
        z_te, pt_te = Z[te], pi_true[te]
        out["true_fraction"] = {t: float(z_te[ct_te == t].mean()) for t in types}
        out["true_pi_fraction"] = {t: float(pt_te[ct_te == t].mean()) for t in types}
        out["spearman_with_pi_true"] = {k: float(spearmanr(p, pt_te).statistic) for k, (p, _) in preds.items()}
        contrast = {}
        for t in types:
            m = ct_te == t
            lo_q, hi_q = np.quantile(tum_te[m], [0.1, 0.9])
            lo, hi = m & (tum_te <= lo_q), m & (tum_te >= hi_q)
            row = {"true": float(_logit(pt_te[hi]).mean() - _logit(pt_te[lo]).mean()),
                   "n_low": int(lo.sum()), "n_high": int(hi.sum())}
            for k, (p, _) in preds.items():
                row[k] = float(_logit(p[hi]).mean() - _logit(p[lo]).mean())
            contrast[t] = row
        out["neighbourhood_contrast"] = contrast
    out["wall_time"] = time.time() - t0
    say(f"  {gen} rep {rep} done in {out['wall_time'] / 60:.1f} min")
    return out
