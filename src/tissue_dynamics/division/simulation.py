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


def run_replicate(task, cfg, say=None):
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
    inp = np.load(f"{OUT}/sim_inputs.npz")
    if str(inp["cell_id_sha256"]) != hashlib.sha256("\n".join(ids).encode()).hexdigest():
        raise RuntimeError("sim_inputs.npz was built on a different cell list")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    if not np.array_equal(fm["cell_id"].astype(str).to_numpy(), ids):
        raise RuntimeError("folds.parquet lists cells in a different order from the table")
    nbc = {k: jnp.asarray(np.asarray(v))
           for k, v in json.load(open(f"{OUT}/nb_params_nbc_full.json")).items()}
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
    LT, cT, _, _ = L.base_t(y, mu0, a0, L.ALPHA_B_PRIMARY)
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
        lname = {"LOG-T": "BASE-T", "LOG-O": "BASE-O", "LOG-S": "SCORE"}[arm]
        preds[arm] = (np.asarray(jax.nn.sigmoid(arms.logistic_eta(p_log[arm], D_te))), lname)

    # ---- metrics -----------------------------------------------------------------------------
    ct_te, tum_te = ct[te], tumour_nbr[te]
    label_scores = {k: _scores(p, labels[lname][te]) for k, (p, lname) in preds.items() if lname}
    fractions = {k: {t: float(p[ct_te == t].mean()) for t in types} for k, (p, _) in preds.items()}
    out = {"generator": gen, "rep": rep, "fold": hold, "n_train": int(len(tr)), "n_test": int(len(te)),
           "label_rates": {k: float(v.mean()) for k, v in labels.items()},
           "null_alpha0": float(a0), "score_threshold": float(thr["threshold"]),
           "g2_alpha1": alpha1, "fits": fits, "label_scores": label_scores, "fractions": fractions,
           "rank": int(spec.rank), "weight_ridge": float(spec.weight_ridge), "b_penalty": spec.b_penalty}
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
