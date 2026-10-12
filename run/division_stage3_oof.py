"""Division classifier comparison, Stage 3: every arm out of fold.

    python run/division_stage3_oof.py

Spec sections 3, 4.1-4.2 and 8. Tests V5 (fold identity) and V7 (LOG
separation), then V4 (push-through identity) and V8 (determinism).

The arms, all on the stored five-fold spatial partition:

* NB-c, NB-d: Stage 2's training-fold fits, reused (`oof_nb.parquet`); the NB
  fits do not depend on any label (spec 4.2).
* MIX: the production mixture refitted on each training fold, with the penalty
  scaled to training size. Each fold fit starts from the full-data point
  estimate and runs to the mixture's convergence criteria, as the bootstrap
  replicates do. A converged fit does not remember its start, and each fold's
  held-out log-likelihood is checked against the Spartan CV fit for the same
  fold and rank, which started from its own multistart.
* LOG-T, LOG-O, LOG-S: logistic regression of the primary labels (BASE-T at
  alpha_B = 0.05, BASE-O per type at Td = 0.5, SCORE) on the shared weight
  block plus g(log N, log A) = b0 + theta log N + s(log A) (spec 1), with the
  mixture's penalty term for term. Fitted on all cells (for V7) and on each
  training fold.

Count-arm predictions are pushed through every threshold in the alpha_B and Td
grids (spec 3): NB p = P(Y >= c), MIX p = pi P1(Y >= c) + (1 - pi) P0(Y >= c),
with the prior pi, never the posterior w.

Writes outputs/division_classifier/{oof_predictions.parquet, stage3_fits.csv,
mix_params_fold*.json, log_params_*.json, stage3_v4.csv} and the Stage 3 note
in report.md. Stops (non-zero exit) on a failing V-test, after writing it.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import mix_p_ge, nb_p_ge       # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.fit import fit_lbfgs                     # noqa: E402
from tissue_dynamics.mixture.model import (build_data, loglik_terms,  # noqa: E402
                                           pi_hat, component_means)

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_FOLDS = 5
LOG_ARMS = {"LOG-T": f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}",
            "LOG-O": f"LO_{L.td_tag(L.TD_PRIMARY)}",
            "LOG-S": "LS"}
# every count-based threshold column in labels.parquet: c^T on the alpha_B grid,
# c^O (per type, the primary BASE-O) on the Td grid
THRESHOLDS = ([f"cT_{L.ab_tag(a)}" for a in L.ALPHA_B_GRID]
              + [f"cO_{L.td_tag(t)}" for t in L.TD_GRID])
PRIMARY = {"BASE-T": f"cT_{L.ab_tag(L.ALPHA_B_PRIMARY)}",
           "BASE-O": f"cO_{L.td_tag(L.TD_PRIMARY)}"}
N_V4, SEED_V4 = 50_000, 20261008


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def fail(test, detail):
    report(f"\n### STOPPED in Stage 3: {test} failed\n\n{detail}\n\nPer spec section 8, "
           "nothing further was run.")
    say(f"\n*** {test} FAILED -- stopping. {detail}")
    sys.exit(1)


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def to_json(p):
    return {k: np.asarray(v).tolist() for k, v in p.items()}


# --------------------------------------------------------------------------
# LOG arms: logit P(L = 1) = b0 + theta log N [+ s(log A)] + eta_state
# (arms.logistic_eta: the area spline only under model.technical: area)
# --------------------------------------------------------------------------
def log_eta(p, D):
    return arms.logistic_eta(p, D)


def log_objective(D, lab, n, spec):
    def f(p):
        eta = log_eta(p, D)
        ll = jnp.sum(lab * eta - jax.nn.softplus(eta))
        return -ll + arms.penalty(p, n, spec.weight_ridge, spec.n_reference,
                                  spec.ridge_tau, spec.b_penalty)
    return f


def init_log(D, lab, rank, seed):
    return arms.init_logistic(D, lab, rank, seed)


def main():
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    fmeta = json.load(open(f"{OUT}/folds_meta.json"))
    oof_nb = pd.read_parquet(f"{OUT}/oof_nb.parquet")
    import glob
    # V5 compares fold SIZES with a Spartan CV run of the mixture -- any CV job:
    # sizes depend only on the cells and the partition. Without one (the area-era
    # CV left the repository with cell area, 2026-10-13), the sizes recorded with
    # the stored folds stand in. Held-out log-likelihoods are compared only with
    # CV of the same rank, penalty and technical covariates as this run.
    cvr = [json.load(open(f)) for j in ("cv_rank", "cv_production", "cv_penalty")
           for f in sorted(glob.glob(f"outputs/hpc/{j}/*.json"))]
    if not cvr:
        cvr = [{"fold": int(f), "n_train": int(s[0]), "n_test": int(s[1])}
               for f, s in fmeta["fold_sizes_train_test"].items()]

    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    n = len(df)
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels.parquet", lab), ("folds.parquet", fm), ("oof_nb.parquet", oof_nb)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            fail("cell alignment", f"{name} lists cells in a different order from the table")
    if int(pe["rank"]) != spec.rank:
        fail("model version", f"point estimate rank {pe['rank']}, config rank {spec.rank}")
    say(f"{n:,} cells; rank {spec.rank}; weight_ridge {spec.weight_ridge:g} x n / "
        f"{spec.n_reference:,}; {spec.b_penalty}, tau {spec.ridge_tau:g}")

    # ---- V5: fold identity --------------------------------------------------
    folds = fm["fold"].to_numpy()
    fold_hash = hashlib.sha256(folds.astype(np.int8).tobytes()).hexdigest()
    nb_folds_ok = np.array_equal(oof_nb["fold"].to_numpy(), folds)
    sizes = {f: (int(np.sum(folds != f)), int(np.sum(folds == f))) for f in range(N_FOLDS)}
    cv_sizes = {}
    for r in cvr:
        cv_sizes.setdefault(int(r["fold"]), set()).add((int(r["n_train"]), int(r["n_test"])))
    v5_ok = (fold_hash == fmeta["sha256"] and nb_folds_ok
             and all(cv_sizes.get(f) == {sizes[f]} for f in range(N_FOLDS)))
    say(f"V5: hash {fold_hash[:16]} (stored {fmeta['sha256'][:16]}), Stage 2 folds identical "
        f"{nb_folds_ok}, sizes match the rank CV {all(cv_sizes.get(f) == {sizes[f]} for f in range(N_FOLDS))}")
    if not v5_ok:
        fail("V5 (fold identity)", f"hash {fold_hash} vs stored {fmeta['sha256']}; Stage 2 "
             f"folds identical: {nb_folds_ok}; sizes {sizes} vs rank CV {cv_sizes}")

    rows = []
    oof = pd.DataFrame({"cell_id": ids, "fold": folds, "cell_type": lab["cell_type"].to_numpy()})

    # ---- MIX: refit on each training fold ------------------------------------
    p_hat = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    maxiter = int(cfg["optim"]["maxiter"])
    pi_oof, l0_oof, l1_oof = np.zeros(n), np.zeros(n), np.zeros(n)
    mix_params = {}
    for f in range(N_FOLDS):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        say(f"  MIX train fold {f} (n = {len(tr):,})")
        res = fit_lbfgs(data.subset(tr), spec, p_hat, maxiter=maxiter)
        Dte = data.subset(te).as_jax()
        pi_oof[te] = np.asarray(pi_hat(res.params, Dte, spec))
        # each component's Poisson mean (lambda_k, plus theta_m m under MisTIC):
        # what the push-through P(Y >= c | component) needs
        lam0, lam1 = component_means(res.params, Dte, spec)
        l0_oof[te], l1_oof[te] = np.asarray(lam0), np.asarray(lam1)
        ho = float(jnp.mean(loglik_terms(res.params, Dte, spec)))
        spartan = [r for r in cvr if int(r["fold"]) == f and r.get("rank") == spec.rank
                   and r.get("weight_ridge") == spec.weight_ridge
                   and r.get("technical", "area") == basis.technical]
        ho_sp = float(spartan[0]["heldout_loglik_per_cell"]) if spartan else np.nan
        diag = arms.eta_diagnostics(res.params, data.subset(tr).as_jax())
        mix_params[f] = res.params
        json.dump(to_json(res.params), open(f"{OUT}/mix_params_fold{f}.json", "w"))
        rows.append({"arm": "MIX", "data": f"train fold {f}", "n": len(tr),
                     "converged": "pass" if res.converged else "FAIL", "iterations": res.n_iter,
                     "grad_inf": res.grad_inf, "loglik (train)": res.loglik,
                     "held-out loglik / cell": ho, "Spartan CV, same fold": ho_sp,
                     "e^delta": float(np.exp(np.asarray(res.params["delta"]))),
                     "mean pi (test)": float(pi_oof[te].mean()),
                     "norm a": diag["norm_a"], "norm c": diag["norm_c"],
                     "sigma1(B)": diag["sigma1_B"], "sd(eta)": diag["sd_eta"],
                     "minutes": res.wall_time / 60})
        say(f"    converged {res.converged} in {res.n_iter} iterations; held-out "
            f"{ho:.6f} / cell (Spartan {ho_sp:.6f}); e^delta "
            f"{rows[-1]['e^delta']:.4f} ({res.wall_time / 60:.1f} min)")
    oof["pi_mix"], oof["lam0_mix"], oof["lam1_mix"] = pi_oof, l0_oof, l1_oof
    mix_ok = all(r["converged"] == "pass" for r in rows)
    if not mix_ok:
        fail("MIX fold convergence", md_table(pd.DataFrame(rows)[["data", "iterations",
                                                                  "grad_inf"]]))

    # ---- LOG arms -------------------------------------------------------------
    log_params = {}

    def fit_log(arm, idx, seed, tag):
        y = lab[LOG_ARMS[arm]].to_numpy(float)
        Ds = data.subset(idx).as_jax() if idx is not None else D
        m = n if idx is None else len(idx)
        yl = jnp.asarray(y if idx is None else y[idx])
        say(f"  {arm} {tag} (n = {m:,}, label rate {float(yl.mean()):.4f})")
        f = log_objective(Ds, yl, m, spec)
        for attempt in range(3):
            p, info = arms.minimise(f, init_log(Ds, yl, spec.rank, seed + 100 * attempt), say=say)
            if info["converged"]:
                break
            say(f"      not converged; retrying from seed {seed + 100 * (attempt + 1)}")
        diag = arms.eta_diagnostics(p, Ds)
        info["seed"] = seed + 100 * attempt
        rows.append({"arm": arm, "data": tag, "n": m, "seed": info["seed"],
                     "converged": "pass" if info["converged"] else "FAIL",
                     "iterations": info["lbfgs_iters"] + info["newton_iters"],
                     "grad_inf": info["grad_inf"],
                     "loglik (train)": -info["f"], "theta (log N)": float(p["theta"]),
                     "norm a": diag["norm_a"], "norm c": diag["norm_c"],
                     "sigma1(B)": diag["sigma1_B"], "sd(eta)": diag["sd_eta"],
                     "sd(eta) > 20 (V7)": "FLAG" if diag["flag_sd_eta"] else "no",
                     "minutes": info["wall_time"] / 60})
        return p, info

    for arm in LOG_ARMS:
        col = arm.replace("-", "").lower()
        p, _ = fit_log(arm, None, 0, "all cells")
        log_params[(arm, "full")] = p
        json.dump(to_json(p), open(f"{OUT}/log_params_{col}_full.json", "w"))
        pred = np.zeros(n)
        for f in range(N_FOLDS):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            p, info = fit_log(arm, tr, 0, f"train fold {f}")
            log_params[(arm, f)] = p
            log_params[(arm, f, "seed")] = info["seed"]
            json.dump(to_json(p), open(f"{OUT}/log_params_{col}_fold{f}.json", "w"))
            pred[te] = np.asarray(jax.nn.sigmoid(log_eta(p, data.subset(te).as_jax())))
        oof[f"p_{col}_{LOG_ARMS[arm]}"] = pred

    fits = pd.DataFrame(rows)
    fits.to_csv(f"{OUT}/stage3_fits.csv", index=False)
    log_fits = fits[fits["arm"].str.startswith("LOG")]
    v7_ok = bool((log_fits["sd(eta) > 20 (V7)"] == "no").all())
    log_conv = bool((log_fits["converged"] == "pass").all())
    say(f"V7: max sd(eta) over LOG fits {log_fits['sd(eta)'].max():.3f}; all converged {log_conv}")

    # ---- count arms pushed through every threshold ---------------------------
    mu_c, al_c = np.exp(oof_nb["log_mu_nbc"].to_numpy()), np.exp(oof_nb["log_alpha_nbc"].to_numpy())
    mu_d, al_d = np.exp(oof_nb["log_mu_nbd"].to_numpy()), np.exp(oof_nb["log_alpha_nbd"].to_numpy())
    oof["mu_nbc"], oof["alpha_nbc"], oof["mu_nbd"], oof["alpha_nbd"] = mu_c, al_c, mu_d, al_d
    for tc in THRESHOLDS:
        c = lab[tc].to_numpy(float)
        lab_col = tc.replace("cT_", "LT_").replace("cO_", "LO_")
        oof[f"p_nbc_{lab_col}"] = nb_p_ge(c, mu_c, al_c)
        oof[f"p_nbd_{lab_col}"] = nb_p_ge(c, mu_d, al_d)
        oof[f"p_mix_{lab_col}"] = mix_p_ge(c, pi_oof, l0_oof, l1_oof)
    oof.to_parquet(f"{OUT}/oof_predictions.parquet", index=False)

    # ---- V4: push-through identity ---------------------------------------------
    def simulate(model, idx, rng):
        if model == "NB-c":
            return L.sim_nb(mu_c[idx], al_c[idx], rng)
        if model == "NB-d":
            return L.sim_nb(mu_d[idx], al_d[idx], rng)
        z = rng.uniform(size=len(idx)) < pi_oof[idx]
        return rng.poisson(np.where(z, l1_oof[idx], l0_oof[idx])).astype(float)

    def v4_table(seed):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, N_V4, replace=False))
        out = []
        for model, col in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix")):
            ysim = simulate(model, idx, rng)
            for tc in THRESHOLDS:
                lab_col = tc.replace("cT_", "LT_").replace("cO_", "LO_")
                p = oof[f"p_{col}_{lab_col}"].to_numpy()[idx]
                rate = float(np.mean(ysim >= lab[tc].to_numpy()[idx]))
                se = float(np.sqrt(np.sum(p * (1 - p)))) / N_V4
                out.append({"model": model, "threshold": lab_col,
                            "primary": tc in PRIMARY.values(),
                            "mean p": float(p.mean()), "simulated rate": rate,
                            "MC SE": se, "z": (rate - p.mean()) / se})
        return pd.DataFrame(out)

    v4 = v4_table(SEED_V4)
    v4["within 3 SE"] = np.where(v4["z"].abs() <= 3, "pass", "FAIL")
    v4.to_csv(f"{OUT}/stage3_v4.csv", index=False)
    v4_ok = bool((v4.loc[v4["primary"], "within 3 SE"] == "pass").all())
    say(v4.round(5).to_string(index=False))

    # ---- V8: determinism --------------------------------------------------------
    tr0 = np.where(folds != 0)[0]
    y0 = jnp.asarray(lab[LOG_ARMS["LOG-T"]].to_numpy(float)[tr0])
    D0 = data.subset(tr0).as_jax()
    # same start as the fit being reproduced (a retry moves the seed on)
    p_again, _ = arms.minimise(log_objective(D0, y0, len(tr0), spec),
                               init_log(D0, y0, spec.rank, log_params[("LOG-T", 0, "seed")]),
                               say=say)
    d_log = max(float(np.max(np.abs(np.asarray(p_again[k]) - np.asarray(log_params[("LOG-T", 0)][k]))))
                for k in p_again)
    nb0 = json.load(open(f"{OUT}/nb_params_nbc_fold0.json"))
    s2 = pd.read_csv(f"{OUT}/stage2_fits.csv")
    nb_seed = 100 * (int(s2[(s2["arm"] == "NB-c") & (s2["data"] == "train fold 0")]
                         ["attempts"].iloc[0]) - 1)
    f_nb = arms.nb_objective(D0, len(tr0), spec.weight_ridge, spec.n_reference,
                             spec.ridge_tau, spec.b_penalty)
    p_nb, _ = arms.minimise(f_nb, arms.init_nb(D0, "const", spec.rank, nb_seed,
                                               theta_free=spec.theta_free), say=say)
    d_nb = max(float(np.max(np.abs(np.asarray(p_nb[k]) - np.asarray(nb0[k])))) for k in p_nb)
    v4_again = v4_table(SEED_V4)
    d_v4 = float(np.max(np.abs(v4_again["simulated rate"].to_numpy()
                               - v4["simulated rate"].to_numpy())))
    v8_ok = d_log == 0.0 and d_nb == 0.0 and d_v4 == 0.0
    say(f"V8: LOG-T fold 0 refit max|diff| {d_log:.1e}; NB-c fold 0 refit vs Stage 2 "
        f"{d_nb:.1e}; V4 simulation rerun {d_v4:.1e}")

    # ---- sanity: overall mean p against the label rate (no scoring here) -------
    sanity = []
    for lab_name, tc in PRIMARY.items():
        lab_col = tc.replace("cT_", "LT_").replace("cO_", "LO_")
        row = {"label": f"{lab_name} ({lab_col})", "label rate": float(lab[lab_col].mean())}
        for arm, col in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix")):
            row[arm] = float(oof[f"p_{col}_{lab_col}"].mean())
        own = "LOG-T" if lab_name == "BASE-T" else "LOG-O"
        row[own] = float(oof[f"p_{own.replace('-', '').lower()}_{lab_col}"].mean())
        sanity.append(row)
    sanity.append({"label": "SCORE (LS)", "label rate": float(lab["LS"].mean()),
                   "LOG-S": float(oof["p_logs_LS"].mean())})
    sanity = pd.DataFrame(sanity)

    mix_rows = fits[fits["arm"] == "MIX"][["data", "converged", "iterations", "grad_inf",
                                           "held-out loglik / cell", "Spartan CV, same fold",
                                           "e^delta", "mean pi (test)", "sd(eta)", "minutes"]]
    log_rows = log_fits[["arm", "data", "converged", "iterations", "grad_inf", "theta (log N)",
                         "norm a", "norm c", "sigma1(B)", "sd(eta)", "sd(eta) > 20 (V7)",
                         "minutes"]]
    max_ho_gap = float(np.nanmax(np.abs(mix_rows["held-out loglik / cell"]
                                        - mix_rows["Spartan CV, same fold"])))
    report(f"""
## Stage 3 -- out-of-fold fits ({time.strftime('%Y-%m-%d %H:%M')})

Every arm fitted on four folds and predicted on the fifth, on the stored
partition. NB-c and NB-d are Stage 2's fold fits. MIX is refitted per training
fold (penalty scaled to training size) from the full-data point estimate, to
the mixture's convergence criteria. LOG-T, LOG-O and LOG-S regress the primary
labels ({LOG_ARMS['LOG-T']}, {LOG_ARMS['LOG-O']}, LS) on the shared weight
block plus b0 + theta log N + s(log A), with the mixture's penalty term for
term. Count arms are pushed through every threshold on the alpha_B and Td
grids: NB p = P(Y >= c); MIX p = pi P1(Y >= c) + (1 - pi) P0(Y >= c), with the
prior pi.

### V5 -- fold identity

Map sha256 `{fold_hash[:16]}...`, equal to the stored map's; Stage 2's
out-of-fold NB predictions use the identical assignment; fold sizes match the
Spartan rank CV on every fold. **{'pass' if v5_ok else 'FAIL'}**

### MIX fold fits

All five converged: **{'yes' if mix_ok else 'NO'}**. Each fold's held-out
log-likelihood per cell agrees with the Spartan CV fit for the same fold (rank
{spec.rank}, lambda {spec.weight_ridge:g}) to {max_ho_gap:.1e}, so starting
from the full-data estimate did not change where the fits end.

{md_table(mix_rows)}

### LOG fits and V7 -- separation

Criterion: sd(eta) <= 20 at the production ridge for every LOG fit.
**{'pass' if v7_ok else 'FAIL'}** (largest sd(eta) {log_fits['sd(eta)'].max():.3f});
all converged: **{'yes' if log_conv else 'NO'}**.

{md_table(log_rows)}

### V4 -- push-through identity

{N_V4:,} cells drawn at random; y simulated from each arm's out-of-fold
predictive distribution; the simulated label rate compared with the mean
pushed-through p. Formal criterion on the primary thresholds ({PRIMARY['BASE-T']},
{PRIMARY['BASE-O']}): within 3 Monte Carlo SE. **{'pass' if v4_ok else 'FAIL'}**.
The other grid thresholds are shown for information; with
{int((~v4['primary']).sum())} further comparisons, one beyond 3 SE would be
expected about {0.0027 * int((~v4['primary']).sum()):.2f} times by chance.

{md_table(v4.drop(columns=['primary']), '{:.5g}')}

### V8 -- determinism

Refitting LOG-T on training fold 0 reproduces its parameters to {d_log:.1e};
refitting NB-c on training fold 0 reproduces Stage 2's saved fit to {d_nb:.1e};
rerunning the V4 simulation with its seed reproduces it to {d_v4:.1e}.
**{'pass' if v8_ok else 'FAIL'}**

### Sanity check, not scoring

Overall mean out-of-fold p against each label's rate. Scoring (log loss,
Brier, AUPRC, calibration by type, intervals) is Stage 4.

{md_table(sanity, '{:.5f}')}

Out-of-fold predictions for every arm and threshold are in
`oof_predictions.parquet`. Stage 3 complete in {(time.time() - t0) / 60:.0f} min.
**Review point** (stages 1, 3 and 6 pause for review).
""")
    for test, ok, detail in (("V7 (LOG separation)", v7_ok and log_conv, md_table(log_rows)),
                             ("V4 (push-through identity)", v4_ok, md_table(v4)),
                             ("V8 (determinism)", v8_ok, f"LOG {d_log}, NB {d_nb}, V4 {d_v4}")):
        if not ok:
            fail(test, detail)
    say(f"\nStage 3 done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
