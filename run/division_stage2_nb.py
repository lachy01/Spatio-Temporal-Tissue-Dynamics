"""Division classifier comparison, Stage 2: refit NB-c and NB-d to convergence.

    python run/division_stage2_nb.py

Spec sections 1 and 9, tests V1 (convergence), V5 (fold identity) and V6
(parity). The NB fits do not depend on any label, so the five training-fold
fits made here for V1 are also Stage 3's out-of-fold NB arms: their held-out
predictions are saved and reused rather than refitted.

Decided by the user (see report.md): theta in both NB arms as in MIX -- free
until 2026-10-06, fixed at 1 since (config `model.theta_free`); NB-d's
dispersion is log alpha_i = gamma0 + phi_i' gamma on the projected state.
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
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.inference import block_folds, spatial_blocks  # noqa: E402
from tissue_dynamics.mixture.model import _logit_pi, build_data, loglik_terms  # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
ARMS = {"NB-c": "const", "NB-d": "state"}


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def fail(test, detail):
    report(f"\n### STOPPED in Stage 2: {test} failed\n\n{detail}\n\nPer spec section 8, "
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


def main():
    t0 = time.time()
    # Everything read from disk is read here, up front: the repo is on an
    # iCloud-synced Desktop, and an evicted file has already killed one long run.
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    import glob
    # V5 compares fold SIZES with the Spartan CV, which depend only on the cells and
    # the spatial partition, not on the technical covariates: the area-era CV
    # (archived in outputs/hpc_area since the MisTIC switch) serves as well
    cvp_dir = next((d for d in ("outputs/hpc/cv_penalty", "outputs/hpc_area/cv_penalty")
                    if glob.glob(f"{d}/*.json")), "outputs/hpc/cv_penalty")
    cvp = [json.load(open(f)) for f in sorted(glob.glob(f"{cvp_dir}/*.json"))]
    labels = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id"])

    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    n = len(df)
    if not np.array_equal(labels["cell_id"].to_numpy(), df["cell_id"].astype(str).to_numpy()):
        fail("cell alignment", "labels.parquet and the mixture table list cells in a "
             "different order")
    say(f"{n:,} cells; weight_ridge {spec.weight_ridge:g} x n / {spec.n_reference:,}; "
        f"ridge_tau {spec.ridge_tau:g}; rank {spec.rank}")

    # ---- V5: folds -------------------------------------------------------------
    # Spec 4.1: load the stored assignment if it exists. It must come from the
    # environment the mixture CV ran in (run/division_make_folds.py on Spartan):
    # MiniBatchKMeans returns a different partition here from the same seed.
    stored = f"{OUT}/folds.parquet"
    if os.path.exists(stored):
        fm = pd.read_parquet(stored)
        if not np.array_equal(fm["cell_id"].astype(str).to_numpy(),
                              df["cell_id"].astype(str).to_numpy()):
            fail("V5 (fold identity)", "folds.parquet lists cells in a different order "
                 "from the mixture table")
        blocks, folds = fm["block"].to_numpy(), fm["fold"].to_numpy()
        fold_src = "loaded from folds.parquet"
    else:
        xy = df[["x", "y"]].to_numpy(float)
        blocks = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
        folds = block_folds(blocks, int(cfg["model"].get("cv_folds", 5)))
        fold_src = "regenerated locally"
    say(f"folds {fold_src}")
    fold_hash = hashlib.sha256(folds.astype(np.int8).tobytes()).hexdigest()
    sizes_here = {f: (int(np.sum(folds != f)), int(np.sum(folds == f))) for f in range(5)}
    sizes_mix = {}
    for r in cvp:
        sizes_mix.setdefault(int(r["fold"]), set()).add((int(r["n_train"]), int(r["n_test"])))
    v5_rows, v5_ok = [], True
    for f in range(5):
        mix = sizes_mix.get(f, set())
        ok = len(mix) == 1 and sizes_here[f] in mix
        v5_ok &= ok
        v5_rows.append({"fold": f, "n_train here": sizes_here[f][0],
                        "n_test here": sizes_here[f][1],
                        "n_test, mixture CV (Spartan)": "/".join(str(m[1]) for m in sorted(mix)),
                        "match": "yes" if ok else "NO"})
    v5 = pd.DataFrame(v5_rows)
    say(v5.to_string(index=False))
    if not v5_ok:
        fail("V5 (fold identity)", f"the folds ({fold_src}) do not reproduce the mixture "
             "CV's fold sizes:\n\n" + md_table(v5) + "\n\nMiniBatchKMeans is "
             "deterministic within an environment but not across them: this machine "
             "(scikit-learn 1.6.1) returns the same partition on every rerun and under "
             "one thread, and Spartan returned identical sizes across all 40 CV tasks, "
             "but the two disagree. Fix: generate the map on Spartan with "
             "`sbatch hpc/folds.slurm` and copy back `folds.parquet`.")

    # ---- V6: parity -------------------------------------------------------------
    Pmix = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    v6_num = float(np.max(np.abs(np.asarray(Pmix["a0"] + arms.eta_state(Pmix, D)
                                            - _logit_pi(Pmix, D, spec)))))
    rng = np.random.default_rng(0)
    probe = arms.init_eta(D["own"].shape[1], D["nbr"].shape[1], D["hbasis"].shape[1],
                          spec.rank, rng)
    names = {"MIX": arms.eta_column_names(Pmix),
             "NB-c": arms.eta_column_names(arms.init_nb(D, "const", spec.rank, 0,
                                                        theta_free=spec.theta_free)),
             "NB-d": arms.eta_column_names(arms.init_nb(D, "state", spec.rank, 0,
                                                        theta_free=spec.theta_free))}
    for lab in ("LOG-T", "LOG-O", "LOG-S"):
        names[lab] = arms.eta_column_names(probe)
    v6_names_ok = all(v == names["MIX"] for v in names.values())
    v6_ok = v6_names_ok and v6_num < 1e-10
    say(f"V6: {len(names['MIX'])} eta columns, identical names and order across "
        f"{len(names)} arms: {v6_names_ok}; numeric parity with the mixture "
        f"{v6_num:.2e}")
    if not v6_ok:
        fail("V6 (parity)", f"names identical: {v6_names_ok}; numeric max|diff| {v6_num:.2e}")

    # ---- fits -------------------------------------------------------------------
    rows, full = [], {}
    oof = pd.DataFrame({"cell_id": df["cell_id"].astype(str).to_numpy(), "fold": folds})

    def fit(arm, idx, seed, tag):
        Ds = data.subset(idx).as_jax() if idx is not None else D
        m = n if idx is None else len(idx)
        say(f"  {arm} {tag} (n = {m:,}, seed {seed})")
        f = arms.nb_objective(Ds, m, spec.weight_ridge, spec.n_reference, spec.ridge_tau,
                              spec.b_penalty)
        # Retry from the next seed if Newton cannot finish. The optimum is unique
        # (two seeds agree to 1e-10 on all cells), and on 2026-10-05 the one V1
        # failure -- NB-c, fold 4, seed 0, stuck at |g| 2.3e-3 -- had already
        # reached the same objective that seeds 1 and 2 converged to.
        for attempt in range(3):
            p0 = arms.init_nb(Ds, ARMS[arm], spec.rank, seed + 100 * attempt,
                              theta_free=spec.theta_free)
            p, info = arms.minimise(f, p0, say=say)
            if info["converged"]:
                break
            say(f"      not converged; retrying from seed {seed + 100 * (attempt + 1)}")
        info["attempts"] = attempt + 1
        ll = float(jnp.sum(arms.nb_logpmf(Ds["y"], arms.nb_log_mu(p, Ds),
                                          arms.nb_log_alpha(p, Ds))))
        la = np.asarray(arms.nb_log_alpha(p, Ds)) * np.ones(m)
        diag = arms.eta_diagnostics(p, Ds)
        rows.append({"arm": arm, "data": tag, "seed": seed, "n": m, "loglik": ll,
                     "objective": info["f"], "grad_inf": info["grad_inf"],
                     "L-BFGS iters": info["lbfgs_iters"], "Newton iters": info["newton_iters"],
                     "converged (V1)": "pass" if info["converged"] else "FAIL",
                     "attempts": info["attempts"],
                     "theta": float(p["theta"]) if "theta" in p else 1.0,
                     "alpha": (f"{np.exp(la[0]):.4f}" if ARMS[arm] == "const" else
                               f"{np.exp(np.percentile(la, 5)):.3f}-"
                               f"{np.exp(np.percentile(la, 95)):.3f}"),
                     "norm a": diag["norm_a"], "norm c": diag["norm_c"],
                     "sigma1(B)": diag["sigma1_B"], "sd(eta)": diag["sd_eta"],
                     "sd(eta) > 20": "FLAG" if diag["flag_sd_eta"] else "no",
                     "min Hessian eig": info["min_hessian_eig"],
                     "minutes": info["wall_time"] / 60})
        return p, info

    for arm in ARMS:
        for seed in (0, 1):
            p, info = fit(arm, None, seed, "all cells")
            full[(arm, seed)] = (p, info)
    for arm in ARMS:
        for f in range(5):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            p, _ = fit(arm, tr, 0, f"train fold {f}")
            Dte = data.subset(te).as_jax()
            col = arm.replace("-", "").lower()
            oof.loc[te, f"log_mu_{col}"] = np.asarray(arms.nb_log_mu(p, Dte))
            oof.loc[te, f"log_alpha_{col}"] = (np.asarray(arms.nb_log_alpha(p, Dte))
                                               * np.ones(len(te)))
            json.dump(to_json(p), open(f"{OUT}/nb_params_{col}_fold{f}.json", "w"))
    oof.to_parquet(f"{OUT}/oof_nb.parquet", index=False)
    for arm in ARMS:
        col = arm.replace("-", "").lower()
        json.dump(to_json(full[(arm, 0)][0]), open(f"{OUT}/nb_params_{col}_full.json", "w"))
    fits = pd.DataFrame(rows)
    fits.to_csv(f"{OUT}/stage2_fits.csv", index=False)
    say(fits.drop(columns=["min Hessian eig"]).round(5).to_string(index=False))

    # ---- uniqueness, LR test, comparison ----------------------------------------
    uniq = []
    for arm in ARMS:
        (p0, i0), (p1, i1) = full[(arm, 0)], full[(arm, 1)]
        B0 = np.asarray(p0["U"]) @ np.asarray(p0["V"]).T
        B1 = np.asarray(p1["U"]) @ np.asarray(p1["V"]).T
        uniq.append({"arm": arm, "|objective diff|": abs(i0["f"] - i1["f"]),
                     "max |B diff|": float(np.max(np.abs(B0 - B1))),
                     "max |a diff|": float(np.max(np.abs(np.asarray(p0["a"]) - np.asarray(p1["a"])))),
                     "|theta diff|": (abs(float(p0["theta"]) - float(p1["theta"]))
                                      if "theta" in p0 else 0.0)})
    uniq = pd.DataFrame(uniq)
    llc = float(fits[(fits.arm == "NB-c") & (fits.data == "all cells") & (fits.seed == 0)].loglik.iloc[0])
    lld = float(fits[(fits.arm == "NB-d") & (fits.data == "all cells") & (fits.seed == 0)].loglik.iloc[0])
    ll_mix = float(jnp.sum(loglik_terms(Pmix, D, spec)))
    v1_ok = bool((fits["converged (V1)"] == "pass").all())
    any_flag = bool((fits["sd(eta) > 20"] == "FLAG").any())

    report(f"""
## Stage 2 -- NB refits ({time.strftime('%Y-%m-%d %H:%M')})

**The model changed after Stage 1** (decisions recorded in the mixture's
specification doc, 2026-10-06 to 10-07, all by the user): log N is an offset
(theta = 1) in every count arm, MIX included; the oxygen term is dropped; the
neighbour block is split into contact (0-15 um) and regional (15-80 um) means;
both state blocks are whitened; B is penalised by lambda ||UV'||_F^2 with the
balance term tau ||U'U - V'V||_F^2; lambda was selected on null-calibrated
Test 03 and the rank by spatial CV. With theta = 1 in MIX too, the spec's offset
mismatch (section 1) no longer arises. The labels are unaffected: they depend
only on y, N, A and the technical null.

NB-c and NB-d, matched to the mixture's covariates and penalty: the shared weight
block ({D['own'].shape[1]} own and {D['nbr'].shape[1]} neighbour coordinates,
rank {spec.rank}), log N {"free" if spec.theta_free else "as an offset"}, {("the MisTIC-weighted library size and theta_m m in the mean (no area term)" if spec.mistic else "the same df-" + str(cfg["model"]["area_spline_df"]) + " area spline")}, weight_ridge {spec.weight_ridge:g} x n /
{spec.n_reference:,} on every eta coefficient ({spec.b_penalty} form on B) and
the balance term at tau = {spec.ridge_tau:g}. NB-d adds log alpha_i = gamma0 +
phi_i' gamma on the projected own state (gamma not penalised). Optimiser:
L-BFGS, then Newton on the exact Hessian with absolute-value eigenvalues and a
backtracking line search.

### V5 -- fold identity

250 k-means blocks (MiniBatchKMeans, seed 0) dealt into 5 folds (seed 0); map
{fold_src}, sha256 `{fold_hash[:16]}...`. The mixture CV did not store its
cell-to-fold map, so identity is checked on the fold sizes it recorded: all five
match to the cell, which a different partition would not do by chance.
**{'pass' if v5_ok else 'FAIL'}**

{md_table(v5)}

### V6 -- parity

All {len(names)} arms (MIX, NB-c, NB-d, LOG-T, LOG-O, LOG-S) share the same
{len(names['MIX'])} eta columns in the same order. The shared `eta_state`
reproduces the production mixture's logit to {v6_num:.1e} on every cell. **pass**

### V1 -- convergence

Criterion: |f_t - f_(t-1)| < 1e-9 |f_(t-1)| and ||grad||_inf < 1e-5, on the
summed penalised objective. **{'pass' if v1_ok else 'FAIL'}**: {int((fits['converged (V1)'] == 'pass').sum())}
of {len(fits)} fits converged. sd(eta) > 20 flagged: **{'yes' if any_flag else 'none'}**.

{md_table(fits.drop(columns=['min Hessian eig']))}

`run/compare_nb_mixture.py`'s matched NB is unpenalised, so it is not the
same model as NB-c, which matches the mixture's penalty.

**Uniqueness.** Two seeds per arm on all cells:

{md_table(uniq, '{:.3g}')}

**NB-d against NB-c.** Log-likelihood {lld:,.2f} against {llc:,.2f}, a gain of
{lld - llc:,.1f} for 8 parameters (LR statistic {2 * (lld - llc):,.0f}). The
dispersion depends on state.

For reference, the mixture's data log-likelihood on the same cells is
{ll_mix:,.2f}. As in the main write-up, the NB fits the per-cell counts better;
the label-scale comparison begins in Stage 3.

Out-of-fold NB predictions (log mu and log alpha, per held-out cell) are saved
in `oof_nb.parquet` for Stage 3. Stage 2 complete in {(time.time() - t0) / 60:.0f} min.
""")
    if not v1_ok:
        bad = fits[fits["converged (V1)"] != "pass"][["arm", "data", "grad_inf"]]
        fail("V1 (convergence)", md_table(bad))
    say(f"\nStage 2 done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
