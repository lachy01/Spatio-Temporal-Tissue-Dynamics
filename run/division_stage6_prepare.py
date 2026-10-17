"""Division classifier comparison, Stage 6 (local part): allocation model and V9.

    python run/division_stage6_prepare.py

The known-truth simulation (spec 7) rebuilds every label from simulated
counts. BASE-T and BASE-O need only the composite count. SCORE needs per-gene
counts, so the simulated total is split across the Tirosh genes by an
allocation model (spec 3.4), holding the control genes at their observed
values.

Allocation model, as adopted (user, 2026-10-08; a deviation from spec 3.4).
The spec's single Dirichlet-multinomial per cell type failed V9: it spreads the
canonical proliferation markers across every cell, when in real data they come
together in the cycling minority (BIRC5 detection over-predicted by 21% in
Cancer Epithelial). The adopted model gives each type two DM profiles, one for
cycling and one for non-cycling cells, each fitted by maximum likelihood with
every cell weighted by the mixture's posterior w (or 1 - w). The simulation
splits a cell's count by the profile of its simulated state (G1, G2) or, where
there is no state (G3), of a state drawn from the fitted mixture's posterior
given the simulated count.

V9 (spec 8), judged on the expected detection rates, which are exact under the
model (each gene's count is beta-binomial): every per-type, per-gene rate
within 10% (relative) of observed, for genes detected in at least 1% of the
type's cells. A single simulated draw is also reported; at detection rates of
1-2% its relative error alone has an SD of about 4%, so one draw is too noisy
to judge a 10% tolerance (decided with the user, 2026-10-08).

SCORE in the simulation. scanpy's score is mean_g log1p(X_g s / N) over the
scored genes minus the mean over its control genes, with N the library size
without the blocklist (fixed under simulation, since the Tirosh genes are in
the blocklist). Holding the control genes at their observed values therefore
holds each cell's control term fixed; this script stores it, as gene term minus
score, and checks the decomposition on scanpy's score recomputed with the
Tirosh columns permuted among cells of equal N (which leaves every gene's
mean, and so scanpy's control draw, unchanged).

Writes outputs/division_classifier/{sim_inputs.npz, stage6_v9.csv} and the V9
note in report.md. Stops on a failing check.
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

from scipy.optimize import minimize_scalar                            # noqa: E402
from scipy.special import gammaln                                     # noqa: E402

from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.simulation import sim_dm                # noqa: E402
from tissue_dynamics.mixture.data import _tirosh_genes, load_config   # noqa: E402

from tissue_dynamics.division import paths as DP                    # noqa: E402

OUT = DP.out_dir()          # DIV_VARIANT=<v>: outputs/division_classifier_<v>
REPORT = f"{OUT}/report.md"
MIN_DETECT, V9_TOL, SEED, N_DRAWS = 0.01, 0.10, 20261008, 10
# Type-gene pairs the user accepted beyond the 10% tolerance, with the bound each
# was accepted at. 2026-10-13, MisTIC model: Cancer Epithelial CENPF at 10.04%
# (the area model's worst pair had been 9.96%).
V9_ACCEPTED = {("Cancer Epithelial", "CENPF"): 0.105}


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def dm_logpmf(X, n, alpha):
    a0 = alpha.sum()
    return (gammaln(n + 1) - gammaln(X + 1).sum(1) + gammaln(a0) - gammaln(n + a0)
            + (gammaln(X + alpha[None, :]) - gammaln(alpha)[None, :]).sum(1))


def fit_weighted(X, wt):
    """Weighted-ML DM profile: proportions q and concentration a."""
    n = X.sum(1)
    q = (wt[:, None] * X).sum(0) + 1e-6
    q /= q.sum()
    res = minimize_scalar(lambda la: -np.sum(wt * dm_logpmf(X, n, np.exp(la) * q)),
                          bounds=(-4.0, 12.0), method="bounded", options={"xatol": 1e-6})
    return q, float(np.exp(res.x))


def p_det(q, a, n):
    """P(count_g >= 1) under DM(n, a q), per cell (rows) and gene (columns)."""
    ar = a * (1 - q)[None, :]
    nn = np.asarray(n, float)[:, None]
    return 1 - np.exp(gammaln(ar + nn) + gammaln(a) - gammaln(ar) - gammaln(a + nn))


def main():
    t0 = time.time()
    import jax.numpy as jnp
    import scanpy as sc
    import scipy.sparse as sp

    from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
    from tissue_dynamics.mixture.model import build_data, responsibilities

    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = types
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "score"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    ids = df["cell_id"].astype(str).to_numpy()
    if not np.array_equal(lab["cell_id"].astype(str).to_numpy(), ids):
        sys.exit("labels.parquet lists cells in a different order from the table")
    ctype = L.cell_type(df, types)
    if not np.array_equal(ctype, lab["cell_type"].astype(str).to_numpy()):
        sys.exit("cell types recomputed from the table differ from Stage 1's")
    N = df["N"].to_numpy(float)
    y = df["y_count"].to_numpy(np.int64)

    # the mixture's posterior, which weights the two allocation profiles
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    w_all = np.asarray(responsibilities({k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()},
                                        D, spec))

    say("reading the Xenium count matrix ...")
    ad = sc.read_10x_h5(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", gex_only=True)
    ad.var_names_make_unique()
    ad = ad[ids].copy()
    genes = np.asarray(ad.var_names)
    tirosh = [g for g in _tirosh_genes() if g in set(genes)]
    if tirosh != meta["tirosh_genes"]:
        sys.exit("the Tirosh gene list differs from the one the labels were built with")
    block = list(dict.fromkeys(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str)))
    block = [g for g in block if g in set(genes)]
    X = ad.X.tocsr()
    gi = {g: i for i, g in enumerate(genes)}
    Yg = np.asarray(X[:, [gi[g] for g in tirosh]].todense()).astype(np.int64)
    if not np.array_equal(Yg.sum(1), y):
        sys.exit("per-gene Tirosh counts do not sum to y_count")

    # ---- SCORE decomposition: control term, checked on a permuted matrix ------
    score, _ = L.module_score(X, genes, N, tirosh, block, seed=0)
    if np.max(np.abs(score - lab["score"].to_numpy())) > 1e-10:
        sys.exit("module score does not reproduce Stage 1's")
    s = float(np.median(N))
    gene_term = lambda Y: np.log1p(Y * (s / N)[:, None]).mean(1)       # noqa: E731
    ctrl = gene_term(Yg) - score
    # Permute the Tirosh rows only among cells with equal N: every gene's mean
    # normalised expression is then unchanged, so scanpy bins the genes, and
    # draws the controls, exactly as before.
    rng = np.random.default_rng(SEED)
    perm = np.arange(len(y))
    order = np.argsort(N, kind="stable")
    for grp in np.split(order, np.flatnonzero(np.diff(N[order]) != 0) + 1):
        perm[grp] = rng.permutation(grp)
    tix = np.array([gi[g] for g in tirosh])
    keep = np.ones(len(genes))
    keep[tix] = 0.0
    S = sp.csr_matrix((np.ones(len(tix)), (np.arange(len(tix)), tix)), shape=(len(tix), len(genes)))
    Xp = (X @ sp.diags(keep) + sp.csr_matrix(Yg[perm].astype(np.float32)) @ S).tocsr()
    Xp.eliminate_zeros()
    score_p, _ = L.module_score(Xp, genes, N, tirosh, block, seed=0)
    dec_err = float(np.max(np.abs(score_p - (gene_term(Yg[perm]) - ctrl))))
    say(f"SCORE = gene term - control term: max |error| on a permuted matrix {dec_err:.1e}")
    if dec_err > 1e-6:
        sys.exit("the score does not decompose as gene term minus a fixed control term")
    ctrl32 = ctrl.astype(np.float32)
    ctrl_err = float(np.max(np.abs(ctrl32.astype(float) - ctrl)))

    # ---- allocation model: cycling and non-cycling profiles per type -----------
    rows, v9_rows = [], []
    Q1, A1, Q0, A0 = [], [], [], []
    sim_rng = np.random.default_rng(SEED + 1)
    for t in types:
        m = ctype == t
        Yt, yt, wt = Yg[m], y[m], w_all[m]
        pos = yt >= 1
        (q1, a1), (q0, a0) = fit_weighted(Yt[pos], wt[pos]), fit_weighted(Yt[pos], 1 - wt[pos])
        obs = (Yt >= 1).mean(0)
        use = obs >= MIN_DETECT
        expected = (wt[:, None] * p_det(q1, a1, yt) + (1 - wt)[:, None] * p_det(q0, a0, yt)).mean(0)
        sims = []
        for _ in range(N_DRAWS):
            z = sim_rng.uniform(size=len(yt)) < wt
            Ys = np.zeros_like(Yt)
            Ys[z], Ys[~z] = sim_dm(yt[z], q1, a1, sim_rng), sim_dm(yt[~z], q0, a0, sim_rng)
            sims.append((Ys >= 1).mean(0))
        sims = np.array(sims)
        for j, g in enumerate(tirosh):
            v9_rows.append({"type": t, "gene": g, "observed": obs[j], "expected": expected[j],
                            "one draw": sims[0, j], f"mean of {N_DRAWS} draws": sims[:, j].mean(),
                            "tested": bool(use[j]),
                            "rel. error (expected)": expected[j] / obs[j] - 1 if obs[j] > 0 else np.nan,
                            "rel. error (one draw)": sims[0, j] / obs[j] - 1 if obs[j] > 0 else np.nan})
        rel = expected[use] / obs[use] - 1
        rows.append({"type": t, "cells": int(m.sum()), "mean w": float(wt.mean()),
                     "concentration, cycling": a1, "concentration, non-cycling": a0,
                     "genes tested": int(use.sum()), "max |rel. error|, expected": float(np.max(np.abs(rel))),
                     "beyond 10%, one draw": int(np.sum(np.abs(sims[0][use] / obs[use] - 1) > V9_TOL))})
        Q1.append(q1)
        A1.append(a1)
        Q0.append(q0)
        A0.append(a0)
        say(f"  {t:<18} a1 {a1:7.2f} a0 {a0:7.2f}; max |rel. error| (expected) {rows[-1]['max |rel. error|, expected']:.3f}")
    alloc = pd.DataFrame(rows)
    v9 = pd.DataFrame(v9_rows)
    bound = np.array([V9_ACCEPTED.get((t, g), V9_TOL) for t, g in zip(v9["type"], v9["gene"])])
    v9["accepted exception"] = [(t, g) in V9_ACCEPTED for t, g in zip(v9["type"], v9["gene"])]
    v9["within bound"] = v9["rel. error (expected)"].abs() <= bound
    v9.to_csv(f"{OUT}/stage6_v9.csv", index=False)
    tested = v9[v9["tested"]]
    v9_ok = bool(tested["within bound"].all())
    excepted = tested[tested["accepted exception"]
                      & (tested["rel. error (expected)"].abs() > V9_TOL)]

    ce = ctype == "Cancer Epithelial"
    vmr_ce = float(y[ce].var() / y[ce].mean())
    np.savez_compressed(f"{OUT}/sim_inputs.npz",
                        cell_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                        ctrl=ctrl32, normaliser_s=s, tirosh=np.array(tirosh), types=np.array(types),
                        q1=np.vstack(Q1), a1=np.array(A1), q0=np.vstack(Q0), a0=np.array(A0),
                        vmr_ce=vmr_ce)
    say(f"Cancer Epithelial variance-to-mean ratio {vmr_ce:.3f}; V9 {'pass' if v9_ok else 'FAIL'}")

    worst = tested.reindex(tested["rel. error (expected)"].abs().sort_values(ascending=False).index).head(6)
    exception_text = ("" if excepted.empty else ", except " + "; ".join(
        f"{r['type']} {r['gene']} at {100 * abs(r['rel. error (expected)']):.2f}%"
        for _, r in excepted.iterrows()) + ", accepted by the user (2026-10-13)")
    report(f"""
### Allocation model as adopted, and V9 ({time.strftime('%Y-%m-%d %H:%M')})

**Decision (user, 2026-10-08):** the state-conditional allocation replaces
spec 3.4's single Dirichlet-multinomial. Each cell type gets one DM profile
for cycling and one for non-cycling cells, each fitted by maximum likelihood
with cells weighted by the mixture's posterior w (or 1 - w). The simulation
splits a cell's count by the profile of its simulated state (G1, G2), or, in
G3, of a state drawn from the fitted mixture's posterior given the simulated
count. The mixture never sees per-gene counts; the allocation borrows its
posterior only to describe how genes split. V9 is judged on the expected
detection rates (exact under the model), since a single draw's noise at 1-2%
detection is comparable to the 10% tolerance.

{md_table(alloc, '{:.4g}')}

**V9: {'pass' if v9_ok else 'FAIL'}**, every one of {len(tested)} tested type-gene pairs within
10% in expectation{exception_text}. The largest:

{md_table(worst[['type', 'gene', 'observed', 'expected', 'one draw', f'mean of {N_DRAWS} draws', 'rel. error (expected)']], '{:.4f}')}

The SCORE decomposition (gene term minus a fixed control term) reproduces
scanpy's own score, recomputed with the Tirosh columns permuted among cells of
equal N, to {dec_err:.1e}; the stored control term (float32) differs from the
float64 one by at most {ctrl_err:.1e}. The observed variance-to-mean ratio of y
in Cancer Epithelial, which G2 matches, is {vmr_ce:.3f} (the spec's 7.29 was the
SpaNorm-era table's). The inputs for the Spartan array are in `sim_inputs.npz`.
""")
    if not v9_ok:
        report("\n### STOPPED in Stage 6: V9 (allocation model) failed\n\nPer spec section 8, "
               "nothing further was run.")
        sys.exit("V9 failed")
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
