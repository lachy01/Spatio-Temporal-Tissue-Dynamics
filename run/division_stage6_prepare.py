"""Division classifier comparison, Stage 6 (local part): allocation model and V9.

    python run/division_stage6_prepare.py

The known-truth simulation (spec 7) rebuilds every label from simulated
counts. BASE-T and BASE-O need only the composite count. SCORE needs per-gene
counts, so the simulated total is split across the Tirosh genes by the
allocation model of spec 3.4, holding the control genes at their observed
values:

* per coarse cell type t, gene proportions q_t pooled over that type's cells
  with y >= 1;
* a Dirichlet-multinomial with one concentration a_t per type, fitted so its
  per-gene detection rates P(count_g >= 1) match the observed ones. Under the
  DM the gene-g count is beta-binomial, so the detection rate is exact:
  1 - B(a q, a(1 - q) + y) / B(a q, a(1 - q)), averaged over the type's cells.

V9 (spec 8): a simulated split of the observed totals must reproduce every
per-type, per-gene detection rate within 10% (relative), for genes detected in
at least 1% of that type's cells. Stops on failure.

SCORE in the simulation. scanpy's score is mean_g log1p(X_g s / N) over the
scored genes minus the mean over its control genes, with N the library size
without the blocklist (fixed under simulation, since the Tirosh genes are in
the blocklist). Holding the control genes at their observed values therefore
holds each cell's control term fixed; this script stores it, derived as
gene-term minus score, and checks the decomposition on a recomputation of
scanpy's score with the Tirosh columns permuted across cells (same gene
averages, so the same control draw).

Also stores the observed variance-to-mean ratio of y in Cancer Epithelial,
which generator G2 matches (spec 7.1).

Writes outputs/division_classifier/{sim_inputs.npz, stage6_v9.csv} and the V9
note in report.md.
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
from tissue_dynamics.mixture.data import _tirosh_genes, load_config   # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
MIN_DETECT, V9_TOL, SEED = 0.01, 0.10, 20261008


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


def detect_rate(q, a, n_vals, n_cnt):
    """Expected fraction of cells with gene count >= 1, per gene, under
    DM(y, a q) for cells whose totals take values n_vals with counts n_cnt."""
    aq, ar = a * q[None, :], a * (1 - q)[None, :]
    n = n_vals[:, None].astype(float)
    log_p0 = gammaln(ar + n) + gammaln(a) - gammaln(ar) - gammaln(a + n)
    p_det = 1.0 - np.exp(log_p0)
    return (n_cnt[:, None] * p_det).sum(0) / n_cnt.sum()


def sim_dm(y, q, a, rng):
    """Split each total y_i across genes, DM(y_i, a q)."""
    out = np.zeros((len(y), len(q)), np.int64)
    m = y > 0
    g = rng.gamma(shape=a * q, size=(int(m.sum()), len(q)))
    p = g / g.sum(1, keepdims=True)
    out[m] = rng.multinomial(y[m].astype(np.int64), p)
    return out


def main():
    t0 = time.time()
    import scanpy as sc

    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "y", "N", "score", "LS"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    ids = df["cell_id"].astype(str).to_numpy()
    if not np.array_equal(lab["cell_id"].astype(str).to_numpy(), ids):
        sys.exit("labels.parquet lists cells in a different order from the table")
    ctype = lab["cell_type"].astype(str).to_numpy()
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    N = df["N"].to_numpy(float)
    y = df["y_count"].to_numpy(np.int64)

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
    score, sinfo = L.module_score(X, genes, N, tirosh, block, seed=0)
    if np.max(np.abs(score - lab["score"].to_numpy())) > 1e-10:
        sys.exit("module score does not reproduce Stage 1's")
    s = float(np.median(N))
    gene_term = lambda Y: np.log1p(Y * (s / N)[:, None]).mean(1)       # noqa: E731
    ctrl = gene_term(Yg) - score
    # Permute the Tirosh rows only among cells with equal N: every gene's mean
    # normalised expression is then unchanged, so scanpy bins the genes, and
    # draws the controls, exactly as before. A free permutation would shift the
    # Tirosh genes' averages (cells differ in N) and the control draw with them.
    rng = np.random.default_rng(SEED)
    perm = np.arange(len(y))
    order = np.argsort(N, kind="stable")
    bounds = np.flatnonzero(np.diff(N[order]) != 0) + 1
    for grp in np.split(order, bounds):
        perm[grp] = rng.permutation(grp)
    import scipy.sparse as sp
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

    # ---- allocation model ------------------------------------------------------
    rows, q_all, a_all = [], [], []
    v9_rows = []
    sim_rng = np.random.default_rng(SEED + 1)
    for t in types:
        m = ctype == t
        Yt, yt = Yg[m], y[m]
        pos = yt >= 1
        q = Yt[pos].sum(0) / Yt[pos].sum()
        q = np.maximum(q, 1e-9)
        q = q / q.sum()
        obs = (Yt >= 1).mean(0)
        n_vals, n_cnt = np.unique(yt, return_counts=True)
        use = obs >= MIN_DETECT

        def loss(la):
            pred = detect_rate(q, np.exp(la), n_vals, n_cnt)
            return float(np.sum((pred[use] / obs[use] - 1) ** 2))
        res = minimize_scalar(loss, bounds=(-4.0, 12.0), method="bounded", options={"xatol": 1e-6})
        a = float(np.exp(res.x))
        pred = detect_rate(q, a, n_vals, n_cnt)
        sim = (sim_dm(yt, q, a, sim_rng) >= 1).mean(0)
        for j, g in enumerate(tirosh):
            v9_rows.append({"type": t, "gene": g, "observed": obs[j], "expected": pred[j],
                            "simulated": sim[j], "tested": bool(use[j]),
                            "rel. error (simulated)": sim[j] / obs[j] - 1 if obs[j] > 0 else np.nan})
        rel = sim[use] / obs[use] - 1
        rows.append({"type": t, "cells": int(m.sum()), "cells with y >= 1": int(pos.sum()),
                     "concentration a": a, "genes tested": int(use.sum()),
                     "max |rel. error|": float(np.max(np.abs(rel))) if use.any() else np.nan,
                     "within 10%": f"{int(np.sum(np.abs(rel) <= V9_TOL))}/{int(use.sum())}"})
        q_all.append(q)
        a_all.append(a)
        say(f"  {t:<18} a = {a:9.3f}; max |rel. error| {rows[-1]['max |rel. error|']:.3f} "
            f"over {use.sum()} genes")
    alloc = pd.DataFrame(rows)
    v9 = pd.DataFrame(v9_rows)
    v9.to_csv(f"{OUT}/stage6_v9.csv", index=False)
    tested = v9[v9["tested"]]
    v9_ok = bool((tested["rel. error (simulated)"].abs() <= V9_TOL).all())

    ce = ctype == "Cancer Epithelial"
    vmr_ce = float(y[ce].var() / y[ce].mean())
    np.savez(f"{OUT}/sim_inputs.npz",
             cell_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
             ctrl=ctrl, normaliser_s=s, tirosh=np.array(tirosh), types=np.array(types),
             q=np.vstack(q_all), a=np.array(a_all), vmr_ce=vmr_ce,
             score_threshold_stage1=meta["score"]["threshold"])
    say(f"Cancer Epithelial variance-to-mean ratio {vmr_ce:.3f}; V9 {'pass' if v9_ok else 'FAIL'}")

    worst = tested.reindex(tested["rel. error (simulated)"].abs().sort_values(ascending=False).index).head(8)
    report(f"""
## Stage 6 -- known-truth simulation ({time.strftime('%Y-%m-%d %H:%M')})

### Allocation model and V9

SCORE is rebuilt in the simulation by splitting each simulated total across
the {len(tirosh)} Tirosh genes in the panel: per cell type, the gene proportions
of the observed counts (pooled over cells with y >= 1) and one
Dirichlet-multinomial concentration, fitted so the per-gene detection rates
match (exact under the model: each gene's count is beta-binomial). Control
genes are held at their observed values, which holds each cell's control term
fixed: scanpy's score is the scored genes' mean log1p(X s / N) minus the
control genes', and N excludes the Tirosh genes. That decomposition
reproduces scanpy's own score, recomputed with the Tirosh columns permuted
among cells of equal N (which leaves every gene's mean, and so scanpy's control
draw, unchanged), to {dec_err:.1e}.

{md_table(alloc, '{:.4g}')}

**V9** (a simulated split of the observed totals reproduces every per-type,
per-gene detection rate within 10%, for genes detected in at least 1% of the
type's cells): **{'pass' if v9_ok else 'FAIL'}**, over {len(tested)} type-gene pairs.
The largest relative errors:

{md_table(worst[['type', 'gene', 'observed', 'simulated', 'rel. error (simulated)']], '{:.4f}')}

The observed variance-to-mean ratio of y in Cancer Epithelial, which G2
matches, is {vmr_ce:.3f} (the spec's 7.29 was the SpaNorm-era table's).
Inputs for the Spartan array are in `sim_inputs.npz`.
""")
    if not v9_ok:
        report("\n### STOPPED in Stage 6: V9 (allocation model) failed\n\nPer spec section 8, "
               "nothing further was run.")
        sys.exit("V9 failed")
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
