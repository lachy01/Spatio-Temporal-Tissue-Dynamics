"""MisTIC step 2: per-cell covariates and the MisTIC-weighted count matrix.

    python run/mistic_covariates.py

Reads `<mistic.dir>/transcripts_mistic.parquet` (run/mistic_gather.py) and the
raw cell_feature_matrix.h5. Writes to `<mistic.dir>/`:

  cell_mistic.parquet   one row per segmented cell, in cell_feature_matrix.h5's
                        order: cell_id, x, y, in_mistic, N, N_nonB, Nt, Nt_nonB,
                        f, Y_cyc, s_cyc, m_cyc (definitions in
                        tissue_dynamics/mistic.py; `_cyc` is the Tirosh S/G2M set)
  cell_feature_matrix_mistic.h5   X~_ig = sum of q_k over cell i's transcripts of
                        gene g, in the layout of cell_feature_matrix.h5 (float
                        data), for the PhiSpace input

and `outputs/mistic/covariates_summary.json`.

A cell outside MisTIC's input (no retained transcript, an invalid polygon, or no
PhiSpace label) has no p; its transcripts count with p = 0, so its Nt equals N,
f = 0 and m = 0, and `in_mistic` is False.

Checks, all fatal: transcripts_mistic.parquet reproduces the raw counts of every
MisTIC cell; p lies in [0, 1]; Nt <= N for every cell.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics import mistic as M                                # noqa: E402
from tissue_dynamics.mixture.data import load_config                   # noqa: E402

OUT = "outputs/mistic"          # outputs/mistic_<dataset> under MIX_DATASET


def main():
    t0 = time.time()
    cfg = load_config()
    P = M.paths(cfg)
    global OUT
    if cfg.get("dataset"):
        OUT = f"outputs/mistic_{cfg['dataset']}"
    os.makedirs(OUT, exist_ok=True)

    counts = M.read_counts(cfg)
    n = len(counts.cell_ids)
    isB = M.blocklist(cfg, counts.genes)
    tirosh = M.tirosh_on_panel(counts.genes)
    print(f"{n:,} cells x {len(counts.genes):,} genes; blocklist {isB.sum()} genes "
          f"(Tirosh {len(tirosh)} on the panel)", flush=True)

    cell, gene, p = M.transcript_codes(P["transcripts"], counts)
    if p.min() < 0 or p.max() > 1 or np.isnan(p).any():
        sys.exit("p outside [0, 1]")
    cov = M.cell_covariates(counts, cell, gene, p, isB)
    N, in_mistic, nb = cov.N, cov.in_mistic, ~isB[gene]
    if not np.array_equal(np.bincount(cell, minlength=n)[in_mistic], N[in_mistic]):
        sys.exit("transcripts_mistic.parquet does not reproduce the raw counts of MisTIC's cells")
    print(f"  {len(p):,} transcripts with p, in {in_mistic.sum():,} cells; reproduce the raw "
          f"counts exactly", flush=True)
    # cells a tile left out of its fit (mistic.rare_type_policy: exclude) were
    # given to MisTIC but never evaluated: p = 0 already, and in_mistic False
    if os.path.exists(P["excluded"]):
        ex = set(c for c in open(P["excluded"]).read().split("\n") if c)
        hit = np.isin(counts.cell_ids, list(ex))
        if int(hit.sum()) != len(ex):
            sys.exit("excluded_cells.txt names cells not in cell_feature_matrix.h5")
        in_mistic = in_mistic & ~hit
        print(f"  {int(hit.sum()):,} cells left out of their tile's MisTIC fit: in_mistic False")
    tol = 1e-6 * np.maximum(N, 1.0)
    if np.any(cov.Nt > N + tol) or np.any(cov.Nt < -tol):
        sys.exit("Nt outside [0, N]")
    m_cyc, s_cyc = cov.m_gene_set(tirosh)
    tab = pd.DataFrame({
        "cell_id": counts.cell_ids, "x": counts.xy[:, 0], "y": counts.xy[:, 1],
        "in_mistic": in_mistic, "N": N, "N_nonB": cov.N_nonB, "Nt": cov.Nt,
        "Nt_nonB": cov.Nt_nonB, "f": cov.f, "Y_cyc": counts.gene_sum(tirosh), "s_cyc": s_cyc,
        "m_cyc": m_cyc})
    tab.to_parquet(P["cells"] + ".tmp", index=False)
    os.replace(P["cells"] + ".tmp", P["cells"])
    print(f"wrote {P['cells']}", flush=True)

    # ---- weighted matrix -------------------------------------------------------
    W = M.weighted_matrix(counts, cell, gene, p)
    if W.data.min() < -1e-4:
        sys.exit("negative weighted counts")
    W.data = np.maximum(W.data, 0.0)
    M.write_10x_h5(P["weighted_h5"], W, counts,
                   f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5")
    print(f"wrote {P['weighted_h5']} ({os.path.getsize(P['weighted_h5']) / 1e6:.0f} MB)", flush=True)

    m = in_mistic
    summ = {
        "cells": n, "cells_in_mistic": int(m.sum()), "transcripts_with_p": int(len(p)),
        "mean_p_all_transcripts": float(p.mean()),
        "mean_p_nonB_transcripts": float(p[nb].mean()),
        "mean_p_cycle_transcripts": float(p[np.isin(gene, counts.gene_index(tirosh))].mean()),
        "share_p_gt_0.5": float(np.mean(p > 0.5)),
        "Nt_over_N_median_mistic_cells": float(np.median(cov.Nt[m] / np.maximum(N[m], 1))),
        "f_quantiles_mistic_cells": dict(zip(("q10", "q50", "q90"),
                                             np.quantile(cov.f[m], [.1, .5, .9]).tolist())),
        "m_cyc_mean": float(m_cyc.mean()), "m_cyc_quantiles": dict(zip(
            ("q50", "q90", "q99"), np.quantile(m_cyc, [.5, .9, .99]).tolist())),
        "Y_cyc_mean": float(tab["Y_cyc"].mean()),
        "corr_log1p_m_cyc_vs_log_N": float(np.corrcoef(np.log1p(m_cyc[N > 0]),
                                                       np.log(N[N > 0]))[0, 1]),
        "seconds": round(time.time() - t0, 1),
    }
    M.summary_json(f"{OUT}/covariates_summary.json", summ)
    for k, v in summ.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
