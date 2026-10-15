"""Ovarian sensitivity: the mixture with a depth-residualised state block.

    MIX_DATASET=ovarian python run/ovarian_depth_residual_check.py [--restarts 4]

On the ovarian section the production fit's prior pi rises with depth within
type (logit pi against log N: Tumour +0.38, Fibroblasts +0.35), where breast
stays mostly within +/-0.1 -- the PhiSpace Tumour score tracks depth within
tumour (rho 0.75). This rebuilds the table with each PhiSpace score
residualised on log N within type (`data.phi_depth_residualise`: the
within-type slope removed, type means kept; neighbour means rebuilt from the
residualised scores; types from the raw scores), copies the MisTIC columns
across (same cells, same order), refits the production specification on all
cells, and compares it with the production point estimate
(outputs/hpc_ovarian/point_estimate.json):

* the global parameters (e^delta, theta_m, cycling fraction, sd of eta);
* within-type Spearman correlation of logit pi and of w with log N;
* per-type cycling fraction (mean w);
* per-type neighbourhood contrast: mean logit pi in cells at or above the
  type's 90th percentile of neighbourhood Tumour score minus those at or
  below its 10th (the raw-score neighbourhood, the same cells in both fits).

Writes data/ovarian/mixture_cells_ovarian_depthres.parquet (+ sidecar) and
outputs/ovarian_depthres/{report.md, comparison.json}.
"""
import argparse
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
from scipy.stats import spearmanr                                     # noqa: E402

from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import (build_table, load_config,   # noqa: E402
                                          phi_cell_types, validate)
from tissue_dynamics.mixture.fit import multistart                    # noqa: E402
from tissue_dynamics.mixture.model import (build_data, pi_hat,        # noqa: E402
                                           responsibilities)
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402

OUT = "outputs/ovarian_depthres"
TABLE = "data/ovarian/mixture_cells_ovarian_depthres.parquet"


def say(*a):
    print(*a, flush=True)


def md_table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def summarise(name, params, df, basis, cfg, spec, ct, tum_nbr):
    D = build_data(df, basis, cfg, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in params.items()}
    pi = np.asarray(pi_hat(P, D, spec))
    w = np.asarray(responsibilities(P, D, spec))
    lN = np.asarray(D["logN"])
    lp = np.log(pi / (1 - pi))
    glob = {"fit": name, "e^delta": float(np.exp(np.asarray(P["delta"]))),
            "rate ratio": float(np.exp(np.exp(np.asarray(P["delta"])))),
            "theta_m": float(np.exp(np.asarray(P["eta_m"]))), "mean w": float(w.mean()),
            "sd logit pi": float(lp.std()),
            "rho(logit pi, log N), all": float(spearmanr(lp, lN).statistic)}
    per = {}
    for t in pd.Series(ct).value_counts().index:
        m = ct == t
        lo_q, hi_q = np.quantile(tum_nbr[m], [0.1, 0.9])
        per[t] = {"cells": int(m.sum()), "mean w": float(w[m].mean()),
                  "rho(logit pi, log N)": float(spearmanr(lp[m], lN[m]).statistic),
                  "rho(w, log N)": float(spearmanr(w[m], lN[m]).statistic),
                  "nbr contrast": float(lp[m & (tum_nbr >= hi_q)].mean() - lp[m & (tum_nbr <= lo_q)].mean())}
    return glob, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--reuse-table", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    if cfg.get("dataset") != "ovarian":
        sys.exit("run with MIX_DATASET=ovarian")
    os.makedirs(OUT, exist_ok=True)
    prod_path = cfg["data"]["table_parquet"]
    pe = json.load(open("outputs/hpc_ovarian/point_estimate.json"))

    # ---- the residualised table -------------------------------------------------------
    if not (a.reuse_table and os.path.exists(TABLE)):
        rcfg = json.loads(json.dumps(cfg))
        rcfg["data"]["phi_depth_residualise"] = True
        df = build_table(rcfg)
        clean, _ = validate(df, rcfg)
        prod = pd.read_parquet(prod_path)
        if not np.array_equal(prod["cell_id"].to_numpy(), clean["cell_id"].to_numpy()):
            sys.exit("the residualised table's cells differ from the production table's")
        add = [c for c in prod.columns if c not in clean.columns]
        for c in add:
            clean[c] = prod[c].to_numpy()
        clean.to_parquet(TABLE, index=False)
        json.dump(list(clean.attrs["phi_names"]), open(TABLE.replace(".parquet", ".phi_names.json"), "w"))
        say(f"wrote {TABLE}: {len(clean):,} cells; copied {len(add)} columns from the production table")

    # ---- fits ---------------------------------------------------------------------------
    df_p, basis_p = _load(cfg)
    rcfg = json.loads(json.dumps(cfg))
    rcfg["data"]["table_parquet"] = TABLE
    df_r, basis_r = _load(rcfg)
    spec_p = spec_from_basis(basis_p, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    spec_r = spec_from_basis(basis_r, rcfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    names = df_p.attrs["phi_names"]
    cols = sorted(c for c in df_p.columns if c.startswith("phi_") and c[4:].isdigit())
    ct = phi_cell_types(df_p[cols].to_numpy(float), names, cfg["data"])
    if not np.array_equal(ct.astype(str), df_r["archetype"].astype(str).to_numpy()):
        sys.exit("the residualised table's types differ from the production types")
    tum_nbr = df_p[f"nbr_phi_{names.index('Tumour'):02d}"].to_numpy(float)
    own_r = df_r[cols].to_numpy(float)
    lN = np.log(df_r["N"].to_numpy(float))
    chk = max(abs(spearmanr(own_r[ct == t, j], lN[ct == t]).statistic) for t in np.unique(ct) for j in range(len(cols)))
    say(f"largest within-type |Spearman(score, log N)| after residualising: {chk:.3f} (linear removal; rank can remain)")

    say(f"multistart on the residualised table ({a.restarts} restarts, {len(df_r):,} cells) ...")
    data_r = build_data(df_r, basis_r, rcfg, spec_r)
    best, res = multistart(data_r, spec_r, rcfg, n_restarts=a.restarts, seed=0, verbose=True,
                           maxiter=int(cfg["optim"]["maxiter"]))
    lls = np.array([r.loglik for r in res])
    g_p, per_p = summarise("production", pe["params"], df_p, basis_p, cfg, spec_p, ct, tum_nbr)
    g_r, per_r = summarise("depth-residualised", best.params, df_r, basis_r, rcfg, spec_r, ct, tum_nbr)
    g_p["penalised loglik"], g_r["penalised loglik"] = float(pe["loglik"]), float(best.loglik)
    glob = pd.DataFrame([g_p, g_r])
    rows = []
    for t in per_p:
        rows.append({"type": t, "cells": per_p[t]["cells"],
                     "w prod": per_p[t]["mean w"], "w resid": per_r[t]["mean w"],
                     "rho(pi,logN) prod": per_p[t]["rho(logit pi, log N)"],
                     "rho(pi,logN) resid": per_r[t]["rho(logit pi, log N)"],
                     "nbr contrast prod": per_p[t]["nbr contrast"],
                     "nbr contrast resid": per_r[t]["nbr contrast"]})
    per = pd.DataFrame(rows)
    json.dump({"global": [g_p, g_r], "per_type": rows, "restart_logliks": lls.tolist()},
              open(f"{OUT}/comparison.json", "w"), indent=1)
    with open(f"{OUT}/report.md", "w") as fh:
        fh.write(f"""# Ovarian: depth-residualised state, against production ({time.strftime('%Y-%m-%d %H:%M')})

`run/ovarian_depth_residual_check.py`. Each PhiSpace score residualised on log N within type
(slope removed, type means kept; neighbour means rebuilt; types from raw scores); production
specification refitted on all {len(df_r):,} cells, {a.restarts} restarts (best of them; restart
log-likelihoods {', '.join(f'{v:,.1f}' for v in sorted(lls, reverse=True))}). The production
column is the Spartan point estimate (20 restarts). Largest within-type |Spearman| of a
residualised score with log N: {chk:.3f}.

## Global

{md_table(glob)}

## By type

Neighbourhood contrast: mean logit pi at or above the type's 90th percentile of neighbourhood
Tumour score minus at or below its 10th (raw-score neighbourhood, same cells in both fits).

{md_table(per)}
""")
    say(open(f"{OUT}/report.md").read())
    say(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
