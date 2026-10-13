"""Division classifier, addendum stage A4: BASE-O-state and LOG-O-state.

    python run/division_addendum_a4_state.py

RUN ONLY IF LACHY APPROVES AFTER THE A3 GATE (addendum section 4.1). Written
ahead of that decision so it is ready; it refuses to run unless --approved is
passed.

Addendum section 4.2. OSDR's per-type s+ made continuous in own state:
on the cells with r > 0 (r = (y - mu0) / sd0, Stage 1's residual), a gamma GLM
with log link for the mean, m1(phi), on the 8-dim projected own state with an
intercept, then a second gamma GLM for (r - m1(phi))^2, v(phi); s+(phi) =
sqrt(v(phi)) (`labels.base_o_state_scale`). a = r / s+(phi); the label
1{a > Td} on the Td grid, with its discrete equivalent c (V10). LOG-O-state is
LOG-O's specification fitted to the primary label (Td 0.5), out of fold.

V14 (blocking): with one-hot type indicators in place of phi the procedure
reproduces primary BASE-O's per-type s+ to within 1e-6.

BASE-O-state is judged in the simulation (section 4.3: run the A5 job with
`division_sim_state: true`), not on label-scale scores; here only the labels,
their rates and LOG-O-state's out-of-fold predictions are produced.

Writes labels_state.parquet and a section of report.md.
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

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data                  # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_FOLDS, TOL_V14 = 5, 1e-6


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved", action="store_true",
                    help="Lachy approved A4 after the A3 gate")
    a = ap.parse_args()
    if not a.approved:
        sys.exit("A4 runs only with Lachy's approval after the A3 gate; pass --approved once given.")
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    n = len(df)
    y, mu0, sd0, r = (lab[k].to_numpy(float) for k in ("y", "mu0", "sd0", "r"))
    ctype = lab["cell_type"].astype(str).to_numpy()
    phi = np.asarray(data.own, float)

    # ---- V14 -------------------------------------------------------------------------
    onehot = (ctype[:, None] == np.array(types)[None, :]).astype(float)
    s_oh, _ = L.base_o_state_scale(r, onehot)
    s_type = L.base_o_scale(r, ctype)
    v14 = float(np.max(np.abs(s_oh - s_type)))
    v14_meta = max(abs(float(s_oh[ctype == t][0]) - meta["s_plus_type"][t]) for t in types)
    v14_ok = v14 < TOL_V14 and v14_meta < TOL_V14
    say(f"V14: one-hot s+ against per-type BASE-O {v14:.2e} (against Stage 1's stored values {v14_meta:.2e})")
    if not v14_ok:
        report(f"\n### STOPPED in addendum stage A4: V14 failed\n\nmax |s+ difference| {v14:.3e} "
               f"(stored {v14_meta:.3e}), tolerance {TOL_V14:g}.")
        sys.exit(1)

    # ---- BASE-O-state ------------------------------------------------------------------
    X = np.column_stack([np.ones(n), phi])
    s_state, info = L.base_o_state_scale(r, X)
    out = pd.DataFrame({"cell_id": ids, "s_plus_state": s_state, "a_state": r / s_state})
    v10_bad = []
    for td in L.TD_GRID:
        tg = L.td_tag(td)
        Lo, c, aa = L.base_o(y, mu0, sd0, s_state, td)
        out[f"LOs_{tg}"], out[f"cOs_{tg}"] = Lo, c
        if np.any(Lo != (aa > td)) or np.any(Lo != (y >= c)):
            v10_bad.append(f"Td {td}")
    if v10_bad:
        report("\n### STOPPED in addendum stage A4: V10 (discrete equivalence) failed at " + ", ".join(v10_bad))
        sys.exit(1)

    # ---- LOG-O-state, out of fold ---------------------------------------------------------
    folds = fm["fold"].to_numpy()
    prim = f"LOs_{L.td_tag(L.TD_PRIMARY)}"
    yl_all = out[prim].to_numpy(float)
    pred, rows = np.zeros(n), []
    for f in range(N_FOLDS):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        Dtr = data.subset(tr).as_jax()
        yl = jnp.asarray(yl_all[tr])
        fobj = arms.logistic_objective(Dtr, yl, len(tr), spec.weight_ridge, spec.n_reference,
                                       spec.ridge_tau, spec.b_penalty)
        for attempt in range(3):
            p, inf = arms.minimise(fobj, arms.init_logistic(Dtr, yl, spec.rank, 100 * attempt), say=say)
            if inf["converged"]:
                break
        dg = arms.eta_diagnostics(p, Dtr)
        rows.append({"fold": f, "converged": "pass" if inf["converged"] else "FAIL",
                     "grad_inf": inf["grad_inf"], "sd(eta)": dg["sd_eta"], "minutes": inf["wall_time"] / 60})
        pred[te] = np.asarray(jax.nn.sigmoid(arms.logistic_eta(p, data.subset(te).as_jax())))
    out["p_logos_" + prim] = pred
    out.to_parquet(f"{OUT}/labels_state.parquet", index=False)
    fits = pd.DataFrame(rows)

    per = []
    lo_prim = f"LO_{L.td_tag(L.TD_PRIMARY)}"
    for t in types:
        m = ctype == t
        per.append({"type": t, "s+ (BASE-O, per type)": float(s_type[m][0]),
                    "s+(phi) median": float(np.median(s_state[m])),
                    "s+(phi) 10th-90th": f"{np.percentile(s_state[m], 10):.3f}-{np.percentile(s_state[m], 90):.3f}",
                    "BASE-O rate (Td 0.5)": float(lab[lo_prim].to_numpy()[m].mean()),
                    "BASE-O-state rate (Td 0.5)": float(yl_all[m].mean()),
                    "LOG-O-state mean p": float(pred[m].mean())})
    per = pd.DataFrame(per)
    report(f"""
## Addendum stage A4 -- BASE-O-state ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_addendum_a4_state.py`, run with Lachy's approval after the A3 gate.
s+(phi) = sqrt(v(phi)), with m1 and v gamma GLMs (log link) on an intercept and the
8-dim projected own state, fitted on the {info['n_pos']:,} cells with r > 0
(`labels.base_o_state_scale`).

V14: with one-hot type indicators the procedure reproduces primary BASE-O's per-type
s+ to {v14:.1e} (Stage 1's stored values to {v14_meta:.1e}): **pass**. V10 (1{{a > Td}} ==
1{{y >= c}} at every Td): **pass**.

{md_table(per)}

LOG-O-state, out of fold on {prim}: all folds converged
**{'yes' if (fits['converged'] == 'pass').all() else 'NO'}**; largest sd(eta)
{fits['sd(eta)'].max():.3f} (V7 criterion 20).

{md_table(fits)}

Judged in the simulation (section 4.3): set `division_sim_state: true` and run the
A5 job. Labels and predictions in `labels_state.parquet`. Stage A4 done in
{(time.time() - t0) / 60:.0f} min.
""")


if __name__ == "__main__":
    main()
