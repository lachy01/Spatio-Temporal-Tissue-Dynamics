"""Division classifier, addendum stage A3: the Wu baseline gate for BASE-O-state.

    python run/division_addendum_a3_wu_gate.py

Addendum section 4.1. In Wu et al. 2021 (GSE176078), restricted to the genes
shared with this Xenium panel:

1. y = the Tirosh composite (the 53 genes used here), N = panel transcripts
   without the Tirosh and blocklist genes -- Stage 1's construction, as in A6.
2. Non-cycling cells, two ways:
   primary   Seurat-called G1: S and G2M scores both below 0;
   stricter  both scores below their 25th percentile within the cell's type.
3. Per type, the composite per 1,000 panel transcripts among the selected
   cells, 1000 sum(y) / sum(N) (a ratio of sums, as A6's rate ratios).
4. Its ratio to the pooled value (all selected cells), with a patient-level
   bootstrap (patients resampled, 1,000 replicates), for all patients and ER+
   patients separately; and max/min across types with its interval.

Suggested reading (addendum): max/min above 1.5, interval excluding 1, in both
selections, supports a state-dependent null; otherwise the global null stands.
The stage then stops for Lachy's decision on A4.

Writes wu_baseline.csv and a section of report.md.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from division_addendum_e import ALIASES, WU_DIR, load_wu, md_table, report, say   # noqa: E402
from tissue_dynamics.mixture.data import load_config                 # noqa: E402

OUT = "outputs/division_classifier"
N_BOOT, SEED, N_MIN = 1000, 20261014, 10


def main():
    t0 = time.time()
    cfg = load_config()
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    wu = pd.read_parquet("data/wu2021/wu_cells.parquet")
    WX, wgenes = load_wu()
    barcodes = pd.read_csv(f"{WU_DIR}/count_matrix_barcodes.tsv", header=None)[0].astype(str).to_numpy()
    if not np.array_equal(barcodes, wu["barcode"].astype(str).to_numpy()):
        sys.exit("wu_cells.parquet is not in the matrix's cell order")
    with h5py.File(f"{cfg['data']['xenium_dir']}/cell_feature_matrix.h5", "r") as h:
        names = h["matrix/features/name"][:].astype(str)
        ftype = h["matrix/features/feature_type"][:].astype(str)
    xgenes = names[ftype == "Gene Expression"]
    wset, inv = set(wgenes), {v: k for k, v in ALIASES.items()}
    x2w = {}
    for g in xgenes:
        for cand in (g, ALIASES.get(g), inv.get(g)):
            if cand in wset:
                x2w[g] = cand
                break
    tirosh = [g for g in meta["tirosh_genes"] if g in x2w]
    block = set(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str))
    wi = {g: i for i, g in enumerate(wgenes)}
    shared = [g for g in xgenes if g in x2w]
    tech = [g for g in shared if g not in set(tirosh) and g not in block]
    y = np.asarray(WX[:, [wi[x2w[g]] for g in tirosh]].sum(1)).ravel()
    N = np.asarray(WX[:, [wi[x2w[g]] for g in tech]].sum(1)).ravel()
    say(f"Wu: {len(wu):,} cells; {len(shared):,} shared genes, Tirosh {len(tirosh)}/"
        f"{len(meta['tirosh_genes'])}, N over {len(tech):,} technical genes")
    ctype = wu["celltype_major"].astype(str).to_numpy()
    S, G2M = wu["S_score"].to_numpy(float), wu["G2M_score"].to_numpy(float)
    keep = N >= N_MIN
    sel = {"primary (Seurat G1: S < 0 and G2M < 0)": (S < 0) & (G2M < 0)}
    q25 = np.zeros(len(wu), bool)
    for t in types:
        m = ctype == t
        q25[m] = (S[m] < np.percentile(S[m], 25)) & (G2M[m] < np.percentile(G2M[m], 25))
    sel["stricter (S and G2M below the type's 25th percentile)"] = q25
    pats, pidx = np.unique(wu["patient"].astype(str).to_numpy(), return_inverse=True)
    er_pat = set(wu.loc[wu["subtype"] == "ER+", "patient"].astype(str))
    rng = np.random.default_rng(SEED)

    rows, summ = [], []
    for scope in ("all patients", "ER+ patients"):
        use_p = np.array([(scope == "all patients") or (p in er_pat) for p in pats])
        pl = np.where(use_p)[0]
        Mb = np.zeros((N_BOOT, len(pats)))
        for b in range(N_BOOT):
            Mb[b] = np.bincount(rng.choice(pl, len(pl), replace=True), minlength=len(pats))
        for sname, s in sel.items():
            base = keep & s & use_p[pidx]
            sy = {t: np.bincount(pidx[base & (ctype == t)], weights=y[base & (ctype == t)], minlength=len(pats))
                  for t in types}
            sn = {t: np.bincount(pidx[base & (ctype == t)], weights=N[base & (ctype == t)], minlength=len(pats))
                  for t in types}
            ty, tn = sum(sy.values()), sum(sn.values())
            pooled, pooled_b = 1000 * ty.sum() / tn.sum(), 1000 * (Mb @ ty) / (Mb @ tn)
            per, per_b = {}, {}
            for t in types:
                with np.errstate(invalid="ignore", divide="ignore"):
                    per[t] = 1000 * sy[t].sum() / sn[t].sum()
                    per_b[t] = 1000 * (Mb @ sy[t]) / (Mb @ sn[t])
                r_b = per_b[t] / pooled_b
                lo, hi = np.nanpercentile(r_b, [2.5, 97.5])
                cells = int((base & (ctype == t)).sum())
                rows.append({"scope": scope, "selection": sname, "type": t, "cells": cells,
                             "patients": int(np.sum(sn[t] > 0)),
                             "composite per 1,000": per[t], "ratio to pooled": per[t] / pooled,
                             "ratio 95% lo": lo, "ratio 95% hi": hi,
                             "excludes 1": bool(lo > 1 or hi < 1)})
            vals = np.array([per[t] for t in types])
            B = np.column_stack([per_b[t] for t in types])
            mm_b = np.nanmax(B, 1) / np.nanmin(B, 1)
            lo, hi = np.nanpercentile(mm_b, [2.5, 97.5])
            summ.append({"scope": scope, "selection": sname, "pooled per 1,000": pooled,
                         "max/min": float(vals.max() / vals.min()),
                         "max/min 95% CI": f"{lo:.2f}-{hi:.2f}",
                         "highest": types[int(vals.argmax())], "lowest": types[int(vals.argmin())],
                         "types excluding 1": int(sum(r["excludes 1"] for r in rows[-len(types):]))})
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/wu_baseline.csv", index=False)
    summ = pd.DataFrame(summ)
    say(summ.to_string(index=False))

    def lohi(r):
        return f"{r['ratio to pooled']:.2f} [{r['ratio 95% lo']:.2f}, {r['ratio 95% hi']:.2f}]"
    wide = {}
    for (scope, sname), g in tab.groupby(["scope", "selection"], sort=False):
        wide[f"{scope.split()[0]}, {sname.split()[0]}"] = g.set_index("type").apply(lohi, axis=1)
    wide = pd.DataFrame(wide).reindex(types).reset_index().rename(columns={"index": "type"})
    cells = tab.pivot_table(index="type", columns=["scope", "selection"], values="cells").reindex(types)
    cells.columns = [f"{a.split()[0]}, {b.split()[0]}" for a, b in cells.columns]
    material = {}
    for scope in ("all patients", "ER+ patients"):
        s = summ[summ["scope"] == scope]
        material[scope] = bool(all(
            (r["max/min"] > 1.5) and float(r["max/min 95% CI"].split("-")[0]) > 1.0 for _, r in s.iterrows()))
    report(f"""
## Addendum stage A3 -- Wu baseline gate for BASE-O-state ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_addendum_a3_wu_gate.py`, addendum section 4.1. Wu et al. 2021,
{len(wu):,} cells and {len(pats)} patients, restricted to the {len(shared):,}
genes shared with this panel. y is the Tirosh composite ({len(tirosh)} of
{len(meta['tirosh_genes'])} genes; the rest are not in Wu), N the panel transcripts
without the Tirosh and blocklist genes ({len(tech):,} genes), as Stage 1 builds
them; cells with N >= {N_MIN}. In non-cycling cells, selected two ways, the
composite per 1,000 panel transcripts per type is 1000 sum(y) / sum(N), and its
ratio to the pooled value is shown with a {N_BOOT:,}-replicate patient bootstrap
(patients resampled; ER+ patients resampled among themselves).

Selections: **primary**, Seurat-called G1 (S and G2M scores both below 0);
**stricter**, both scores below their 25th percentile within the cell's type.

Ratio to pooled [95% patient-bootstrap interval]:

{md_table(wide)}

Cells selected:

{md_table(cells.reset_index(), '{:.0f}')}

Summary (max/min across the nine types, with its bootstrap interval):

{md_table(summ, '{:.3g}')}

Against the suggested reading (max/min above 1.5 with an interval excluding 1,
in both selections): all patients **{'met' if material['all patients'] else 'not met'}**;
ER+ patients **{'met' if material['ER+ patients'] else 'not met'}**.
{"These baseline differences support a state-dependent null. Note: MIX's rates also omit state, so this is a specification question for MIX as well, out of scope here." if any(material.values()) else "On this reading the global null stands."}

Every type, selection and scope is in `wu_baseline.csv`. Stage A3 complete in
{(time.time() - t0) / 60:.0f} min. **Stopped for Lachy's decision** on whether
A4 (BASE-O-state) runs.
""")


if __name__ == "__main__":
    main()
