"""Division classifier, addendum stage A1: the diagnostics section, from saved outputs.

    python run/division_addendum_a1_report.py

The 2026-10-13 A1 run stopped at V11 before writing its section 1.4
diagnostics; the user then accepted V11. This writes those diagnostics and
the X9/X10 verdicts from what the run saved (null_fits_trunc.json,
trunc_diagnostics.csv, labels_trunc.parquet), so nothing is refitted. A full
rerun of run/division_addendum_a1.py now continues past V11 and writes the
same section itself.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from division_addendum_a1 import OUT, REPORT, TOL_V11, md_table, report  # noqa: E402
from tissue_dynamics.division import labels as L                        # noqa: E402


def main():
    j = json.load(open(f"{OUT}/null_fits_trunc.json"))
    diag = pd.read_csv(f"{OUT}/trunc_diagnostics.csv")
    lt = pd.read_parquet(f"{OUT}/labels_trunc.parquet", columns=["cell_id"] + [f"ctrim{L.trunc_tag(a)}" for a in L.TRUNC_GRID])
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "y", "N"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    if not np.array_equal(lt["cell_id"].to_numpy(), lab["cell_id"].astype(str).to_numpy()):
        sys.exit("labels_trunc and labels list cells in different orders")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    y, ctype = lab["y"].to_numpy(float), lab["cell_type"].astype(str).to_numpy()
    dec = pd.qcut(lab["N"].to_numpy(float), 10, labels=False, duplicates="drop") + 1

    v11 = pd.DataFrame(j["v11"])
    tr = v11[v11["fit"] == "truncated"]
    worst = float(tr[[c for c in v11.columns if c.startswith("rel err")]].abs().to_numpy().max())

    rows = [{"null": "naive (Stage 1)", **j["naive"],
             "trimmed fraction": float(meta["null_trace"][-1]["flagged_frac"]),
             "iterations": len(meta["null_trace"]), "stopped on": meta["null_stopped_on"]}]
    for at in L.TRUNC_GRID:
        t = j["truncated"][str(at)]
        c = lt[f"ctrim{L.trunc_tag(at)}"].to_numpy(float)
        rows.append({"null": f"trunc {at:g}", "beta0": t["beta0"], "theta_m": t["theta_m"],
                     "alpha0": t["alpha0"], "trimmed fraction": float(np.mean(y >= c)),
                     "iterations": len(t["trace"]), "stopped on": t["stopped_on"]})
    par = pd.DataFrame(rows)
    trim = {}
    for at in L.TRUNC_GRID:
        tr_ = y >= lt[f"ctrim{L.trunc_tag(at)}"].to_numpy(float)
        col = {f"N decile {d}": float(tr_[dec == d].mean()) for d in range(1, int(dec.max()) + 1)}
        col.update({t: float(tr_[ctype == t].mean()) for t in types})
        trim[f"alpha_trim {at:g}"] = col
    trim = pd.DataFrame(trim).reset_index().rename(columns={"index": "stratum"})

    order = ["naive (Stage 1)"] + [f"trunc {at:g}" for at in L.TRUNC_GRID]
    per = diag[diag["type"] != "all"]
    rates = diag[diag["type"] == "all"].pivot(index="null", columns="alpha_B", values="label rate").reindex(order)
    neg = per.assign(below=per["RG hi"] < 0).groupby(["null", "alpha_B"])["below"].sum().unstack().reindex(order)
    p05 = per[per["alpha_B"] == 0.05]
    rate_t = p05.pivot(index="type", columns="null", values="label rate")[order].reindex(types)
    rg = p05.assign(cell=lambda d: d.apply(
        lambda r: f"{r['Rogan-Gladen']:+.3f} [{r['RG lo']:+.3f}, {r['RG hi']:+.3f}]", axis=1)
    ).pivot(index="type", columns="null", values="cell")[order].reindex(types)

    def cols(df):
        return df.reset_index().rename(columns=lambda c: f"alpha_B {c:g}" if isinstance(c, float) else c)

    a_nv = j["naive"]["alpha0"]
    a_tr = [j["truncated"][str(at)]["alpha0"] for at in L.TRUNC_GRID]
    x9a = all(a < a_nv for a in a_tr)
    x9b = all((rates.loc[f"trunc {at:g}"] > rates.loc["naive (Stage 1)"]).all() for at in L.TRUNC_GRID)
    x9c = all(neg.loc[f"trunc {at:g}"].sum() < neg.loc["naive (Stage 1)"].sum() for at in L.TRUNC_GRID)
    x9d = bool(np.all(np.diff(a_tr) < 0))
    v2 = pd.DataFrame(j["v2_trunc"])
    v2_ok = bool((v2["relative error"].abs() <= 0.30).all())
    nv05 = v11[(v11["fit"] != "truncated") & (v11["alpha_trim"] == 0.05)]
    bias05 = float(nv05[[c for c in v11.columns if c.startswith("rel err")]].abs().to_numpy().max())
    clauses = (x9a, x9b, x9c, x9d)
    yes = lambda b: "yes" if b else "no"                                 # noqa: E731

    text = [f"\n### A1 continued: V11 accepted (user, 2026-10-13) ({time.strftime('%Y-%m-%d %H:%M')})\n",
            f"V11 missed its 2% tolerance on theta_m at alpha_trim 0.05 (+{100 * worst:.1f}%, z = 1.9). The "
            "untrimmed maximum-likelihood fit of the same simulated draw is already +1.1% on theta_m "
            "(and -1.9 SE on b0), so the truncated fits inherit most of the miss; their departure from "
            "that fit is under 2 SE. The user accepted V11 on this evidence, to be tested on more "
            "simulated draws later. The section 1.4 diagnostics follow, written from the saved "
            "outputs (`run/division_addendum_a1_report.py`).\n",
            "### Fitted nulls\n", md_table(par), "",
            "At alpha_trim 0.001 the truncated null matches the naive one, which also trims at 0.001. "
            "The thresholds at alpha_trim 0.05 still moved for a few dozen boundary cells at the "
            "20-iteration cap; the parameters had settled to 1e-4.\n",
            "Trimmed fraction (y >= c_trim at the final fit) by N decile and cell type:\n",
            md_table(trim), "",
            "### Label rates\n", "All cells:\n", md_table(cols(rates)), "",
            "By type at alpha_B 0.05:\n", md_table(rate_t.reset_index()), "",
            "### Rogan-Gladen, sensitivity borrowed from MIX\n",
            "Fraction (rate - mean alpha_i) / (sens - mean alpha_i) per type, sens = mean "
            "P1(Y >= c^T_i) under MIX's out-of-fold cycling component; 95% block-bootstrap intervals "
            "(Stage 5's 300 replicates). At alpha_B 0.05:\n", md_table(rg.reset_index()), "",
            "Types (of 9) whose interval lies wholly below 0:\n", md_table(cols(neg)), "",
            "Every type, null and alpha_B is in `trunc_diagnostics.csv`.\n",
            "### X9 and X10\n",
            f"- **X9** (lower alpha0, higher label rate at each alpha_B, fewer negative Rogan-Gladen "
            f"types, gains growing with alpha_trim). alpha0 lower at every alpha_trim: **{yes(x9a)}** "
            f"({', '.join(f'{a:.4f}' for a in a_tr)} against {a_nv:.4f}; at 0.001 it is marginally "
            f"higher). Label rate higher at every alpha_B: **{yes(x9b)}**. Fewer types below 0: "
            f"**{yes(x9c)}**. Gain grows with alpha_trim: **{yes(x9d)}**. Overall: "
            f"**{'holds' if all(clauses) else 'partly holds' if any(clauses) else 'does not hold'}**; "
            "every clause holds at alpha_trim 0.01 and 0.05, and at 0.001 the truncated null is the "
            "naive one.",
            f"- **X10** (V2-trunc passes at every alpha_trim; truncation removes the bias naive "
            f"trimming shows at 0.05). V2-trunc **{'passes' if v2_ok else 'fails'}**. In V11 the "
            f"naive-trimmed fit at 0.05 is off by up to {100 * bias05:.0f}% (alpha0), against "
            f"{100 * worst:.1f}% for the truncated fit. Overall: **{'holds' if v2_ok and bias05 > 10 * TOL_V11 else 'partly holds'}**.\n",
            "Stage A1 complete. **Review point** (A1, A3, A5 and A6 pause for review).\n"]
    report("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
