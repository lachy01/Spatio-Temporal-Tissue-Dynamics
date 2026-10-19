"""BASE-T and BASE-T-trunc calibrated on thinned Wu: every cell scored, and the
technical null with and without cell type.

    python run/division_wu_calibration_v2.py [--seeds 3]

Second Wu calibration (user, 2026-10-19). Two changes from the first
(run/division_wu_calibration.py, whose results stay in report.md):

1. **No confident-call restriction.** The truth is Seurat's own call, S/G2M
   where max(S, G2M score) > 0, on EVERY cell. The first run scored only cells
   with a score above 0.1 or at most 0 and at least 3,000 UMIs, which favours
   cells with many cycle-gene counts and so pushes the chosen threshold strict.
   The first run's truth is kept as a sensitivity, with two others.
2. **Cell type in the null.** Cell types differ in their basal expression of the
   cycle genes, so the null may carry a per-type intercept:
       none   log mu0 = b0 + log N             (+ theta_m m on the section)
       type   log mu0 = b0 + b_type + log N    (+ theta_m m on the section)
   Wu has cell-type labels but no PhiSpace state, so "state" is the type; this
   section uses its own types (standardised PhiSpace argmax), so the two
   designs match. A per-type intercept also absorbs between-type differences
   in CYCLING; truncation (alpha_trim) limits that by fitting the null to each
   type's lower body. The `type-rate` columns check what survives: Spearman, over
   types, of the label's per-type rate with the truth's.

Truncated nulls never trim a count of 1 (`c_floor` 2): with a per-type intercept,
iterated truncation otherwise runs away on shallow types at heavy alpha_trim (every
non-zero count trimmed, the type's intercept falls without limit). The floor is
applied to every null here, both designs and both datasets, so they compare like
for like; the section's depth-only nulls are refitted with it rather than reused.

Everything else is the first run's: Wu restricted to the shared panel genes and
thinned within type to this section's depth (3 seeds), the plain log N null on
Wu (no MisTIC term there), every null refitted per seed and scope (ER+ primary,
all patients), labels at every alpha_B on the grid, cells weighted to this
section's type mix. Settings transfer as (design, null, alpha_B); the label
rate they give on this section is reported under the production MisTIC null
with the same design.

Selection (fixed before the results are read): for each design, the (null,
alpha_B) with the highest mean weighted MCC against the primary truth (ER+).
Reported beside it: Youden's J pick, MCC at this section's own Rogan-Gladen
fraction, and the type-rate agreement. Writes
outputs/division_classifier/wu_calibration_v2/ and a section of report.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from division_addendum_e import thin                                  # noqa: E402
from division_trunc_calibration import AB_GRID, NULLS, PARAM_TOL, TRIMS  # noqa: E402
from division_wu_calibration import (PANEL_CACHE, md_table, mcc_at,   # noqa: E402
                                     rogan_gladen, weighted_curve, y_and_N)
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge                 # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/division_classifier"
WOUT = f"{OUT}/wu_calibration_v2"
V1 = f"{OUT}/wu_calibration"
N_MIN = 10
SEED = 20261017                         # the first run's thinning seeds
DESIGNS = ("none", "type")
FIT_MAXITER = 400                       # inner L-BFGS cap per truncated refit (see labels.truncated_null)
C_FLOOR = 2                             # a count of 1 is never trimmed (labels.truncated_null, c_floor)

say = lambda *a: print(*a, flush=True)                                # noqa: E731


def type_design(types_cell, levels):
    """Intercept plus a dummy per type except the first (the reference)."""
    X = [np.ones(len(types_cell))] + [(types_cell == t).astype(float) for t in levels[1:]]
    return np.column_stack(X)


def nulls_for(y, X, off, m, tag="", cache=None):
    """p-values of every null (naive robust, truncated at each alpha_trim) and alpha0.
    `cache` (a path): load if present, else compute and store, so a rerun resumes."""
    import pickle
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = {}
    t1 = time.time()
    nv = L.robust_null(y, X, off, m=m)
    out["naive"] = (nb_p_ge(y, nv["mu"], nv["alpha"]), float(nv["alpha"]))
    full = L.fit_nb_additive_null(y, X, off, m)
    for t in TRIMS:
        ft = L.truncated_null(y, X, off, m, t, full, param_tol=PARAM_TOL, fit_maxiter=FIT_MAXITER,
                              c_floor=C_FLOOR)
        out[f"trunc {t:g}"] = (nb_p_ge(y, ft["mu"], ft["alpha"]), float(ft["alpha"]))
        say(f"    {tag} trunc {t:g}: alpha0 {ft['alpha']:.4g}, {len(ft['trace'])} it, "
            f"k_max {ft.get('k_max', '?')} ({time.time() - t1:.0f}s)")
    if cache:
        pickle.dump(out, open(cache, "wb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(f"{WOUT}/cache", exist_ok=True)
    cfg = load_config()
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_id", "cell_type", "y", "N"])
    meta = json.load(open(f"{OUT}/labels_meta.json"))
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    wu = pd.read_parquet("data/wu2021/wu_cells.parquet")
    block_x = list(dict.fromkeys(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str)))
    for f in (PANEL_CACHE, f"{V1}/section_panel.npz", f"{OUT}/trunc_calibration/real_nulls.npz"):
        if not os.path.exists(f):
            sys.exit(f"{f} missing: run run/division_wu_calibration.py (and the G4 calibration) first")
    shared = json.load(open(PANEL_CACHE.replace(".npz", ".genes.json")))
    WP = sp.load_npz(PANEL_CACHE).tocsr()
    tir_sh = [g for g in meta["tirosh_genes"] if g in set(shared)]
    xdepth = np.load(f"{V1}/section_panel.npz", allow_pickle=True)["depth_by_gene_set"].item()["shared_depth"]
    wtype = wu["celltype_major"].astype(str).to_numpy()
    xtype = lab["cell_type"].astype(str).to_numpy()
    wdepth = np.asarray(WP.sum(1)).ravel()
    target = np.zeros(len(wu))
    for t in types:
        wc, xc = wtype == t, xtype == t
        q = (pd.Series(wdepth[wc]).rank(method="average").to_numpy() - 0.5) / wc.sum()
        target[wc] = np.quantile(xdepth[xc], q)
    p_thin = np.minimum(1.0, target / np.maximum(wdepth, 1e-12))
    sec_frac = pd.Series(xtype).value_counts(normalize=True)

    # ---- truths: every cell (primary), and three sensitivities -----------------------------
    score = np.maximum(wu["S_score"].to_numpy(float), wu["G2M_score"].to_numpy(float))
    umi = wu["nCount_RNA"].to_numpy(float)
    deep = umi >= 3000
    seurat = (score > 0).astype(int)
    truths = {"Seurat call, every cell": seurat,
              "Seurat call, >= 3k UMI": np.where(deep, seurat, -1),
              "score > 0.05, every cell": (score > 0.05).astype(int),
              "confident (first run)": np.where(deep, np.where(score > 0.1, 1, np.where(score <= 0, 0, -1)), -1)}
    PRIMARY = "Seurat call, every cell"

    # ---- this section: production MisTIC null, with and without type ------------------------
    sec_path = f"{WOUT}/section_nulls.npz"
    if os.path.exists(sec_path):
        sec = np.load(sec_path, allow_pickle=True)["p"].item()
    else:
        df = pd.read_parquet(cfg["data"]["table_parquet"])
        df.attrs["phi_names"] = types
        basis = build_basis(df, cfg, verbose=False)
        X1, off, m = L.null_design(df, basis)
        y = df["y_count"].to_numpy(float)
        if not np.array_equal(df["cell_id"].astype(str).to_numpy(), lab["cell_id"].astype(str).to_numpy()):
            sys.exit("labels and table list cells in different orders")
        say("this section, MisTIC null without and with cell type ...")
        sec = {"none": nulls_for(y, X1, off, m, tag="section none", cache=f"{WOUT}/cache/section_none.pkl"),
               "type": nulls_for(y, np.column_stack([X1, type_design(xtype, types)[:, 1:]]), off, m,
                                 tag="section type", cache=f"{WOUT}/cache/section_type.pkl")}
        np.savez(sec_path, p=np.array(sec, dtype=object))
    say(f"section nulls ready ({time.time() - t0:.0f}s)")

    def section_rate(design, null, ab):
        return float(np.mean(sec[design][null][0] < ab))

    # ---- thinned Wu ----------------------------------------------------------------------------
    y_full, N_full = y_and_N(WP, shared, tir_sh, block_x)
    scopes = {"ER+": (wu["subtype"] == "ER+").to_numpy(), "all patients": np.ones(len(wu), bool)}
    rows, fid, trows = [], [], []
    for sd in range(a.seeds):
        Wt = thin(WP, p_thin, np.random.default_rng(SEED + sd))
        yw, Nw = y_and_N(Wt, shared, tir_sh, block_x)
        for scope, sc in scopes.items():
            cells = sc & (Nw >= N_MIN)
            ct = wtype[cells]
            levels = [t for t in types if (ct == t).any()]
            for design in DESIGNS:
                X = np.ones((cells.sum(), 1)) if design == "none" else type_design(ct, levels)
                nl = nulls_for(yw[cells], X, np.log(Nw[cells]), np.zeros(cells.sum()),
                               tag=f"seed {sd} {scope} {design}",
                               cache=f"{WOUT}/cache/nulls_s{sd}_{scope.replace(' ', '_').replace('+', 'p')}_{design}.pkl")
                for k, (p, al) in nl.items():
                    fid.append({"seed": sd, "scope": scope, "design": design, "null": k, "alpha0": al})
                    for tname, tr_all in truths.items():
                        tr = tr_all[cells]
                        s_ = tr >= 0
                        wfrac = pd.Series(ct[s_]).value_counts(normalize=True)
                        w = np.array([sec_frac.get(t, 0) / wfrac.get(t, np.inf) for t in ct])
                        cur = weighted_curve(p, tr, w, AB_GRID)
                        # does the label keep the between-type differences in cycling?
                        tr_rate = pd.Series(tr[s_] == 1).groupby(ct[s_]).mean()
                        rho = []
                        for ab in AB_GRID:
                            lr = pd.Series(p[s_] < ab).groupby(ct[s_]).mean()
                            rho.append(float(lr.corr(tr_rate, method="spearman")))
                        for i, ab in enumerate(AB_GRID):
                            rows.append({"seed": sd, "scope": scope, "design": design, "truth": tname,
                                         "null": k, "alpha_B": float(ab),
                                         **{m_: float(v[i]) for m_, v in cur.items()},
                                         "type-rate Spearman": rho[i],
                                         "truth rate": float((tr[s_] == 1).mean())})
                say(f"  seed {sd} {scope} {design}: {cells.sum():,} cells ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(f"{WOUT}/grid.csv", index=False)
    grid = pd.DataFrame(rows)
    fid = pd.DataFrame(fid)
    fid.to_csv(f"{WOUT}/fidelity.csv", index=False)

    # ---- selection -----------------------------------------------------------------------------
    def pick(design, truth=PRIMARY, scope="ER+"):
        g = grid[(grid["design"] == design) & (grid["truth"] == truth) & (grid["scope"] == scope)]
        gm = g.groupby(["null", "alpha_B"], sort=False).agg(
            MCC=("MCC", "mean"), sens=("sensitivity", "mean"), fpr=("fpr", "mean"),
            prec=("precision", "mean"), rho=("type-rate Spearman", "mean"),
            truth_rate=("truth rate", "mean")).reset_index()
        gm["section rate"] = [section_rate(design, k, ab) for k, ab in zip(gm["null"], gm["alpha_B"])]
        gm["J"] = gm["sens"] - gm["fpr"]
        gm["RG"] = rogan_gladen(gm["section rate"], gm["sens"], gm["fpr"])
        b = gm.loc[gm["MCC"].idxmax()]
        yj = gm.loc[gm["J"].idxmax()]
        f_sec = float(yj["RG"])
        gm["MCC at section fraction"] = [float(mcc_at(s_, f_, f_sec)) for s_, f_ in zip(gm["sens"], gm["fpr"])]
        pm = gm.loc[gm["MCC at section fraction"].idxmax()]
        row = lambda r, crit: {"design": design, "truth": truth, "scope": scope, "criterion": crit,   # noqa: E731
                               "null": r["null"], "alpha_B": float(r["alpha_B"]), "MCC (Wu)": float(r["MCC"]),
                               "sensitivity": float(r["sens"]), "precision (Wu)": float(r["prec"]),
                               "type-rate Spearman": float(r["rho"]), "section rate": float(r["section rate"]),
                               "RG section fraction": float(r["RG"]), "truth rate (Wu)": float(r["truth_rate"])}
        return [row(b, "MCC (rule)"), row(yj, "Youden J"), row(pm, "MCC at section fraction")], gm

    out, gms = [], {}
    for design in DESIGNS:
        rs, gms[design] = pick(design)
        out += rs
        for truth in list(truths)[1:]:
            out += pick(design, truth)[0][:1]
        out += pick(design, scope="all patients")[0][:1]
    res = pd.DataFrame(out)
    res.to_csv(f"{WOUT}/choices.csv", index=False)
    prim = res[(res["truth"] == PRIMARY) & (res["scope"] == "ER+")]
    best = prim[prim["criterion"] == "MCC (rule)"].set_index("design")
    say(prim.round(4).to_string(index=False))

    # ---- figure: weighted MCC on Wu against the rate on this section, per design --------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#07244a"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True, facecolor=SURF)
    for ax, design in zip(axes, DESIGNS):
        gm = gms[design]
        ax.set_facecolor(SURF)
        for k in NULLS:
            s = gm[gm["null"] == k]
            col = "#898781" if k == "naive" else SEQ[TRIMS.index(float(k.split()[1]))]
            ax.plot(s["section rate"], s["MCC"], color=col, lw=2.2 if k == best.loc[design, "null"] else 1.3,
                    ls="--" if k == "naive" else "-", label=k)
        b = best.loc[design]
        ax.plot([b["section rate"]], [b["MCC (Wu)"]], "o", ms=8, mfc=SEQ[5], mec=SURF, mew=2)
        ax.axvline(0.10, color=INK2, lw=1, ls=":")
        ax.set_xlim(0, 0.4)
        ax.set_xlabel("label rate those settings give on THIS section")
        ax.grid(color=GRIDC, lw=0.8)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(f"null: {'depth only' if design == 'none' else 'depth + cell type'}"
                     f"  (pick {b['null']}, alpha_B {b['alpha_B']:.3g}, {b['section rate']:.1%})",
                     loc="left", fontsize=10)
    axes[0].set_ylabel("weighted MCC with Seurat's call, thinned Wu (ER+)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.suptitle("BASE-T on thinned Wu, every cell scored", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{WOUT}/mcc_vs_rate.png", dpi=170, facecolor=SURF)

    # ---- report ---------------------------------------------------------------------------------
    fm = fid.groupby(["scope", "design", "null"], sort=False)["alpha0"].mean().unstack(["scope", "design"])
    fm.columns = [f"Wu {s}, {d}" for s, d in fm.columns]
    fm.insert(0, "section, none", [sec["none"][k][1] for k in fm.index])
    fm.insert(1, "section, type", [sec["type"][k][1] for k in fm.index])
    tr_rates = pd.DataFrame({t: {tn: float(np.mean(v[(wtype == t) & scopes["ER+"] & (v >= 0)] == 1))
                                 for tn, v in truths.items()} for t in types}).T
    tr_tab = tr_rates.reset_index().rename(columns={"index": "type"})
    show = ["design", "criterion", "null", "alpha_B", "MCC (Wu)", "sensitivity", "precision (Wu)",
            "type-rate Spearman", "section rate", "RG section fraction"]
    sens = res[(res["criterion"] == "MCC (rule)") & ~((res["truth"] == PRIMARY) & (res["scope"] == "ER+"))]
    text = f"""
## BASE-T calibrated on Wu again: every cell scored, cell type in the null ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_wu_calibration_v2.py`. Requested by the user (2026-10-19): score every cell rather than
confident calls (which favour cells with many cycle-gene counts), and test a technical null that
carries cell type, since types differ in basal cycle-gene expression. Primary truth: Seurat's own
call (max(S, G2M) > 0) on every ER+ Wu cell ({best.iloc[0]['truth rate (Wu)']:.1%} of them called S/G2M;
cells are weighted to this section's type mix in every score). Null designs: `none` = depth only (as before); `type` = depth + a per-type
intercept (Wu: celltype_major; this section: its types). Thinning, seeds, grids and weights as in the
first Wu run; {a.seeds} seeds.

### Fitted nulls: alpha0 by design

Mean over seeds. The section's nulls are the production MisTIC null (theta_m m included); Wu's are
plain-depth (scRNA-seq has no misassignment term).

{md_table(fm.reset_index())}

### Picks, primary truth (ER+, Seurat call on every cell)

Section rate: the label rate those settings give on this section with the same design. RG: Rogan-Gladen
fraction of this section, from Wu's sensitivity and FPR. Type-rate Spearman: over the nine types, the
label's per-type rate against the truth's on Wu (1 = the between-type ordering is kept).

{md_table(prim[show], fmt="{:.4g}")}

### How the rule's pick moves with the truth and the scope

{md_table(sens[["design", "truth", "scope", "null", "alpha_B", "MCC (Wu)", "section rate", "type-rate Spearman"]], fmt="{:.4g}")}

### Truth rates by type on ER+ Wu

{md_table(tr_tab, fmt="{:.3f}")}

![MCC against section rate](wu_calibration_v2/mcc_vs_rate.png)

Done in {(time.time() - t0) / 60:.0f} min. Not propagated to Stages 3-7; the user decides.
"""
    with open(f"{OUT}/report.md", "a") as fh:
        fh.write(text)
    say(text)


if __name__ == "__main__":
    main()
