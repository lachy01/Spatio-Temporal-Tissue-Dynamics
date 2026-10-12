"""MisTIC T3 (blocking): does the new technical null hold on genes with no
division signal?

    python run/mistic_t3_negative_control.py

Twenty random gene sets, each as large as the Tirosh set on the panel, drawn
from the genes outside the blocklist B and matched to the Tirosh genes gene by
gene on mean count and detection rate over the analysis cells (each Tirosh
gene is replaced by one of its 10 nearest non-B genes on log mean and logit
detection, never twice in a set). Each set's summed count goes through BASE-T
and BASE-O exactly as the division labels do -- the robustly trimmed null
mu = Nt e^b0 + theta m^G, with m^G the set's own expected misassigned count --
and the call rates are reported overall, by cell-area decile and by decile of
the 80 um neighbourhood's Cancer Epithelial score. Area is a diagnostic axis
only; it is not in the model.

Pass (spec), on the rates pooled over the 20 sets:
  * overall call rate within [alpha / 2, 2 alpha];
  * within each binning, the largest decile rate over the smallest below 2.
alpha is the nominal level: alpha_B = 0.05 for BASE-T. BASE-O has no nominal
level -- its rule (Td = 0.5 on the per-type-scaled residual) calls several
percent of cells even under a perfect null -- so its alpha is the call rate
the same rule gives on counts drawn from each set's fitted null. BASE-T's
attainable level (the mean realised alpha_i, below alpha_B by the discreteness
of counts) is reported beside it.

Writes outputs/mistic/t3.md, t3_sets.csv, t3_deciles.csv and t3_deciles.png;
exits non-zero on a failure.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics import mistic as M                               # noqa: E402
from tissue_dynamics.additive_glm import sim_nb                       # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402

N_SETS, K_NEAR, SEED = 20, 10, 20261012
ALPHA_B, TD = L.ALPHA_B_PRIMARY, L.TD_PRIMARY


def matched_sets(counts, cix, isB, tirosh, rng):
    """N_SETS gene sets, each matched gene by gene to the Tirosh genes."""
    X = counts.X[cix]
    mean = np.asarray(X.mean(0)).ravel()
    det = np.asarray((X > 0).mean(0)).ravel()
    f = np.column_stack([np.log(mean + 1e-6),
                         np.log(np.clip(det, 1e-6, 1 - 1e-6) / (1 - np.clip(det, 1e-6, 1 - 1e-6)))])
    f = (f - f.mean(0)) / f.std(0)
    cand = np.flatnonzero(~isB)
    tix = counts.gene_index(tirosh)
    near = {g: cand[np.argsort(((f[cand] - f[g]) ** 2).sum(1))[:K_NEAR]] for g in tix}
    sets = []
    for _ in range(N_SETS):
        used, s = set(), []
        for g in rng.permutation(tix):
            opts = [c for c in near[g] if c not in used]
            pick = int(rng.choice(opts if opts else [c for c in cand if c not in used]))
            used.add(pick)
            s.append(pick)
        sets.append(np.array(sorted(s)))
    gap = {"mean (Tirosh)": float(mean[tix].mean()), "detection (Tirosh)": float(det[tix].mean()),
           "mean (sets)": float(np.mean([mean[s].mean() for s in sets])),
           "detection (sets)": float(np.mean([det[s].mean() for s in sets]))}
    return sets, gap


def labels_for(y, X, off, m, ctype, rng):
    """BASE-T and BASE-O on one count, with the null's own reference rates."""
    null = L.robust_null(y, X, off, m=m)
    mu0, a0 = null["mu"], null["alpha"]
    sd0 = np.sqrt(mu0 + a0 * mu0 ** 2)
    LT, _c, ai, _p = L.base_t(y, mu0, a0, ALPHA_B)
    s_plus = L.base_o_scale((y - mu0) / sd0, ctype)
    LO, _cO, _a = L.base_o(y, mu0, sd0, s_plus, TD)
    ys = sim_nb(mu0, a0, rng)                     # BASE-O's rule on null counts
    LO_null, _c2, _a2 = L.base_o(ys, mu0, sd0, L.base_o_scale((ys - mu0) / sd0, ctype), TD)
    return LT, LO, float(ai.mean()), float(LO_null.mean()), null


def main():
    t0 = time.time()
    cfg, df, out = M.test_setup()
    basis = build_basis(df, cfg, verbose=False)
    X, off, _m = L.null_design(df, basis)
    cov = M.load_covariates(cfg)
    ids = df["cell_id"].astype(str).to_numpy()
    cix = pd.Index(cov.counts.cell_ids).get_indexer(ids)
    if (cix < 0).any():
        sys.exit("table cells missing from cell_mistic.parquet")
    isB = M.blocklist(cfg, cov.counts.genes)
    tirosh = M.tirosh_on_panel(cov.counts.genes)
    names = df.attrs["phi_names"]
    ctype = L.cell_type(df, names)
    area = df["area"].to_numpy(float)
    tum = df[f"nbr_phi_{names.index('Cancer Epithelial'):02d}"].to_numpy(float)
    bins = {"cell area": pd.qcut(area, 10, labels=False),
            "neighbourhood Cancer Epithelial score": pd.qcut(tum, 10, labels=False)}

    rng = np.random.default_rng(SEED)
    sets, gap = matched_sets(cov.counts, cix, isB, tirosh, rng)
    print(f"{N_SETS} matched sets of {len(tirosh)} genes: " + ", ".join(f"{k} {v:.3f}"
                                                                     for k, v in gap.items()),
          flush=True)
    rows, dec = [], []
    for k, s in enumerate(sets):
        genes = list(cov.counts.genes[s])
        y = cov.counts.gene_sum(genes)[cix]
        m_set = cov.m_gene_set(genes)[0][cix]
        LT, LO, abar, o_null, null = labels_for(y, X, off, m_set, ctype, rng)
        rows.append({"set": k, "mean count": float(y.mean()), "mean m": float(m_set.mean()),
                     "theta_m": null["theta_m"], "alpha0": null["alpha"],
                     "BASE-T rate": float(LT.mean()), "BASE-T attainable level (mean alpha_i)": abar,
                     "BASE-O rate": float(LO.mean()), "BASE-O null rate": o_null})
        for bn, b in bins.items():
            for d in range(10):
                mm = b == d
                dec.append({"set": k, "binning": bn, "decile": d + 1, "n": int(mm.sum()),
                            "BASE-T calls": int(LT[mm].sum()), "BASE-O calls": int(LO[mm].sum())})
        print(f"  set {k:2d}: mean y {y.mean():.3f}, theta_m {null['theta_m']:.3f}, BASE-T "
              f"{LT.mean():.4f} (attainable {abar:.4f}), BASE-O {LO.mean():.4f} "
              f"(null {o_null:.4f})", flush=True)
    S = pd.DataFrame(rows)
    D = pd.DataFrame(dec).groupby(["binning", "decile"], as_index=False).sum(numeric_only=True)
    D["BASE-T rate"] = D["BASE-T calls"] / D["n"]
    D["BASE-O rate"] = D["BASE-O calls"] / D["n"]
    D = D.drop(columns=["set"])
    S.to_csv(f"{out}/t3_sets.csv", index=False)
    D.to_csv(f"{out}/t3_deciles.csv", index=False)

    ref = {"BASE-T": ALPHA_B, "BASE-O": float(S["BASE-O null rate"].mean())}
    checks = {}
    for lab in ("BASE-T", "BASE-O"):
        r = float(S[f"{lab} rate"].mean())
        checks[f"{lab}: overall rate {r:.4f} within [{ref[lab] / 2:.4f}, {2 * ref[lab]:.4f}]"] = \
            bool(ref[lab] / 2 <= r <= 2 * ref[lab])
        for bn in bins:
            v = D.loc[D["binning"] == bn, f"{lab} rate"].to_numpy()
            ratio = float(v.max() / v.min()) if v.min() > 0 else np.inf
            checks[f"{lab}: max/min decile rate by {bn} = {ratio:.2f} < 2"] = bool(ratio < 2)
    ok = all(checks.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, bn in zip(axes, bins):
        sub = D[D["binning"] == bn]
        for lab, col in (("BASE-T", "#1f4e79"), ("BASE-O", "#c0504d")):
            ax.plot(sub["decile"], sub[f"{lab} rate"], "o-", color=col, label=lab)
            ax.axhspan(ref[lab] / 2, 2 * ref[lab], color=col, alpha=0.08)
            ax.axhline(ref[lab], color=col, lw=0.8, ls=":")
        ax.set_xlabel(f"decile of {bn}")
        ax.set_ylabel("call rate, pooled over the 20 sets")
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8)
        ax.set_title(f"Negative-control call rates by {bn}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/t3_deciles.png", dpi=150)
    plt.close(fig)

    def md(frame, fmt="{:.4f}"):
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in frame.iterrows():
            lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v)
                                           for v in r.to_numpy()) + " |")
        return "\n".join(lines)

    piv = D.pivot(index="decile", columns="binning", values=["BASE-T rate", "BASE-O rate"])
    piv.columns = [f"{a}, {b}" for a, b in piv.columns]
    text = [f"## T3. Negative control: null calibration — **{'PASS' if ok else 'FAIL'}**", "",
            f"{N_SETS} sets of {len(tirosh)} genes outside the blocklist, matched gene by gene to "
            f"the Tirosh genes (mean count {gap['mean (sets)']:.3f} against "
            f"{gap['mean (Tirosh)']:.3f}; detection {gap['detection (sets)']:.3f} against "
            f"{gap['detection (Tirosh)']:.3f}). Each set: the trimmed null "
            "mu = Nt e^b0 + theta m^G with its own m^G, BASE-T at alpha_B = "
            f"{ALPHA_B:g}, BASE-O at Td = {TD:g} with per-type s+.", "",
            *[f"- {k}: **{'pass' if v else 'FAIL'}**" for k, v in checks.items()], "",
            f"BASE-T's attainable level, the mean realised alpha_i, is "
            f"{S['BASE-T attainable level (mean alpha_i)'].mean():.4f} (pooled); BASE-O's null "
            f"rate is {ref['BASE-O']:.4f}. Fitted theta_m across sets: "
            f"{S['theta_m'].min():.3f} to {S['theta_m'].max():.3f} (median "
            f"{S['theta_m'].median():.3f}).", "",
            "Per set:", "", md(S.round(4)), "",
            "By decile (pooled over the sets):", "", md(piv.reset_index().astype(float).round(4)), "",
            "![T3](t3_deciles.png)"]
    open(f"{out}/t3.md", "w").write("\n".join(text) + "\n")
    print("\n".join(f"  {k}: {'pass' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"T3: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
