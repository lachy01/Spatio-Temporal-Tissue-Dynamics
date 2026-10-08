"""Division classifier comparison, Stage 7: sensitivities (spec 9, item 7).

    python run/division_stage7_sensitivity.py

Out of fold on the stored partition, scored as Stage 4 scores (250-block
bootstrap, 300 replicates, paired differences within replicate):

1. The alpha_B and Td grids. Stage 4 pushed the count arms through every grid
   threshold; here each grid label also gets its own LOG arm (LOG-T at
   alpha_B 0.01, 0.001 and 1e-4; LOG-O at Td 0, 0.25, 0.75 and 1), with the
   production ridge, so every threshold has the full set of arms.
2. BASE-O-global (one s+ for all cells) and BASE-O-neg (the negative-codeword
   floor, decoding noise only) at Td 0.5: the count arms pushed through their
   thresholds, and a LOG arm for each.
3. Unpenalised arms: NB-c and LOG-T, LOG-O, LOG-S refitted with
   weight_ridge = 0 (the balance term, which fixes only the U/V
   parameterisation, stays). Tests whether the shared penalty, chosen for the
   mixture's weight model, handicapped the label-trained arms (Stage 4's
   caveat on X1).
4. MIX with theta = 1: the production mixture has fixed theta = 1 since
   2026-10-06, so this row is the production row; nothing to rerun.
5. The nucleus-only matrix (spillover check): waits for the transcript-level
   file, as decided in Stage 1.

V7 applies to every LOG arm at the production ridge (sd(eta) <= 20, else
stop). Unpenalised fits report sd(eta) without a criterion; separation there
is a finding, not a failure.

Each fit's held-out predictions are cached in outputs/division_classifier/
cache_stage7/, so an interrupted run resumes. Writes stage7_fits.csv,
stage7_scores.csv, stage7_grid.png and the Stage 7 note in report.md.
"""
import importlib.util
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
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division.survival import mix_p_ge, nb_p_ge       # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data                  # noqa: E402

_s = importlib.util.spec_from_file_location("s4", os.path.join(HERE, "division_stage4_scores.py"))
s4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s4)

OUT = "outputs/division_classifier"
CACHE = f"{OUT}/cache_stage7"
REPORT = f"{OUT}/report.md"
N_FOLDS = 5
GRID_T = [a for a in L.ALPHA_B_GRID if a != L.ALPHA_B_PRIMARY]
GRID_O = [t for t in L.TD_GRID if t != L.TD_PRIMARY]
PRIMARY_T, PRIMARY_O = f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}", f"LO_{L.td_tag(L.TD_PRIMARY)}"
VARIANTS_O = {"BASE-O-global": f"LO_global_{L.td_tag(L.TD_PRIMARY)}",
              "BASE-O-neg": f"LO_neg_{L.td_tag(L.TD_PRIMARY)}"}


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


md_table = s4.md_table


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = types
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    oof = pd.read_parquet(f"{OUT}/oof_predictions.parquet")
    ids = df["cell_id"].astype(str).to_numpy()
    for name, frame in (("labels", lab), ("folds", fm), ("oof_predictions", oof)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), ids):
            sys.exit(f"{name} lists cells in a different order from the table")
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    folds = fm["fold"].to_numpy()
    n = len(df)
    fold_data = {}
    for f in range(N_FOLDS):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        fold_data[f] = (tr, te, data.subset(tr).as_jax(), data.subset(te).as_jax())
    say(f"{n:,} cells; production weight_ridge {spec.weight_ridge:g}")

    rows = []

    def oof_log(name, label_col, ridge):
        """Out-of-fold LOG arm on `label_col` at `ridge`; cached."""
        path = f"{CACHE}/{name}.npz"
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            rows.extend(json.loads(str(z["rows"])))
            return z["p"]
        yl_all = lab[label_col].to_numpy(float)
        pred, rr = np.zeros(n), []
        for f in range(N_FOLDS):
            tr, te, D_tr, D_te = fold_data[f]
            yl = jnp.asarray(yl_all[tr])
            obj = arms.logistic_objective(D_tr, yl, len(tr), ridge, spec.n_reference, spec.ridge_tau,
                                          spec.b_penalty)
            for attempt in range(3):
                p, info = arms.minimise(obj, arms.init_logistic(D_tr, yl, spec.rank, 100 * attempt))
                if info["converged"]:
                    break
            diag = arms.eta_diagnostics(p, D_tr)
            pred[te] = np.asarray(jax.nn.sigmoid(arms.logistic_eta(p, D_te)))
            rr.append({"fit": name, "label": label_col, "ridge": ridge, "fold": f,
                       "converged": bool(info["converged"]), "grad_inf": info["grad_inf"],
                       "sd(eta)": diag["sd_eta"], "norm a": diag["norm_a"], "sigma1(B)": diag["sigma1_B"],
                       "minutes": info["wall_time"] / 60})
        np.savez(path, p=pred, rows=json.dumps(rr))
        rows.extend(rr)
        say(f"  {name}: {sum(r['converged'] for r in rr)}/5 converged, max sd(eta) "
            f"{max(r['sd(eta)'] for r in rr):.2f} ({(time.time() - t0) / 60:.0f} min)")
        return pred

    def oof_nb_unpen():
        path = f"{CACHE}/NB-c_unpenalised.npz"
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            rows.extend(json.loads(str(z["rows"])))
            return z["mu"], z["alpha"]
        mu, al, rr = np.zeros(n), np.zeros(n), []
        for f in range(N_FOLDS):
            tr, te, D_tr, D_te = fold_data[f]
            obj = arms.nb_objective(D_tr, len(tr), 0.0, spec.n_reference, spec.ridge_tau, spec.b_penalty)
            for attempt in range(3):
                p, info = arms.minimise(obj, arms.init_nb(D_tr, "const", spec.rank, 100 * attempt,
                                                          theta_free=spec.theta_free))
                if info["converged"]:
                    break
            diag = arms.eta_diagnostics(p, D_tr)
            mu[te] = np.exp(np.asarray(arms.nb_log_mu(p, D_te)))
            al[te] = float(np.exp(np.asarray(arms.nb_log_alpha(p, D_te))))
            rr.append({"fit": "NB-c unpenalised", "label": "-", "ridge": 0.0, "fold": f,
                       "converged": bool(info["converged"]), "grad_inf": info["grad_inf"],
                       "sd(eta)": diag["sd_eta"], "norm a": diag["norm_a"], "sigma1(B)": diag["sigma1_B"],
                       "minutes": info["wall_time"] / 60})
        np.savez(path, mu=mu, alpha=al, rows=json.dumps(rr))
        rows.extend(rr)
        say(f"  NB-c unpenalised: {sum(r['converged'] for r in rr)}/5 converged "
            f"({(time.time() - t0) / 60:.0f} min)")
        return mu, al

    # ---- fits ------------------------------------------------------------------------
    preds = {}
    for a in GRID_T:
        col = f"LT_{L.ab_tag(a)}"
        preds[("BASE-T", f"alpha_B {a:g}", "LOG-T")] = oof_log(f"LOG-T_{col}", col, spec.weight_ridge)
    for t in GRID_O:
        col = f"LO_{L.td_tag(t)}"
        preds[("BASE-O", f"Td {t:g}", "LOG-O")] = oof_log(f"LOG-O_{col}", col, spec.weight_ridge)
    for vname, col in VARIANTS_O.items():
        preds[(vname, "Td 0.5", "LOG-O")] = oof_log(f"LOG-O_{col}", col, spec.weight_ridge)
    unpen = {"LOG-T": oof_log("LOG-T_unpenalised", PRIMARY_T, 0.0),
             "LOG-O": oof_log("LOG-O_unpenalised", PRIMARY_O, 0.0),
             "LOG-S": oof_log("LOG-S_unpenalised", "LS", 0.0)}
    mu_u, al_u = oof_nb_unpen()
    fits = pd.DataFrame(rows)
    fits.to_csv(f"{OUT}/stage7_fits.csv", index=False)
    pen_log = fits[(fits["fit"].str.startswith("LOG")) & (fits["ridge"] > 0)]
    v7_ok = bool((pen_log["sd(eta)"] <= 20).all())
    all_conv = bool(fits["converged"].all())
    say(f"fits done ({(time.time() - t0) / 60:.0f} min); V7 {'pass' if v7_ok else 'FAIL'}; "
        f"all converged {all_conv}")
    if not v7_ok:
        report("\n## Stage 7 -- sensitivities\n\n### STOPPED: V7 (LOG separation) failed\n\n"
               + md_table(pen_log[pen_log["sd(eta)"] > 20]) + "\n\nPer spec section 8, nothing further was run.")
        sys.exit("V7 failed")

    # count arms pushed through the BASE-O variants
    for vname, col in VARIANTS_O.items():
        c = lab[col.replace("LO_", "cO_")].to_numpy(float)
        preds[(vname, "Td 0.5", "NB-c")] = nb_p_ge(c, oof["mu_nbc"].to_numpy(), oof["alpha_nbc"].to_numpy())
        preds[(vname, "Td 0.5", "NB-d")] = nb_p_ge(c, oof["mu_nbd"].to_numpy(), oof["alpha_nbd"].to_numpy())
        preds[(vname, "Td 0.5", "MIX")] = mix_p_ge(c, oof["pi_mix"].to_numpy(), oof["lam0_mix"].to_numpy(),
                                                  oof["lam1_mix"].to_numpy())
    # count arms through the grid thresholds come from Stage 3
    for a in GRID_T:
        col = f"LT_{L.ab_tag(a)}"
        for arm, cc in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix")):
            preds[("BASE-T", f"alpha_B {a:g}", arm)] = oof[f"p_{cc}_{col}"].to_numpy()
    for t in GRID_O:
        col = f"LO_{L.td_tag(t)}"
        for arm, cc in (("NB-c", "nbc"), ("NB-d", "nbd"), ("MIX", "mix")):
            preds[("BASE-O", f"Td {t:g}", arm)] = oof[f"p_{cc}_{col}"].to_numpy()
    # unpenalised arms against their penalised counterparts, at the primary labels
    for lname, col, own in (("BASE-T", PRIMARY_T, "LOG-T"), ("BASE-O", PRIMARY_O, "LOG-O")):
        c = lab[col.replace("LT_", "cT_").replace("LO_", "cO_")].to_numpy(float)
        preds[(lname, "primary", "NB-c unpenalised")] = nb_p_ge(c, mu_u, al_u)
        preds[(lname, "primary", f"{own} unpenalised")] = unpen[own]
        for arm, cc in (("NB-c", "nbc"), ("MIX", "mix")):
            preds[(lname, "primary", arm)] = oof[f"p_{cc}_{col}"].to_numpy()
        preds[(lname, "primary", own)] = oof[f"p_{own.replace('-', '').lower()}_{col}"].to_numpy()
    preds[("SCORE", "primary", "LOG-S")] = oof["p_logs_LS"].to_numpy()
    preds[("SCORE", "primary", "LOG-S unpenalised")] = unpen["LOG-S"]

    # ---- scoring (Stage 4 machinery) -------------------------------------------------
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb_ = int(bidx.max()) + 1
    rng = np.random.default_rng(s4.SEED)
    M = np.stack([np.bincount(rng.integers(0, nb_, nb_), minlength=nb_) for _ in range(s4.N_BOOT)]).astype(float)
    scorer = s4.Scorer(bidx, nb_, M)

    def label_col(lname, thr):
        if lname in VARIANTS_O:
            return VARIANTS_O[lname]
        if lname == "SCORE":
            return "LS"
        if thr == "primary":
            return PRIMARY_T if lname == "BASE-T" else PRIMARY_O
        v = float(thr.split()[-1])
        return f"LT_{L.ab_tag(v)}" if lname == "BASE-T" else f"LO_{L.td_tag(v)}"

    sc_rows, boots = [], {}
    for (lname, thr, arm), p in preds.items():
        yv = lab[label_col(lname, thr)].to_numpy(float)
        point, boot, _ = scorer.score(np.asarray(p, float), yv)
        boots[(lname, thr, arm)] = boot
        for m in ("log loss", "Brier", "AUROC", "AUPRC", "mean p / rate"):
            lo, hi = s4.ci(boot[m])
            sc_rows.append({"label": lname, "threshold": thr, "arm": arm, "metric": m,
                            "estimate": point[m], "ci_lo": lo, "ci_hi": hi,
                            "label rate": float(yv.mean())})
    scores = pd.DataFrame(sc_rows)

    def diff(lname, thr, a, b, m):
        est = (scores.query("label == @lname and threshold == @thr and arm == @a and metric == @m")["estimate"].iloc[0]
               - scores.query("label == @lname and threshold == @thr and arm == @b and metric == @m")["estimate"].iloc[0])
        lo, hi = s4.ci(boots[(lname, thr, a)][m] - boots[(lname, thr, b)][m])
        return est, lo, hi

    for (lname, thr, arm) in list(preds):
        own = {"BASE-T": "LOG-T", "SCORE": "LOG-S"}.get(lname, "LOG-O")
        if arm != own or lname == "SCORE":
            continue
        for other in ("MIX", "NB-c"):
            for m in ("log loss", "AUPRC"):
                est, lo, hi = diff(lname, thr, own, other, m)
                sc_rows.append({"label": lname, "threshold": thr, "arm": f"{own} - {other}", "metric": m,
                                "estimate": est, "ci_lo": lo, "ci_hi": hi, "label rate": np.nan})
    for lname, own in (("BASE-T", "LOG-T"), ("BASE-O", "LOG-O"), ("SCORE", "LOG-S")):
        pairs = [(f"{own} unpenalised", own)]
        if lname != "SCORE":
            pairs += [("NB-c unpenalised", "NB-c"), (f"{own} unpenalised", "MIX"),
                      (f"{own} unpenalised", "NB-c unpenalised")]
        for a, b in pairs:
            for m in ("log loss", "AUPRC"):
                est, lo, hi = diff(lname, "primary", a, b, m)
                sc_rows.append({"label": lname, "threshold": "primary", "arm": f"{a} - {b}", "metric": m,
                                "estimate": est, "ci_lo": lo, "ci_hi": hi, "label rate": np.nan})
    scores = pd.DataFrame(sc_rows)
    scores.to_csv(f"{OUT}/stage7_scores.csv", index=False)

    # ---- figure: every arm across the grids ------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for row, (lname, grid, own) in enumerate((("BASE-T", [f"alpha_B {a:g}" for a in L.ALPHA_B_GRID], "LOG-T"),
                                              ("BASE-O", [f"Td {t:g}" for t in L.TD_GRID], "LOG-O"))):
        prim = f"alpha_B {L.ALPHA_B_PRIMARY:g}" if lname == "BASE-T" else f"Td {L.TD_PRIMARY:g}"
        for ax, m in zip(axes[row], ("log loss", "AUPRC", "mean p / rate")):
            for arm in ("NB-c", "NB-d", "MIX", own):
                est, lo, hi = [], [], []
                for thr in grid:
                    if thr == prim:
                        s = s4_primary(lname, arm, m)
                    else:
                        s = scores.query("label == @lname and threshold == @thr and arm == @arm and metric == @m").iloc[0]
                    est.append(s["estimate"]); lo.append(s["ci_lo"]); hi.append(s["ci_hi"])
                x = np.arange(len(grid))
                ax.plot(x, est, "o-", ms=4, color=s4.COL.get(arm, "#8a8984"), label=arm)
                ax.fill_between(x, lo, hi, color=s4.COL.get(arm, "#8a8984"), alpha=0.15, lw=0)
            ax.set_xticks(np.arange(len(grid)))
            ax.set_xticklabels(grid, rotation=30, ha="right", fontsize=8)
            if m == "mean p / rate":
                ax.axhline(1, color="#0b0b0b", lw=0.8, ls="--")
                ax.set_yscale("log")
            if m == "log loss":
                ax.set_yscale("log")
            ax.set_title(f"{lname}: {m}")
            ax.grid(True, color="#e4e3df", lw=0.6)
        axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage7_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----------------------------------------------------------------------
    def fmt(s, d=4):
        return f"{s['estimate']:.{d}f} [{s['ci_lo']:.{d}f}, {s['ci_hi']:.{d}f}]"

    def grid_table(lname, own):
        out = []
        grid = [f"alpha_B {a:g}" for a in GRID_T] if lname == "BASE-T" else [f"Td {t:g}" for t in GRID_O]
        for thr in grid:
            r = {"threshold": thr,
                 "label rate": float(scores.query("label == @lname and threshold == @thr").iloc[0]["label rate"])}
            for arm in ("NB-c", "MIX", own):
                r[f"{arm} log loss"] = fmt(scores.query("label == @lname and threshold == @thr and arm == @arm "
                                                        "and metric == 'log loss'").iloc[0])
            for arm in ("NB-c", "MIX", own):
                r[f"{arm} AUPRC"] = fmt(scores.query("label == @lname and threshold == @thr and arm == @arm "
                                                     "and metric == 'AUPRC'").iloc[0], 3)
            d_arm = f"{own} - MIX"
            d = scores.query("label == @lname and threshold == @thr and arm == @d_arm and metric == 'log loss'")
            r[f"{own} - MIX log loss"] = fmt(d.iloc[0])
            out.append(r)
        return pd.DataFrame(out)

    def variant_table():
        out = []
        for vname in VARIANTS_O:
            for arm in ("NB-c", "NB-d", "MIX", "LOG-O"):
                g = scores.query("label == @vname and arm == @arm").set_index("metric")
                out.append({"label": vname, "arm": arm, "label rate": float(g["label rate"].iloc[0]),
                            **{m: fmt(g.loc[m], 3 if m != "log loss" else 4)
                               for m in ("log loss", "AUPRC", "mean p / rate")}})
        return pd.DataFrame(out)

    def unpen_table():
        out = []
        for lname, own in (("BASE-T", "LOG-T"), ("BASE-O", "LOG-O"), ("SCORE", "LOG-S")):
            arms_ = [own, f"{own} unpenalised"] + (["NB-c", "NB-c unpenalised", "MIX"] if lname != "SCORE" else [])
            for arm in arms_:
                g = scores.query("label == @lname and threshold == 'primary' and arm == @arm").set_index("metric")
                out.append({"label": lname, "arm": arm,
                            **{m: fmt(g.loc[m], 3 if m != "log loss" else 4)
                               for m in ("log loss", "AUPRC", "mean p / rate")}})
        return pd.DataFrame(out)

    def unpen_diff_table():
        d = scores[scores["arm"].str.contains("unpenalised -")]
        return pd.DataFrame([{"label": r["label"], "difference": r["arm"], "metric": r["metric"],
                              "estimate": fmt(r)} for _, r in d.iterrows()])

    unp = fits[fits["ridge"] == 0].groupby("fit").agg(converged=("converged", "sum"),
                                                     max_sd_eta=("sd(eta)", "max"),
                                                     max_norm_a=("norm a", "max")).reset_index()
    x1_unpen = []
    for lname, own in (("BASE-T", "LOG-T"), ("BASE-O", "LOG-O")):
        d_arms = [f"{own} unpenalised - MIX", f"{own} unpenalised - NB-c unpenalised"]
        d = scores.query("label == @lname and arm in @d_arms and metric == 'log loss'")
        x1_unpen.append(f"{lname}: " + ("the unpenalised LOG arm now has the best log loss"
                                        if (d["ci_hi"] < 0).all() else "the unpenalised LOG arm still does not have "
                                        "the best log loss"))
    report(f"""
## Stage 7 -- sensitivities ({time.strftime('%Y-%m-%d %H:%M')})

Out of fold, scored as in Stage 4 (250-block bootstrap, {s4.N_BOOT} replicates;
paired differences within replicate). {len(fits)} new fits, all converged:
**{'yes' if all_conv else 'NO'}**. V7 at the production ridge for every new LOG
arm: **{'pass' if v7_ok else 'FAIL'}** (largest sd(eta) {pen_log['sd(eta)'].max():.2f}).

### 1. The threshold grids, every arm

Each grid label now has its own LOG arm; the count arms are Stage 3's push-through.
The last column is the LOG arm's paired log-loss difference against MIX
(negative = LOG better).

**BASE-T**

{md_table(grid_table('BASE-T', 'LOG-T'))}

**BASE-O**

{md_table(grid_table('BASE-O', 'LOG-O'))}

The primary thresholds are in Stage 4; all of it is drawn in `stage7_grid.png`.

### 2. BASE-O variants (Td 0.5)

BASE-O-global uses one s+ for all cells; BASE-O-neg the negative-codeword floor
(decoding noise only, effectively 1{{y >= 2}}).

{md_table(variant_table())}

### 3. Unpenalised arms

weight_ridge = 0 (balance term kept). Convergence and size of the fits:

{md_table(unp, '{:.3f}')}

{md_table(unpen_table())}

Paired differences (first minus second):

{md_table(unpen_diff_table())}

X1 without the penalty: {'; '.join(x1_unpen)}.

### 4. MIX with theta = 1

The production mixture has fixed theta = 1 since 2026-10-06, so this row is
the production row in Stages 3-6; there is nothing separate to run.

### 5. Nucleus-only matrix

Not run: it needs the transcript-level file (nucleus overlap per
transcript), which is not on this machine (decided in Stage 1).

Every score and interval is in `stage7_scores.csv`; every fit in
`stage7_fits.csv`. Stage 7 complete in {(time.time() - t0) / 60:.0f} min.
""")
    say(f"\nStage 7 done in {(time.time() - t0) / 60:.0f} min")


def s4_primary(lname, arm, m):
    """A primary-threshold score from Stage 4's scores.csv."""
    sc = pd.read_csv(f"{OUT}/scores.csv")
    return sc[(sc["label"] == lname) & (sc["threshold"] == "primary") & (sc["arm"] == arm)
              & (sc["stratum"] == "all") & (sc["metric"] == m)].iloc[0]


if __name__ == "__main__":
    main()
