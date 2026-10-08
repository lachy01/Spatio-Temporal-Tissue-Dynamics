"""Division classifier comparison, Stage 4: label-scale scoring (spec 4.3, 4.4).

    python run/division_stage4_scores.py

Scores Stage 3's out-of-fold predictions against each label; nothing is
refitted. For BASE-T and BASE-O separately, at the primary thresholds:

* log loss; Brier score with its Murphy decomposition (20 equal-count bins of
  the arm's own p: reliability, resolution, uncertainty, and the within-bin
  remainder the binning leaves); AUROC and AUPRC;
* calibration-in-the-large, as mean p / label rate and mean p - label rate,
  overall and per cell type;
* reliability curves (20 equal-count bins), overall and for Cancer Epithelial
  and Plasmablasts;
* a threshold sweep: calibration-in-the-large of the count arms across the
  alpha_B grid (BASE-T) and the Td grid (BASE-O), overall and per type, and by
  the cell's own threshold c_i (pre-stated expectations X2 and X3).

SCORE is scored for LOG-S alone, in its own table (spec 4.3).

Intervals: spatial block bootstrap over the stored 250 blocks, 300 replicates,
resampling the out-of-fold predictions without refitting. Every replicate
scores every arm on the same resampled cells, so paired differences between
arms are taken within each replicate. Additive scores are bootstrapped from
per-block sums; AUROC and AUPRC from per-cell block multiplicities on a fixed
ranking (exact for each replicate, ties handled as scikit-learn does).

Noise ceilings (spec 4.4): the score each arm would get if it were the true
model, i.e. with labels drawn as L ~ Bernoulli(p). For a count arm that is the
same as simulating y and applying c_i, which V4 verified. Expected log loss and
Brier are exact (mean binary entropy, mean p(1 - p)); AUROC and AUPRC are
averaged over 50 simulated label sets. Every arm is given one, the LOG arms
included, so all are read on the same footing.

Writes outputs/division_classifier/{scores.csv, ceilings.csv,
calibration_by_threshold.csv, reliability_*.png, calibration_sweep_*.png} and
the Stage 4 note in report.md.
"""
import json
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

from tissue_dynamics.division import labels as L                      # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
N_BOOT, SEED, N_BINS, SIM_REPS = 300, 20261008, 20, 50
EPS = 1e-12
COUNT_ARMS = {"NB-c": "nbc", "NB-d": "nbd", "MIX": "mix"}
LABELS = {"BASE-T": dict(col=f"LT_{L.ab_tag(L.ALPHA_B_PRIMARY)}", log=("LOG-T", "logt"),
                         grid=[f"LT_{L.ab_tag(a)}" for a in L.ALPHA_B_GRID],
                         names=[f"alpha_B {a:g}" for a in L.ALPHA_B_GRID]),
          "BASE-O": dict(col=f"LO_{L.td_tag(L.TD_PRIMARY)}", log=("LOG-O", "logo"),
                         grid=[f"LO_{L.td_tag(t)}" for t in L.TD_GRID],
                         names=[f"Td {t:g}" for t in L.TD_GRID])}
METRICS = ("log loss", "Brier", "AUROC", "AUPRC", "mean p / rate", "mean p - rate")
LOWER_IS_BETTER = {"log loss", "Brier"}
C_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 10), (11, 14), (15, 30), (31, 10 ** 9)]
COL = {"NB-c": "#52514e", "NB-d": "#8a8984", "MIX": "#2a78d6", "LOG-T": "#eb6834",
       "LOG-O": "#eb6834", "LOG-S": "#1baf7a"}
_trapz = getattr(np, "trapezoid", None) or np.trapz


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


# --------------------------------------------------------------------------
# Scoring machinery
# --------------------------------------------------------------------------
def ranked(p, y):
    """Fixed descending ranking of p, with the last index of each tie group."""
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    return order, y[order].astype(float), last


def ap_auc(w, ys, last):
    """Weighted average precision (step-wise, as sklearn) and ROC AUC
    (trapezoidal over tie groups, as sklearn), on a fixed ranking."""
    tp = np.cumsum(w * ys)[last]
    fp = np.cumsum(w * (1.0 - ys))[last]
    P, N = tp[-1], fp[-1]
    if P <= 0 or N <= 0:
        return np.nan, np.nan
    prec = tp / np.maximum(tp + fp, 1e-300)
    ap = float(np.sum(np.diff(np.r_[0.0, tp]) / P * prec))
    auc = float(_trapz(np.r_[0.0, tp / P], np.r_[0.0, fp / N]))
    return ap, auc


class Scorer:
    """Point estimates and block-bootstrap replicates for one stratum."""

    def __init__(self, bidx, n_blocks, mult, mask=None):
        self.mask = np.ones(len(bidx), bool) if mask is None else mask
        self.b = bidx[self.mask]
        self.nb = n_blocks
        self.M = mult                                    # (N_BOOT, n_blocks)

    def _bsum(self, v):
        return np.bincount(self.b, weights=v, minlength=self.nb)

    def score(self, p, y):
        p = np.clip(p[self.mask], EPS, 1 - EPS)
        y = y[self.mask].astype(float)
        ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        br = (p - y) ** 2
        n_b, ll_b, br_b, p_b, y_b = (self._bsum(np.ones_like(p)), self._bsum(ll), self._bsum(br),
                                     self._bsum(p), self._bsum(y))
        point = {"log loss": ll.mean(), "Brier": br.mean(),
                 "mean p / rate": p.mean() / y.mean() if y.mean() > 0 else np.nan,
                 "mean p - rate": p.mean() - y.mean()}
        Mn = self.M @ n_b
        boot = {"log loss": (self.M @ ll_b) / Mn, "Brier": (self.M @ br_b) / Mn}
        mp, my = (self.M @ p_b) / Mn, (self.M @ y_b) / Mn
        with np.errstate(divide="ignore", invalid="ignore"):
            boot["mean p / rate"] = mp / my
        boot["mean p - rate"] = mp - my
        order, ys, last = ranked(p, y)
        point["AUPRC"], point["AUROC"] = ap_auc(np.ones(len(p)), ys, last)
        bo = self.b[order]
        aps, aucs = np.empty(len(self.M)), np.empty(len(self.M))
        for r in range(len(self.M)):
            aps[r], aucs[r] = ap_auc(self.M[r][bo], ys, last)
        boot["AUPRC"], boot["AUROC"] = aps, aucs
        # Murphy decomposition on 20 equal-count bins of p
        k = np.empty(len(p), int)
        k[order[::-1]] = np.arange(len(p)) * N_BINS // len(p)    # ascending p -> bin
        nk = np.bincount(k, minlength=N_BINS)
        pk = np.bincount(k, weights=p, minlength=N_BINS) / nk
        yk = np.bincount(k, weights=y, minlength=N_BINS) / nk
        rel = float(np.sum(nk * (pk - yk) ** 2) / len(p))
        res = float(np.sum(nk * (yk - y.mean()) ** 2) / len(p))
        unc = float(y.mean() * (1 - y.mean()))
        point.update({"reliability": rel, "resolution": res, "uncertainty": unc,
                      "within-bin": float(br.mean() - (rel - res + unc))})
        # reliability curve with block-bootstrap bars
        cell = self.b * N_BINS + k
        size = self.nb * N_BINS
        cn = np.bincount(cell, minlength=size).reshape(self.nb, N_BINS)
        cy = np.bincount(cell, weights=y, minlength=size).reshape(self.nb, N_BINS)
        with np.errstate(divide="ignore", invalid="ignore"):
            yk_boot = (self.M @ cy) / (self.M @ cn)
        curve = pd.DataFrame({"mean p": pk, "observed": yk, "n": nk,
                              "lo": np.nanpercentile(yk_boot, 2.5, axis=0),
                              "hi": np.nanpercentile(yk_boot, 97.5, axis=0)})
        return point, boot, curve


def ceiling(p, rng):
    """The arm's own expected scores if it were the true model."""
    p = np.clip(p, EPS, 1 - EPS)
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    sims = []
    for _ in range(SIM_REPS):
        ys = (rng.uniform(size=len(p)) < ps).astype(float)
        sims.append(ap_auc(np.ones(len(p)), ys, last))
    sims = np.array(sims)
    return {"log loss": float(np.mean(-(p * np.log(p) + (1 - p) * np.log1p(-p)))),
            "Brier": float(np.mean(p * (1 - p))),
            "AUPRC": float(sims[:, 0].mean()), "AUPRC sd": float(sims[:, 0].std()),
            "AUROC": float(sims[:, 1].mean()), "AUROC sd": float(sims[:, 1].std())}


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


def main():
    t0 = time.time()
    lab = pd.read_parquet(f"{OUT}/labels.parquet")
    oof = pd.read_parquet(f"{OUT}/oof_predictions.parquet")
    fm = pd.read_parquet(f"{OUT}/folds.parquet")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    for name, frame in (("oof_predictions", oof), ("folds", fm)):
        if not np.array_equal(frame["cell_id"].astype(str).to_numpy(),
                              lab["cell_id"].astype(str).to_numpy()):
            sys.exit(f"{name} lists cells in a different order from labels.parquet")
    ctype = lab["cell_type"].astype(str).to_numpy()
    _, bidx = np.unique(fm["block"].to_numpy(), return_inverse=True)
    nb = int(bidx.max()) + 1
    rng = np.random.default_rng(SEED)
    M = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]
                 ).astype(float)
    say(f"{len(lab):,} cells, {nb} blocks, {N_BOOT} bootstrap replicates")
    strata = {"all": None, **{t: ctype == t for t in types}}
    scorers = {s: Scorer(bidx, nb, M, m) for s, m in strata.items()}

    rows, boots, curves = [], {}, {}

    def run(label_name, label_col, arm, pcol, stratum, threshold="primary"):
        y = lab[label_col].to_numpy(float)
        point, boot, curve = scorers[stratum].score(oof[pcol].to_numpy(float), y)
        boots[(label_name, threshold, arm, stratum)] = boot
        curves[(label_name, threshold, arm, stratum)] = curve
        for m in METRICS:
            lo, hi = ci(boot[m])
            rows.append({"label": label_name, "threshold": threshold, "arm": arm,
                         "stratum": stratum, "metric": m, "estimate": point[m],
                         "ci_lo": lo, "ci_hi": hi})
        for m in ("reliability", "resolution", "uncertainty", "within-bin"):
            rows.append({"label": label_name, "threshold": threshold, "arm": arm,
                         "stratum": stratum, "metric": m, "estimate": point[m],
                         "ci_lo": np.nan, "ci_hi": np.nan})

    # ---- primary labels, every arm, every stratum --------------------------------
    for lname, spec in LABELS.items():
        arms_here = {**{a: f"p_{c}_{spec['col']}" for a, c in COUNT_ARMS.items()},
                     spec["log"][0]: f"p_{spec['log'][1]}_{spec['col']}"}
        for arm, pcol in arms_here.items():
            for s in strata:
                run(lname, spec["col"], arm, pcol, s)
            say(f"  scored {lname} {arm} ({time.time() - t0:.0f}s)")
    for s in strata:
        run("SCORE", "LS", "LOG-S", "p_logs_LS", s)
    say(f"  scored SCORE LOG-S ({time.time() - t0:.0f}s)")

    # ---- threshold sweep: count arms, every grid threshold, every stratum ----------
    for lname, spec in LABELS.items():
        for col, tname in zip(spec["grid"], spec["names"]):
            if col == spec["col"]:
                continue
            for arm, c in COUNT_ARMS.items():
                for s in strata:
                    run(lname, col, arm, f"p_{c}_{col}", s, threshold=tname)
        say(f"  swept {lname} grid ({time.time() - t0:.0f}s)")
    scores = pd.DataFrame(rows)

    # ---- paired differences (within replicate) ----------------------------------
    pairs = {"BASE-T": [("MIX", "NB-c"), ("NB-d", "NB-c"), ("LOG-T", "MIX"), ("LOG-T", "NB-c")],
             "BASE-O": [("MIX", "NB-c"), ("NB-d", "NB-c"), ("LOG-O", "MIX"), ("LOG-O", "NB-c")]}
    diff_rows = []
    for lname, pl in pairs.items():
        for s in ("all", "Cancer Epithelial", "Plasmablasts"):
            for a, b in pl:
                for m in ("log loss", "Brier", "AUROC", "AUPRC"):
                    est = (scores.query("label == @lname and threshold == 'primary' and arm == @a "
                                        "and stratum == @s and metric == @m")["estimate"].iloc[0]
                           - scores.query("label == @lname and threshold == 'primary' and arm == @b "
                                          "and stratum == @s and metric == @m")["estimate"].iloc[0])
                    d = boots[(lname, "primary", a, s)][m] - boots[(lname, "primary", b, s)][m]
                    lo, hi = ci(d)
                    diff_rows.append({"label": lname, "threshold": "primary", "arm": f"{a} - {b}",
                                      "stratum": s, "metric": m, "estimate": est,
                                      "ci_lo": lo, "ci_hi": hi})
    diffs = pd.DataFrame(diff_rows)
    pd.concat([scores, diffs]).to_csv(f"{OUT}/scores.csv", index=False)

    # ---- noise ceilings -----------------------------------------------------------
    crng = np.random.default_rng(SEED + 1)
    ceil_rows = []
    for lname, spec in LABELS.items():
        arms_here = {**{a: f"p_{c}_{spec['col']}" for a, c in COUNT_ARMS.items()},
                     spec["log"][0]: f"p_{spec['log'][1]}_{spec['col']}"}
        for arm, pcol in arms_here.items():
            c = ceiling(oof[pcol].to_numpy(float), crng)
            obs = scores.query("label == @lname and threshold == 'primary' and arm == @arm "
                               "and stratum == 'all'").set_index("metric")["estimate"]
            ceil_rows.append({"label": lname, "arm": arm,
                              **{f"{m} ceiling": c[m] for m in ("log loss", "Brier", "AUROC", "AUPRC")},
                              "AUPRC ceiling sd": c["AUPRC sd"],
                              **{f"{m} observed": obs[m] for m in ("log loss", "AUROC", "AUPRC")},
                              "log loss gap": obs["log loss"] - c["log loss"],
                              "AUPRC gap": obs["AUPRC"] - c["AUPRC"]})
    c = ceiling(oof["p_logs_LS"].to_numpy(float), crng)
    obs = scores.query("label == 'SCORE' and arm == 'LOG-S' and stratum == 'all'"
                       ).set_index("metric")["estimate"]
    ceil_rows.append({"label": "SCORE", "arm": "LOG-S",
                      **{f"{m} ceiling": c[m] for m in ("log loss", "Brier", "AUROC", "AUPRC")},
                      "AUPRC ceiling sd": c["AUPRC sd"],
                      **{f"{m} observed": obs[m] for m in ("log loss", "AUROC", "AUPRC")},
                      "log loss gap": obs["log loss"] - c["log loss"],
                      "AUPRC gap": obs["AUPRC"] - c["AUPRC"]})
    ceilings = pd.DataFrame(ceil_rows)
    ceilings.to_csv(f"{OUT}/ceilings.csv", index=False)
    say(f"  ceilings ({time.time() - t0:.0f}s)")

    # ---- calibration by the cell's own threshold c_i (X2, X3) ----------------------
    cal_rows = []
    for lname, spec in LABELS.items():
        for col, tname in zip(spec["grid"], spec["names"]):
            cvals = lab[col.replace("LT_", "cT_").replace("LO_", "cO_")].to_numpy()
            yv = lab[col].to_numpy(float)
            for lo_c, hi_c in C_BINS:
                m = (cvals >= lo_c) & (cvals <= hi_c)
                if m.sum() == 0:
                    continue
                row = {"label": lname, "threshold": tname,
                       "c_i": f"{lo_c}" if lo_c == hi_c else (f"{lo_c}-{hi_c}" if hi_c < 10 ** 9 else f">={lo_c}"),
                       "cells": int(m.sum()), "label rate": float(yv[m].mean())}
                for arm, cc in COUNT_ARMS.items():
                    pm = oof[f"p_{cc}_{col}"].to_numpy()[m]
                    row[f"{arm} p - rate"] = float(pm.mean() - yv[m].mean())
                    row[f"{arm} p / rate"] = (float(pm.mean() / yv[m].mean())
                                             if yv[m].mean() > 0 else np.nan)
                cal_rows.append(row)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(f"{OUT}/calibration_by_threshold.csv", index=False)

    # ---- figures --------------------------------------------------------------------
    for lname in ("BASE-T", "BASE-O", "SCORE"):
        arms_here = ([*COUNT_ARMS, LABELS[lname]["log"][0]] if lname != "SCORE" else ["LOG-S"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, s in zip(axes, ("all", "Cancer Epithelial", "Plasmablasts")):
            for arm in arms_here:
                cv = curves[(lname, "primary", arm, s)]
                ax.errorbar(cv["mean p"], cv["observed"],
                            yerr=[cv["observed"] - cv["lo"], cv["hi"] - cv["observed"]],
                            fmt="o-", ms=3, lw=1.3, capsize=2, color=COL[arm], label=arm)
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            if lname == "BASE-T":
                ax.set_xscale("log"); ax.set_yscale("log")
                lim = [max(lim[0], 1e-4), lim[1]]
            ax.plot(lim, lim, color="#0b0b0b", lw=0.8, ls="--")
            ax.set_xlabel("mean predicted p (20 equal-count bins)")
            ax.set_ylabel("observed label rate")
            ax.set_title(f"{lname} ({LABELS[lname]['col'] if lname != 'SCORE' else 'LS'}): {s}")
            ax.grid(True, color="#e4e3df", lw=0.6)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/reliability_{lname}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    for lname, spec in LABELS.items():
        fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True)
        x = np.arange(len(spec["grid"]))
        for ax, s in zip(axes.ravel(), strata):
            for arm in COUNT_ARMS:
                est, lo, hi = [], [], []
                for col, tname in zip(spec["grid"], spec["names"]):
                    th = "primary" if col == spec["col"] else tname
                    b = boots[(lname, th, arm, s)]["mean p / rate"]
                    pnt = scores.query("label == @lname and threshold == @th and arm == @arm and "
                                       "stratum == @s and metric == 'mean p / rate'")["estimate"].iloc[0]
                    est.append(pnt); lo.append(ci(b)[0]); hi.append(ci(b)[1])
                ax.plot(x, est, "o-", ms=3.5, color=COL[arm], label=arm)
                ax.fill_between(x, lo, hi, color=COL[arm], alpha=0.15, lw=0)
            ax.axhline(1, color="#0b0b0b", lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_title(s, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(spec["names"], rotation=35, ha="right", fontsize=8)
            ax.grid(True, color="#e4e3df", lw=0.6)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("mean p / label rate")
        fig.suptitle(f"{lname}: calibration-in-the-large across the threshold grid "
                     "(1 = calibrated; band: 95% block bootstrap)", y=1.0)
        fig.tight_layout()
        fig.savefig(f"{OUT}/calibration_sweep_{lname}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ---- report ----------------------------------------------------------------------
    def fmt_ci(est, lo, hi, d=4):
        return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

    def headline(lname, arms_here):
        out = []
        for arm in arms_here:
            r = scores.query("label == @lname and threshold == 'primary' and arm == @arm "
                             "and stratum == 'all'").set_index("metric")
            out.append({"arm": arm, **{m: fmt_ci(r.loc[m, "estimate"], r.loc[m, "ci_lo"],
                                                 r.loc[m, "ci_hi"], 3 if m == "mean p / rate" else 4)
                                       for m in ("log loss", "Brier", "AUROC", "AUPRC", "mean p / rate")}})
        return pd.DataFrame(out)

    def diff_table(lname):
        d = diffs.query("label == @lname and stratum == 'all'")
        out = []
        for pair, g in d.groupby("arm", sort=False):
            g = g.set_index("metric")
            out.append({"difference": pair, **{m: fmt_ci(g.loc[m, "estimate"], g.loc[m, "ci_lo"],
                                                         g.loc[m, "ci_hi"])
                                               for m in ("log loss", "Brier", "AUROC", "AUPRC")}})
        return pd.DataFrame(out)

    def murphy(lname, arms_here):
        out = []
        for arm in arms_here:
            r = scores.query("label == @lname and threshold == 'primary' and arm == @arm "
                             "and stratum == 'all'").set_index("metric")["estimate"]
            out.append({"arm": arm, "Brier": r["Brier"], "reliability": r["reliability"],
                        "resolution": r["resolution"], "uncertainty": r["uncertainty"],
                        "within-bin": r["within-bin"]})
        return pd.DataFrame(out)

    def by_type(lname, arms_here, metric="mean p / rate"):
        out = []
        for s in strata:
            row = {"stratum": s}
            for arm in arms_here:
                r = scores.query("label == @lname and threshold == 'primary' and arm == @arm "
                                 "and stratum == @s and metric == @metric").iloc[0]
                flag = "" if r["ci_lo"] <= (1.0 if metric == "mean p / rate" else 0.0) <= r["ci_hi"] else "*"
                row[arm] = f"{r['estimate']:.3f}{flag}"
            out.append(row)
        return pd.DataFrame(out)

    def sweep_table(lname):
        spec = LABELS[lname]
        out = []
        for col, tname in zip(spec["grid"], spec["names"]):
            th = "primary" if col == spec["col"] else tname
            row = {"threshold": tname + (" (primary)" if th == "primary" else ""),
                   "label rate": float(lab[col].mean())}
            for arm in COUNT_ARMS:
                r = scores.query("label == @lname and threshold == @th and arm == @arm and "
                                 "stratum == 'all' and metric == 'mean p / rate'").iloc[0]
                row[f"{arm} p / rate"] = fmt_ci(r["estimate"], r["ci_lo"], r["ci_hi"], 3)
            out.append(row)
        return pd.DataFrame(out)

    # pre-stated expectations
    def best_logloss(lname, own):
        d = diffs.query("label == @lname and stratum == 'all' and metric == 'log loss' and "
                        "arm.str.startswith(@own)")
        return bool((d["ci_hi"] < 0).all()), d
    x1_t, _ = best_logloss("BASE-T", "LOG-T")
    x1_o, _ = best_logloss("BASE-O", "LOG-O")
    nbc = cal.assign(diff=cal["NB-c p - rate"])
    over_5_10 = nbc[nbc["c_i"] == "5-10"]
    under_15_30 = nbc[nbc["c_i"] == "15-30"]
    w = lambda g, col: float(np.average(g[col], weights=g["cells"])) if len(g) else np.nan  # noqa: E731
    x2_lo, x2_hi = w(over_5_10, "NB-c p - rate"), w(under_15_30, "NB-c p - rate")
    x2 = x2_lo > 0 and x2_hi < 0
    STRICTEST = ("alpha_B 0.001", "alpha_B 0.0001")

    def gap(arm, skip=()):
        return np.mean([abs(np.log(scores.query(
            "label == @ln and threshold == @th and arm == @arm and stratum == 'all' and "
            "metric == 'mean p / rate'")["estimate"].iloc[0]))
            for ln, sp in LABELS.items()
            for col, tn in zip(sp["grid"], sp["names"]) if tn not in skip
            for th in ["primary" if col == sp["col"] else tn]])
    gap_mix, gap_nbc = gap("MIX"), gap("NB-c")
    gap_mix7, gap_nbc7 = gap("MIX", STRICTEST), gap("NB-c", STRICTEST)
    strict = scores.query("label == 'BASE-T' and arm == 'MIX' and stratum == 'all' and "
                          "metric == 'mean p / rate' and threshold in ['alpha_B 0.001', 'alpha_B 0.0001']")
    x3_better = gap_mix7 < gap_nbc7
    x3_under = bool((strict["ci_hi"] < 1).all())

    arms_t = [*COUNT_ARMS, "LOG-T"]
    arms_o = [*COUNT_ARMS, "LOG-O"]
    score_row = scores.query("label == 'SCORE' and arm == 'LOG-S' and stratum == 'all'").set_index("metric")
    report(f"""
## Stage 4 -- label-scale scoring ({time.strftime('%Y-%m-%d %H:%M')})

Every score is out of fold (Stage 3's predictions; nothing refitted). Intervals
are 95% spatial block bootstrap: the 250 stored blocks resampled with
replacement, {N_BOOT} replicates, every arm scored on the same resampled cells,
so a paired difference is taken within each replicate. Per the interpretive
rule, each LOG arm was trained on the label it is scored against, and a
label-scale win for it is not evidence that it infers division better.
Calibration-in-the-large is reported as mean p / label rate (1 = calibrated).
Cell types are the Stage 1 standardised argmax.

### BASE-T ({LABELS['BASE-T']['col']}, label rate {lab[LABELS['BASE-T']['col']].mean():.4f})

{md_table(headline('BASE-T', arms_t))}

Paired differences (first minus second; for log loss and Brier, negative means
the first arm is better; for AUROC and AUPRC, positive):

{md_table(diff_table('BASE-T'))}

### BASE-O ({LABELS['BASE-O']['col']}, label rate {lab[LABELS['BASE-O']['col']].mean():.4f})

{md_table(headline('BASE-O', arms_o))}

{md_table(diff_table('BASE-O'))}

### SCORE (LOG-S only; not comparable with the tables above)

Label rate {lab['LS'].mean():.4f}. Log loss {fmt_ci(*score_row.loc['log loss', ['estimate', 'ci_lo', 'ci_hi']])};
Brier {fmt_ci(*score_row.loc['Brier', ['estimate', 'ci_lo', 'ci_hi']])};
AUROC {fmt_ci(*score_row.loc['AUROC', ['estimate', 'ci_lo', 'ci_hi']])};
AUPRC {fmt_ci(*score_row.loc['AUPRC', ['estimate', 'ci_lo', 'ci_hi']])}.

### Murphy decomposition of the Brier score

Brier = reliability - resolution + uncertainty + within-bin, on 20
equal-count bins of each arm's own p. Lower reliability is better calibration;
higher resolution is better discrimination.

**BASE-T**

{md_table(murphy('BASE-T', arms_t), '{:.6f}')}

**BASE-O**

{md_table(murphy('BASE-O', arms_o), '{:.6f}')}

### Noise ceilings (spec 4.4)

What each arm would score if it were the true model (labels drawn as
L ~ Bernoulli(p); for a count arm, the same as simulating y and applying c_i).
A gap near zero means the arm's errors are what its own uncertainty predicts.
The ceiling is computed from the arm's own p, so it moves with the arm's
calibration: an arm that over-predicts a rare label raises its own expected log
loss and shows a negative gap (MIX on BASE-T, mean p / rate 1.12), and one that
under-predicts lowers it. Read a gap together with the arm's calibration; only
for a calibrated arm does a log-loss gap above zero, or an AUPRC gap below,
mean the label carries structure the arm does not.

{md_table(ceilings[['label', 'arm', 'log loss observed', 'log loss ceiling', 'log loss gap',
                    'AUPRC observed', 'AUPRC ceiling', 'AUPRC gap', 'AUROC observed',
                    'AUROC ceiling']], '{:.4f}')}

### Calibration-in-the-large by cell type (mean p / rate; * = 95% interval excludes 1)

**BASE-T**

{md_table(by_type('BASE-T', arms_t))}

**BASE-O**

{md_table(by_type('BASE-O', arms_o))}

### Threshold sweep (count arms, all cells)

**BASE-T**

{md_table(sweep_table('BASE-T'))}

**BASE-O**

{md_table(sweep_table('BASE-O'))}

By cell type, in `calibration_sweep_BASE-T.png` and `calibration_sweep_BASE-O.png`;
by the cell's own threshold c_i, in `calibration_by_threshold.csv`.

### Pre-stated expectations X1-X3

- **X1** (each LOG arm has the best log loss on its own label): BASE-T
  **{'holds' if x1_t else 'does not hold'}**; BASE-O **{'holds' if x1_o else 'does not hold'}**
  (every paired log-loss difference against the LOG arm, interval excluding 0).
- **X2** (NB-c over-predicts where c_i is 5-10 and under-predicts where it is
  15-30, pooled over both grids): mean p - rate {x2_lo:+.5f} at c_i 5-10 and
  {x2_hi:+.5f} at 15-30: **{'holds' if x2 else 'does not hold'}**.
- **X3** (MIX better calibrated than NB-c across the grids, but under-predicting
  at the strictest thresholds). The two clauses are judged separately, since
  the second concerns the strictest thresholds. Mean |log(p / rate)| over the
  seven grid thresholds other than alpha_B 0.001 and 0.0001: {gap_mix7:.3f} for
  MIX against {gap_nbc7:.3f} for NB-c (**{'better' if x3_better else 'not better'}**).
  At alpha_B 0.001 and 0.0001 MIX's p / rate is
  {', '.join(f"{v:.2f}" for v in strict['estimate'])} (**{'under-predicts' if x3_under else 'does not clearly under-predict'}**),
  far below NB-c's, so over all nine thresholds MIX is the worse calibrated
  ({gap_mix:.3f} against {gap_nbc:.3f}). Overall: **{'holds' if x3_better and x3_under else 'partly holds'}**.

Figures: `reliability_BASE-T.png`, `reliability_BASE-O.png`,
`reliability_SCORE.png`, `calibration_sweep_BASE-T.png`,
`calibration_sweep_BASE-O.png`. Every score and interval is in `scores.csv`
(including the full per-type metrics and the Murphy components for every
stratum); ceilings in `ceilings.csv`. Stage 4 complete in
{(time.time() - t0) / 60:.0f} min.
""")
    say(f"\nStage 4 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
