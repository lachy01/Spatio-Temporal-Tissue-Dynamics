"""How well each BASE-T label agrees with the mixture (MIX), across label variants.

    python run/division_mix_agreement.py

MIX's cycling prior pi (out-of-fold, outputs/division_classifier/oof_predictions.parquet)
uses no counts, so agreement with it is agreement in what predicts cycling, not in
the count each label is thresholded on. For every BASE-T label of the three variants
(original: naive null; t10: trunc 0.2 at 10%; w30: floored trunc 0.3, Wu v2 pick
and 10%) and every alpha_B on its grid:

  label rate             share of cells labelled; MIX's mean pi beside it
  per-type rank          Spearman of the nine per-type label rates with MIX's
                         per-type mean pi
  per-type distortion    SD across types of log(label rate / MIX mean pi): 0 means
                         the label keeps MIX's between-type contrasts up to one scale
  AUC of MIX pi          how well MIX's prior ranks the labelled cells above the rest
  MIX calibration        mean of MIX's pushed-through P(label) minus the label rate,
                         where the variant's OOF predictions hold that column
  LOG-T vs MIX           Spearman of the label's own LOG-T OOF prediction with MIX's
                         pi (primary label of each variant)

Writes outputs/division_classifier_w30/mix_agreement.{csv,md,png}.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

VARIANTS = {"original (naive null)": "outputs/division_classifier",
            "t10 (trunc 0.2)": "outputs/division_classifier_t10",
            "w30 (floored trunc 0.3)": "outputs/division_classifier_w30"}
PRIMARY = {"original (naive null)": "LT_a05", "t10 (trunc 0.2)": "LT_a1085", "w30 (floored trunc 0.3)": "LT_a1547"}
COL = {"original (naive null)": "#6b6a66", "t10 (trunc 0.2)": "#2563eb", "w30 (floored trunc 0.3)": "#c2410c"}
OUT = "outputs/division_classifier_w30"


def auc(score, y):
    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    base = pd.read_parquet(f"{VARIANTS['original (naive null)']}/oof_predictions.parquet",
                           columns=["cell_id", "cell_type", "pi_mix"])
    pi, ct = base["pi_mix"].to_numpy(float), base["cell_type"].astype(str).to_numpy()
    types = sorted(set(ct))
    pi_type = np.array([pi[ct == t].mean() for t in types])
    rows = []
    for v, d in VARIANTS.items():
        lab = pd.read_parquet(f"{d}/labels.parquet")
        lab = lab.set_index(lab["cell_id"].astype(str)).loc[base["cell_id"].astype(str)]
        oof = pd.read_parquet(f"{d}/oof_predictions.parquet")
        if not np.array_equal(oof["cell_id"].astype(str).to_numpy(), base["cell_id"].astype(str).to_numpy()):
            raise SystemExit(f"{v}: OOF cells differ from the original's")
        if not np.allclose(oof["pi_mix"], pi):
            raise SystemExit(f"{v}: MIX's OOF pi differs from the original's")
        for col in sorted(c for c in lab.columns if c.startswith("LT_")):
            y = lab[col].to_numpy(bool)
            rt = np.array([y[ct == t].mean() for t in types])
            pmix = f"p_mix_{col}"
            plog = f"p_logt_{col}"
            rows.append({"variant": v, "label": col, "primary": col == PRIMARY[v],
                         "label rate": float(y.mean()), "MIX mean pi": float(pi.mean()),
                         "per-type Spearman": float(spearmanr(rt, pi_type)[0]),
                         "per-type log-ratio SD": float(np.std(np.log(np.clip(rt, 1e-6, None) / pi_type))),
                         "AUC of MIX pi": auc(pi, y),
                         "MIX calibration (pred - obs)": (float(oof[pmix].mean() - y.mean())
                                                          if pmix in oof else np.nan),
                         "LOG-T vs MIX pi (Spearman)": (float(spearmanr(oof[plog], pi)[0])
                                                        if plog in oof else np.nan)})
    tab = pd.DataFrame(rows).sort_values(["variant", "label rate"])
    tab.to_csv(f"{OUT}/mix_agreement.csv", index=False)
    with open(f"{OUT}/mix_agreement.md", "w") as fh:
        cols = list(tab.columns)
        fh.write("| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n")
        for _, r in tab.iterrows():
            fh.write("| " + " | ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in r) + " |\n")
    print(tab.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), layout="constrained")
    for v, g in tab.groupby("variant", sort=False):
        for ax, m in zip(axes, ("per-type Spearman", "per-type log-ratio SD", "AUC of MIX pi")):
            ax.plot(100 * g["label rate"], g[m], "o-", color=COL[v], lw=2, ms=5, label=v)
            p = g[g["primary"]]
            ax.plot(100 * p["label rate"], p[m], "o", color=COL[v], ms=11, mfc="none", mew=2)
    titles = {"per-type Spearman": "Per-type ordering vs MIX (higher = closer)",
              "per-type log-ratio SD": "Per-type contrast distortion vs MIX (lower = closer)",
              "AUC of MIX pi": "MIX prior's AUC for the label"}
    for ax, m in zip(axes, titles):
        ax.set_xscale("log")
        ax.axvline(100 * pi.mean(), color="0.6", lw=1, ls=":")
        ax.set_xlabel("cells labelled cycling (%)")
        ax.set_title(titles[m], fontsize=10.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Agreement of each BASE-T label with the mixture (circled: each variant's primary; "
                 "dotted: MIX's mean pi)", fontsize=11)
    fig.savefig(f"{OUT}/mix_agreement.png", dpi=150)


if __name__ == "__main__":
    main()
