"""Division classifier comparison, Stage 6: summarise the known-truth simulation.

    python run/division_stage6_gather.py [--dir outputs/hpc/division_sim] [--no-report]

Reads the Spartan array's per-replicate results (job `division_sim`, written by
src/tissue_dynamics/division/simulation.py) and summarises spec 7.3 across
replicates:

1. Rank agreement: Spearman correlation of each prediction with pi_true on the
   held-out fold (G1, G2).
2. Neighbourhood effect: per own type, the predicted over the true logit
   contrast between tumour-rich (>= 90th percentile of the 80 um disc's Cancer
   Epithelial score, within the type) and tumour-poor (<= 10th) neighbourhoods.
   Reported as the per-type ratio (median over replicates), and pooled across
   types as the slope of predicted on true contrast through the origin. Types
   whose true contrast is small (|logit difference| < 0.1) make the ratio
   unstable; they are flagged and left out of the pooled summaries.
3. Fraction error: per type, |mean prediction - realised cycling fraction|,
   averaged over types. Only MIX's pi targets this quantity; the label-scale
   arms are shown with that caveat.
G3 (no latent classes): label-scale scores and each arm's implied cycling
fraction.

Pre-stated expectation X8 is judged here. Writes outputs/division_classifier/
{sim_recovery.csv, sim_recovery.png} and the Stage 6 results in report.md.
"""
import argparse
import glob
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

from tissue_dynamics.division import paths as DP                    # noqa: E402

OUT = DP.out_dir()          # DIV_VARIANT=<v>: outputs/division_classifier_<v>
REPORT = f"{OUT}/report.md"
PREDS = ["MIX pi", "MIX, BASE-T", "MIX, BASE-O", "NB-c, BASE-T", "NB-c, BASE-O",
         "LOG-T", "LOG-O", "LOG-S"]
MIN_TRUE = 0.1
COL = {"MIX pi": "#2a78d6", "MIX, BASE-T": "#7fb0e8", "MIX, BASE-O": "#a9c8ee",
       "NB-c, BASE-T": "#52514e", "NB-c, BASE-O": "#8a8984", "LOG-T": "#f5a87e",
       "LOG-O": "#eb6834", "LOG-S": "#1baf7a"}
MK = {"MIX pi": "o", "MIX, BASE-T": "o", "MIX, BASE-O": "o", "NB-c, BASE-T": "s",
      "NB-c, BASE-O": "s", "LOG-T": "^", "LOG-O": "^", "LOG-S": "D"}


def say(*a):
    print(*a, flush=True)


def md_table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def med_rng(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return f"{np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]" if len(v) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{DP.hpc_dir()}/division_sim"
                    + (f"_{DP.variant()}" if DP.variant() else ""))
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "0*.json")))]
    if not res:
        sys.exit(f"no results in {a.dir}")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    df = pd.read_parquet("data/mixture_cells_major.parquet", columns=["N", "archetype"])
    lab = pd.read_parquet(f"{OUT}/labels.parquet", columns=["cell_type"])
    med_N = df.groupby(lab["cell_type"].to_numpy())["N"].median().reindex(types)
    low_depth = set(med_N[med_N < df["N"].median()].index)
    gens = sorted({r["generator"] for r in res})
    say(f"{len(res)} replicates: " + ", ".join(f"{g} {sum(r['generator'] == g for r in res)}" for g in gens))
    conv = pd.DataFrame([{"generator": r["generator"], "rep": r["rep"],
                          **{f"{k} converged": v["converged"] for k, v in r["fits"].items()},
                          "LOG max sd(eta)": max(r["fits"][k]["sd_eta"] for k in ("LOG-T", "LOG-O", "LOG-S")),
                          "MIX e^delta": r["fits"]["MIX"]["e_delta"], "minutes": r["wall_time"] / 60}
                         for r in res])

    long = []
    for r in res:
        g, rep = r["generator"], r["rep"]
        for k, v in r["label_scores"].items():
            for m in ("log loss", "AUPRC", "mean p / rate"):
                long.append({"generator": g, "rep": rep, "metric": f"label: {m}", "prediction": k,
                             "stratum": "all", "value": v[m]})
        for k, fr in r["fractions"].items():
            for t, v in fr.items():
                long.append({"generator": g, "rep": rep, "metric": "fraction", "prediction": k,
                             "stratum": t, "value": v})
        if "spearman_with_pi_true" not in r:
            continue
        for t in types:
            long.append({"generator": g, "rep": rep, "metric": "true fraction", "prediction": "truth",
                         "stratum": t, "value": r["true_fraction"][t]})
        for k, v in r["spearman_with_pi_true"].items():
            long.append({"generator": g, "rep": rep, "metric": "spearman", "prediction": k,
                         "stratum": "all", "value": v})
        for t, row in r["neighbourhood_contrast"].items():
            long.append({"generator": g, "rep": rep, "metric": "true contrast", "prediction": "truth",
                         "stratum": t, "value": row["true"]})
            for k in PREDS:
                long.append({"generator": g, "rep": rep, "metric": "contrast", "prediction": k,
                             "stratum": t, "value": row[k]})
                long.append({"generator": g, "rep": rep, "metric": "contrast ratio", "prediction": k,
                             "stratum": t, "value": row[k] / row["true"] if row["true"] != 0 else np.nan})
            for k in PREDS:
                long.append({"generator": g, "rep": rep, "metric": "fraction error", "prediction": k,
                             "stratum": t, "value": abs(r["fractions"][k][t] - r["true_fraction"][t])})
    L = pd.DataFrame(long)
    L.to_csv(f"{OUT}/sim_recovery.csv", index=False)

    def q(metric, gen, pred=None, stratum=None):
        s = L[(L["metric"] == metric) & (L["generator"] == gen)]
        if pred is not None:
            s = s[s["prediction"] == pred]
        if stratum is not None:
            s = s[s["stratum"] == stratum]
        return s

    state_gens = [g for g in gens if g in ("G1", "G2")]
    # true contrasts, to flag unstable ratios
    true_c = {g: q("true contrast", g).groupby("stratum")["value"].median().reindex(types) for g in state_gens}
    stable = {g: [t for t in types if abs(true_c[g][t]) >= MIN_TRUE] for g in state_gens}

    rank_tab, slope_tab, frac_tab = [], [], []
    for k in PREDS:
        rank_tab.append({"prediction": k, **{g: med_rng(q("spearman", g, k)["value"]) for g in state_gens}})
        row_s, row_f = {"prediction": k}, {"prediction": k}
        for g in state_gens:
            slopes = []
            for rep, grp in q("contrast", g, k).groupby("rep"):
                tc = q("true contrast", g).query("rep == @rep").set_index("stratum")["value"]
                pc = grp.set_index("stratum")["value"]
                ts = [t for t in stable[g]]
                slopes.append(float(np.sum(pc[ts] * tc[ts]) / np.sum(tc[ts] ** 2)))
            row_s[g] = med_rng(slopes)
            fe = q("fraction error", g, k).groupby("rep")["value"].mean()
            row_f[g] = med_rng(fe)
        slope_tab.append(row_s)
        frac_tab.append(row_f)
    rank_tab, slope_tab, frac_tab = (pd.DataFrame(x) for x in (rank_tab, slope_tab, frac_tab))

    ratio_tabs = {}
    for g in state_gens:
        rt = q("contrast ratio", g).groupby(["stratum", "prediction"])["value"].median().unstack()
        rt = rt.reindex(index=types, columns=PREDS)
        rt.insert(0, "true contrast", true_c[g])
        rt.insert(1, "median N", med_N)
        ratio_tabs[g] = rt

    # X8: under G2, LOG-O and LOG-S further from 1 than MIX (median |log ratio| over
    # stable types), with the largest distortion in small, low-depth types
    x8_txt, x8 = "G2 not run", "not judged"
    if "G2" in ratio_tabs:
        rt = ratio_tabs["G2"].loc[stable["G2"]]
        dist = {k: float(np.median(np.abs(np.log(np.clip(rt[k].abs(), 1e-6, None))))) for k in PREDS}
        worst = {k: rt[k].apply(lambda v: abs(np.log(max(abs(v), 1e-6)))).idxmax() for k in ("LOG-O", "LOG-S")}
        further = dist["LOG-O"] > dist["MIX pi"] and dist["LOG-S"] > dist["MIX pi"]
        low = all(worst[k] in low_depth for k in worst)
        x8 = "holds" if further and low else ("partly holds" if further or low else "does not hold")
        x8_txt = (f"median |log ratio| over the {len(stable['G2'])} types with a true contrast of at "
                  f"least {MIN_TRUE}: MIX pi {dist['MIX pi']:.3f}, LOG-O {dist['LOG-O']:.3f}, "
                  f"LOG-S {dist['LOG-S']:.3f} ({'both further from 1' if further else 'not both further from 1'}); "
                  f"largest distortion in {worst['LOG-O']} (LOG-O) and {worst['LOG-S']} (LOG-S); low-depth types "
                  f"(median N below the overall {df['N'].median():.0f}): {', '.join(sorted(low_depth))}")

    g3_tab = None
    if "G3" in gens:
        rows = []
        for k in PREDS:
            fr = q("fraction", "G3", k).groupby("stratum")["value"].median().reindex(types)
            rows.append({"prediction": k, **{t: fr[t] for t in types}})
        g3_tab = pd.DataFrame(rows)
    label_tab = []
    for g in gens:
        for k in PREDS:
            s = q("label: log loss", g, k)
            if len(s):
                label_tab.append({"generator": g, "prediction": k, "log loss": med_rng(s["value"]),
                                  "AUPRC": med_rng(q("label: AUPRC", g, k)["value"]),
                                  "mean p / rate": med_rng(q("label: mean p / rate", g, k)["value"])})
    label_tab = pd.DataFrame(label_tab)

    # ---- figure: the headline neighbourhood-effect ratio, by arm and generator ----
    show = ["MIX pi", "NB-c, BASE-O", "MIX, BASE-O", "LOG-T", "LOG-O", "LOG-S"]
    fig, axes = plt.subplots(1, len(state_gens) + 1, figsize=(6.2 * (len(state_gens) + 1), 5.6),
                             gridspec_kw={"width_ratios": [1.2] * len(state_gens) + [1]})
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, state_gens):
        rt = ratio_tabs[g]
        order = rt["true contrast"].abs().sort_values().index.tolist()
        yk = np.arange(len(order))
        for j, k in enumerate(show):
            vals = rt.loc[order, k].to_numpy()
            ax.plot(vals, yk + (j - len(show) / 2) * 0.09, MK[k], color=COL[k], ms=6, label=k)
        ax.axvline(1, color="#0b0b0b", lw=0.8, ls="--")
        ax.axvline(0, color="#8a8984", lw=0.6)
        ax.set_yticks(yk)
        ax.set_yticklabels([f"{t}{'' if t in stable[g] else ' (weak truth)'}" for t in order], fontsize=8.5)
        ax.set_xlabel("predicted / true neighbourhood effect (logit, tumour-rich vs tumour-poor)")
        ax.set_title(f"{g}: neighbourhood-effect ratio, median over replicates")
        ax.grid(True, color="#e4e3df", lw=0.6)
    axes[0].legend(fontsize=7.5, loc="lower right")
    ax = axes[-1]
    for gi_, g in enumerate(state_gens):
        for j, k in enumerate(PREDS):
            v = q("spearman", g, k)["value"].to_numpy()
            ax.plot(np.full(len(v), j + (gi_ - 0.5) * 0.25), v, MK[k], color=COL[k], ms=4,
                    alpha=0.8 if g == "G2" else 0.4, mfc="none" if g == "G1" else COL[k])
    ax.set_xticks(np.arange(len(PREDS)))
    ax.set_xticklabels(PREDS, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Spearman with pi_true (held-out fold)")
    ax.set_title("rank agreement (open: G1, filled: G2)")
    ax.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sim_recovery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    say(conv.round(3).to_string(index=False))
    say(rank_tab.to_string(index=False))
    say(slope_tab.to_string(index=False))
    say(frac_tab.to_string(index=False))
    for g, rt in ratio_tabs.items():
        say(f"{g} ratios:\n{rt.round(3).to_string()}")
    say("X8: " + x8_txt)
    if a.no_report:
        return

    n_conv = {k: int(conv[f"{k} converged"].sum()) for k in ("NB-c", "MIX", "LOG-T", "LOG-O", "LOG-S")}
    with open(REPORT, "a") as fh:
        fh.write(f"""
### Stage 6 results ({time.strftime('%Y-%m-%d %H:%M')})

{len(res)} replicates ({', '.join(f"{g}: {sum(r['generator'] == g for r in res)}" for g in gens)}), each
on {res[0]['n_train'] + res[0]['n_test']:,} real cells (held-out fold about
{np.median([r['n_test'] for r in res]):,.0f} cells). Converged fits: {', '.join(f'{k} {v}/{len(res)}' for k, v in n_conv.items())};
largest LOG sd(eta) {conv['LOG max sd(eta)'].max():.2f}. Values are the median
[min, max] over replicates.

**1. Rank agreement with pi_true** (Spearman, held-out fold):

{md_table(rank_tab)}

**2. Neighbourhood effect.** Pooled over the types with a true contrast of at
least {MIN_TRUE} in logit, the slope of predicted on true contrast (1 =
recovered, below 1 = attenuated):

{md_table(slope_tab)}

Per type, the ratio of predicted to true contrast (median over replicates):

""" + "\n\n".join(f"**{g}**\n\n" + md_table(rt.reset_index().rename(columns={"stratum": "type", "index": "type"}))
                    for g, rt in ratio_tabs.items()) + f"""

**3. Fraction error** (mean over types of |mean prediction - realised cycling
fraction|; only MIX's pi targets this quantity):

{md_table(frac_tab)}

**G3 (no latent classes): implied cycling fraction by type** (median over
replicates; the truth has no cycling class):

{md_table(g3_tab) if g3_tab is not None else 'not run'}

**Label-scale scores, held-out fold:**

{md_table(label_tab)}

**X8** (under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from
1 than MIX, with the largest distortion in small, low-depth types), judged
clause by clause: {x8_txt}. **{x8}**.

Figure: `sim_recovery.png`, the neighbourhood-effect ratio by arm and
generator, and the rank agreement. Every replicate value is in
`sim_recovery.csv`. **Review point** (stages 1, 3 and 6 pause for review).
""")
    say(f"report appended")


if __name__ == "__main__":
    main()
