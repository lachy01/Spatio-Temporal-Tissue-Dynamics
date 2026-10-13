"""Division classifier, addendum stage A5: collate the simulation extension.

    python run/division_addendum_a5_gather.py [--dir outputs/hpc/division_sim_addendum] [--no-report]

Reads the Spartan array `division_sim_addendum` (src/tissue_dynamics/division/
simulation.py with addendum=True: G1, G2, G3 x 10 replicates, n = 100,000, the
same draws as Stage 6's `division_sim`) and reports:

* Section 2.3: the alpha_B surface -- for every null (naive; truncated at
  alpha_trim 0.001, 0.01, 0.05) and alpha_B in {0.001, 0.01, 0.05, 0.1, 0.2},
  the median over replicates of LOG-T's pooled neighbourhood-effect slope under
  G1 and G2 (slope through the origin of predicted on true per-type logit
  contrast, over types whose median true contrast is at least 0.1 in absolute
  value, as Stage 6). NB-c, MIX, TWO-P and TWO-PNB pushed through every grid
  label are shown beside it but do not enter the rule. Then the pre-registered
  rule: the (null, alpha_B) pair maximising LOG-T's median G2 slope, ties within
  0.02 going to the smaller alpha_B.
* Section 3.5: TWO-P and TWO-PNB against MIX and LOG-T -- rank agreement with
  pi_true, pooled slope, fraction error (G1, G2), and under G3 TWO-PNB's implied
  cycling fraction per type beside MIX's.
* Section 6, for every arm: B direction recovery (cosine of vec(B-hat) with
  vec(B_true)) and the per-type ratio spread, median and max of
  |log(predicted / true contrast)| over the same types (a sign flip counts as
  infinite).
* A check that each replicate reproduces Stage 6's draws: its label rates and
  the shared arms' rank agreement against `division_sim`'s, where present.

X11, X13 and X14's G3 clause are judged. X15 waits on A4.
Writes sim_recovery_addendum.csv, sim_alpha_surface.png and a section of report.md.
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

from tissue_dynamics.division import labels as L                      # noqa: E402

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
MIN_TRUE, TIE = 0.1, 0.02
NULLS = ["naive"] + [f"trunc {a:g}" for a in L.TRUNC_GRID]
GRID = list(L.ALPHA_B_GRID_EXT)
STATE_ARMS = ["MIX pi", "TWO-P pi", "TWO-PNB pi", "LOG-T", "LOG-O", "LOG-S",
              "MIX, BASE-T", "TWO-P, BASE-T", "TWO-PNB, BASE-T", "NB-c, BASE-T"]
MODELS = ["MIX", "TWO-P", "TWO-PNB", "NB-c", "LOG-T", "LOG-O", "LOG-S"]
COL = {"naive": "#52514e", "trunc 0.001": "#2a78d6", "trunc 0.01": "#eb6834", "trunc 0.05": "#1baf7a"}
INK2 = "#52514e"


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def md_table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def med_rng(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return ""
    f = lambda x: "inf" if np.isinf(x) else f"{x:.3f}"            # noqa: E731
    return f"{f(np.median(v))} [{f(v.min())}, {f(v.max())}]"


def gkey(null, a):
    return f"{null}|{a:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/hpc/division_sim_addendum")
    ap.add_argument("--stage6", default="outputs/hpc/division_sim")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "0*.json")))]
    res = [r for r in res if r.get("addendum")]
    if not res:
        sys.exit(f"no addendum results in {a.dir}")
    types = json.load(open("data/mixture_cells_major.phi_names.json"))
    gens = sorted({r["generator"] for r in res})
    sgens = [g for g in gens if g in ("G1", "G2")]
    has_state = any("LOG-O-state" in r["b_cosine"] for r in res)
    state_arms = STATE_ARMS + (["LOG-O-state"] if has_state else [])
    models = MODELS + (["LOG-O-state"] if has_state else [])
    say(f"{len(res)} replicates: " + ", ".join(f"{g} {sum(r['generator'] == g for r in res)}" for g in gens))

    # ---- pairing with Stage 6 ----------------------------------------------------------
    s6 = {(r["generator"], r["rep"]): r for r in
          (json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.stage6, "0*.json"))))}
    pair_rows = []
    for r in res:
        o = s6.get((r["generator"], r["rep"]))
        if o is None or o["n_train"] != r["n_train"]:
            continue
        d_rate = max(abs(r["label_rates"][k] - o["label_rates"][k]) for k in o["label_rates"])
        d_sp = (max(abs(r["spearman_with_pi_true"][k] - o["spearman_with_pi_true"][k])
                    for k in o["spearman_with_pi_true"]) if "spearman_with_pi_true" in o else np.nan)
        pair_rows.append({"generator": r["generator"], "rep": r["rep"], "max |label rate diff|": d_rate,
                          "max |Spearman diff|, Stage 6 arms": d_sp})
    pair = pd.DataFrame(pair_rows)

    # ---- long table -------------------------------------------------------------------------
    long = []
    for r in res:
        g, rep = r["generator"], r["rep"]
        for k, v in r["b_cosine"].items():
            long.append({"generator": g, "rep": rep, "metric": "B cosine", "prediction": k,
                         "stratum": "all", "value": v})
        for k, fr in r["fractions"].items():
            for t, v in fr.items():
                long.append({"generator": g, "rep": rep, "metric": "fraction", "prediction": k,
                             "stratum": t, "value": v})
        for k, v in r["label_scores"].items():
            for m in ("log loss", "AUPRC", "mean p / rate"):
                long.append({"generator": g, "rep": rep, "metric": f"label: {m}", "prediction": k,
                             "stratum": "all", "value": v[m]})
        if "spearman_with_pi_true" not in r:
            continue
        for k, v in r["spearman_with_pi_true"].items():
            long.append({"generator": g, "rep": rep, "metric": "spearman", "prediction": k,
                         "stratum": "all", "value": v})
        for t in types:
            long.append({"generator": g, "rep": rep, "metric": "true fraction", "prediction": "truth",
                         "stratum": t, "value": r["true_fraction"][t]})
        for t, row in r["neighbourhood_contrast"].items():
            long.append({"generator": g, "rep": rep, "metric": "true contrast", "prediction": "truth",
                         "stratum": t, "value": row["true"]})
            for k, v in row.items():
                if k in ("true", "n_low", "n_high"):
                    continue
                long.append({"generator": g, "rep": rep, "metric": "contrast", "prediction": k,
                             "stratum": t, "value": v})
    Lg = pd.DataFrame(long)
    Lg.to_csv(f"{OUT}/sim_recovery_addendum.csv", index=False)

    def sub(metric, g, pred=None):
        s = Lg[(Lg["metric"] == metric) & (Lg["generator"] == g)]
        return s if pred is None else s[s["prediction"] == pred]

    true_c = {g: sub("true contrast", g).groupby("stratum")["value"].median().reindex(types) for g in sgens}
    stable = {g: [t for t in types if abs(true_c[g][t]) >= MIN_TRUE] for g in sgens}

    def per_rep(g, pred):
        """{rep: (slope, median |log ratio|, max |log ratio|)} over stable types."""
        out = {}
        tc_all = sub("true contrast", g)
        for rep, grp in sub("contrast", g, pred).groupby("rep"):
            tc = tc_all[tc_all["rep"] == rep].set_index("stratum")["value"][stable[g]]
            pc = grp.set_index("stratum")["value"][stable[g]]
            ratio = (pc / tc).to_numpy()
            lr = np.where(ratio > 0, np.abs(np.log(np.where(ratio > 0, ratio, 1.0))), np.inf)
            out[rep] = (float(np.sum(pc * tc) / np.sum(tc ** 2)), float(np.median(lr)), float(np.max(lr)))
        return out

    def slope_med(g, pred, j=0):
        v = [x[j] for x in per_rep(g, pred).values()]
        return float(np.median(v)) if v else np.nan

    # ---- alpha_B surface ------------------------------------------------------------------------
    surf = []
    for g in sgens:
        for null in NULLS:
            for ab in GRID:
                k = gkey(null, ab)
                row = {"generator": g, "null": null, "alpha_B": ab}
                for arm, pred in (("LOG-T", f"LOG-T [{k}]"), ("NB-c", f"NB-c, BASE-T [{k}]"),
                                  ("MIX", f"MIX, BASE-T [{k}]"), ("TWO-P", f"TWO-P, BASE-T [{k}]"),
                                  ("TWO-PNB", f"TWO-PNB, BASE-T [{k}]")):
                    pr = per_rep(g, pred)
                    sl = [x[0] for x in pr.values()]
                    row[f"{arm} slope"] = float(np.median(sl)) if sl else np.nan
                    if arm == "LOG-T":
                        row["LOG-T slope range"] = med_rng(sl)
                        row["LOG-T spread (median |log ratio|)"] = float(np.median([x[1] for x in pr.values()]))
                        bc = sub("B cosine", g, f"LOG-T [{k}]")["value"]
                        row["LOG-T B cosine"] = float(np.median(bc)) if len(bc) else np.nan
                surf.append(row)
    surf = pd.DataFrame(surf)

    # the pre-registered rule (section 2.3)
    chosen, rule_txt = None, "G2 not run"
    if "G2" in sgens:
        s2 = surf[surf["generator"] == "G2"].dropna(subset=["LOG-T slope"])
        top = s2["LOG-T slope"].max()
        near = s2[s2["LOG-T slope"] >= top - TIE]
        smallest = near["alpha_B"].min()
        cand = near[near["alpha_B"] == smallest].sort_values("LOG-T slope", ascending=False)
        chosen = cand.iloc[0]
        rule_txt = (f"Largest median G2 slope {top:.3f}; {len(near)} (null, alpha_B) pairs lie within "
                    f"{TIE} of it; the smallest alpha_B among them is {smallest:g}, and there the "
                    f"highest is {chosen['null']} ({chosen['LOG-T slope']:.3f}). **Primary: alpha_B "
                    f"{chosen['alpha_B']:g}, null {chosen['null']}.**")
        if top > 1:
            # The rule was written expecting attenuation (slopes below 1, X11). A slope
            # above 1 is over-recovery, not better recovery; the rule is applied as
            # registered and the case is flagged for Lachy, with the pair whose slope
            # is closest to 1 shown beside it. Not a substitute for the rule.
            near1 = s2.iloc[(s2["LOG-T slope"] - 1).abs().argsort()].iloc[0]
            rule_txt += (f"\n\n**Flag for Lachy:** the maximum exceeds 1, i.e. LOG-T over-states the "
                         f"neighbourhood effect there (median per-type spread "
                         f"{chosen['LOG-T spread (median |log ratio|)']:.3f}). The rule assumed "
                         f"attenuation. For comparison only, the pair closest to a slope of 1 is "
                         f"{near1['null']} at alpha_B {near1['alpha_B']:g} ({near1['LOG-T slope']:.3f}, "
                         f"spread {near1['LOG-T spread (median |log ratio|)']:.3f}). Which governs is "
                         f"Lachy's decision; the registered rule's pick stands until then.")

    # X11
    x11_rows, x11 = [], "not judged"
    if "G2" in sgens:
        ok_all = True
        for null in NULLS:
            s = surf[(surf["generator"] == "G2") & (surf["null"] == null)].set_index("alpha_B")["LOG-T slope"]
            peak_a, peak = float(s.idxmax()), float(s.max())
            ok = 0.05 <= peak_a <= 0.1 and peak < 0.8
            ok_all &= ok
            x11_rows.append(f"{null}: peak {peak:.3f} at alpha_B {peak_a:g}")
        x11 = "holds" if ok_all else "does not hold"

    # ---- state recovery, TWO against MIX and LOG-T --------------------------------------------------
    rec = []
    for arm in state_arms:
        row = {"prediction": arm}
        for g in sgens:
            row[f"{g} Spearman"] = med_rng(sub("spearman", g, arm)["value"])
            pr = per_rep(g, arm)
            row[f"{g} slope"] = med_rng([x[0] for x in pr.values()])
            row[f"{g} spread median"] = med_rng([x[1] for x in pr.values()])
            row[f"{g} spread max"] = med_rng([x[2] for x in pr.values()])
            fe = []
            for rep, grp in sub("fraction", g, arm).groupby("rep"):
                tf = sub("true fraction", g)
                tf = tf[tf["rep"] == rep].set_index("stratum")["value"]
                fe.append(float(np.mean(np.abs(grp.set_index("stratum")["value"][types] - tf[types]))))
            row[f"{g} fraction error"] = med_rng(fe)
        rec.append(row)
    rec = pd.DataFrame(rec)
    bcos = pd.DataFrame([{"model": m, **{g: med_rng(sub("B cosine", g, m)["value"]) for g in gens}}
                         for m in models])

    # X13
    x13, x13_txt = "not judged", ""
    if sgens:
        sp = {g: {k: float(np.median(sub("spearman", g, k)["value"])) for k in ("TWO-P pi", "MIX pi")} for g in sgens}
        rank_ok = all(abs(sp[g]["TWO-P pi"] - sp[g]["MIX pi"]) <= 0.05 for g in sgens)
        best_log = {g: float(surf[surf["generator"] == g]["LOG-T slope"].max()) for g in sgens}
        sl = {g: {k: slope_med(g, k) for k in ("TWO-P pi", "MIX pi")} for g in sgens}
        between = all(min(best_log[g], sl[g]["MIX pi"]) < sl[g]["TWO-P pi"] < max(best_log[g], sl[g]["MIX pi"])
                      for g in sgens)
        x13 = "holds" if rank_ok and between else ("partly holds" if rank_ok or between else "does not hold")
        x13_txt = "; ".join(
            f"{g}: Spearman TWO-P {sp[g]['TWO-P pi']:.3f} vs MIX {sp[g]['MIX pi']:.3f}, slope TWO-P "
            f"{sl[g]['TWO-P pi']:.3f} vs LOG-T best {best_log[g]:.3f} and MIX {sl[g]['MIX pi']:.3f}"
            for g in sgens)

    # G3: implied cycling fractions
    g3_tab, x14c = None, "not judged"
    if "G3" in gens:
        g3 = []
        for t in types:
            row = {"type": t}
            for k in ("MIX pi", "TWO-P pi", "TWO-PNB pi"):
                v = Lg[(Lg["metric"] == "fraction") & (Lg["generator"] == "G3") & (Lg["prediction"] == k)
                       & (Lg["stratum"] == t)]["value"]
                row[k] = float(np.median(v))
            g3.append(row)
        g3_tab = pd.DataFrame(g3)
        x14c = "holds" if bool((g3_tab["TWO-PNB pi"] >= g3_tab["MIX pi"]).all()) else "does not hold"

    # X15 (only if A4 ran)
    x15 = "waits on A4 (BASE-O-state), which runs only if Lachy approves after A3"
    if has_state and "G2" in sgens and chosen is not None:
        d_slope = slope_med("G2", "LOG-O-state") - slope_med("G2", "LOG-O")
        prim = f"LOG-T [{gkey(chosen['null'], chosen['alpha_B'])}]"
        sp_state, sp_t = slope_med("G2", "LOG-O-state", 1), slope_med("G2", prim, 1)
        ok = d_slope < 0.1 and sp_state >= sp_t
        x15 = (f"G2 slope LOG-O-state minus LOG-O {d_slope:+.3f}; median spread LOG-O-state "
               f"{sp_state:.3f} against LOG-T at the primary alpha_B {sp_t:.3f}: "
               f"**{'holds' if ok else 'does not hold'}**")

    conv = pd.DataFrame([{"generator": r["generator"], "rep": r["rep"],
                          "all fits converged": all(v.get("converged", True) for v in r["fits"].values()),
                          "LOG max sd(eta)": max(v["sd_eta"] for k, v in r["fits"].items()
                                                 if k.startswith("LOG")),
                          "TWO-PNB starts agree": (max(r["fits"]["TWO-PNB"]["step1_objectives"])
                                                   - min(r["fits"]["TWO-PNB"]["step1_objectives"])) <=
                                                  1e-4 * abs(min(r["fits"]["TWO-PNB"]["step1_objectives"])),
                          "TWO-PNB alpha1": r["fits"]["TWO-PNB"]["alpha1"],
                          "minutes": r["wall_time"] / 60} for r in res])

    # ---- figure ------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(len(sgens), 1), figsize=(6.2 * max(len(sgens), 1), 4.6), sharey=True,
                             squeeze=False)
    x = np.arange(len(GRID))
    for ax, g in zip(axes[0], sgens):
        for null in NULLS:
            s = surf[(surf["generator"] == g) & (surf["null"] == null)]
            ax.plot(x, s["LOG-T slope"], "o-", color=COL[null], lw=2, ms=6, label=f"LOG-T, {null}")
        mix = surf[(surf["generator"] == g) & (surf["null"] == "naive")]["MIX slope"]
        ax.plot(x, mix, "s--", color="#898781", lw=1.2, ms=4, label="MIX pushed through (naive)")
        ax.axhline(slope_med(g, "MIX pi"), color="#0b0b0b", lw=1, ls=":", label="MIX pi")
        ax.axhline(1, color="#c3c2bd", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in GRID])
        ax.set_xlabel("alpha_B")
        ax.set_title(f"{g}: pooled neighbourhood slope (median over replicates)", fontsize=10)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0, 0].set_ylabel("slope of predicted on true contrast (1 = recovered)", color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sim_alpha_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    say(surf.round(3).to_string(index=False))
    say(rule_txt)
    if a.no_report:
        return
    surf_show = surf[["generator", "null", "alpha_B", "LOG-T slope", "LOG-T slope range",
                      "LOG-T spread (median |log ratio|)", "LOG-T B cosine", "NB-c slope", "MIX slope",
                      "TWO-P slope", "TWO-PNB slope"]]
    report(f"""
## Addendum stage A5 -- simulation extension ({time.strftime('%Y-%m-%d %H:%M')})

Spartan job `division_sim_addendum`: {len(res)} replicates
({', '.join(f"{g} {sum(r['generator'] == g for r in res)}" for g in gens)}), n = {res[0]['n_train'] + res[0]['n_test']:,} cells each,
the same cells, truth and Stage 6 labels as `division_sim` (same seeds), plus the
extended BASE-T grid for the naive null and the truncated null refitted on the
simulated y at each alpha_trim, LOG-T on every grid label, TWO-P and TWO-PNB,
and the section 6 metrics. Every arm fitted on four folds and scored on the
fifth. Medians over replicates [min, max]. The slope is that of predicted on true
per-type logit contrast through the origin, over types whose median true contrast
is at least {MIN_TRUE} (G1: {len(stable.get('G1', []))} types; G2: {len(stable.get('G2', []))}).

### Fits

{md_table(conv)}

{"Pairing with Stage 6 (same draws): largest label-rate difference " + f"{pair['max |label rate diff|'].max():.1e}" + ", largest Spearman difference on the shared arms " + f"{pair['max |Spearman diff|, Stage 6 arms'].max():.1e}" + "." if len(pair) else "Stage 6 results not found beside these, so pairing was not checked."}

### The alpha_B surface (section 2.3)

{md_table(surf_show)}

Figure: `sim_alpha_surface.png`.

**Rule (pre-registered):** {rule_txt}

### State recovery (G1, G2)

Spread = |log(predicted / true contrast)| over the same types, median and max within a
replicate (a sign flip counts as infinite).

{md_table(rec)}

### B direction recovery (cosine of vec(B-hat) with vec(B_true))

B_true is the production MIX's U V' under G1 and G2 and the full-data NB-c's under G3.
LOG-T here is the naive alpha_B 0.05 label; every grid label's is in the surface table
and `sim_recovery_addendum.csv`.

{md_table(bcos)}

{"### G3: implied cycling fractions (median over replicates)" + chr(10) + chr(10) + md_table(g3_tab) if g3_tab is not None else ""}

### Pre-stated expectations

- **X11** (under G2, LOG-T's slope peaks at alpha_B 0.05-0.1 for every null, below 0.8):
  {'; '.join(x11_rows)}. **{x11}**.
- **X13** (TWO-P's rank agreement within 0.05 of MIX's under G1 and G2; its slope strictly
  between LOG-T's best and MIX's): {x13_txt}. **{x13}**.
- **X14**, G3 clause (TWO-PNB's implied cycling fraction at least MIX's in every type): **{x14c}**.
- **X15** (LOG-O-state's G2 slope improves on LOG-O's by less than 0.1; its spread not
  smaller than LOG-T's at the primary alpha_B): {x15}.

Every replicate, metric and prediction is in `sim_recovery_addendum.csv`. Collated in
{(time.time() - t0) / 60:.1f} min. **Review point** (A1, A3, A5 and A6 pause for review).
""")


if __name__ == "__main__":
    main()
