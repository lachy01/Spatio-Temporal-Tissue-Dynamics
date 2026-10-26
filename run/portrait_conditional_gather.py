"""Collect the conditional-slice sensitivity and bootstrap jobs (Spartan).

    python run/portrait_conditional_gather.py [--hpc outputs/hpc] [--out <dir>]

Reads <hpc>/portrait_cond_sens/*.json and <hpc>/portrait_cond_boot/*.json
(src/tissue_dynamics/portrait/conditional.py). The "base" sensitivity tasks are the
point estimate (they reproduce fixed_points_<bound>.csv) and carry the slice grids
and support masks every replicate is read on.

Matching: a replicate recovers a point-estimate fixed point when it has a fixed
point of the same class within 1 log2 unit (Euclidean on the log2 axes, a factor of
2 in density) -- in support or not; "in support" is then reported separately.

Writes outputs/portrait/conditional_slices/
  sensitivity.csv                every variant x bound x priority pair: fixed points,
                                 and whether each point-estimate fixed point recurs
  sensitivity_<bound>.png        nullclines per pair under the bandwidth, conditioning
                                 set, anchor and radius variants (--simple-sensitivity:
                                 sensitivity_<bound>_simple.png, short title, bandwidth as h)
  all_pairs.csv                  every one of the 36 pairs at the base settings
  all_pairs_<bound>.png          the 36 conditional slices as an upper-triangle sheet
  bootstrap_fixed_points.csv     per source x bound x point-estimate fixed point:
                                 recurrence, in-support share, location 95% range
  bootstrap_topology.csv         per source x bound x pair: share of replicates with
                                 the point estimate's in-support classes, share with
                                 any in-support saddle / stable point, median circular
                                 SD of arrow direction inside support
  bootstrap_<bound>.png          per pair x source: point-estimate nullclines over 40
                                 replicates' nullclines, and every replicate's fixed
                                 points
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402

OUT = "outputs/portrait/conditional_slices"
BOUNDS = ("none", "osdr")
TAG = {"none": "unbounded", "osdr": "OSDR bound, p = 8"}
SOURCES = {"param": "parametric (sandwich draws)", "block": "spatial block (refit + surface)",
           "ref": "surface only (block-resampled reference)"}
MARK = {"stable": "o", "saddle": "s", "unstable": "^"}
C_A, C_B, INK, MUTED = "#2a78d6", "#e34948", "#1f1f1e", "#8a8985"
MATCH_LOG2 = 1.0
GROUPS = {"bandwidth (log2)": ["bw 0.10", "bw 0.15", "base", "bw 0.40", "bw 0.60"],
          "conditioning set": ["base", "both >= 2 cells", "zeros kept", "bins 320"],
          "death anchor": ["anchor x0.95", "base", "anchor x1.05"],
          "radius": ["radius 60 um", "base", "radius 100 um"]}
LABEL = {"base": "base (bw 0.25, both > 0, 160 bins, 80 um)"}


def label_h(v):
    """--simple legend labels: the bandwidth written as h."""
    if v == "base":
        return "base (h = 0.25, both > 0, 160 bins, 80 µm)"
    return f"h = {v[3:]}" if v.startswith("bw ") else v.replace(" um", " µm")


def say(*a):
    print(*a, flush=True)


def load(hpc, job):
    files = sorted(glob.glob(f"{hpc}/{job}/*.json"))
    say(f"{job}: {len(files)} results")
    return [json.load(open(f)) for f in files]


def log2xy(f):
    return np.log2([f["X_a"], f["X_b"]])


def match(f0, fps):
    """The nearest same-class fixed point within MATCH_LOG2, or None."""
    best, dbest = None, MATCH_LOG2
    for f in fps:
        if f["classification"] != f0["classification"]:
            continue
        d = float(np.hypot(*(log2xy(f) - log2xy(f0))))
        if d <= dbest:
            best, dbest = f, d
    return best


def fmt_fps(fps):
    return "; ".join(f"{f['classification']} ({f['X_a']:.1f}, {f['X_b']:.1f}){'' if f['in_support'] else ' out'}"
                     for f in fps) or "none"


def sup_key(fps):
    return tuple(sorted(f["classification"] for f in fps if f["in_support"]))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_slice(ax, s, lw=1.8, alpha=1.0, shade=True, ca=C_A, cb=C_B, fps=True, ls_b="solid"):
    La, Lb = np.log2(s["ga"]), np.log2(s["gb"])
    M = np.asarray(s["M"], bool)
    if shade:
        ax.pcolormesh(La, Lb, np.where(M, np.nan, 1.0), cmap="Greys", vmin=0, vmax=3, shading="auto",
                      alpha=.55, zorder=0)
    U, V = np.asarray(s["U"], float), np.asarray(s["V"], float)
    ax.contour(La, Lb, np.where(M, U, np.nan), [0], colors=[ca], linewidths=lw, alpha=alpha, zorder=3)
    ax.contour(La, Lb, np.where(M, V, np.nan), [0], colors=[cb], linewidths=lw, alpha=alpha, zorder=3,
               linestyles=ls_b)
    if fps:
        for f in s["fixed_points"]:
            x, y = log2xy(f)
            ax.plot(x, y, MARK[f["classification"]], ms=8, mfc=INK if f["classification"] == "stable" else "white",
                    mec=INK, mew=1.4, alpha=1.0 if f["in_support"] else 0.35, zorder=6)
    ax.set_xlim(La[0], La[-1])
    ax.set_ylim(Lb[0], Lb[-1])


# ----------------------------------------------------------------------------- sensitivity

def sensitivity(sens):
    by = {(r["variant"], r["bound"]): r for r in sens if r["variant"] != "all pairs"}
    rows = []
    for bd in BOUNDS:
        base = by.get(("base", bd))
        if base is None:
            raise SystemExit(f"the base sensitivity task for bound {bd!r} is missing")
        for v, r in sorted(((v, r) for (v, b), r in by.items() if b == bd), key=lambda t: t[0] != "base"):
            for pair, s in r["slices"].items():
                fps0 = base["slices"][pair]["fixed_points"]
                rec = [match(f0, s["fixed_points"]) for f0 in fps0]
                rows.append({"variant": v, "bound": bd, "pair": pair, "neighbourhoods conditioned on": s["n_cond"],
                             "fixed points": fmt_fps(s["fixed_points"]),
                             "in-support classes": ", ".join(sup_key(s["fixed_points"])) or "none",
                             "same in-support classes as base": sup_key(s["fixed_points"]) == sup_key(fps0),
                             "base fixed points recovered": "; ".join(
                                 f"{f0['classification']} ({f0['X_a']:.1f}, {f0['X_b']:.1f}): "
                                 + ("no" if m is None else f"({m['X_a']:.1f}, {m['X_b']:.1f})"
                                    + ("" if m["in_support"] else " out"))
                                 for f0, m in zip(fps0, rec)) or "-"})
        plot_sensitivity(by, bd)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/sensitivity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500):
        say(tab[["variant", "bound", "pair", "in-support classes", "same in-support classes as base",
                 "base fixed points recovered"]].to_string(index=False))
    return by


def plot_sensitivity(by, bd, simple=False):
    base = by[("base", bd)]
    pairs = list(base["slices"])
    fig, axes = plt.subplots(len(pairs), len(GROUPS), figsize=(5.0 * len(GROUPS), 4.6 * len(pairs)),
                             squeeze=False, layout="constrained")
    for c, (gname, vs) in enumerate(GROUPS.items()):
        vs = [v for v in vs if (v, bd) in by]
        vs = [v for v in vs if v != "base"] + ["base"]                # the base drawn last, in black
        cols = list(plt.get_cmap("viridis")(np.linspace(0.05, 0.85, max(len(vs) - 1, 1))))[:len(vs) - 1] + [INK]
        for p, pair in enumerate(pairs):
            ax = axes[p, c]
            draw_slice(ax, base["slices"][pair], lw=0, fps=False)              # support shading only
            for v, col in zip(vs, cols):
                s = by[(v, bd)]["slices"][pair]
                lw = 2.6 if v == "base" else 1.4
                draw_slice(ax, s, lw=lw, shade=False, ca=col, cb=col, fps=False, ls_b="dashed")
                for f in s["fixed_points"]:
                    x, y = log2xy(f)
                    ax.plot(x, y, MARK[f["classification"]], ms=8, mfc=col if f["classification"] == "stable"
                            else "white", mec=col, mew=1.6, alpha=1.0 if f["in_support"] else 0.35, zorder=6)
            a, b = pair.split(" x ")
            ax.set_xlabel(f"log2 {a} density")
            ax.set_ylabel(f"log2 {b} density")
            gtitle = gname.replace("bandwidth (log2)", "bandwidth h (log2)") if simple else gname
            ax.set_title(f"{pair}: {gtitle}", fontsize=10, loc="left")
            style(ax)
        axes[0, c].legend(handles=[Line2D([], [], color=col, lw=2.6 if v == "base" else 1.4,
                                          label=label_h(v) if simple else LABEL.get(v, v))
                                   for v, col in zip(vs, cols)],
                          fontsize=7.5, frameon=False, loc="upper left")
    if simple:
        fig.suptitle(f"Sensitivity of the conditional slices, {TAG[bd]}", x=0.01, ha="left", fontsize=12)
        fig.savefig(f"{OUT}/sensitivity_{bd}_simple.png", dpi=120)
        plt.close(fig)
        return
    fig.suptitle(f"Conditional slices, {TAG[bd]}: each variant's nullclines in its colour (one setting "
                 "changed from the base; base in black)\nsolid = x-nullcline, dashed = y-nullcline; circle stable, square saddle, triangle unstable; faded = outside "
                 "support; grey = outside the base support", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/sensitivity_{bd}.png", dpi=120)
    plt.close(fig)


def all_pairs(sens, by):
    rows = []
    for bd in BOUNDS:
        sl = dict(by[("base", bd)]["slices"])
        for r in sens:
            if r["variant"] == "all pairs" and r["bound"] == bd:
                sl.update(r["slices"])
        for pair, s in sl.items():
            rows.append({"bound": bd, "pair": pair, "fixed points": fmt_fps(s["fixed_points"]),
                         "in-support classes": ", ".join(sup_key(s["fixed_points"])) or "none"})
        plot_all_pairs(sl, bd)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/all_pairs.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 200):
        say(tab.to_string(index=False))


def plot_all_pairs(sl, bd):
    names = json.load(open("data/mixture_cells_major.phi_names.json"))
    K = len(names)
    fig, axes = plt.subplots(K - 1, K - 1, figsize=(2.6 * (K - 1), 2.5 * (K - 1)))
    for ax in axes.ravel():
        ax.set_axis_off()
    for i in range(K):
        for j in range(i + 1, K):
            key = f"{names[i]} x {names[j]}" if f"{names[i]} x {names[j]}" in sl else f"{names[j]} x {names[i]}"
            if key not in sl:
                continue
            s = sl[key]
            ax = axes[i, j - 1]
            ax.set_axis_on()
            if key.startswith(names[j]):                     # stored with the axes the other way round
                s = {**s, "ga": s["gb"], "gb": s["ga"], "U": np.asarray(s["V"]).T, "V": np.asarray(s["U"]).T,
                     "M": np.asarray(s["M"]).T, "fixed_points": [{**f, "X_a": f["X_b"], "X_b": f["X_a"]}
                                                              for f in s["fixed_points"]]}
            draw_slice(ax, s, lw=1.3)
            n_in = sum(f["in_support"] for f in s["fixed_points"])
            ax.set_title(f"{n_in} in support" if n_in else "", fontsize=7.5, loc="right", color=INK)
            ax.tick_params(labelsize=6)
            if i == j - 1:
                ax.set_xlabel(f"log2 {names[j]}", fontsize=8)
                ax.set_ylabel(f"log2 {names[i]}", fontsize=8)
    fig.suptitle(f"All 36 conditional slices at the base settings, {TAG[bd]}: row = y-axis type, column = "
                 "x-axis type\nblue = x-nullcline, red = y-nullcline; circle stable, square saddle, triangle "
                 "unstable (faded = outside support); grey = outside support", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/all_pairs_{bd}.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- bootstrap

def circ_sd(theta):
    R = np.abs(np.mean(np.exp(1j * theta), axis=0))
    return np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))))


def bootstrap(boot, by):
    reps = defaultdict(list)                       # source -> replicate dicts
    for r in boot:
        reps[r["source"]] += r["replicates"]
    say("replicates: " + ", ".join(f"{s} {len(v)}" for s, v in reps.items()))
    rows_fp, rows_top = [], []
    for bd in BOUNDS:
        base = by[("base", bd)]["slices"]
        for src, R in reps.items():
            for pair, s0 in base.items():
                fps0 = s0["fixed_points"]
                key0 = sup_key(fps0)
                keys = [sup_key(r["slices"][bd][pair]["fixed_points"]) for r in R]
                M = np.asarray(s0["M"], bool)
                th = np.array([np.arctan2(np.asarray(r["slices"][bd][pair]["V"]),
                                          np.asarray(r["slices"][bd][pair]["U"])) for r in R])
                sd = circ_sd(th)[M]
                common = Counter(keys).most_common(3)
                rows_top.append({"source": src, "bound": bd, "pair": pair, "replicates": len(R),
                                 "point in-support classes": ", ".join(key0) or "none",
                                 "share same": float(np.mean([k == key0 for k in keys])),
                                 "share with in-support saddle": float(np.mean(["saddle" in k for k in keys])),
                                 "share with in-support stable": float(np.mean(["stable" in k for k in keys])),
                                 "share with in-support unstable": float(np.mean(["unstable" in k for k in keys])),
                                 "commonest in-support sets": "; ".join(f"[{', '.join(k) or 'none'}] {c / len(R):.0%}"
                                                                         for k, c in common),
                                 "median circular SD in support (deg)": float(np.median(sd)) if len(sd) else np.nan})
                for f0 in fps0:
                    m = [match(f0, r["slices"][bd][pair]["fixed_points"]) for r in R]
                    hit = [x for x in m if x is not None]
                    xa, xb = np.array([x["X_a"] for x in hit]), np.array([x["X_b"] for x in hit])
                    q = lambda v: f"{np.percentile(v, 2.5):.1f} to {np.percentile(v, 97.5):.1f}" if len(v) else "-"  # noqa: E731
                    rows_fp.append({"source": src, "bound": bd, "pair": pair, "class": f0["classification"],
                                    "X_a": f0["X_a"], "X_b": f0["X_b"], "in support (point)": f0["in_support"],
                                    "share recovered": len(hit) / len(R),
                                    "share recovered in support": float(np.mean([x["in_support"] for x in hit]))
                                    * len(hit) / len(R) if hit else 0.0,
                                    "X_a 95%": q(xa), "X_b 95%": q(xb)})
        plot_bootstrap(reps, base, bd)
    a, b = pd.DataFrame(rows_fp), pd.DataFrame(rows_top)
    a.to_csv(f"{OUT}/bootstrap_fixed_points.csv", index=False)
    b.to_csv(f"{OUT}/bootstrap_topology.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        say(a.round(3).to_string(index=False))
        say(b.round(3).to_string(index=False))


def plot_bootstrap(reps, base, bd, n_lines=40):
    srcs = [s for s in SOURCES if s in reps]
    pairs = list(base)
    fig, axes = plt.subplots(len(pairs), len(srcs), figsize=(5.0 * len(srcs), 4.6 * len(pairs)),
                             squeeze=False, layout="constrained")
    for c, src in enumerate(srcs):
        R = reps[src]
        pick = np.random.default_rng(0).choice(len(R), size=min(n_lines, len(R)), replace=False)
        for p, pair in enumerate(pairs):
            ax = axes[p, c]
            s0 = base[pair]
            draw_slice(ax, s0, lw=0, fps=False)
            for j in pick:
                s = {**s0, "U": R[j]["slices"][bd][pair]["U"], "V": R[j]["slices"][bd][pair]["V"]}
                draw_slice(ax, s, lw=0.6, alpha=0.18, shade=False, fps=False)
            for r in R:
                for f in r["slices"][bd][pair]["fixed_points"]:
                    x, y = log2xy(f)
                    ax.plot(x, y, MARK[f["classification"]], ms=3.5, mfc="none", mec=MUTED,
                            mew=0.6, alpha=0.6 if f["in_support"] else 0.2, zorder=5)
            draw_slice(ax, s0, lw=2.4, shade=False)
            a, b = pair.split(" x ")
            ax.set_xlabel(f"log2 {a} density")
            ax.set_ylabel(f"log2 {b} density")
            ax.set_title(f"{pair}\n{SOURCES[src]}, {len(R)} replicates", fontsize=10, loc="left")
            style(ax)
    fig.suptitle(f"Conditional slices, {TAG[bd]}: point estimate (bold) over {n_lines} replicates' nullclines "
                 "(thin); every replicate's fixed points in grey\nblue = x-nullcline, red = y-nullcline; "
                 "circle stable, square saddle, triangle unstable; grey = outside support",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/bootstrap_{bd}.png", dpi=120)
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpc", default="outputs/hpc")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--simple-sensitivity", action="store_true",
                    help="only redraw sensitivity_<bound>_simple.png (short title, bandwidth as h)")
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    sens = load(a.hpc, "portrait_cond_sens")
    if a.simple_sensitivity:
        by = {(r["variant"], r["bound"]): r for r in sens if r["variant"] != "all pairs"}
        for bd in BOUNDS:
            plot_sensitivity(by, bd, simple=True)
        return
    by = sensitivity(sens)
    all_pairs(sens, by)
    boot = load(a.hpc, "portrait_cond_boot")
    if boot:
        bootstrap(boot, by)


if __name__ == "__main__":
    main()
