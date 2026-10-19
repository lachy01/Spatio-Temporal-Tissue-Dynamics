"""Gather the portrait uncertainty jobs (portrait spec 8.3).

    python run/portrait_uncertainty_gather.py

Reads outputs/hpc/portrait_param_boot/*.json (spec 8.1, MVN draws) and
outputs/hpc/portrait_block_boot/*.json (spec 8.2, block-bootstrap refits), and the
point estimate's summary (recomputed here with the same code). For each source and
state (barycentric, counts):

  topology preservation   share of replicates whose in-support (slice, class) key
                          equals the point estimate's -- the spec's headline number;
                          also for all interior fixed points, in or out of support
  fixed-point scatter     each point-estimate interior fixed point matched to the
                          nearest one in the same slice of each replicate (log
                          distance): share found, share with the same class, and the
                          2.5-97.5% range of its total density n
  anchors                 2.5-97.5% of pi_bar per archetype
  thresholds              per archetype, the share of replicates with a stable
                          ceiling / an unstable threshold at its own composition, and
                          the range of the first of each
  tumour ceiling          the composition path's ceiling range (min and max n)

Writes outputs/portrait/uncertainty/{report.md, summary.json}. Not done: the
per-grid-cell circular SD of arrow direction and the overlaid bootstrap arrows
(spec 8.3's figures), which need the replicates' velocity grids.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

OUT = "outputs/portrait/uncertainty"
STATES = ("barycentric", "counts")


def md(df, fmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r.to_numpy()) + " |")
    return "\n".join(out)


def load(job):
    reps = []
    for f in sorted(glob.glob(f"outputs/hpc/{job}/*.json")):
        r = json.load(open(f))
        if "draws" in r:
            reps += [d["summary"] for d in r["draws"]]
        elif "summary" in r:
            reps.append(r["summary"])
    return reps


def q(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (f"{np.percentile(v, 2.5):.3g} to {np.percentile(v, 97.5):.3g}" if len(v) else "none")


def analyse(point, reps, state):
    p = point[state]
    S = [r[state] for r in reps]
    out = {"n_replicates": len(S),
           "topology_preserved": float(np.mean([s["topology"] == p["topology"] for s in S])),
           "topology_interior_preserved": float(np.mean([s["topology_interior"] == p["topology_interior"]
                                                         for s in S]))}
    rows = []
    for f in [f for f in p["fixed_points"] if f["classification"] != "face"]:
        found, same, ns = 0, 0, []
        for s in S:
            cand = [g for g in s["fixed_points"] if g["slice"] == f["slice"] and g["q_ref"] == f["q_ref"]
                    and g["classification"] != "face"]
            if not cand:
                continue
            g = min(cand, key=lambda g: np.hypot(np.log(g["X_a"] / f["X_a"]), np.log(g["X_b"] / f["X_b"])))
            found += 1
            same += g["classification"] == f["classification"]
            ns.append(g["n"])
        rows.append({"slice": f"{f['slice']} q{f['q_ref']}", "class": f["classification"],
                     "in support": f["in_support"], "n (point)": f["n"],
                     "found": found / len(S), "same class": same / len(S), "n 95% range": q(ns)})
    out["fixed_points"] = rows
    names = list(p["pi_bar"])
    anc, thr = [], []
    for n in names:
        anc.append({"archetype": n, "point": p["pi_bar"][n], "95% range": q([s["pi_bar"][n] for s in S])})
        st = [s["own_sweep"].get(n, {}).get("crossings", []) for s in S]
        first = lambda cr, kind: next((c["n"] for c in cr if c["kind"] == kind), None)  # noqa: E731
        pc = p["own_sweep"].get(n, {}).get("crossings", [])
        thr.append({"archetype": n,
                    "point": "; ".join(f"{c['kind']} {c['n']:.0f}" for c in pc) or p["own_sweep"][n]["sign"],
                    "share with ceiling": float(np.mean([first(c, "stable") is not None for c in st])),
                    "ceiling n 95%": q([first(c, "stable") for c in st]),
                    "share with threshold": float(np.mean([first(c, "unstable") is not None for c in st])),
                    "threshold n 95%": q([first(c, "unstable") for c in st])})
    out["anchors"], out["thresholds"] = anc, thr
    cp = [s.get("ce_path", {}) for s in S]
    out["ce_path"] = {"point": f"{p['ce_path']['ceiling_min']}-{p['ce_path']['ceiling_max']}",
                      "ceiling min 95%": q([c.get("ceiling_min") for c in cp]),
                      "ceiling max 95%": q([c.get("ceiling_max") for c in cp]),
                      "share with folds": float(np.mean([c.get("folds", 0) > 0 for c in cp]))}
    return out


def main():
    from tissue_dynamics.mixture.data import load_config
    from tissue_dynamics.portrait.uncertainty import _setup, _summaries

    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    ctx = _setup(cfg)
    point = _summaries(ctx, ctx["pe"]["params"],
                       np.asarray(__import__("tissue_dynamics.mixture.model", fromlist=["x"]).responsibilities(
                           {k: np.asarray(v) for k, v in ctx["pe"]["params"].items()},
                           ctx["data"].as_jax(), ctx["spec"])))
    res, text = {}, ["# Portrait uncertainty (portrait spec 8)\n"]
    for job, label in (("portrait_param_boot", "Parametric bootstrap (spec 8.1, block-cluster sandwich)"),
                       ("portrait_block_boot", "Spatial block bootstrap with refits (spec 8.2)")):
        reps = load(job)
        if not reps:
            text.append(f"## {label}\n\nno results in outputs/hpc/{job}/\n")
            continue
        text.append(f"## {label}: {len(reps)} replicates\n")
        for st in STATES:
            a = analyse(point, reps, st)
            res[f"{job}|{st}"] = a
            text.append(f"### {st} state\n\nTopology preserved (in-support fixed points, spec 8.3 headline): "
                        f"**{100 * a['topology_preserved']:.0f}%**; all interior fixed points: "
                        f"{100 * a['topology_interior_preserved']:.0f}%. Point estimate's in-support key: "
                        f"{point[st]['topology'] or 'none'}.\n")
            if a["fixed_points"]:
                text.append(md(pd.DataFrame(a["fixed_points"])) + "\n")
            text.append(md(pd.DataFrame(a["thresholds"])) + "\n")
            text.append(md(pd.DataFrame(a["anchors"])) + "\n")
            text.append(f"Tumour ceiling on the composition path: point {a['ce_path']['point']}; "
                        f"lowest ceiling 95% {a['ce_path']['ceiling min 95%']}, highest "
                        f"{a['ce_path']['ceiling max 95%']}; replicates with folds "
                        f"{100 * a['ce_path']['share with folds']:.0f}%.\n")
    json.dump({"point": point, "results": res}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    open(f"{OUT}/report.md", "w").write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
