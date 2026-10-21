"""Score-weighted own-type death anchor against the hard-label one (FINDINGS §14).

    python run/portrait_anchor_weighted.py [--starts 400]

The production anchor is `pi_bar_k = mean of pi_i over cells labelled k` (argmax of
the standardised PhiSpace scores). The score-weighted anchor replaces the hard label
by a soft membership S_ik: `pi_bar_k = sum_i S_ik pi_i / sum_i S_ik`
(`portrait.core.type_score_weights`). Variants:
  hard           the production anchor (the t -> 0 limit)
  softmax t      softmax of the standardised scores, t = 0.25, 0.5, 1
  membership     the scores themselves, (s + 1) / 2, clipped at 0 and renormalised
Only the anchor changes: centroids, states and the bound's pi-cap keep the hard
labels. For each variant, on the count-based state:
  - the anchors, and the share of each anchor's weight from the type's own cells;
  - thresholds at each archetype's own composition and the tumour's composition-path
    ceiling, unbounded and under OSDR's bound (p = 8, refitted to the anchor);
  - where trajectories from `--starts` observed states end (t = 3000), as in §13:
    the attractors they settle on (clustered in log2(1 + X)), collapse, run away. A type
    below 0.5 cells per (100 um)^2 at the end and still declining is going extinct (it
    decays exponentially, so its rate never reaches 0, and slowly: at r = -0.001 it is
    still far from 0 at t = 3000); the rest must have |r_k| < 1e-5 to count as settled.

Writes outputs/portrait/sensitivity/anchor_weighted.{csv,txt,json}.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import OUT, context                         # noqa: E402
from portrait_stable_state_figure import integrate                    # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import (build_archetypes, fit_osdr_bound,  # noqa: E402
                                           net_rate, type_score_weights)
from tissue_dynamics.portrait.summary import summarise                # noqa: E402

VARIANTS = (("hard", 0.0), ("softmax t=0.25", 0.25), ("softmax t=0.5", 0.5), ("softmax t=1", 1.0),
            ("membership", None))
SHOW = ("Cancer Epithelial", "CAFs", "Myeloid", "T-cells", "B-cells", "Endothelial")
TOL = 0.05          # log2(1 + X) distance within which two end states are the same
EXTINCT = 0.5       # a type below this density at t = 3000 and still declining is going extinct
SETTLED = 1e-5      # max |r_k| over the surviving types


def thresholds(s):
    sw, cp = s["own_sweep"], s["ce_path"]
    r = {n: ("; ".join(f"{'s' if c['kind'] == 'stable' else 'u'}{c['n']:.0f} (p{c['pct']:.0f})"
                       for c in sw[n]["crossings"]) or sw[n]["sign"]) for n in SHOW}
    r["CE path ceiling"] = (f"{cp['ceiling_min']:.0f}-{cp['ceiling_max']:.0f}"
                            if cp["ceiling_min"] is not None else "none")
    return r


def fates(X0, model, arch, ox, bound, nmax):
    path, Xend, _speed = integrate(X0, model, arch, ox, bound)
    nend, nprev = Xend.sum(1), path[-20].sum(1)
    runaway = ~np.isfinite(Xend).all(1) | ((nend > nmax) & (nend > nprev))
    collapse = (~runaway) & (nend < 1.0)
    # a type decaying to extinction keeps a negative rate for ever (and at rates near
    # -0.001 is still far from 0 at t = 3000), so settling is judged on the survivors
    # and the declining types below EXTINCT are set to 0. Types absent from the start
    # stay at 0 (the face is invariant), whatever their invasion rate.
    r, _ = net_rate(np.where(np.isfinite(Xend), Xend, 0.0), model, arch, ox, bound)
    alive = (Xend >= EXTINCT) | ((r >= 0) & (X0 > 0))
    settled = ~(runaway | collapse) & (np.where(alive, np.abs(r), 0.0).max(1) < SETTLED)
    Xend = np.where(alive, Xend, 0.0)
    L = np.log2(1 + np.clip(Xend, 0, 1e6))
    states = []                                     # greedy clusters of settled end points
    for i in np.flatnonzero(settled):
        for st in states:
            if np.sqrt(((L[i] - st["L"]) ** 2).sum()) < TOL:
                st["idx"].append(i)
                break
        else:
            states.append({"L": L[i], "idx": [i]})
    states.sort(key=lambda st: -len(st["idx"]))
    out = {"collapse": float(collapse.mean()), "runaway": float(runaway.mean()),
           "still moving": float((~(runaway | collapse | settled)).mean()), "attractors": []}
    for st in states[:3]:
        X = np.median(Xend[st["idx"]], 0)
        out["attractors"].append({"share": len(st["idx"]) / len(X0), "n": float(X.sum()),
                                  "types present": int((X > 0.5).sum()),
                                  "X": dict(zip(arch.names, np.round(X, 2).tolist()))})
    return out


def fmt_fates(o):
    a = "; ".join(f"{100 * x['share']:.0f}% n={x['n']:.0f} ({x['types present']} types)"
                  for x in o["attractors"]) or "none"
    return (f"{a} | collapse {100 * o['collapse']:.0f}%, runaway {100 * o['runaway']:.0f}%, "
            f"moving {100 * o['still moving']:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx = context(load_config())
    df, names = ctx["df"], list(ctx["basis"].phi_names)
    H = (ctx["arche"][:, None] == np.array(names)[None]).astype(float)
    base, _W = build_archetypes(df, ctx["basis"], pi_hat=ctx["w"], state="counts", radius_um=ctx["radius"])
    Xo = np.clip(base.X_obs, 0, None)
    X0 = Xo[np.random.default_rng(0).choice(len(Xo), size=a.starts, replace=False)]
    nmax = float(np.percentile(ctx["dens"], 99.9))
    args = (ctx["ox"], [], ctx["dens"], ctx["nbr_own"], ctx["arche"])

    anchors, shares, rows, raw = {}, {}, [], {}
    for lab, t in VARIANTS:
        S = H if t == 0.0 else type_score_weights(df, names, temperature=t)
        arch, _ = build_archetypes(df, ctx["basis"], pi_hat=ctx["w"], state="counts", radius_um=ctx["radius"],
                                   anchor="score_weighted", anchor_weights=S)
        if t == 0.0:
            assert np.allclose(arch.pi_bar, base.pi_bar)
        anchors[lab] = arch.pi_bar
        shares[lab] = (S * H).sum(0) / S.sum(0)
        bound = fit_osdr_bound(ctx["model"], arch, ctx["ox"], cell_archetype=ctx["arche"], power=8)
        for bl, bd in (("unbounded", None), ("OSDR p=8", bound)):
            s = summarise(ctx["model"], arch, *args, bound=bd)
            f = fates(X0, ctx["model"], arch, ctx["ox"], bd, nmax)
            rows.append({"anchor": lab, "bound": bl, **thresholds(s), "trajectory ends": fmt_fates(f)})
            raw[f"{lab}|{bl}"] = {"thresholds": thresholds(s), "fates": f}
            print(rows[-1], flush=True)

    A = pd.DataFrame(anchors, index=names).round(3)
    Sh = pd.DataFrame(shares, index=names).round(2)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/anchor_weighted.csv", index=False)
    json.dump({"anchors": A.to_dict(), "own-cell share": Sh.to_dict(), "runs": raw},
              open(f"{OUT}/anchor_weighted.json", "w"), indent=1, default=float)
    with pd.option_context("display.width", 320, "display.max_colwidth", 90, "display.max_columns", 20):
        txt = ("Score-weighted own-type anchor, count-based state. Hard = production (argmax labels).\n\n"
               "Anchors pi_bar_k:\n" + A.to_string() + "\n\n"
               "Share of each anchor's weight from cells labelled k:\n" + Sh.to_string() + "\n\n"
               "Thresholds at each archetype's own composition: s/u = stable/unstable crossing, n (density pct);\n"
               f"trajectory ends from {a.starts} observed states at t = 3000:\n\n"
               + tab.to_string(index=False) + "\n")
    open(f"{OUT}/anchor_weighted.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
