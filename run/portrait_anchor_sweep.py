"""Fine sweep of the death-anchor multiplier (portrait spec 9's pi_bar axis).

    python run/portrait_anchor_sweep.py

`run/portrait_sensitivity.py` runs the spec's x0.7 and x1.3; this fills in
0.8-1.2 in steps of 0.05 for the barycentric state, without the fixed-point
search (thresholds and the tumour's composition path only), to show how far the
anchor can move before the thresholds do. Writes
outputs/portrait/sensitivity/anchor_sweep.{csv,txt}.
"""
import dataclasses
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import OUT, context                         # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import build_archetypes            # noqa: E402
from tissue_dynamics.portrait.summary import summarise                # noqa: E402

MULTS = (0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)
SHOW = ("Cancer Epithelial", "CAFs", "Myeloid", "T-cells", "B-cells")


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = context(load_config())
    arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state="barycentric",
                                radius_um=ctx["radius"])
    rows = []
    for m in MULTS:
        s = summarise(ctx["model"], dataclasses.replace(arch, pi_bar=arch.pi_bar * m), ctx["ox"], [],
                      ctx["dens"], ctx["nbr_own"], ctx["arche"])
        sw, cp = s["own_sweep"], s["ce_path"]
        r = {"multiplier": m, "CE anchor": round(float(arch.pi_bar[arch.names.index("Cancer Epithelial")] * m), 4),
             "CE path ceiling (n)": (f"{cp['ceiling_min']:.0f}-{cp['ceiling_max']:.0f}"
                                     if cp["ceiling_min"] is not None else "none"),
             "CE path folds": cp["folds"]}
        for n in SHOW:
            r[n] = ("; ".join(f"{'stable' if c['kind'] == 'stable' else 'UNSTABLE'} {c['n']:.0f} (p{c['pct']:.0f})"
                              for c in sw[n]["crossings"]) or sw[n]["sign"])
        rows.append(r)
        print(r, flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/anchor_sweep.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 60):
        txt = ("Death-anchor multiplier sweep, barycentric state, unbounded. Thresholds at each archetype's own\n"
               "mean composition (n, density percentile); the tumour's composition-path ceiling range.\n\n"
               + tab.to_string(index=False) + "\n")
    open(f"{OUT}/anchor_sweep.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
