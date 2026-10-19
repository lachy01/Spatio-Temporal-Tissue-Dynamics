"""Stationary points of the full 9-archetype portrait, F(X) = 0 in every coordinate.

    python run/portrait_full_equilibria.py [--state counts|barycentric]
        [--bound none|osdr] [--power 8] [--target-at-max] [--seeds 1500]

The slice search (run/portrait_preliminary.py) holds seven densities at a reference
state that is not itself at rest, so the points it finds are fixed only in their
slice; most are unstable in the held directions. This searches the whole system.

Search. A point is stationary when, for some set S of present archetypes,
X_k = 0 for k outside S and the net rate r_k(X) = 0 for k in S (F_k = X_k r_k / tau_k).
Seeds are observed neighbourhood states: for a random sample of cells, S = the
archetypes present (X_k >= 0.5, about one cell in the 80 um disc), and also S = all
nine. r_S = 0 is solved in log X_S (scipy `hybr`), converged at max |r| < 1e-9.
Barycentric states can be negative; only their positive part seeds the search.

Classification, by the full 9 x 9 Jacobian of F (`portrait.core.jacobian`): at a
face point the rows of absent types carry their invasion rate r_k, so an
equilibrium is stable only if it is stable within S AND no absent type can invade.

Support. In log2(1 + X), a point is inside the data if its distance to its 20th
nearest observed state is at most the 95th percentile of the same distance among
observed states.

Writes outputs/portrait/equilibria/<state>_<bound>.{csv,txt}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait_sensitivity import context                              # noqa: E402

from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.portrait.core import (build_archetypes, fit_osdr_bound,  # noqa: E402
                                           jacobian, net_rate)

OUT = "outputs/portrait/equilibria"
PRESENT, KNN = 0.5, 20


def solve(seed_x, S, model, arch, ox, bound, lo, hi):
    S = np.asarray(S)

    def resid(u):
        X = np.zeros(arch.K)
        X[S] = np.exp(np.clip(u, lo, hi[S]))
        r = net_rate(X[None], model, arch, ox, bound)[0][0, S]
        return np.where(np.isfinite(r), r, 1e3)

    sol = root(resid, np.log(np.maximum(seed_x[S], PRESENT)), method="hybr")
    if not sol.success:
        return None
    if np.any(sol.x < lo + 1e-6) or np.any(sol.x > hi[S] - 1e-6):
        return None                                   # pinned at the search box
    if np.max(np.abs(resid(sol.x))) > 1e-9:
        return None
    X = np.zeros(arch.K)
    X[S] = np.exp(sol.x)
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", choices=("counts", "barycentric"), default="counts")
    ap.add_argument("--bound", choices=("none", "osdr"), default="osdr")
    ap.add_argument("--power", type=int, default=8)
    ap.add_argument("--target-at-max", action="store_true")
    ap.add_argument("--seeds", type=int, default=1500)
    a = ap.parse_args()
    tag = f"{a.state}_{'unbounded' if a.bound == 'none' else f'osdr_p{a.power}'}" + \
          ("_target_at_max" if a.target_at_max else "")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    ctx = context(load_config())
    arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state=a.state,
                                radius_um=ctx["radius"])
    model, ox, names, K = ctx["model"], ctx["ox"], arch.names, arch.K
    bound = None
    if a.bound == "osdr":
        bound = fit_osdr_bound(model, arch, ox, cell_archetype=ctx["arche"], power=a.power,
                               target_at_max=a.target_at_max)
    Xo = np.clip(arch.X_obs, 0, None)
    lo = np.log(0.05)
    hi = np.log(3 * Xo.max(0) + 1)

    # support reference
    rng = np.random.default_rng(0)
    L = np.log2(1 + Xo)
    ref = L[rng.choice(len(L), size=min(200_000, len(L)), replace=False)]
    tree = cKDTree(ref)
    d_obs = tree.query(ref[rng.choice(len(ref), 5000, replace=False)], k=KNN + 1)[0][:, -1]
    d95 = float(np.percentile(d_obs, 95))

    seeds = []
    for i in rng.choice(len(Xo), size=a.seeds, replace=False):
        x = Xo[i]
        S = np.where(x >= PRESENT)[0]
        if len(S):
            seeds.append((x, S))
        seeds.append((x, np.arange(K)))
    print(f"{tag}: {len(seeds)} seeds ({(time.time() - t0) / 60:.1f} min)", flush=True)

    found, hits = [], []
    for j, (x, S) in enumerate(seeds):
        X = solve(x, S, model, arch, ox, bound, lo, hi)
        if X is None:
            continue
        key = np.log2(1 + X)
        for m, Y in enumerate(found):
            if np.max(np.abs(np.log2(1 + Y) - key)) < 1e-3:
                hits[m] += 1
                break
        else:
            found.append(X)
            hits.append(1)
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{len(seeds)} seeds, {len(found)} distinct points", flush=True)

    rows = []
    for X, h in zip(found, hits):
        J = jacobian(X, model, arch, ox, bound=bound)
        ev = np.linalg.eigvals(J)
        absent = X == 0
        rates = net_rate(X[None], model, arch, ox, bound)[0][0]
        invaders = [names[k] for k in np.where(absent & (rates > 1e-12))[0]]
        mx = float(ev.real.max())
        cls = "stable" if mx < -1e-12 else ("unstable" if np.all(ev.real[~absent] > 1e-12) and not
                                             np.any(absent) else "saddle")
        dist = float(tree.query(np.log2(1 + X), k=KNN)[0][-1])
        n = float(X.sum())
        comp = sorted(((X[k] / n, names[k]) for k in range(K) if X[k] > 0), reverse=True)
        rows.append({"n": round(n, 1), "density pct": round(float(100 * np.mean(ctx["dens"] < n)), 1),
                     "class": cls, "max Re eig": mx, "n present": int((~absent).sum()),
                     "invaded by": ", ".join(invaders) or "-",
                     "in support": dist <= d95, "kNN dist / q95": round(dist / d95, 2),
                     "seeds converging": h,
                     "composition": ", ".join(f"{nm} {100 * w:.0f}%" for w, nm in comp if w >= 0.02),
                     **{f"X_{nm}": float(X[k]) for k, nm in enumerate(names)}})
    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values(["class", "in support", "seeds converging"], ascending=[False, False, False])
    tab.to_csv(f"{OUT}/{tag}.csv", index=False)
    head = (f"Full-system stationary points: {tag}. {len(seeds)} seeds, {len(tab)} distinct points "
            f"({(time.time() - t0) / 60:.1f} min).\n"
            + (f"Bound (r_k at x_max_k): {json.dumps({k: [round(float(r), 3), round(float(x), 1)] for k, r, x in zip(names, bound.max_flux, bound.x_max)})}\n"
               if bound is not None else "Unbounded.\n")
            + "class from the full Jacobian (face points: absent types' invasion rates included); "
              "in support = 20-NN distance <= q95 of observed states'.\n\n")
    cols = ["class", "in support", "kNN dist / q95", "n", "density pct", "n present", "invaded by",
            "seeds converging", "max Re eig", "composition"]
    with pd.option_context("display.width", 300, "display.max_colwidth", 120):
        txt = head + (tab[cols].to_string(index=False) if len(tab) else "none found") + "\n"
    open(f"{OUT}/{tag}.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
