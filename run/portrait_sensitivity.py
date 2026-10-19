"""Phase portrait sensitivity grid (portrait spec 9), one portrait summary per cell.

    python run/portrait_sensitivity.py

Axes that apply to the production model (both states unless noted), each against
the baseline (own-type anchor, unbounded):
  anchor multiplier   pi_bar x 0.7 and x 1.3
  delta               the proportional-rates offset at +/- 1 SD (SD over the 300
                      spatial-block bootstrap replicates); delta enters the portrait
                      only through the anchor (the posterior pi it implies)
  tau                 20 draws of per-archetype tau_k, log-uniform on [0.5, 2]: fixed
                      points do not move (F_k = X_k r_k / tau_k), so the point
                      estimate's are reclassified and the share whose class holds is
                      reported
  K                   7 archetypes (B-cells + Plasmablasts, CAFs + PVL) and 6 (also
                      Normal Epithelial into Cancer Epithelial), on the counts state
                      (merged types sum their counts; centroids and anchors are the
                      merged groups')
  radius              60 and 100 um (refits: config_r60/r100.yaml,
                      run/portrait_radius_tables.py, Spartan `multistart`), included
                      when outputs/hpc_r60|r100/point_estimate.json exist
Not applicable: the calibration alpha/beta and oxygen axes (no calibration step, no
oxygen term in production) and L0. `own_state_mode = loading_weighted` equals the
vertex for centroid archetypes with hard types.

Writes outputs/portrait/sensitivity/{grid.csv, grid.json} and prints the table.
"""
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data, responsibilities  # noqa: E402
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402
from tissue_dynamics.portrait.core import (Archetypes, PortraitModel,  # noqa: E402
                                           build_archetypes, jacobian, type_count_loadings)
from tissue_dynamics.portrait.summary import PAIRS, build_slices, summarise  # noqa: E402

OUT = "outputs/portrait/sensitivity"
STATES = ("barycentric", "counts")
MERGES = {7: {"B-cells": "B/Plasma", "Plasmablasts": "B/Plasma", "CAFs": "CAFs+PVL", "PVL": "CAFs+PVL"},
          6: {"B-cells": "B/Plasma", "Plasmablasts": "B/Plasma", "CAFs": "CAFs+PVL", "PVL": "CAFs+PVL",
              "Normal Epithelial": "Cancer Epithelial"}}


def say(*a):
    print(*a, flush=True)


def context(cfg, hpc="outputs/hpc"):
    df, basis = _load(cfg)
    pe = json.load(open(f"{hpc}/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    w = np.asarray(responsibilities(P, D, spec))
    return dict(df=df, basis=basis, pe=pe, spec=spec, D=D, P=P, w=w,
                model=PortraitModel.from_fit(pe["params"], basis),
                arche=df["archetype"].astype(str).to_numpy(), dens=df["density"].to_numpy(float),
                nbr_own=basis.disc_in_own(df), ox=float(np.median(df["oxygen"])),
                radius=float(cfg["data"]["neighbour_radius_um"]))


def row(axis, value, state, s, base_key):
    fps = [f for f in s["fixed_points"] if f["classification"] != "face"]
    sup = [f for f in fps if f["in_support"]]
    stable_sup = [f for f in sup if f["classification"] == "stable"]
    prin = max(stable_sup, key=lambda f: f["n"]) if stable_sup else None
    sw = s["own_sweep"]
    fmt = lambda n: ("; ".join(f"{c['kind'][0]}{c['n']:.0f}" for c in sw[n]["crossings"])  # noqa: E731
                     or sw[n]["sign"]) if n in sw else "-"
    cp = s.get("ce_path", {})
    return {"axis": axis, "value": value, "state": state,
            "fixed points (interior / in support)": f"{len(fps)} / {len(sup)}",
            "in-support classes": ", ".join(s["topology"]) or "none",
            "principal stable point": (f"{prin['slice']} q{prin['q_ref']}: n {prin['n']:.0f}" if prin else "none"),
            "topology = baseline": s["topology"] == base_key,
            "CE ceiling on path (n)": (f"{cp['ceiling_min']:.0f}-{cp['ceiling_max']:.0f}"
                                      if cp.get("ceiling_min") is not None else "none"),
            "CAFs": fmt("CAFs") if "CAFs" in sw else fmt("CAFs+PVL"), "Myeloid": fmt("Myeloid"),
            "Cancer Epithelial": fmt("Cancer Epithelial")}


def merged_arch(ctx, groups):
    """Counts-state archetypes with some types merged."""
    df = ctx["df"].copy()
    lab = np.array([groups.get(t, t) for t in ctx["arche"]])
    names = list(dict.fromkeys(groups.get(t, t) for t in ctx["basis"].phi_names))
    df["archetype"] = lab
    own = ctx["basis"].own(df)
    A = np.vstack([own[lab == n].mean(0) for n in names])
    W = type_count_loadings(df, names, ctx["radius"])
    nbr = ctx["nbr_own"]
    r2 = float(1 - np.sum((nbr - W @ A) ** 2) / np.sum((nbr - nbr.mean(0)) ** 2))
    pb = np.array([ctx["w"][lab == n].mean() for n in names])
    arch = Archetypes(names=names, A=A, Q=ctx["basis"].Q_own, pi_bar=pb, X_obs=ctx["dens"][:, None] * W,
                      r2_recon=r2, pi_bar_clipped=pb)
    pairs = [(groups.get(a, a), groups.get(b, b)) for a, b in PAIRS]
    pairs = [p for p in dict.fromkeys(pairs) if p[0] != p[1]]
    return arch, lab, pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    ctx = context(cfg)
    rows, raw = [], {}
    base = {}
    for st in STATES:
        arch, _W = build_archetypes(ctx["df"], ctx["basis"], pi_hat=ctx["w"], state=st, radius_um=ctx["radius"])
        slices = build_slices(arch.X_obs, arch.names)
        args = (ctx["ox"], slices, ctx["dens"], ctx["nbr_own"], ctx["arche"])
        s0 = summarise(ctx["model"], arch, *args)
        base[st] = (arch, slices, s0)
        rows.append(row("baseline", "own-type anchor", st, s0, s0["topology"]))
        raw[f"baseline|{st}"] = s0
        say(f"{st} baseline: {s0['topology']}")
        for mult in (0.7, 1.3):
            s = summarise(ctx["model"], dataclasses.replace(arch, pi_bar=arch.pi_bar * mult), *args)
            rows.append(row("anchor multiplier", mult, st, s, s0["topology"]))
            raw[f"anchor x{mult}|{st}"] = s
        # delta +/- 1 SD (bootstrap), through the posterior pi and so the anchor
        bt = [json.load(open(f))["e_delta"] for f in sorted(glob.glob("outputs/hpc/bootstrap/*.json"))]
        sd = float(np.std(np.log(bt), ddof=1))
        for sgn in (-1, 1):
            P2 = dict(ctx["P"])
            P2["delta"] = ctx["P"]["delta"] + sgn * sd
            w2 = np.asarray(responsibilities(P2, ctx["D"], ctx["spec"]))
            pb = np.array([w2[ctx["arche"] == n].mean() for n in arch.names])
            s = summarise(ctx["model"], dataclasses.replace(arch, pi_bar=pb), *args)
            rows.append(row("delta", f"{'+' if sgn > 0 else '-'}1 SD ({sd:.4f})", st, s, s0["topology"]))
            raw[f"delta {sgn:+d}sd|{st}"] = s
        # tau: reclassify the point estimate's interior fixed points
        rng = np.random.default_rng(0)
        fps = [f for f in s0["fixed_points"] if f["classification"] != "face"]
        held = []
        for _ in range(20):
            tau = np.exp(rng.uniform(np.log(0.5), np.log(2.0), len(arch.names)))
            same = 0
            for f in fps:
                sl = next(x for x in slices if x.pair == f["slice"] and x.qref == f["q_ref"])
                Z = sl.ref.copy()
                Z[sl.ka], Z[sl.kb] = f["X_a"], f["X_b"]
                J = jacobian(Z, ctx["model"], arch, ctx["ox"], tau=tau)
                ev = np.linalg.eigvals(J[np.ix_([sl.ka, sl.kb], [sl.ka, sl.kb])])
                cls = "stable" if np.all(ev.real < 0) else "unstable" if np.all(ev.real > 0) else "saddle"
                same += cls == f["classification"]
            held.append(same / max(len(fps), 1))
        rows.append({"axis": "tau", "value": "20 draws, log-U[0.5, 2]", "state": st,
                     "fixed points (interior / in support)": f"{len(fps)} (locations unchanged)",
                     "in-support classes": f"classification kept in {100 * np.mean(held):.0f}% of fixed-point x draw",
                     "topology = baseline": bool(np.all(np.array(held) == 1))})
    # K: merged archetypes, counts state
    s0c = base["counts"][2]
    for K, groups in MERGES.items():
        arch, lab, pairs = merged_arch(ctx, groups)
        slices = build_slices(arch.X_obs, arch.names, pairs=pairs)
        s = summarise(ctx["model"], arch, ctx["ox"], slices, ctx["dens"], ctx["nbr_own"], lab)
        r = row("K", f"{K} ({', '.join(sorted(set(groups.values())))})", "counts", s, s0c["topology"])
        r["topology = baseline"] = "n/a (different slices)"
        rows.append(r)
        raw[f"K{K}|counts"] = s
        say(f"K {K}: R2 {arch.r2_recon:.3f}; {s['topology']}")
    # radius refits, if fitted
    for r_um in (60, 100):
        hpc = f"outputs/hpc_r{r_um}"
        if not os.path.exists(f"{hpc}/point_estimate.json"):
            rows.append({"axis": "radius", "value": f"{r_um} um", "state": "-",
                         "in-support classes": f"pending: {hpc}/point_estimate.json not found"})
            continue
        os.environ["MIX_DATASET"] = f"r{r_um}"
        rcfg = load_config()
        os.environ.pop("MIX_DATASET")
        rc = context(rcfg, hpc=hpc)
        for st in STATES:
            arch, _W = build_archetypes(rc["df"], rc["basis"], pi_hat=rc["w"], state=st, radius_um=rc["radius"])
            slices = build_slices(arch.X_obs, arch.names)
            s = summarise(rc["model"], arch, rc["ox"], slices, rc["dens"], rc["nbr_own"], rc["arche"])
            rows.append(row("radius", f"{r_um} um", st, s, base[st][2]["topology"]))
            raw[f"radius {r_um}|{st}"] = s
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/grid.csv", index=False)
    json.dump(raw, open(f"{OUT}/grid.json", "w"), indent=1, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        say(tab.to_string(index=False))


if __name__ == "__main__":
    main()
