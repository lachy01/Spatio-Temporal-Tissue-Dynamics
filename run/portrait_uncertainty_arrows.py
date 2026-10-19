"""Arrow-direction uncertainty for the portrait (portrait spec 8.3's figures).

    python run/portrait_uncertainty_arrows.py [--replot]

--replot redraws the figures from arrow_grids.npz without replaying replicates.

The Spartan jobs stored summaries, not velocity grids, but every replicate can be
replayed exactly:
  portrait_block_boot   each result keeps its refitted parameters;
  portrait_param_boot   the draws are regenerated from their seeds (chunk c uses
                        default_rng(20261020 + c), 10 draws in order, as in
                        portrait/uncertainty.py) -- checked by recomputing one
                        draw's anchors against its stored summary.
Each replicate's anchors are read from its stored summary.

For each state (barycentric, counts) and priority slice at q25 and q50, the
velocity on log2 axes, (pi_k - pi_bar_k) / ln 2 per axis (spec 6: per-capita,
what the quiver draws), is evaluated on a 25 x 25 sub-grid of the slice. Per grid
cell: the circular SD of arrow direction, sqrt(-2 ln R) with R the mean resultant
length over replicates.

Writes outputs/portrait/uncertainty/
  arrows_<source>_<state>.png   per slice: circular SD shaded inside support, the
                                point-estimate arrows in black and 30 replicates'
                                arrows in grey over them
  arrows_summary.csv            per source x state x slice: median circular SD
                                inside support, share of cells under 30 and 60 deg
  arrow_grids.npz               the angles, for re-plotting
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

OUT = "outputs/portrait/uncertainty"
SUB = slice(2, None, 4)          # 100 x 100 slice grid -> 25 x 25
N_OVERLAY = 30


def say(*a):
    print(*a, flush=True)


def angles(model, arch, ox, slices):
    """Arrow angle per slice on the sub-grid, shape (n_slices, 25, 25)."""
    from tissue_dynamics.portrait.core import net_rate

    out = []
    for sl in slices:
        GA, GB = sl.GA[SUB, SUB], sl.GB[SUB, SUB]
        X = np.repeat(sl.ref[None], GA.size, axis=0)
        X[:, sl.ka], X[:, sl.kb] = GA.ravel(), GB.ravel()
        R, _ = net_rate(X, model, arch, ox)
        out.append(np.arctan2(R[:, sl.kb], R[:, sl.ka]).reshape(GA.shape))
    return np.array(out)


def circ_sd(theta):
    R = np.abs(np.mean(np.exp(1j * theta), axis=0))
    return np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1)))


def main():
    import dataclasses

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from tissue_dynamics.mixture.data import load_config
    from tissue_dynamics.mixture.model import responsibilities
    from tissue_dynamics.portrait.core import PortraitModel
    from tissue_dynamics.portrait.uncertainty import COV_PATH, N_DRAWS_PER_TASK, STATES, _setup

    cfg = load_config()
    ctx = _setup(cfg)
    if "--replot" in sys.argv:
        g = np.load(f"{OUT}/arrow_grids.npz")
        for st in STATES:
            slices = ctx["geo"][st][1]
            for src in ("param", "block"):
                th = g[f"{src}__{st}"]
                plot(slices, g[f"point__{st}"], th, np.degrees(circ_sd(th)), src, st, th.shape[0])
        return
    names = ctx["geo"]["barycentric"][0].names

    def model_of(params):
        return PortraitModel.from_fit({k: np.asarray(v) for k, v in params.items()}, ctx["basis"])

    # replicates: (params, {state: pi_bar})
    reps = {"param": [], "block": []}
    z = np.load(COV_PATH)
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in ctx["pe"]["params"].items()}
    flat0, unravel = ravel_pytree(P0)
    assert np.allclose(np.asarray(flat0), z["theta"])
    for f in sorted(glob.glob("outputs/hpc/portrait_param_boot/*.json")):
        r = json.load(open(f))
        rng = np.random.default_rng(20261020 + int(r["chunk"]))
        for b in range(N_DRAWS_PER_TASK):
            th = z["theta"] + z["evecs"] @ (np.sqrt(np.clip(z["evals"], 0, None))
                                            * rng.standard_normal(len(z["evals"])))
            d = r["draws"][b]
            reps["param"].append((unravel(jnp.asarray(th)), {
                st: np.array([d["summary"][st]["pi_bar"][n] for n in names]) for st in STATES}))
    for f in sorted(glob.glob("outputs/hpc/portrait_block_boot/*.json")):
        r = json.load(open(f))
        reps["block"].append((r["params"], {
            st: np.array([r["summary"][st]["pi_bar"][n] for n in names]) for st in STATES}))
    say(f"replicates: {len(reps['param'])} parametric, {len(reps['block'])} block")

    # check the regenerated draws: recompute the first one's anchors
    p, pb = reps["param"][0]
    w = np.asarray(responsibilities(p, ctx["data"].as_jax(), ctx["spec"]))
    chk = np.array([w[ctx["arche"] == n].mean() for n in names])
    err = float(np.max(np.abs(chk - pb["barycentric"])))
    say(f"regenerated draw 0: max anchor difference from the stored summary {err:.2e}")
    if err > 1e-8:
        raise RuntimeError("regenerated parametric draws do not match the Spartan run")

    rows, store = [], {}
    pe_model = model_of(ctx["pe"]["params"])
    for st in STATES:
        arch, slices = ctx["geo"][st]
        th_pe = angles(pe_model, arch, ctx["ox"], slices)
        store[f"point|{st}"] = th_pe
        for src, R in reps.items():
            th = np.array([angles(model_of(p), dataclasses.replace(arch, pi_bar=pb[st]), ctx["ox"], slices)
                           for p, pb in R])
            store[f"{src}|{st}"] = th
            sd = np.degrees(circ_sd(th))                        # (n_slices, 25, 25)
            say(f"{src} {st}: {len(R)} replicates")
            for i, sl in enumerate(slices):
                m = sl.M[SUB, SUB]
                v = sd[i][m]
                rows.append({"source": src, "state": st, "slice": f"{sl.pair} q{sl.qref}",
                             "cells in support": int(m.sum()),
                             "median circular SD (deg)": float(np.median(v)) if len(v) else np.nan,
                             "share < 30 deg": float(np.mean(v < 30)) if len(v) else np.nan,
                             "share < 60 deg": float(np.mean(v < 60)) if len(v) else np.nan})
            plot(slices, th_pe, th, sd, src, st, len(R))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/arrows_summary.csv", index=False)
    np.savez_compressed(f"{OUT}/arrow_grids.npz", **{k.replace("|", "__"): v.astype(np.float32)
                                                     for k, v in store.items()})
    say(tab.round(3).to_string(index=False))


def plot(slices, th_pe, th, sd, src, st, n_rep):
    ns = len(slices)
    qs = sorted({s.qref for s in slices})
    pairs = list(dict.fromkeys(s.pair for s in slices))
    fig, axes = plt.subplots(len(qs), len(pairs), figsize=(5.4 * len(pairs), 5.0 * len(qs)), squeeze=False,
                             layout="constrained")
    rng = np.random.default_rng(0)
    pick = rng.choice(th.shape[0], size=min(N_OVERLAY, th.shape[0]), replace=False)
    cmap = plt.get_cmap("Blues").copy()
    for i in range(ns):
        sl = slices[i]
        ax = axes[qs.index(sl.qref), pairs.index(sl.pair)]
        la, lb = np.log2(sl.ga[SUB]), np.log2(sl.gb[SUB])
        m = sl.M[SUB, SUB]
        img = np.where(m, sd[i], np.nan)
        a, b = sl.pair.split(" x ")
        ax.set_facecolor("0.88")
        pc = ax.pcolormesh(la, lb, img, cmap=cmap, vmin=0, vmax=90, shading="nearest")
        LA, LB = np.meshgrid(la, lb)
        ev = np.zeros_like(m)
        ev[::2, ::2] = True
        sel = m & ev
        for j in pick:
            ax.quiver(LA[sel], LB[sel], np.cos(th[j, i][sel]), np.sin(th[j, i][sel]), color="0.45",
                      alpha=0.25, angles="xy", scale=13, width=0.003, headwidth=0, headlength=0,
                      headaxislength=0)
        ax.quiver(LA[sel], LB[sel], np.cos(th_pe[i][sel]), np.sin(th_pe[i][sel]), color="black",
                  angles="xy", scale=13, width=0.005, headwidth=4, headlength=4.5)
        ax.set_title(f"{a} \u00d7 {b}, held at q{sl.qref}\nmedian circular SD {np.nanmedian(img):.0f}\u00b0",
                     fontsize=10)
        ax.set_xlabel(f"log2 {a} density")
        ax.set_ylabel(f"log2 {b} density")
    fig.colorbar(pc, ax=axes, shrink=0.6, label="circular SD of arrow direction (\u00b0)")
    lab = {"param": "parametric bootstrap (block-cluster sandwich)", "block": "spatial block bootstrap"}[src]
    fig.suptitle(f"Arrow-direction uncertainty, {st} state: {lab}, {n_rep} replicates\n"
                 f"black: point estimate; grey: {min(N_OVERLAY, n_rep)} replicates; grey background: "
                 f"outside support", fontsize=12)
    fig.savefig(f"{OUT}/arrows_{src}_{st}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
