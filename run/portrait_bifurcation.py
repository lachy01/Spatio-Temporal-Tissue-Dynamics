"""Cancer Epithelial equilibria as a function of neighbourhood composition.

    python run/portrait_bifurcation.py                    # unbounded (the default)
    python run/portrait_bifurcation.py --bound osdr [--power 8] [--target-at-max]

Resolves an apparent contradiction in the portrait: at the q50 mixed composition
the tumour compartment has a stable density ceiling, but at its own
tumour-dominated composition it has none. Blending between the two shows why --
the ceiling is born and dies in a pair of saddle-node bifurcations, so it exists
only inside a window of tumour fraction.

Runs UNBOUNDED by default, like the rest of the portrait. `--bound osdr` adds
OSDR's maximal-density correction as a flagged sensitivity variant (spec 5) and
writes to outputs/portrait/osdr_p<power>[_target_at_max]/, as
run/portrait_preliminary.py does.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                      # noqa: E402
from tissue_dynamics.mixture.model import build_data, responsibilities    # noqa: E402
from tissue_dynamics.portrait.core import (PortraitModel, build_archetypes,  # noqa: E402
                                           fit_osdr_bound, net_rate)

OUT = "outputs/portrait"
N_BLEND, N_DENS = 301, 12000


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bound", choices=("none", "osdr"), default="none")
    ap.add_argument("--state", choices=("barycentric", "counts"), default="barycentric",
                    help="counts: X_k = density x the disc's type-k fraction (a literal density, "
                         "OSDR's state), written to outputs/portrait/counts/")
    ap.add_argument("--power", type=int, default=8)
    ap.add_argument("--target-at-max", action="store_true")
    a = ap.parse_args()
    base = "outputs/portrait" + ("/counts" if a.state == "counts" else "")
    OUT = base
    if a.bound == "osdr":
        OUT = f"{base}/osdr_p{a.power}{'_target_at_max' if a.target_at_max else ''}"
    os.makedirs(OUT, exist_ok=True)
    flag = ("  (UNBOUNDED: no OSDR density correction)" if a.bound == "none" else
            f"  (SENSITIVITY: OSDR density correction, power {a.power})")
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    pi_hat = np.asarray(responsibilities(P, data.as_jax(), spec))
    model = PortraitModel.from_fit(pe["params"], basis)
    arch, _W = build_archetypes(df, basis, pi_hat=pi_hat, state=a.state,
                                 radius_um=float(cfg["data"]["neighbour_radius_um"]))

    names, Xo = arch.names, arch.X_obs
    ox = float(np.median(df["oxygen"]))
    dens = df["density"].to_numpy(float)
    bound = (None if a.bound == "none" else
             fit_osdr_bound(model, arch, ox, cell_archetype=df["archetype"].astype(str).to_numpy(),
                            power=a.power, target_at_max=a.target_at_max))
    k = names.index("Cancer Epithelial")

    ref = np.percentile(Xo, 50, axis=0)
    w0 = ref / ref.sum()                       # q50 mixed neighbourhood
    w1 = np.zeros(len(names)); w1[k] = 1.0     # pure tumour neighbourhood
    grid = np.linspace(np.percentile(dens, .5), np.percentile(dens, 99.9), N_DENS)

    # For each blend, find every density at which pi_k crosses its death anchor.
    # Inside the window there are three: escape, ceiling, escape.
    per_t = []
    for t in np.linspace(0, 1, N_BLEND):
        w = (1 - t) * w0 + t * w1
        phibar = w @ arch.A
        if bound is None:
            eta = model.eta(np.repeat(arch.A[k][None], len(grid), axis=0),
                            np.repeat(phibar[None], len(grid), axis=0), grid, ox)
            f = 1.0 / (1.0 + np.exp(-eta)) - arch.pi_bar[k]
        else:      # the correction acts on the type's own density, X_k = n w_k
            f = net_rate(grid[:, None] * w[None, :], model, arch, ox, bound)[0][:, k]
        cr = sorted((grid[i], bool(f[i] > 0)) for i in np.where(np.diff(np.sign(f)))[0])
        per_t.append((float(w[k]), cr))

    # Classify every crossing rather than assume a shape: where pi falls below
    # the anchor as density rises, the crossing is a stable ceiling; where it
    # rises above it, an unstable escape threshold. The 2026-10-03 fit had three
    # crossings inside a window of tumour fraction (escape, ceiling, escape) and
    # one outside it; the rank-3 split fit of 2026-10-07 has a single ceiling at
    # every blend. Folds are wherever the number of crossings changes.
    stab = np.array([(wk, d) for wk, c in per_t for d, pos in c if pos]).reshape(-1, 2)
    unst = np.array([(wk, d) for wk, c in per_t for d, pos in c if not pos]).reshape(-1, 2)
    n_cross = np.array([len(c) for _, c in per_t])
    wks = np.array([wk for wk, _ in per_t])
    folds = [float((wks[i] + wks[i + 1]) / 2) for i in np.where(np.diff(n_cross) != 0)[0]]
    obs = Xo[:, k] / Xo.sum(1)
    lines = []
    say = lambda m: (print(m), lines.append(m))
    if len(stab):
        lo, hi = stab[:, 0].min(), stab[:, 0].max()
        frac = (100 * np.mean(obs < lo), 100 * np.mean((obs >= lo) & (obs <= hi)),
                100 * np.mean(obs > hi))
        say(f"stable ceiling exists for {lo:.3f} <= w_CE <= {hi:.3f} "
              f"(blends run {wks.min():.3f} to {wks.max():.3f})")
        say(f"observed neighbourhoods: {frac[0]:.1f}% below, {frac[1]:.1f}% inside, "
              f"{frac[2]:.1f}% above")
        say(f"ceiling density spans n = {stab[:, 1].min():.0f} to {stab[:, 1].max():.0f} "
              f"(pct {100*np.mean(dens < stab[:, 1].min()):.1f} to "
              f"{100*np.mean(dens < stab[:, 1].max()):.1f})")
    else:
        lo = hi = None
        say("no stable ceiling at any blend")
    say(f"crossings per blend: {sorted(set(n_cross.tolist()))}; "
          f"unstable thresholds at {len(unst)} of {len(per_t)} blends; "
          f"folds at w_CE = {[round(f, 3) for f in folds] or 'none'}")

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.axhspan(*np.percentile(dens, [25, 75]), color="0.93", zorder=0,
               label="observed density IQR")
    if lo is not None and (lo > wks.min() or hi < wks.max()):
        ax.axvspan(lo, hi, color="C0", alpha=.10, zorder=0)
    if len(unst):
        ax.plot(unst[:, 0], unst[:, 1], ".", color="C3", ms=3,
                label="escape threshold (unstable)")
    if len(stab):
        ax.plot(stab[:, 0], stab[:, 1], ".", color="k", ms=3,
                label="stable density ceiling")
    for x in folds:
        ax.axvline(x, color="C0", lw=1, ls=":")
    ax.set_xlabel("tumour fraction of the neighbourhood  $\\bar{w}_{CE}$"
                  "  (q50 mixed neighbourhood to pure tumour)", fontsize=9.5)
    ax.set_ylabel("total neighbourhood density $n$  (cells per $(100\\,\\mu m)^2$)")
    ax.set_title("Cancer Epithelial equilibria vs neighbourhood composition"
                 + (f"\nfolds at $\\bar{{w}}_{{CE}}$ = {', '.join(f'{f:.3f}' for f in folds)}"
                    if folds else "\nno folds: the equilibrium structure is the same at every blend")
                 + flag, fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/composition_bifurcation.png", dpi=150)

    pd.DataFrame({"w_CE": [w for w, _ in per_t],
                  "crossings": [json.dumps(c) for _, c in per_t]}).to_csv(
        f"{OUT}/composition_bifurcation.csv", index=False)
    with open(f"{OUT}/composition_bifurcation.txt", "w") as fh:
        fh.write(f"Blend from the q50 reference neighbourhood (w_CE = {wks.min():.3f}) to a pure "
                 "tumour neighbourhood, sweeping total density at each blend.\n"
                 + "\n".join(lines) + "\n")
    print(f"wrote {OUT}/composition_bifurcation.{{png,csv,txt}}")


if __name__ == "__main__":
    main()
