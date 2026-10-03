"""Cancer Epithelial equilibria as a function of neighbourhood composition.

    python run/portrait_bifurcation.py

Resolves an apparent contradiction in the portrait: at the q50 mixed composition
the tumour compartment has a stable density ceiling, but at its own
tumour-dominated composition it has none. Blending between the two shows why --
the ceiling is born and dies in a pair of saddle-node bifurcations, so it exists
only inside a window of tumour fraction.

Runs UNBOUNDED, like the rest of the portrait: no OSDR density correction.
"""
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
from tissue_dynamics.portrait.core import PortraitModel, build_archetypes  # noqa: E402

OUT = "outputs/portrait"
N_BLEND, N_DENS = 301, 12000


def main():
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
    arch, _W = build_archetypes(df, basis, pi_hat=pi_hat)

    names, Xo = arch.names, arch.X_obs
    ox = float(np.median(df["oxygen"]))
    dens = df["density"].to_numpy(float)
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
        eta = model.eta(np.repeat(arch.A[k][None], len(grid), axis=0),
                        np.repeat(phibar[None], len(grid), axis=0), grid, ox)
        f = 1.0 / (1.0 + np.exp(-eta)) - arch.pi_bar[k]
        cr = sorted((grid[i], bool(f[i] > 0)) for i in np.where(np.diff(np.sign(f)))[0])
        per_t.append((float(w[k]), cr))

    stab = np.array([(wk, c[1][0]) for wk, c in per_t if len(c) == 3])
    lo, hi = stab[:, 0].min(), stab[:, 0].max()
    # Away from the window a single crossing survives: the lower branch below the
    # left fold, the upper branch above the right fold. Assigning both to one
    # list would draw a spurious vertical jump across the fold.
    low = np.array([(wk, c[0][0]) for wk, c in per_t
                    if len(c) == 3 or (len(c) == 1 and wk < lo)])
    upp = np.array([(wk, c[-1][0]) for wk, c in per_t
                    if len(c) == 3 or (len(c) == 1 and wk > hi)])

    obs = Xo[:, k] / Xo.sum(1)
    frac = (100 * np.mean(obs < lo), 100 * np.mean((obs >= lo) & (obs <= hi)),
            100 * np.mean(obs > hi))
    print(f"stable ceiling exists for {lo:.3f} <= w_CE <= {hi:.3f}")
    print(f"observed neighbourhoods: {frac[0]:.1f}% below, {frac[1]:.1f}% inside, "
          f"{frac[2]:.1f}% above")
    print(f"ceiling density spans n = {stab[:, 1].min():.0f} to {stab[:, 1].max():.0f} "
          f"(pct {100*np.mean(dens < stab[:, 1].min()):.0f} to "
          f"{100*np.mean(dens < stab[:, 1].max()):.0f})")

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.axhspan(*np.percentile(dens, [25, 75]), color="0.93", zorder=0,
               label="observed density IQR")
    ax.axvspan(lo, hi, color="C0", alpha=.10, zorder=0)
    ax.plot(low[:, 0], low[:, 1], "--", color="C3", lw=2.4,
            label="escape threshold (unstable)")
    ax.plot(upp[:, 0], upp[:, 1], "--", color="C3", lw=2.4)
    ax.plot(stab[:, 0], stab[:, 1], "-", color="k", lw=2.6,
            label="stable density ceiling")
    for x in (lo, hi):
        ax.axvline(x, color="C0", lw=1, ls=":")
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * .96, "growth\ncontrolled", ha="center",
            va="top", fontsize=9.5, color="C0")
    ax.annotate(f"saddle-node\n$\\bar{{w}}_{{CE}}={lo:.3f}$",
                (lo, stab[stab[:, 0].argmin(), 1]), textcoords="offset points",
                xytext=(-88, 20), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.9))
    ax.annotate(f"saddle-node\n$\\bar{{w}}_{{CE}}={hi:.3f}$",
                (hi, stab[stab[:, 0].argmax(), 1]), textcoords="offset points",
                xytext=(26, -30), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.9))
    ax.set_xlabel("tumour fraction of the neighbourhood  $\\bar{w}_{CE}$"
                  f"     ({frac[0]:.0f}% of cells below the window, {frac[1]:.0f}% inside,"
                  f" {frac[2]:.0f}% above)", fontsize=9.5)
    ax.set_ylabel("total neighbourhood density $n$  (cells per $(100\\,\\mu m)^2$)")
    ax.set_title("Cancer Epithelial equilibria vs neighbourhood composition\n"
                 "a stable ceiling exists only inside the shaded window "
                 "(UNBOUNDED: no OSDR density correction)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/composition_bifurcation.png", dpi=150)

    pd.DataFrame({"w_CE": [w for w, _ in per_t],
                  "crossings": [json.dumps(c) for _, c in per_t]}).to_csv(
        f"{OUT}/composition_bifurcation.csv", index=False)
    print(f"wrote {OUT}/composition_bifurcation.{{png,csv}}")


if __name__ == "__main__":
    main()
