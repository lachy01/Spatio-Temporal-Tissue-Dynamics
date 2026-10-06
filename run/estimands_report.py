"""Spec 8.2's archetype-conditional neighbourhood table, plus spatial maps.

    python run/estimands_report.py

Spec 8.2 calls this the headline result and the direct generalisation of the
discrete interaction matrix: for a reference own-state archetype `a`, the
neighbourhood coefficient vector is modified to `c + B^T phi_a`, and the
reported quantity is the odds ratio per IQR of each neighbourhood component.

CIs come from the bootstrap replicates already on disk -- each carries its own
`c` and `B`, so no refitting is needed.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import jax.numpy as jnp                                             # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis   # noqa: E402
from tissue_dynamics.mixture.data import load_config                     # noqa: E402
from tissue_dynamics.mixture.diagnostics import pearson_residuals        # noqa: E402
from tissue_dynamics.mixture.model import (build_data, marginal_mean,    # noqa: E402
                                           pi_hat, responsibilities)


def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    names = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    p = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    D = data.as_jax()
    # eff below is a NEIGHBOUR-space vector (c + B' phi), so it maps back to
    # cell types through Q_nbr. Q_own gave the same answer only while the two
    # projections were identical, which whitening ends.
    Q = basis.Q_nbr

    own = basis.own(df)                     # projected, centred
    nbr_raw = df[[f"nbr_phi_{j:02d}" for j in range(len(names))]].to_numpy(float)
    delta = np.percentile(nbr_raw, 75, axis=0) - np.percentile(nbr_raw, 25, axis=0)

    boot = [json.load(open(f)) for f in sorted(glob.glob("outputs/hpc/bootstrap/*.json"))]
    arche = df["archetype"].astype(str).to_numpy()

    def table(c_vec, B_mat):
        """Odds ratio per IQR of each neighbour component, per own archetype."""
        out = {}
        for a in names:
            m = arche == a
            if m.sum() < 200:
                continue
            phi_a = own[m].mean(0)                       # centroid in the projected basis
            eff = np.asarray(c_vec) + np.asarray(B_mat).T @ phi_a
            out[a] = np.exp((Q @ eff) * delta)           # back to cell types, per IQR
        return out

    point = table(pe["params"]["c"], np.asarray(pe["params"]["U"]) @ np.asarray(pe["params"]["V"]).T)
    reps = [table(r["c"], np.asarray(r["B"])) for r in boot if r.get("B") is not None]
    arche_list = list(point)

    print(f"=== spec 8.2: odds ratio of cycling per IQR of each neighbour component ===")
    print("rows = own-state archetype, columns = neighbourhood component\n")
    tab = pd.DataFrame({a: point[a] for a in arche_list}, index=names).T
    print(tab.round(3).to_string())

    lo = pd.DataFrame({a: np.percentile([r[a] for r in reps], 2.5, axis=0)
                       for a in arche_list}, index=names).T
    hi = pd.DataFrame({a: np.percentile([r[a] for r in reps], 97.5, axis=0)
                       for a in arche_list}, index=names).T
    sig = (lo > 1.0) | (hi < 1.0)
    print(f"\nentries whose 95% CI excludes 1: {int(sig.to_numpy().sum())} of {sig.size}")

    # --- spatial maps --------------------------------------------------------
    resid = pearson_residuals(data.y, p, D, spec)
    w = np.asarray(responsibilities(p, D, spec))
    pi = np.asarray(pi_hat(p, D, spec))
    mu = np.asarray(marginal_mean(p, D, spec))
    xy = df[["x", "y"]].to_numpy(float)
    TILE = 100.0
    ix = ((xy[:, 0] - xy[:, 0].min()) // TILE).astype(int)
    iy = ((xy[:, 1] - xy[:, 1].min()) // TILE).astype(int)
    key = pd.DataFrame({"ix": ix, "iy": iy, "resid": resid, "w": w, "pi": pi,
                        "y": data.y, "mu": mu})
    grp = key.groupby(["ix", "iy"]).agg(
        resid=("resid", "mean"), w=("w", "mean"), pi=("pi", "mean"),
        obs=("y", "mean"), fit=("mu", "mean"), n=("y", "size")).reset_index()
    grp = grp[grp["n"] >= 5]
    print(f"\n{len(grp):,} tiles of {TILE:.0f} um with >=5 cells")

    def panel(ax, col, title, cmap, sym=False):
        v = grp[col].to_numpy()
        lim = np.percentile(np.abs(v), 99) if sym else None
        sc = ax.scatter(grp["ix"] * TILE, grp["iy"] * TILE, c=v, s=3, cmap=cmap,
                        vmin=-lim if sym else np.percentile(v, 1),
                        vmax=lim if sym else np.percentile(v, 99), linewidths=0)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046)

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    panel(ax[0, 0], "obs", "observed mean count", "viridis")
    panel(ax[0, 1], "fit", "fitted marginal mean", "viridis")
    panel(ax[0, 2], "resid", "Pearson residual (mean)", "RdBu_r", sym=True)
    panel(ax[1, 0], "pi", r"$\hat\pi$ — prior P(cycling)", "magma")
    panel(ax[1, 1], "w", r"$\hat w$ — posterior P(cycling)", "magma")
    a2 = ax[1, 2]
    a2.hist(resid, bins=120, range=(-4, 6), color="0.5")
    a2.set_title("residual distribution", fontsize=10)
    a2.set_xlabel("Pearson residual")
    fig.suptitle("Stage B: spatial fit and residuals (100 um tiles)", y=0.99)
    fig.tight_layout()
    fig.savefig("outputs/figures/spatial_residuals.png", dpi=140)
    print("wrote outputs/figures/spatial_residuals.png")

    with open("outputs/estimands_8_2.md", "w") as fh:
        fh.write("# Spec 8.2 — archetype-conditional neighbourhood effects\n\n")
        fh.write("Odds ratio of cycling per IQR of each neighbourhood component, "
                 "for a cell whose own state sits at the given archetype centroid. "
                 "`c + B^T phi_a`, mapped back to cell types. 95% CIs from the "
                 "spatial block bootstrap; **bold** where the CI excludes 1.\n\n")
        fh.write("| own archetype | " + " | ".join(n[:13] for n in names) + " |\n")
        fh.write("|---" * (len(names) + 1) + "|\n")
        for a in arche_list:
            cells = []
            for j, _n in enumerate(names):
                v, l_, h_ = tab.loc[a].iloc[j], lo.loc[a].iloc[j], hi.loc[a].iloc[j]
                cells.append(f"**{v:.2f}**" if (l_ > 1 or h_ < 1) else f"{v:.2f}")
            fh.write(f"| {a} | " + " | ".join(cells) + " |\n")
        fh.write(f"\n{int(sig.to_numpy().sum())} of {sig.size} entries have a CI "
                 "excluding 1.\n")
    print("wrote outputs/estimands_8_2.md")


if __name__ == "__main__":
    main()
