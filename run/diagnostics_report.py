"""Tests 09 and 14 on the fitted point estimate (spec 6.9, 6.14).

    python run/diagnostics_report.py

Runs locally: no refitting, just a forward pass through the stored parameters.
Writes outputs/diagnostics_stage_b.md and figures under outputs/figures/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import jax.numpy as jnp                                               # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config                    # noqa: E402
from tissue_dynamics.mixture.diagnostics import (confound_diagnostic,   # noqa: E402
                                                 correlogram, mixture_rootogram,
                                                 pearson_residuals,
                                                 poisson_rootogram, randomised_pit,
                                                 single_component_reference)
from tissue_dynamics.mixture.model import (build_data, normalised_entropy,  # noqa: E402
                                           responsibilities)

KMAX = 20


def rms(d):
    return float(np.sqrt(np.mean(np.asarray(d) ** 2)))


def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    side = os.path.splitext(cfg["data"]["table_parquet"])[0] + ".phi_names.json"
    df.attrs["phi_names"] = json.load(open(side))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B",
                           weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    p = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    D = data.as_jax()
    y = data.y
    w = np.asarray(responsibilities(p, D, spec))
    print(f"{len(y):,} cells | mean w {w.mean():.4f} | entropy {normalised_entropy(w):.4f}")

    # --- Test 09: rootogram + randomised PIT, mixture vs single component -----
    ks, obs, exp_m, disc_m = mixture_rootogram(y, p, D, spec, max_count=KMAX)
    mu1 = single_component_reference(y, data, spec)
    _k, _o, exp_1, disc_1 = poisson_rootogram(y, mu1, max_count=KMAX)
    _u, pit, _e = randomised_pit(y, p, D, spec, n_bins=10)

    print("\n=== Test 09: marginal fit ===")
    print(f"  RMS hanging discrepancy: single component {rms(disc_1):8.1f} "
          f"-> mixture {rms(disc_m):8.1f}")
    print(f"  counts 1-2 discrepancy:  single {disc_1[1]:+.1f}/{disc_1[2]:+.1f} "
          f"-> mixture {disc_m[1]:+.1f}/{disc_m[2]:+.1f}")
    print(f"  randomised PIT max|dev from uniform|: {np.max(np.abs(pit - 1)):.4f}")

    rows = []
    for a in df["archetype"].value_counts().index:
        m = (df["archetype"] == a).to_numpy()
        sub = data.subset(np.where(m)[0])
        _k2, _o2, _e2, dm = mixture_rootogram(sub.y, p, sub.as_jax(), spec, max_count=KMAX)
        _k3, _o3, _e3, d1 = poisson_rootogram(sub.y, mu1[m], max_count=KMAX)
        rows.append({"archetype": a, "n": int(m.sum()), "single": rms(d1),
                     "mixture": rms(dm), "change_%": 100 * (rms(dm) - rms(d1)) / rms(d1)})
    per = pd.DataFrame(rows)
    print("\n  RMS discrepancy by archetype (spec 6.9: Cancer Epithelial is the "
          "stratum that matters):")
    print(per.round(2).to_string(index=False))

    # --- Test 14: residual spatial autocorrelation ---------------------------
    from tissue_dynamics.evaluate import morans_i
    resid = pearson_residuals(y, p, D, spec)
    xy = df[["x", "y"]].to_numpy(float)
    rng = np.random.default_rng(0)
    sub_i = rng.choice(len(resid), min(120_000, len(resid)), replace=False)
    mi = morans_i(resid[sub_i], coords=xy[sub_i], n_neighbors=8, permutations=99, seed=0)
    print("\n=== Test 14: residual spatial autocorrelation ===")
    print(f"  Moran's I (8-NN, 120k subsample) = {mi['I']:.4f}  p = {mi['p_value']:.3f}")
    cg = correlogram(resid, xy, max_dist=500.0, n_bands=10)
    print("  correlogram:")
    for dist, I, npair in cg:
        print(f"    {dist:6.0f} um   I = {I:+.4f}   ({npair:,} pairs)")
    band = next((d for d, I, _n in cg if np.isfinite(I) and abs(I) < 0.01), None)
    print(f"  -> correlation range ~{band if band else '>500'} um "
          "(the Conley bandwidth, and the floor on block size for spec 7.1)")

    conf = confound_diagnostic(w, df, basis)

    # --- figures -------------------------------------------------------------
    os.makedirs("outputs/figures", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].axhline(0, color="k", lw=1)
    ax[0].plot(ks, disc_1, "o-", color="0.5", label=f"single component (RMS {rms(disc_1):.0f})")
    ax[0].plot(ks, disc_m, "s-", color="C2", label=f"mixture (RMS {rms(disc_m):.0f})")
    ax[0].set_xlabel("count"); ax[0].set_ylabel(r"$\sqrt{obs}-\sqrt{exp}$")
    ax[0].set_title("Test 09: hanging rootogram"); ax[0].legend(fontsize=8)
    ax[1].bar(np.arange(10) + .5, pit, width=.9, color="C0", alpha=.8)
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set_xlabel("PIT bin"); ax[1].set_ylabel("relative frequency")
    ax[1].set_title(f"Randomised PIT (max|dev| {np.max(np.abs(pit-1)):.3f})")
    dd = [c[0] for c in cg]; ii = [c[1] for c in cg]
    ax[2].plot(dd, ii, "o-", color="C3"); ax[2].axhline(0, color="k", lw=1)
    ax[2].set_xlabel("distance (um)"); ax[2].set_ylabel("Moran's I")
    ax[2].set_title("Test 14: residual correlogram")
    fig.tight_layout(); fig.savefig("outputs/figures/diagnostics_stage_b.png", dpi=140)
    print("\nwrote outputs/figures/diagnostics_stage_b.png")

    with open("outputs/diagnostics_stage_b.md", "w") as fh:
        fh.write(f"""# Stage B diagnostics (spec 6.9, 6.14, 8.7, 8.8)

Computed on the fitted point estimate, {len(y):,} cells.

## Test 09 — marginal fit

RMS hanging discrepancy over counts 0-{KMAX}: single component
**{rms(disc_1):.1f}** -> mixture **{rms(disc_m):.1f}**
({100*(rms(disc_m)-rms(disc_1))/rms(disc_1):+.1f}%).

Counts 1-2, which spec 6.9 singles out: {disc_1[1]:+.1f} / {disc_1[2]:+.1f}
-> {disc_m[1]:+.1f} / {disc_m[2]:+.1f}.

Randomised PIT max deviation from uniform: **{np.max(np.abs(pit-1)):.4f}**.

### By archetype

{per.round(2).to_markdown(index=False)}

## Test 14 — residual spatial autocorrelation

Moran's I on mixture Pearson residuals (8-NN, 120k subsample):
**{mi['I']:.4f}** (p = {mi['p_value']:.3f}).

| distance (um) | Moran's I | pairs |
|---:|---:|---:|
""")
        for dist, I, npair in cg:
            fh.write(f"| {dist:.0f} | {I:+.4f} | {npair:,} |\n")
        fh.write(f"""
## Spec 8.7 / 8.8 (mandatory with every reported fit)

- normalised entropy **{normalised_entropy(w):.4f}** (floor 0.30)
- confound R^2: nuisance **{conf['confound_r2_nuisance']:.4f}**, state
  **{conf['confound_r2_state']:.4f}**, ratio
  **{conf['confound_ratio_state_over_nuisance']:.2f}x**
- size/detection classifier: **{conf['is_size_detection_classifier']}**
""")
    print("wrote outputs/diagnostics_stage_b.md")


if __name__ == "__main__":
    main()
