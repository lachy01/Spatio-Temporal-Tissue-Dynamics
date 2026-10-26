"""The contact effect of one neighbour type on every own type, with and without MisTIC.

    python run/contact_effect_mistic_vs_plain.py [--neighbour "Cancer Epithelial"] [--block contact]

The full neighbour effect of run/interaction_full_effects.py, c + B' phi_a mapped to
cell-type scores and scaled by the score's IQR (log odds ratio of cycling per IQR),
for the production fit (technical: mistic, outputs/hpc) and the plain baseline
(technical: plain, outputs/hpc_plain). Both share the neighbour basis, the own-type
centroids phi_a and the IQRs, so the effects compare directly.

95% intervals: percentiles over each fit's 300 spatial block bootstrap replicates.
Replicate r of both jobs resamples the same blocks (default_rng(r) over the same
spatial_blocks partition, both run on Spartan), so the difference's interval is
taken over the paired replicate differences.

Writes outputs/interaction/contact_effect_<neighbour>_mistic_vs_plain.csv.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tissue_dynamics.mixture.basis import build_basis                 # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402

OUT = "outputs/interaction"
ORDER = ["Cancer Epithelial", "Normal Epithelial", "T-cells", "B-cells", "Plasmablasts",
         "Myeloid", "CAFs", "Endothelial", "PVL"]
FITS = {"MisTIC (production)": "outputs/hpc", "plain (no MisTIC)": "outputs/hpc_plain"}


def column(c, B, phi_c, Q, iqr, j):
    """log OR per IQR of neighbour coordinate j, for every own type (rows of phi_c)."""
    eff = np.asarray(c)[None, :] + phi_c @ np.asarray(B)      # (own types, 16): c + B' phi_a
    return (eff @ Q.T)[:, j] * iqr[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbour", default="Cancer Epithelial")
    ap.add_argument("--block", choices=("contact", "regional"), default="contact")
    a = ap.parse_args()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open(os.path.splitext(cfg["data"]["table_parquet"])[0] + ".phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    Q = basis.Q_nbr
    j = list(basis.nbr_names).index(f"{a.block}:{a.neighbour}")
    iqr = np.subtract(*np.percentile(df[basis.nbr_cols].to_numpy(float), [75, 25], axis=0))
    own = basis.own(df)
    lab = df["archetype"].astype(str).to_numpy()
    phi_c = np.stack([own[lab == t].mean(0) for t in ORDER])

    point, reps = {}, {}
    for name, d in FITS.items():
        P = json.load(open(f"{d}/point_estimate.json"))["params"]
        point[name] = column(P["c"], np.asarray(P["U"]) @ np.asarray(P["V"]).T, phi_c, Q, iqr, j)
        reps[name] = {r["rep"]: column(r["c"], r["B"], phi_c, Q, iqr, j)
                      for r in map(lambda f: json.load(open(f)), sorted(glob.glob(f"{d}/bootstrap/*.json")))
                      if r.get("B") is not None}
    (m, mr), (p, pr) = [(point[k], reps[k]) for k in FITS]
    common = sorted(set(mr) & set(pr))
    diff_b = np.stack([mr[r] - pr[r] for r in common])
    ci = lambda x: np.percentile(np.stack(list(x)), [2.5, 97.5], axis=0)  # noqa: E731
    (mlo, mhi), (plo, phi_), (dlo, dhi) = ci(mr.values()), ci(pr.values()), ci(diff_b)
    rows = [{"own type": t, "MisTIC": m[i], "MisTIC 95%": f"{mlo[i]:+.3f} to {mhi[i]:+.3f}",
             "plain": p[i], "plain 95%": f"{plo[i]:+.3f} to {phi_[i]:+.3f}",
             "difference (MisTIC - plain)": m[i] - p[i], "difference 95% (paired)": f"{dlo[i]:+.3f} to {dhi[i]:+.3f}",
             "difference excludes 0": bool(dlo[i] > 0 or dhi[i] < 0)} for i, t in enumerate(ORDER)]
    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    slug = a.neighbour.replace(" ", "_")
    path = f"{OUT}/{a.block}_effect_{slug}_mistic_vs_plain.csv"
    tab.to_csv(path, index=False)
    print(f"{a.block} effect of {a.neighbour} neighbours, log OR of cycling per IQR "
          f"(IQR {iqr[j]:.4f}); bootstrap: {len(mr)} MisTIC, {len(pr)} plain, {len(common)} paired")
    with pd.option_context("display.width", 220):
        print(tab.round(3).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
