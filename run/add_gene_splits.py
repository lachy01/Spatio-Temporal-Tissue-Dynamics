"""Add the Test 08 half-set counts to an existing table (spec 6.8).

    python run/add_gene_splits.py

The parquet carries only the summed Tirosh count, but the held-out gene test
needs the two halves separately. This reads the Xenium matrix, computes y_A and
y_B for each of the 5 phase-stratified splits, and joins them onto the existing
table -- avoiding a full rebuild, which would re-solve the oxygen PDE for nothing.

Run locally (the raw sample lives here, not on the cluster) and re-copy the
parquet afterwards.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_dynamics.gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES   # noqa: E402
from tissue_dynamics.mixture.data import load_config                     # noqa: E402
from tissue_dynamics.mixture.heldout import split_gene_set               # noqa: E402

N_SPLITS = 5


def main():
    cfg = load_config()
    path = cfg["data"]["table_parquet"]
    df = pd.read_parquet(path)
    print(f"{path}: {len(df):,} cells, {df.shape[1]} columns")

    import h5py
    import scipy.sparse as sp

    h5 = os.path.join(cfg["data"]["xenium_dir"], "cell_feature_matrix.h5")
    with h5py.File(h5, "r") as f:
        k = list(f.keys())[0]
        g = f[k]
        names = np.array([n.decode() for n in g["features"]["name"][:]])
        bc = np.array([b.decode() for b in g["barcodes"][:]])
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                          shape=tuple(g["shape"][:]))

    present = set(names)
    pos = {n: i for i, n in enumerate(names)}
    order = pd.Index(bc).get_indexer(df["cell_id"].astype(str))
    if (order < 0).any():
        raise SystemExit(f"{int((order < 0).sum())} cells in the table are absent "
                         "from the matrix; is this the right sample?")

    for s in range(N_SPLITS):
        A, B = split_gene_set(TIROSH_S_GENES, TIROSH_G2M_GENES, present, seed=s)
        ya = np.asarray(M[[pos[g_] for g_ in A], :].sum(0)).ravel()[order]
        yb = np.asarray(M[[pos[g_] for g_ in B], :].sum(0)).ravel()[order]
        df[f"y_A{s}"] = np.rint(ya).astype(np.int64)
        df[f"y_B{s}"] = np.rint(yb).astype(np.int64)
        print(f"  split {s}: |A|={len(A):2d} |B|={len(B):2d}  "
              f"mean y_A {ya.mean():.3f}  mean y_B {yb.mean():.3f}")

    tot = df["y_count"].to_numpy()
    recon = df["y_A0"].to_numpy() + df["y_B0"].to_numpy()
    print(f"\ncheck: y_A0 + y_B0 == y_count for "
          f"{100 * np.mean(recon == tot):.1f}% of cells "
          "(100% expected -- the halves partition the signature)")

    df.to_parquet(path, index=False)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
