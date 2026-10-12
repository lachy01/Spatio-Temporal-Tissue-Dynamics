"""Add the MisTIC technical covariates to the mixture table.

    python run/add_mistic_columns.py            # in place
    python run/add_mistic_columns.py --out X    # write a copy instead

`model.technical: mistic` reads, per analysis cell:

  Nt_nonB, Nt      MisTIC-weighted library size, over genes outside the blocklist
                   (the models' library size) and over all genes
  N_all            raw count over all genes (the spec's N_i)
  f_mistic         contamination fraction (mean p over genes outside the blocklist)
  s_cyc, m_cyc     neighbour share and expected misassigned count, Tirosh set
  in_mistic        whether the cell was in MisTIC's input (False: p = 0)
  m_A{s}, m_B{s}   m for each half of the held-out-gene splits (Test 08), so the
                   half-A fit and the half-B nuisance model carry their own m

from data/mistic/cell_mistic.parquet (run/mistic_covariates.py). Every other
column, and the cell set and order, stay as they are, so the CV folds and
the stored labels still line up.

Checks, all fatal: every table cell is in cell_mistic.parquet; its N_nonB
equals the table's N and its Y_cyc the table's y_count (the same library size
and outcome); Nt_nonB > 0.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics import mistic as M                                # noqa: E402
from tissue_dynamics.gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES  # noqa: E402
from tissue_dynamics.mixture.data import load_config                   # noqa: E402
from tissue_dynamics.mixture.heldout import split_gene_set             # noqa: E402

N_SPLITS = 5            # as run/add_gene_splits.py
COLS = ["Nt_nonB", "Nt", "N_all", "f_mistic", "s_cyc", "m_cyc", "in_mistic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="write here instead of in place")
    a = ap.parse_args()

    cfg = load_config()
    path = cfg["data"]["table_parquet"]
    df = pd.read_parquet(path)
    ids = df["cell_id"].astype(str).to_numpy()
    print(f"{path}: {len(df):,} cells, {df.shape[1]} columns", flush=True)

    cov = M.load_covariates(cfg)
    cix = pd.Index(cov.counts.cell_ids).get_indexer(ids)
    if (cix < 0).any():
        sys.exit(f"{int((cix < 0).sum())} table cells are missing from cell_mistic.parquet")
    tab = M.load_cell_table(cfg).reindex(ids)
    if not np.array_equal(tab["N_nonB"].to_numpy(), df["N"].to_numpy(float)):
        sys.exit("cell_mistic N_nonB differs from the table's N")
    if not np.array_equal(tab["Y_cyc"].to_numpy(), df["y_count"].to_numpy(float)):
        sys.exit("cell_mistic Y_cyc differs from the table's y_count")
    if not (tab["Nt_nonB"].to_numpy() > 0).all():
        sys.exit("Nt_nonB is not positive for every analysis cell")

    new = pd.DataFrame({"Nt_nonB": tab["Nt_nonB"].to_numpy(), "Nt": tab["Nt"].to_numpy(),
                        "N_all": tab["N"].to_numpy(), "f_mistic": tab["f"].to_numpy(),
                        "s_cyc": tab["s_cyc"].to_numpy(), "m_cyc": tab["m_cyc"].to_numpy(),
                        "in_mistic": tab["in_mistic"].to_numpy(bool)})
    present = set(cov.counts.genes)
    for s in range(N_SPLITS):
        A, B = split_gene_set(TIROSH_S_GENES, TIROSH_G2M_GENES, present, seed=s)
        for half, genes in (("A", A), ("B", B)):
            col = f"y_{half}{s}"
            if col in df.columns:
                yc = cov.counts.gene_sum(genes)[cix]
                if not np.array_equal(yc, df[col].to_numpy(float)):
                    sys.exit(f"{col}: the split's genes do not reproduce the table's counts")
            new[f"m_{half}{s}"] = cov.m_gene_set(genes)[0][cix]
    for c in new.columns:
        df[c] = new[c].to_numpy()

    out = a.out or path
    df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    m = new["m_cyc"].to_numpy()
    print(f"wrote {out}: +{len(new.columns)} columns; m_cyc mean {m.mean():.4f}, "
          f"Nt_nonB / N median {np.median(new['Nt_nonB'] / df['N']):.3f}, "
          f"{int((~new['in_mistic']).sum())} cells outside MisTIC's input (p = 0)")


if __name__ == "__main__":
    main()
