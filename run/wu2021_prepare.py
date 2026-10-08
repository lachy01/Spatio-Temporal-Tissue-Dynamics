"""Per-cell cell-cycle phase and receptor genes for the Wu et al. 2021 atlas.

    python run/wu2021_prepare.py

The division comparison's spec 5.2 anchors the per-type cycling fractions to
Wu et al. 2021 (GSE176078; 26 primary breast tumours, 10x Chromium). Its
per-cell metadata carries patient, clinical subtype and `celltype_major` -- the
same nine types as the Xenium PhiSpace reference -- but no phase calls, so the
spec's fallback applies: Seurat's CellCycleScoring, run here through scanpy's
port of it (`score_genes_cell_cycle`: the Tirosh S and G2M sets, control genes
drawn from 25 expression bins, `ctrl_size` = the smaller set's size, as
Seurat's own default; phase S or G2M where the larger score is positive, else
G1). Expression is Seurat's LogNormalize: counts per 10,000, then log1p.

Inputs, from GEO's FTP (the GEO web page itself is behind a CAPTCHA):
  data/wu2021/Wu_etal_2021_BRCA_scRNASeq/{count_matrix_sparse.mtx,
      count_matrix_genes.tsv, count_matrix_barcodes.tsv, metadata.csv}
  from GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz (533 MB).

Writes data/wu2021/wu_cells.parquet (one row per cell: the metadata, S and
G2M scores, phase, and raw counts of ESR1, PGR and ERBB2) and
data/wu2021/wu_cells.json (gene-set coverage and the scoring settings). The
2.4 GB matrix is read once, here; later stages read the parquet.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES  # noqa: E402

DIR = "data/wu2021/Wu_etal_2021_BRCA_scRNASeq"
OUT = "data/wu2021"
# HGNC renames since Tirosh et al. 2016, resolved against the atlas's gene list
ALIASES = {"MLF1IP": "CENPU", "FAM64A": "PIMREG", "HN1": "JPT1"}
RECEPTORS = ("ESR1", "PGR", "ERBB2")


def say(*a):
    print(*a, flush=True)


def resolve(genes, present):
    out, missing = [], []
    for g in genes:
        if g in present:
            out.append(g)
        elif ALIASES.get(g) in present:
            out.append(ALIASES[g])
        else:
            missing.append(g)
    return out, missing


def main():
    import anndata as ad
    import pyarrow.csv as pacsv
    import scanpy as sc
    import scipy.sparse as sp

    t0 = time.time()
    genes = pd.read_csv(f"{DIR}/count_matrix_genes.tsv", header=None)[0].astype(str).to_numpy()
    barcodes = pd.read_csv(f"{DIR}/count_matrix_barcodes.tsv", header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(f"{DIR}/metadata.csv", index_col=0)
    say(f"{len(genes):,} genes x {len(barcodes):,} cells; metadata {meta.shape}")

    # MatrixMarket coordinate file: 2 header lines (banner, then dimensions)
    with open(f"{DIR}/count_matrix_sparse.mtx") as fh:
        fh.readline()
        n_g, n_c, nnz = (int(v) for v in fh.readline().split())
    tab = pacsv.read_csv(f"{DIR}/count_matrix_sparse.mtx",
                         read_options=pacsv.ReadOptions(skip_rows=2, autogenerate_column_names=True),
                         parse_options=pacsv.ParseOptions(delimiter=" "),
                         convert_options=pacsv.ConvertOptions(column_types={"f0": "int32", "f1": "int32",
                                                                           "f2": "float32"}))
    if tab.num_rows != nnz or n_g != len(genes) or n_c != len(barcodes):
        sys.exit(f"matrix header ({n_g} x {n_c}, {nnz} entries) does not match the files "
                 f"({len(genes)} genes, {len(barcodes)} cells, {tab.num_rows} entries)")
    gi = tab.column("f0").to_numpy() - 1
    ci = tab.column("f1").to_numpy() - 1
    v = tab.column("f2").to_numpy()
    del tab
    X = sp.csr_matrix((v, (ci, gi)), shape=(n_c, n_g), dtype=np.float32)
    del gi, ci, v
    say(f"read {nnz:,} entries in {time.time() - t0:.0f}s")

    meta = meta.reindex(barcodes)
    if meta["celltype_major"].isna().any():
        sys.exit("barcodes missing from metadata.csv")
    adata = ad.AnnData(X=X, obs=meta.copy(), var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    present = set(adata.var_names)
    raw = {g: np.asarray(adata[:, g].X.todense()).ravel() for g in RECEPTORS}

    s_genes, s_miss = resolve(TIROSH_S_GENES, present)
    g2m_genes, g2m_miss = resolve(TIROSH_G2M_GENES, present)
    say(f"S genes {len(s_genes)}/{len(TIROSH_S_GENES)} (missing {s_miss}); "
        f"G2M genes {len(g2m_genes)}/{len(TIROSH_G2M_GENES)} (missing {g2m_miss})")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, random_state=0)

    out = adata.obs[["orig.ident", "subtype", "celltype_major", "celltype_minor",
                     "nCount_RNA", "nFeature_RNA", "S_score", "G2M_score", "phase"]].copy()
    out = out.rename(columns={"orig.ident": "patient"})
    for g in RECEPTORS:
        out[g] = raw[g].astype(np.int32)
    out.index.name = "barcode"
    out.reset_index().to_parquet(f"{OUT}/wu_cells.parquet", index=False)
    json.dump(dict(n_cells=int(len(out)), n_genes=int(len(genes)), s_genes=s_genes, g2m_genes=g2m_genes,
                   s_missing=s_miss, g2m_missing=g2m_miss, aliases=ALIASES,
                   normalisation="normalize_total(1e4) + log1p (Seurat LogNormalize)",
                   scoring="scanpy.tl.score_genes_cell_cycle (port of Seurat CellCycleScoring), "
                           "random_state 0, default n_bins 25, ctrl_size = min(|S|, |G2M|)"),
              open(f"{OUT}/wu_cells.json", "w"), indent=2)
    frac = out.assign(cyc=out["phase"].isin(["S", "G2M"])).groupby("celltype_major")["cyc"].mean()
    say("S/G2M fraction by type (all patients):\n" + frac.sort_values().round(3).to_string())
    say(f"wrote {OUT}/wu_cells.parquet in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
