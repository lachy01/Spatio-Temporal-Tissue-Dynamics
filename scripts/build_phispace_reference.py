"""
build_phispace_reference.py

Fetch a breast scRNA-seq reference from the CELLxGENE Census, restricted to the
genes on a Xenium panel, and audit whether it is usable as a Phi-Space ST
training reference.

Why Census rather than raw GEO:
  - obs['cell_type'] is a standardised CL ontology field -> this is the Y matrix
    for the PLS regression. No guessing which of 14 .obs columns is the label.
  - You can filter by *gene* at read time. Since Phi-Space can only use genes
    present in both reference and query, and your Xenium panel is ~300 genes,
    you never need to materialise the full 30k-gene matrix. This turns a
    multi-GB atlas into a few hundred MB (or less, with subsampling).
  - is_primary_data == True de-duplicates cells that appear in several studies.

Install:
    pip install cellxgene-census scanpy anndata pandas

Usage:
    # 1. See what breast references exist and how they are annotated
    python build_phispace_reference.py survey --panel gene_panel.json

    # 2. Build the reference (writes ref_breast_panel.h5ad)
    python build_phispace_reference.py build --panel gene_panel.json \
        --out ref_breast_panel.h5ad --max-cells-per-type 2000

    # 3. Panel-only checks (no download): Tirosh cell-cycle overlap
    python build_phispace_reference.py panel-check --panel gene_panel.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CENSUS_VERSION = "2025-01-30"  # pin this; "stable" moves under you
ORGANISM = "Homo sapiens"

# Coarse lineages you need represented for an invasive-margin model.
# If a reference collapses everything into 4 labels, the PLS phenotype space
# will be too low-rank to resolve EMT-adjacent drift.
REQUIRED_LINEAGES = {
    "epithelial": ["epithelial", "luminal", "basal", "myoepithelial", "malignant"],
    "fibroblast": ["fibroblast", "stromal", "myofibroblast"],
    "endothelial": ["endothelial", "vascular", "lymphatic"],
    "myeloid": ["macrophage", "monocyte", "dendritic", "myeloid"],
    "lymphoid": ["T cell", "B cell", "NK", "plasma", "lymphocyte"],
}


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------
def load_panel(path: Path) -> list[str]:
    """Read gene symbols from a Xenium gene_panel.json, or a plain CSV/TXT."""
    if path.suffix == ".json":
        with open(path) as f:
            panel = json.load(f)
        genes = set()
        # Xenium panel JSON nests targets under payload -> targets -> type -> data
        for t in panel.get("payload", {}).get("targets", []):
            tt = t.get("type", {})
            data = tt.get("data", {})
            name = data.get("name")
            if name and tt.get("descriptor") == "gene":
                genes.add(name)
        if not genes:  # fall back: brute-force scan for any 'name' fields
            def walk(o):
                if isinstance(o, dict):
                    if "name" in o and isinstance(o["name"], str):
                        genes.add(o["name"])
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(panel)
        return sorted(genes)

    df = pd.read_csv(path, header=None)
    return sorted(set(df.iloc[:, 0].astype(str).str.strip()))


# ---------------------------------------------------------------------------
# Tirosh cell-cycle overlap (no download needed)
# ---------------------------------------------------------------------------
TIROSH_S = """MCM5 PCNA TYMS FEN1 MCM7 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1
HELLS RFC2 POLR1B NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2
RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B
MRPL36 E2F8""".split()

TIROSH_G2M = """HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B
MKI67 TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E
TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2
DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF
NEK2 G2E3 GAS2L3 CBX5 CENPA""".split()


def panel_check(panel_genes: list[str]) -> None:
    pset = set(panel_genes)
    print(f"Panel: {len(panel_genes)} genes\n")
    for name, genes in [("Tirosh S-phase", TIROSH_S), ("Tirosh G2/M", TIROSH_G2M)]:
        hit = sorted(pset & set(genes))
        print(f"{name}: {len(hit)}/{len(genes)} present")
        print(f"  present: {', '.join(hit) if hit else '(none)'}")
        print(f"  missing: {', '.join(sorted(set(genes) - pset))}\n")

    # A cell-cycle score built on <5 genes per phase is noise. Flag it.
    n_s = len(pset & set(TIROSH_S))
    n_g2m = len(pset & set(TIROSH_G2M))
    if n_s < 5 or n_g2m < 5:
        print("!! WARNING: too few cell-cycle genes for a stable phase score.")
        print("   Consider MKI67/TOP2A raw counts as the proliferation proxy instead.")
    else:
        print(f"OK: {n_s} S + {n_g2m} G2/M genes -> scanpy score_genes_cell_cycle is viable.")


# ---------------------------------------------------------------------------
# Census survey / build
# ---------------------------------------------------------------------------
def open_census():
    try:
        import cellxgene_census
    except ImportError:
        sys.exit("pip install cellxgene-census")
    return cellxgene_census, cellxgene_census.open_soma(census_version=CENSUS_VERSION)


OBS_FILTER = (
    "tissue_general == 'breast' and is_primary_data == True "
    "and assay in ['10x 3\\' v2', '10x 3\\' v3', '10x 5\\' v1', '10x 5\\' v2']"
)


def survey(panel_genes: list[str]) -> None:
    """Pull only obs metadata (cheap) to see what's available and how it's labelled."""
    cellxgene_census, census = open_census()
    with census:
        obs = cellxgene_census.get_obs(
            census,
            ORGANISM,
            value_filter=OBS_FILTER,
            column_names=["cell_type", "disease", "assay", "donor_id", "dataset_id"],
        )
    print(f"{len(obs):,} breast cells across {obs.dataset_id.nunique()} datasets, "
          f"{obs.donor_id.nunique()} donors\n")

    print("--- disease ---")
    print(obs.disease.value_counts().to_string(), "\n")

    print("--- cell_type (top 40) ---")
    ct = obs.cell_type.value_counts()
    print(ct.head(40).to_string())
    print(f"\n{len(ct)} distinct cell types; "
          f"{(ct >= 50).sum()} with >=50 cells (usable as PLS classes)\n")

    print("--- lineage coverage ---")
    labels = " | ".join(ct.index).lower()
    for lineage, keys in REQUIRED_LINEAGES.items():
        found = [k for k in keys if k.lower() in labels]
        mark = "OK " if found else "MISSING"
        print(f"  {mark} {lineage:12s} {found}")

    print("\n--- dataset breakdown (pick tumour + normal to combine) ---")
    print(obs.groupby(["dataset_id", "disease"]).size()
             .sort_values(ascending=False).head(15).to_string())


def build(panel_genes: list[str], out: Path, max_cells_per_type: int,
          disease: str | None) -> None:
    import numpy as np
    import scanpy as sc

    cellxgene_census, census = open_census()

    obs_filter = OBS_FILTER
    if disease:
        obs_filter += f" and disease == '{disease}'"

    var_filter = "feature_name in [" + ", ".join(f"'{g}'" for g in panel_genes) + "]"

    print("Querying Census (panel genes only)...")
    with census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=ORGANISM,
            obs_value_filter=obs_filter,
            var_value_filter=var_filter,
            obs_column_names=["cell_type", "disease", "assay", "donor_id",
                              "dataset_id", "tissue"],
        )

    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()

    found = set(adata.var_names)
    missing = sorted(set(panel_genes) - found)
    print(f"\nPanel genes recovered: {len(found)}/{len(panel_genes)}")
    if missing:
        print(f"Missing (symbol mismatch or not in Census): {', '.join(missing[:20])}"
              f"{' ...' if len(missing) > 20 else ''}")
        print("  -> check these are not aliases (e.g. FYB1/FYB, JPT1/HN1) before "
              "accepting the loss; alias mismatch silently drops informative genes.")

    # Drop rare types: a PLS class with 20 cells contributes an unstable column to Y.
    counts = adata.obs.cell_type.value_counts()
    keep_types = counts[counts >= 50].index
    dropped = sorted(set(counts.index) - set(keep_types))
    if dropped:
        print(f"\nDropping {len(dropped)} cell types with <50 cells: {', '.join(dropped[:10])}")
    adata = adata[adata.obs.cell_type.isin(keep_types)].copy()

    # Balance classes. PLS is sensitive to class imbalance in Y; a reference that
    # is 60% T cells will bias the phenotype embedding toward the lymphoid axis.
    rng = np.random.default_rng(0)
    idx = []
    for ct, grp in adata.obs.groupby("cell_type", observed=True).groups.items():
        grp = np.asarray(grp)
        if len(grp) > max_cells_per_type:
            grp = rng.choice(grp, max_cells_per_type, replace=False)
        idx.extend(grp)
    adata = adata[adata.obs_names.isin(idx)].copy()

    # Standard Phi-Space input: log-normalised counts.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.write_h5ad(out, compression="gzip")
    size_mb = out.stat().st_size / 1e6

    print(f"\nWrote {out} ({size_mb:.1f} MB)")
    print(f"  {adata.n_obs:,} cells x {adata.n_vars} genes")
    print(f"  {adata.obs.cell_type.nunique()} cell types "
          f"(these become the {adata.obs.cell_type.nunique()} columns of Y)")
    print(f"  {adata.obs.donor_id.nunique()} donors "
          f"-> use donor as the CV fold, not random split, or you overstate accuracy")
    print("\nFinal class balance:")
    print(adata.obs.cell_type.value_counts().to_string())


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["survey", "build", "panel-check"])
    p.add_argument("--panel", type=Path, required=True,
                   help="Xenium gene_panel.json, or a CSV/TXT with one gene per line")
    p.add_argument("--out", type=Path, default=Path("ref_breast_panel.h5ad"))
    p.add_argument("--max-cells-per-type", type=int, default=2000)
    p.add_argument("--disease", default=None,
                   help="e.g. 'breast cancer' or 'normal'. Omit to keep both.")
    args = p.parse_args()

    panel_genes = load_panel(args.panel)
    if not panel_genes:
        sys.exit(f"No genes parsed from {args.panel}")

    if args.mode == "panel-check":
        panel_check(panel_genes)
    elif args.mode == "survey":
        survey(panel_genes)
    else:
        build(panel_genes, args.out, args.max_cells_per_type, args.disease)


if __name__ == "__main__":
    main()