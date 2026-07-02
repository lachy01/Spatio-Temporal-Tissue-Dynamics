"""
Download and process Xenium data from GSE250346 (Vannan et al., pulmonary
fibrosis lung atlas, Banovich lab) into per-cell transcript tables.

GSE250346 supplementary files are the standard 10x Xenium "outs" bundle per
sample (cell_feature_matrix, cells.csv.gz, transcripts.csv.gz, images) packed
as a per-GSM tar.gz, plus standalone transcripts.csv.gz for some multi-region
runs. This script:

  1. Downloads a sample's supplementary bundle from GEO given a GSM accession.
  2. Extracts the Xenium outs/ folder.
  3. Loads it two ways:
       (a) fast path  -> prebuilt cell x gene count matrix (cell_feature_matrix)
       (b) manual path -> per-cell counts built directly from transcripts.csv.gz,
           with QC (quality-value filtering, control-probe exclusion)

NOTE: NCBI/GEO domains are not reachable from Anthropic's sandboxed bash tool
that generated this script, so this has not been executed end-to-end here.
Run it in your own environment (needs internet access to ncbi.nlm.nih.gov).

Dependencies:
    pip install pandas scipy anndata spatialdata spatialdata-io requests

Note: squidpy.read.xenium() has been removed from recent squidpy releases.
The current, actively maintained loader lives in spatialdata-io instead
(spatialdata_io.xenium), which is what this script uses now.
"""

import gzip
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import requests
from scipy import io as sio
from scipy.sparse import csr_matrix
import anndata as ad


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------

def download_gsm_bundle(gsm_id: str, dest_dir: str = "./xenium_downloads") -> Path:
    """
    Download all supplementary files for a given GSM accession from GEO.

    GEO's "format=file" endpoint returns a tar containing every supplementary
    file attached to that record (for Xenium samples: the outs/ bundle).

    Parameters
    ----------
    gsm_id : str
        e.g. "GSM7990532"
    dest_dir : str
        Local directory to save into.

    Returns
    -------
    Path to the downloaded tar file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{gsm_id}.tar"

    url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm_id}&format=file"
    print(f"Downloading {gsm_id} from {url} ...")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print(f"\nSaved to {out_path}")
    return out_path


def extract_bundle(tar_path: Path, dest_dir: str = "./xenium_downloads") -> Path:
    """
    Extract a downloaded GSM tar, then unpack any nested .tar.gz Xenium
    'outs' bundle found inside it. Returns the path to the extracted outs/
    folder (or the extraction root if structure differs).
    """
    dest_dir = Path(dest_dir)
    extract_root = dest_dir / tar_path.stem
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_root)

    # GEO bundles often contain a nested *_output-XETG*.tar.gz — unpack that too
    for nested in extract_root.glob("*.tar.gz"):
        print(f"Unpacking nested archive {nested.name} ...")
        with tarfile.open(nested) as tf:
            tf.extractall(extract_root)

    return extract_root


# --------------------------------------------------------------------------
# 2a. Fast path: load the prebuilt cell x gene matrix
# --------------------------------------------------------------------------

def load_via_prebuilt_matrix(outs_dir: Path) -> ad.AnnData:
    """
    Load a Xenium outs/ folder using the prebuilt cell_feature_matrix and
    cells.csv.gz.

    Uses spatialdata-io's xenium() reader (the current, maintained loader —
    squidpy.read.xenium() was removed from recent squidpy releases and no
    longer exists). Falls back to a manual mtx read if spatialdata-io isn't
    installed.
    """
    try:
        from spatialdata_io import xenium as read_xenium

        # gex_only=True keeps just the "Gene Expression" features (drops
        # control-probe / control-codeword feature types automatically).
        # We skip the heavy image/label layers since we only need counts +
        # coordinates for neighborhood construction.
        sdata = read_xenium(
            str(outs_dir),
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_labels=False,
            nucleus_labels=False,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
            transcripts=False,  # we read transcripts.csv.gz separately if needed
            cells_table=True,
            gex_only=True,
        )
        adata = sdata.tables["table"]

    except ImportError:
        adata = _read_10x_mtx_manual(outs_dir)

    # spatialdata-io already puts centroids in adata.obsm['spatial'];
    # if using the manual fallback, merge cells.csv.gz coords in ourselves.
    if "spatial" not in adata.obsm:
        cells = pd.read_csv(outs_dir / "cells.csv.gz")
        cells = cells.set_index("cell_id").loc[adata.obs_names]
        adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].values
        adata.obs["cell_area"] = cells["cell_area"].values

    # Belt-and-braces: drop any remaining control probes (blanks, negative
    # controls) from the gene panel. gex_only=True should already exclude
    # these, but this is a cheap safety net in case a panel version doesn't
    # tag them consistently.
    control_prefixes = ("NegControlProbe_", "NegControlCodeword_", "BLANK_",
                         "antisense_", "UnassignedCodeword_")
    is_control = adata.var_names.str.startswith(control_prefixes)
    if is_control.any():
        adata = adata[:, ~is_control].copy()

    return adata


def _read_10x_mtx_manual(outs_dir: Path) -> ad.AnnData:
    """Manual fallback if squidpy isn't available: read the raw mtx triplet."""
    mtx_dir = outs_dir / "cell_feature_matrix"
    matrix = sio.mmread(mtx_dir / "matrix.mtx.gz").tocsr().T.tocsr()  # cells x genes
    features = pd.read_csv(mtx_dir / "features.tsv.gz", sep="\t", header=None,
                            names=["gene_id", "gene_name", "feature_type"])
    barcodes = pd.read_csv(mtx_dir / "barcodes.tsv.gz", sep="\t", header=None,
                            names=["cell_id"])
    adata = ad.AnnData(
        X=csr_matrix(matrix),
        obs=barcodes.set_index("cell_id"),
        var=features.set_index("gene_name"),
    )
    return adata


# --------------------------------------------------------------------------
# 2b. Manual path: build per-cell counts directly from transcripts.csv.gz
# --------------------------------------------------------------------------

def build_percell_counts_from_transcripts(
    outs_dir: Path,
    qv_threshold: float = 20.0,
    require_nuclear: bool = False,
) -> pd.DataFrame:
    """
    Build a per-cell x gene count matrix directly from the raw transcript
    detection file, applying your own QC thresholds instead of trusting
    10x's default assignment blindly.

    transcripts.csv.gz columns (10x Xenium Onboard Analysis output):
        transcript_id, cell_id, overlaps_nucleus, feature_name,
        x_location, y_location, z_location, qv, ...

    Parameters
    ----------
    qv_threshold : float
        Minimum Phred-like quality value (10x default filter is qv >= 20,
        i.e. ~1% expected decoding error rate). Raise this if you want
        stricter per-transcript confidence for a noisy proxy like a
        cell-cycle module score.
    require_nuclear : bool
        If True, keep only transcripts assigned within the nuclear boundary
        (stricter, throws away real cytoplasmic signal — usually not what
        you want, but sometimes used to sanity-check segmentation quality).

    Returns
    -------
    DataFrame: rows = cell_id, columns = gene, values = transcript counts.
    """
    transcripts_path = outs_dir / "transcripts.csv.gz"
    if not transcripts_path.exists():
        # standalone per-slide files in this GSE are named *_transcripts.csv.gz
        candidates = list(outs_dir.glob("*_transcripts.csv.gz"))
        if not candidates:
            raise FileNotFoundError(f"No transcripts.csv.gz found under {outs_dir}")
        transcripts_path = candidates[0]

    control_prefixes = ("NegControlProbe_", "NegControlCodeword_", "BLANK_",
                         "antisense_", "UnassignedCodeword_")

    counts = {}
    chunksize = 2_000_000
    print(f"Streaming {transcripts_path.name} in chunks of {chunksize:,} rows ...")

    for i, chunk in enumerate(pd.read_csv(transcripts_path, chunksize=chunksize)):
        # basic QC
        chunk = chunk[chunk["qv"] >= qv_threshold]
        chunk = chunk[chunk["cell_id"] != -1]  # unassigned to any cell
        chunk = chunk[~chunk["feature_name"].str.startswith(control_prefixes)]
        if require_nuclear and "overlaps_nucleus" in chunk.columns:
            chunk = chunk[chunk["overlaps_nucleus"] == 1]

        chunk_counts = (
            chunk.groupby(["cell_id", "feature_name"]).size()
            .rename("count").reset_index()
        )
        for cell_id, gene, n in chunk_counts.itertuples(index=False):
            counts.setdefault(cell_id, {})[gene] = counts.get(cell_id, {}).get(gene, 0) + n

        print(f"\r  processed chunk {i + 1}", end="")

    print("\nPivoting to per-cell x gene matrix ...")
    df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)
    df.index.name = "cell_id"
    return df


def load_cell_centroids(outs_dir: Path) -> pd.DataFrame:
    """Load per-cell spatial centroids and QC metadata (for neighborhood construction)."""
    cells = pd.read_csv(outs_dir / "cells.csv.gz")
    return cells.set_index("cell_id")[["x_centroid", "y_centroid", "cell_area"]]


# --------------------------------------------------------------------------
# Example end-to-end usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    GSM_ID = "GSM7990532"  # THD0008_Control_TMA3-Run2_5mm — swap for any sample

    tar_path = download_gsm_bundle(GSM_ID)
    outs_dir = extract_bundle(tar_path)

    # If the outs/ folder is nested one level deeper, adjust here:
    if not (outs_dir / "cells.csv.gz").exists():
        nested_dirs = [d for d in outs_dir.iterdir() if d.is_dir()]
        if nested_dirs:
            outs_dir = nested_dirs[0]

    # --- Fast path ---
    adata = load_via_prebuilt_matrix(outs_dir)
    print(adata)
    adata.write_h5ad(outs_dir / f"{GSM_ID}_processed.h5ad")

    # --- Manual path (slower, but full QC control) ---
    percell_counts = build_percell_counts_from_transcripts(outs_dir, qv_threshold=20.0)
    centroids = load_cell_centroids(outs_dir)
    percell_counts = percell_counts.join(centroids, how="inner")
    percell_counts.to_csv(outs_dir / f"{GSM_ID}_percell_counts_manual.csv")

    print(f"\nDone. {percell_counts.shape[0]} cells x "
          f"{percell_counts.shape[1] - 3} genes (manual QC path).")
