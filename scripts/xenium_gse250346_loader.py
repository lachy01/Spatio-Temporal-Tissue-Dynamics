"""
Build a per-cell x gene count matrix directly from raw Xenium transcript
detections (GSE250346, Vannan et al., pulmonary fibrosis lung atlas,
Banovich lab), applying your own QC thresholds instead of trusting 10x's
onboard per-cell assignment blindly.

Downloading and reading the standard prebuilt cell_feature_matrix is handled
by the tissue_dynamics package now (`download_xenium_sample`, `read_xenium`)
-- this script only covers what isn't part of that package: an alternative
cell x gene matrix built straight from `transcripts.csv.gz`, with your own
quality-value filtering and (optionally) a stricter nuclear-overlap
requirement.

Dependencies (from the repo root):
    pip install -e ".[download]"
"""

from pathlib import Path

import pandas as pd


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


# --------------------------------------------------------------------------
# Example end-to-end usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from tissue_dynamics import download_xenium_sample, read_xenium

    GSM_ID = "GSM7990532"  # THD0008_Control_TMA3-Run2_5mm — swap for any sample

    # include_transcripts=True since that's the one file this script needs
    # that download_xenium_sample doesn't fetch by default.
    outs_dir = download_xenium_sample(GSM_ID, include_transcripts=True)

    adata = read_xenium(outs_dir)
    print(adata)
    adata.write_h5ad(outs_dir / f"{GSM_ID}_processed.h5ad")

    percell_counts = build_percell_counts_from_transcripts(outs_dir, qv_threshold=20.0)

    centroids = pd.DataFrame(
        adata.obsm["spatial"], columns=["x_centroid", "y_centroid"], index=adata.obs_names,
    )
    centroids["cell_area"] = adata.obs["cell_area"].to_numpy()
    percell_counts = percell_counts.join(centroids, how="inner")
    percell_counts.to_csv(outs_dir / f"{GSM_ID}_percell_counts_manual.csv")

    print(f"\nDone. {percell_counts.shape[0]} cells x "
          f"{percell_counts.shape[1] - 3} genes (manual QC path).")
