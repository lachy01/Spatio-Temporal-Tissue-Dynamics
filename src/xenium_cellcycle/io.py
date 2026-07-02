"""Load a 10x Xenium sample's cell x gene matrix into an AnnData object."""

from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io as sio
from scipy.sparse import csr_matrix

CONTROL_PREFIXES = (
    "NegControlProbe_", "NegControlCodeword_", "BLANK_",
    "antisense_", "UnassignedCodeword_",
)


def find_outs_dir(path):
    """Locate the Xenium `outs/` folder given a sample root or the outs dir itself."""
    path = Path(path)
    if (path / "cells.csv.gz").exists():
        return path
    matches = list(path.rglob("cells.csv.gz"))
    if not matches:
        raise FileNotFoundError(f"No cells.csv.gz found under {path}")
    return matches[0].parent


def read_xenium(path, gex_only=True, drop_controls=True):
    """Read a Xenium sample's prebuilt cell x gene matrix into an AnnData.

    Parameters
    ----------
    path : str or Path
        Sample root directory or its `outs/` folder.
    gex_only : bool
        Keep only "Gene Expression" features via spatialdata-io (if
        installed); has no effect on the manual mtx fallback, where control
        features are always excluded by name via `drop_controls`.
    drop_controls : bool
        Drop any remaining negative-control / blank / unassigned-codeword
        features by name, as a safety net regardless of loader path.

    Returns
    -------
    AnnData with cells x genes counts, spatial centroids in `obsm["spatial"]`,
    and `obs["cell_area"]`.
    """
    outs_dir = find_outs_dir(path)

    try:
        from spatialdata_io import xenium as read_xenium_sdata

        sdata = read_xenium_sdata(
            str(outs_dir),
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_labels=False,
            nucleus_labels=False,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
            transcripts=False,
            cells_table=True,
            gex_only=gex_only,
        )
        adata = sdata.tables["table"]
    except ImportError:
        adata = _read_10x_mtx_manual(outs_dir)

    if "spatial" not in adata.obsm:
        cells = pd.read_csv(outs_dir / "cells.csv.gz")
        cells = cells.set_index("cell_id").loc[adata.obs_names]
        adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].values
        adata.obs["cell_area"] = cells["cell_area"].values

    if drop_controls:
        is_control = adata.var_names.str.startswith(CONTROL_PREFIXES)
        if is_control.any():
            adata = adata[:, ~is_control].copy()

    return adata


def _read_10x_mtx_manual(outs_dir):
    """Fallback reader if spatialdata-io isn't installed: raw mtx triplet."""
    mtx_dir = outs_dir / "cell_feature_matrix"
    matrix = sio.mmread(mtx_dir / "matrix.mtx.gz").tocsr().T.tocsr()  # cells x genes
    features = pd.read_csv(
        mtx_dir / "features.tsv.gz", sep="\t", header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    barcodes = pd.read_csv(
        mtx_dir / "barcodes.tsv.gz", sep="\t", header=None, names=["cell_id"],
    )
    return ad.AnnData(
        X=csr_matrix(matrix),
        obs=barcodes.set_index("cell_id"),
        var=features.set_index("gene_name"),
    )
