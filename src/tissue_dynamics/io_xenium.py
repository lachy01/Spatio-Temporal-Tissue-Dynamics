"""Load a 10x Xenium sample's cell x gene matrix into an AnnData object."""

import warnings
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


def read_xenium(path, gex_only=True, drop_controls=True, sample_id=None):
    """Read a Xenium sample's prebuilt cell x gene matrix into an AnnData.

    Parameters
    ----------
    path : str or Path
        Sample root directory or its `outs/` folder.
    gex_only : bool
        Keep only "Gene Expression" features via spatialdata-io (if
        installed and it succeeds); has no effect on the manual fallback
        reader, where control features are always excluded by name via
        `drop_controls`.
    drop_controls : bool
        Drop any remaining negative-control / blank / unassigned-codeword
        features by name, as a safety net regardless of loader path.
    sample_id : str, optional
        Section/slide identifier stored in `obs["sample_id"]`, used
        downstream to keep spatial-neighborhood computations from crossing
        tissue sections. Defaults to the `outs/` folder's parent directory
        name (i.e. the sample root folder name) if not given.

    Returns
    -------
    AnnData with cells x genes counts, spatial centroids in `obsm["spatial"]`,
    `obs["cell_area"]`, `obs["sample_id"]`, and `obs["cell_id"]` (the raw
    per-section Xenium cell barcode, preserved separately from `obs_names`
    since `load_xenium_samples` may need to suffix `obs_names` to keep them unique
    across sections).
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
    except Exception as exc:
        # spatialdata-io is installed but failed for some other reason (e.g.
        # a file its particular version expects -- which has shifted across
        # versions, such as preferring a zarr-backed cells table over
        # cells.parquet/cells.csv.gz -- isn't present in this outs/ folder).
        # The manual reader only needs the matrix + cells.csv.gz, so fall
        # back to it rather than hard-failing.
        warnings.warn(
            f"spatialdata-io failed to read this sample ({exc!r}); falling back "
            "to the manual cell_feature_matrix reader.",
            stacklevel=2,
        )
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

    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = sample_id if sample_id is not None else outs_dir.parent.name
    return adata


def load_xenium_samples(paths, sample_ids=None, **read_kwargs):
    """Read several Xenium sections and concatenate them into one AnnData.

    Parameters
    ----------
    paths : sequence of str or Path
        Sample root directories or their `outs/` folders, one per section.
    sample_ids : sequence of str, optional
        Section identifiers, one per entry in `paths` (defaults to each
        section's own `read_xenium`-derived `sample_id`). Must be unique.
    **read_kwargs
        Forwarded to `read_xenium` for every sample.

    Returns
    -------
    AnnData with all sections stacked (`anndata.concat`, `join="outer"`),
    `obs["sample_id"]` distinguishing sections, and `obsm["spatial"]`
    preserved. Cells keep each section's contiguous block, but downstream
    code (see `neighbors.build_adjacency`) does not assume this ordering.
    """
    if sample_ids is None:
        sample_ids = [None] * len(paths)
    if len(sample_ids) != len(paths):
        raise ValueError("sample_ids must be the same length as paths")

    adatas = [read_xenium(p, sample_id=sid, **read_kwargs) for p, sid in zip(paths, sample_ids)]
    resolved_ids = [a.obs["sample_id"].iloc[0] for a in adatas]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError(f"sample_id values must be unique; got {resolved_ids}")

    combined = ad.concat(adatas, join="outer", index_unique="-", keys=resolved_ids)
    return combined


def _read_10x_mtx_manual(outs_dir):
    """Fallback reader used when spatialdata-io isn't installed, or is
    installed but fails to read this sample. Prefers the single-file
    `cell_feature_matrix.h5` (via scanpy, already a core dependency; this is
    the same CellRanger-style h5 format `sc.read_10x_h5` expects), falling
    back to the raw mtx triplet if the h5 isn't present.
    """
    h5_path = outs_dir / "cell_feature_matrix.h5"
    if h5_path.exists():
        import scanpy as sc

        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
        return adata

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
