"""Read externally-trained Phi-Space cell-state embeddings and attach them to an AnnData."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_phi_table(path, cell_id_col="cell_id", sample_col=None, phi_prefix="phi_"):
    """Read a Phi-Space embedding table (CSV/CSV.gz/Parquet) into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Table with one row per cell: a `cell_id_col` column, an optional
        `sample_col` column (section/slide identifier), and one column per
        embedding dimension named `{phi_prefix}0`, `{phi_prefix}1`, ...
    cell_id_col, sample_col : str
        Column names identifying the cell (and, if given, its section).
    phi_prefix : str
        Prefix shared by every embedding-dimension column; all columns
        starting with this prefix are taken as phi coordinates, in the
        order they appear in the file.

    Returns
    -------
    DataFrame indexed by `cell_id_col` (and carrying `sample_col`, if any),
    with only the matched phi columns retained, renamed `phi_0..phi_{k-1}`.
    """
    path = Path(path)
    suffix = "".join(path.suffixes)
    if suffix.endswith(".parquet"):
        table = pd.read_parquet(path)
    elif suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        table = pd.read_csv(path)
    else:
        raise ValueError(f"Unrecognized Phi-Space table format for {path!r}; expected .csv, .csv.gz, or .parquet")

    if cell_id_col not in table.columns:
        raise ValueError(f"cell_id_col={cell_id_col!r} not found in Phi-Space table columns: {list(table.columns)}")
    if sample_col is not None and sample_col not in table.columns:
        raise ValueError(f"sample_col={sample_col!r} not found in Phi-Space table columns: {list(table.columns)}")

    phi_cols = [c for c in table.columns if c.startswith(phi_prefix)]
    if not phi_cols:
        raise ValueError(f"No columns starting with phi_prefix={phi_prefix!r} found in {list(table.columns)}")

    keep_cols = [cell_id_col] + ([sample_col] if sample_col else []) + phi_cols
    table = table[keep_cols].rename(columns={c: f"phi_{i}" for i, c in enumerate(phi_cols)})
    return table.set_index(cell_id_col)


def attach_phi(adata, phi, cell_id_col="cell_id", sample_col=None,
                sample_key="sample_id", phi_key="phi"):
    """Attach a Phi-Space embedding table to an AnnData's `obsm[phi_key]`.

    Joins on the raw per-section `cell_id` (see `read_xenium`'s
    `obs["cell_id"]`), combined with `obs[sample_key]` when `sample_col` is
    given — Xenium cell barcodes are only guaranteed unique *within* one
    section, so the composite key guards against silent cross-section
    collisions when `adata` holds multiple sections (e.g. from
    `load_xenium_samples`).

    Parameters
    ----------
    adata : AnnData
        Must have `obs["cell_id"]` (set by `read_xenium`) and, if
        `sample_col` is given, `obs[sample_key]`.
    phi : DataFrame
        As returned by `read_phi_table` (indexed by `cell_id_col`, phi
        columns named `phi_0..phi_{k-1}`, optionally carrying `sample_col`).

    Returns
    -------
    `adata`, modified in place, with `obsm[phi_key]` set (float array,
    n_obs x k) and `uns[f"{phi_key}_dim"]` recording k.
    """
    if "cell_id" not in adata.obs.columns:
        raise ValueError('adata.obs["cell_id"] not found; read the sample with read_xenium (or set it manually) before attaching phi')

    phi_cols = [c for c in phi.columns if c.startswith("phi_")]
    if not phi_cols:
        raise ValueError(f"No phi_* columns found in the phi table: {list(phi.columns)}")

    if sample_col is not None:
        if sample_key not in adata.obs.columns:
            raise ValueError(f"adata.obs[{sample_key!r}] not found; required when sample_col is given")
        phi_key_index = pd.MultiIndex.from_arrays([phi.index, phi[sample_col]])
        adata_key_index = pd.MultiIndex.from_arrays([adata.obs["cell_id"], adata.obs[sample_key]])
    else:
        phi_key_index = phi.index
        adata_key_index = pd.Index(adata.obs["cell_id"])

    phi_lookup = pd.DataFrame(phi[phi_cols].to_numpy(), index=phi_key_index, columns=phi_cols)
    phi_lookup = phi_lookup[~phi_lookup.index.duplicated(keep="first")]

    missing = ~adata_key_index.isin(phi_lookup.index)
    if missing.any():
        raise ValueError(
            f"{missing.sum()}/{adata.n_obs} cells in adata have no matching Phi-Space row "
            "(joined on cell_id" + (" + sample" if sample_col is not None else "") + ")."
        )

    aligned = phi_lookup.loc[adata_key_index]
    adata.obsm[phi_key] = aligned.to_numpy(dtype=np.float64)
    adata.uns[f"{phi_key}_dim"] = len(phi_cols)
    return adata
