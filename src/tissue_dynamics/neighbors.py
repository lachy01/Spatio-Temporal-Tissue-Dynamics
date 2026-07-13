"""Build a spatial neighborhood graph and average neighbor Phi-Space embeddings.

Adapted from """

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import radius_neighbors_graph


def build_adjacency(adata, radius=80.0, sample_key="sample_id", spatial_key="spatial",
                      connectivity_key="spatial_connectivity"):
    """Build a sparse 0/1 spatial-neighborhood adjacency matrix.

    Two cells are adjacent iff they are in the same `obs[sample_key]` group
    (tissue section/slide) and within `radius` (Euclidean, same units as
    `obsm[spatial_key]`, typically micrometers) of each other. Neighbors are
    never computed across different sections.

    Parameters
    ----------
    adata : AnnData
        Must have `obs[sample_key]` and `obsm[spatial_key]`.
    radius : float
        Neighborhood radius (e.g. 80.0 micrometers).

    Returns
    -------
    `adata`, modified in place: `obsp[connectivity_key]` (sparse, symmetric,
    0/1, csr, zero diagonal), `obs["n_spatial_neighbors"]`, `obs["is_isolated"]`,
    `uns[f"{connectivity_key}_params"]`.
    """
    if sample_key not in adata.obs.columns:
        raise ValueError(f'adata.obs[{sample_key!r}] not found')
    if spatial_key not in adata.obsm:
        raise ValueError(f'adata.obsm[{spatial_key!r}] not found')

    coords = np.asarray(adata.obsm[spatial_key])
    sample_ids = adata.obs[sample_key].to_numpy()
    n = adata.n_obs

    order_chunks = []
    blocks = []
    for s in pd.unique(sample_ids):
        idx = np.flatnonzero(sample_ids == s)
        order_chunks.append(idx)
        if len(idx) > 1:
            block = radius_neighbors_graph(
                coords[idx], radius, mode="connectivity", include_self=False,
            )
        else:
            block = sp.csr_matrix((len(idx), len(idx)))
        blocks.append(block)

    order = np.concatenate(order_chunks)
    A = sp.block_diag(blocks, format="csr")
    inv = np.argsort(order)
    A = A[inv][:, inv]

    # Defensive symmetrization: a Euclidean radius query is mathematically
    # symmetric, but don't rely on that holding bit-for-bit across sklearn's
    # internal tree implementations.
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocsr()

    degree = np.asarray(A.sum(axis=1)).ravel()
    is_isolated = degree == 0
    n_isolated = int(is_isolated.sum())
    if n_isolated == n and n > 0:
        warnings.warn(
            f"All {n} cells have no spatial neighbors within radius={radius} -- every "
            "neighbor-averaged embedding will be the zero vector, silently dropping the "
            "c^T phi_bar and bilinear interaction terms from the model entirely. This "
            "usually means radius is mismatched to this data's spatial resolution (e.g. "
            "the 80um default is tuned for Xenium's single-cell spacing, not Visium's "
            "~100um spot pitch) rather than a real absence of neighbors -- check the "
            "typical nearest-neighbor spacing in obsm['spatial'] and raise radius to "
            "comfortably exceed it.",
            stacklevel=2,
        )
    elif n_isolated:
        warnings.warn(
            f"{n_isolated}/{n} cells have no spatial neighbors within radius={radius} "
            "(isolated); their neighbor-averaged embedding will be the zero vector.",
            stacklevel=2,
        )

    adata.obsp[connectivity_key] = A
    adata.obs["n_spatial_neighbors"] = degree.astype(int)
    adata.obs["is_isolated"] = is_isolated
    adata.uns[f"{connectivity_key}_params"] = {
        "radius": radius, "sample_key": sample_key, "n_isolated": n_isolated,
    }
    return adata


def average_neighbors(adata, phi_key="phi", phi_bar_key="phi_bar",
                        connectivity_key="spatial_connectivity"):
    """Compute each cell's row-normalized neighbor-averaged Phi-Space embedding.

    `phi_bar_i = (D^-1 A) @ phi`, where `A` is the adjacency built by
    `build_adjacency` and `D` its degree diagonal. Isolated cells (degree 0)
    get `phi_bar = 0`.

    Returns
    -------
    `adata`, modified in place, with `obsm[phi_bar_key]` set.
    """
    if connectivity_key not in adata.obsp:
        raise ValueError(f"adata.obsp[{connectivity_key!r}] not found; call build_adjacency first")
    if phi_key not in adata.obsm:
        raise ValueError(f"adata.obsm[{phi_key!r}] not found; call attach_phi first")

    A = adata.obsp[connectivity_key]
    phi = np.asarray(adata.obsm[phi_key])

    degree = np.asarray(A.sum(axis=1)).ravel()
    inv_degree = np.zeros_like(degree, dtype=np.float64)
    np.divide(1.0, degree, out=inv_degree, where=degree > 0)
    W = sp.diags(inv_degree) @ A

    adata.obsm[phi_bar_key] = np.asarray(W @ phi)
    return adata
