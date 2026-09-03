"""Per-cell technical covariates (library size, local density) for the fate model.

Any count-derived outcome correlates with sequencing depth, and local cell
density correlates with both depth and segmentation quality — and, in the OSDR
picture, density is itself a substantive driver (logistic saturation at maximum
tissue density). Without these in the model, density-driven detection artefacts
and genuine density effects are entangled. These helpers compute the two
covariates and assemble a standardized matrix to feed the model as linear terms.
"""

import warnings

import numpy as np


def compute_library_size(adata, counts_layer="counts", key="total_counts",
                          log_key="log_total_counts"):
    """Per-cell library size (total raw counts) and its log.

    Raw counts are read from `layers[counts_layer]` if present (where
    `ensure_lognormalized` stashes them), else from `X`. Adds `obs[key]` and
    `obs[log_key]`; cells with zero library size are clipped to 1 before the
    log (with a warning) so `log_total_counts` stays finite — but they are
    degenerate and usually already dropped upstream.
    """
    counts = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    total = np.asarray(counts.sum(1)).ravel().astype(float)

    n_zero = int((total <= 0).sum())
    if n_zero:
        warnings.warn(
            f"{n_zero}/{adata.n_obs} cells have zero library size; clipping to 1 for the log "
            "(these near-empty cells are best filtered out before modeling).",
            stacklevel=2,
        )
    adata.obs[key] = total
    adata.obs[log_key] = np.log(np.clip(total, 1.0, None))
    return adata


def compute_local_density(adata, radius=50.0, spatial_key="spatial", sample_key="sample_id",
                           key="local_density"):
    """Per-cell local cell density: cells within `radius`, per (100 µm)^2.

    Counts each cell's neighbors within `radius` (Euclidean, same units as
    `obsm[spatial_key]`, including the cell itself) per section, and converts
    to a density in cells per (100 µm)^2 area = count / (pi * radius^2 / 1e4).
    Computed independently of the fate-model adjacency, so its radius can
    differ (density is a local property; the model neighborhood is typically
    wider). Adds `obs[key]`, records the radius in `uns[key + "_radius"]`.
    """
    import pandas as pd
    from sklearn.neighbors import radius_neighbors_graph

    coords = np.asarray(adata.obsm[spatial_key])
    samples = (adata.obs[sample_key].to_numpy() if sample_key in adata.obs.columns
               else np.zeros(adata.n_obs, dtype=int))

    degree = np.zeros(adata.n_obs, dtype=float)
    for s in pd.unique(samples):
        idx = np.flatnonzero(samples == s)
        if len(idx) > 1:
            graph = radius_neighbors_graph(coords[idx], radius, mode="connectivity", include_self=False)
            degree[idx] = np.asarray(graph.sum(axis=1)).ravel()

    area_100um2 = np.pi * radius ** 2 / 1e4        # (100 µm)^2 = 1e4 µm^2
    adata.obs[key] = (degree + 1.0) / area_100um2  # +1 counts the cell itself
    adata.uns[f"{key}_radius"] = radius
    return adata


def compute_ambient_background(adata, control_key="control_counts", gene_key="gene_counts",
                                radius=500.0, bin_size=25.0, spatial_key="spatial",
                                sample_key="sample_id", key="ambient_pi", log_key="log_ambient_pi"):
    r"""Local ambient-background fraction from control-probe counts.

    For each cell, over the `radius`-um neighbourhood N^amb_i,

        pi_tilde_i = (sum_{j in N} u_j) / (sum_{j in N} (m_j + u_j))

    where u_j = `obs[control_key]` (control-probe counts) and m_j =
    `obs[gene_key]` (total gene expression). This is the fraction of local counts
    that are non-specific control signal — a proxy for ambient/background
    contamination, which is spatially structured. Adds `obs[key]` (pi_tilde) and
    `obs[log_key]` = log(pi_tilde).

    A 500-um neighbourhood holds thousands of cells, so an exact radius graph
    would have billions of edges; instead the per-cell u and (m+u) are binned to
    a `bin_size`-um grid and disk-convolved (radius `radius`) per section — an
    accurate approximation for a smooth background field. Cells whose
    neighbourhood sums no control counts (pi = 0) are clipped to the smallest
    positive pi before the log, with a warning.
    """
    import pandas as pd
    from scipy.signal import fftconvolve

    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    u = adata.obs[control_key].to_numpy(dtype=float)
    total = u + adata.obs[gene_key].to_numpy(dtype=float)          # m_j + u_j
    samples = (adata.obs[sample_key].to_numpy() if sample_key in adata.obs.columns
               else np.zeros(adata.n_obs, dtype=int))

    rb = radius / bin_size
    kk = np.arange(-int(np.ceil(rb)), int(np.ceil(rb)) + 1)
    KX, KY = np.meshgrid(kk, kk, indexing="ij")
    disk = ((KX ** 2 + KY ** 2) <= rb ** 2).astype(float)          # hard 500-um disk kernel

    pi = np.full(adata.n_obs, np.nan)
    for s in pd.unique(samples):
        idx = np.flatnonzero(samples == s)
        x, y = coords[idx, 0], coords[idx, 1]
        ix = np.floor((x - x.min()) / bin_size).astype(int)
        iy = np.floor((y - y.min()) / bin_size).astype(int)
        grid = (ix.max() + 1, iy.max() + 1)
        U = np.zeros(grid); T = np.zeros(grid)
        np.add.at(U, (ix, iy), u[idx])
        np.add.at(T, (ix, iy), total[idx])
        # round: the convolution of integer counts with a 0/1 disk is integer-valued,
        # so this removes FFT roundoff (~1e-13) that would otherwise make log(pi) absurd
        Uc = np.rint(fftconvolve(U, disk, mode="same"))
        Tc = np.rint(fftconvolve(T, disk, mode="same"))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(Tc > 0, Uc / Tc, np.nan)
        pi[idx] = ratio[ix, iy]

    adata.obs[key] = pi
    positive = pi[np.isfinite(pi) & (pi > 0)]
    floor = positive.min() if positive.size else 1e-8
    n_bad = int((np.isfinite(pi) & (pi <= 0)).sum())
    if n_bad:
        warnings.warn(
            f"{n_bad}/{adata.n_obs} cells have zero control counts in their {radius:g}um "
            f"neighbourhood; clipping pi to the smallest positive value ({floor:.2e}) for the log.",
            stacklevel=2,
        )
    adata.obs[log_key] = np.log(np.where(pi > 0, pi, floor))
    adata.uns[f"{key}_radius"] = radius
    return adata


def build_covariate_matrix(adata, covariate_keys=("log_total_counts", "local_density"),
                            standardize=True, covar_key="covariates"):
    """Assemble (and optionally z-score) an obs-column covariate matrix.

    Stacks the named `obs` columns into an (n, C) matrix. Standardizing
    (default) puts every covariate on a common per-SD scale — important
    because raw `log_total_counts` (~4) and `local_density` (~tens) are on
    very different scales from Phi-Space embeddings (~0.1), which would
    otherwise make the model ill-conditioned and its weight decay uneven. The
    per-column mean/SD are stored so the same transform can be reapplied.

    Adds `obsm[covar_key]` (float matrix), `uns[covar_key + "_names"]`, and
    (if standardized) `uns[covar_key + "_stats"]` = {"mean", "std"}. Returns
    the (matrix, names).
    """
    missing = [c for c in covariate_keys if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"covariate columns {missing} not found in obs; compute them first "
                         "(compute_library_size / compute_local_density)")

    Z = np.column_stack([adata.obs[c].to_numpy(dtype=float) for c in covariate_keys])
    names = list(covariate_keys)

    if standardize:
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std == 0] = 1.0
        Z = (Z - mean) / std
        adata.uns[f"{covar_key}_stats"] = {"mean": mean, "std": std}

    adata.obsm[covar_key] = Z
    adata.uns[f"{covar_key}_names"] = names
    return Z, names
