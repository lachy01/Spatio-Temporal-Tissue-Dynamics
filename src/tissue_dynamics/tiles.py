"""Tile the slide into a spatial grid and aggregate per tile.

The per-cell fate labels are noisy (see cellcycle/death), and adjacent cells
share almost their whole neighborhood, so the *effective* sample size of a
per-cell model is far below the cell count — closer to slide-area / (pi r^2),
a few thousand independent neighborhoods for a 100 mm^2 section, not 700k.
Aggregating to ~100 um tiles collapses the data to roughly that effective n
and turns sparse per-cell binary calls into per-tile *fractions* / counts,
which is the honest scale at which to test "does neighborhood composition
predict proliferation?" before committing to any per-cell architecture.
"""

import numpy as np
import pandas as pd


def assign_tiles(adata, tile_size=100.0, spatial_key="spatial", sample_key="sample_id",
                 tile_key="tile"):
    """Assign each cell to a square tile of side `tile_size` (same units as coords).

    Tiles are indexed per `sample_key` so they never span two sections. Adds
    `obs[tile_key]` (string id `"{sample}:{ix}:{iy}"`) plus integer
    `obs[tile_key + "_ix"]` / `["_iy"]`, and returns `adata`.
    """
    coords = np.asarray(adata.obsm[spatial_key])
    samples = (adata.obs[sample_key].to_numpy() if sample_key in adata.obs.columns
               else np.zeros(adata.n_obs, dtype=int))

    ix = np.zeros(adata.n_obs, dtype=np.int64)
    iy = np.zeros(adata.n_obs, dtype=np.int64)
    for s in pd.unique(samples):
        m = samples == s
        c = coords[m]
        ix[m] = np.floor((c[:, 0] - c[:, 0].min()) / tile_size).astype(np.int64)
        iy[m] = np.floor((c[:, 1] - c[:, 1].min()) / tile_size).astype(np.int64)

    adata.obs[tile_key] = [f"{s}:{a}:{b}" for s, a, b in zip(samples, ix, iy)]
    adata.obs[f"{tile_key}_ix"] = ix
    adata.obs[f"{tile_key}_iy"] = iy
    adata.uns[f"{tile_key}_size"] = tile_size
    return adata


def aggregate_tiles(adata, tile_key="tile", phi_key="phi", phi_bar_key=None,
                     obs_means=(), gene_counts=(), min_cells=5, spatial_key="spatial",
                     counts_layer="counts"):
    """Aggregate cells to one row per tile.

    Parameters
    ----------
    phi_key : str or None
        If set, adds per-tile mean of each `obsm[phi_key]` dimension as
        columns `phi_mean_0..`. `phi_bar_key` does the same for the
        neighbor-averaged embedding (prefix `phibar_mean_`).
    obs_means : sequence of str
        `obs` columns to average per tile (e.g. `"is_dividing"` ->
        division fraction, `"proliferation_score"`, `"cell_area"`). Boolean
        columns become fractions.
    gene_counts : sequence of str
        Genes to *sum* (raw counts) per tile, added as `{gene}_sum`; the
        tile's summed library size is added as `total_counts_sum` whenever
        any gene is requested (the natural offset for a tile-level count
        model). Raw counts are read from `layers[counts_layer]` if present,
        else `X`.
    min_cells : int
        Drop tiles with fewer than this many cells (sparse edge tiles).

    Returns
    -------
    DataFrame indexed by tile id with `n_cells`, tile-centroid `x`/`y`, the
    requested means/sums, and (if `phi_key`) the mean-phi columns.
    """
    import scipy.sparse as sp

    obs = adata.obs
    tiles = obs[tile_key].to_numpy()
    order = pd.unique(tiles)

    coords = np.asarray(adata.obsm[spatial_key])
    df = pd.DataFrame(index=pd.Index(order, name=tile_key))
    grp = pd.Series(np.arange(adata.n_obs), index=obs.index).groupby(tiles, sort=False)

    idx_lists = {t: g.to_numpy() for t, g in grp}
    df["n_cells"] = [len(idx_lists[t]) for t in order]
    df["x"] = [coords[idx_lists[t], 0].mean() for t in order]
    df["y"] = [coords[idx_lists[t], 1].mean() for t in order]

    for col in obs_means:
        values = obs[col].to_numpy().astype(float)
        df[f"{col}_mean"] = [values[idx_lists[t]].mean() for t in order]

    if phi_key is not None:
        phi = np.asarray(adata.obsm[phi_key])
        means = np.vstack([phi[idx_lists[t]].mean(0) for t in order])
        for j in range(phi.shape[1]):
            df[f"phi_mean_{j}"] = means[:, j]
    if phi_bar_key is not None:
        phibar = np.asarray(adata.obsm[phi_bar_key])
        means = np.vstack([phibar[idx_lists[t]].mean(0) for t in order])
        for j in range(phibar.shape[1]):
            df[f"phibar_mean_{j}"] = means[:, j]

    if gene_counts:
        counts = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
        total = np.asarray(counts.sum(1)).ravel()
        df["total_counts_sum"] = [total[idx_lists[t]].sum() for t in order]
        for gene in gene_counts:
            col = adata.var_names.get_loc(gene)
            g = counts[:, col]
            g = g.toarray().ravel() if sp.issparse(g) else np.asarray(g).ravel()
            df[f"{gene}_sum"] = [g[idx_lists[t]].sum() for t in order]

    return df[df["n_cells"] >= min_cells].copy()


def binned_conditional_mean(x, y, n_bins=10, binning="quantile"):
    """Observed mean of `y` within bins of `x` — the OSDR sanity plot's data.

    Bins cells/tiles by `x` (a predicted score or a leading neighborhood
    axis), then reports the observed mean of `y` (e.g. a 0/1 dividing call,
    so the mean is the division fraction) per bin with a normal-approximation
    standard error. A monotone, wide-spanning curve is the neighborhood
    signal OSDR needs — even when per-cell accuracy is near the majority
    baseline.

    Parameters
    ----------
    binning : "quantile" or "uniform"
        Equal-count (quantile) or equal-width (uniform) bins of `x`.

    Returns
    -------
    DataFrame with `x_mean`, `y_mean`, `y_se`, `n` per bin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if binning == "quantile":
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    elif binning == "uniform":
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning {binning!r}; expected 'quantile' or 'uniform'")
    if len(edges) < 2:
        raise ValueError("x has too few distinct values to bin")

    bin_idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            continue
        ybar = y[m].mean()
        # normal-approx SE; for a 0/1 outcome this is the binomial SE.
        se = np.sqrt(ybar * (1 - ybar) / n) if set(np.unique(y[m])) <= {0.0, 1.0} else y[m].std(ddof=1) / np.sqrt(n)
        rows.append({"x_mean": x[m].mean(), "y_mean": ybar, "y_se": se, "n": n})
    return pd.DataFrame(rows)
