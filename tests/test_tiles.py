import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.tiles import aggregate_tiles, assign_tiles, binned_conditional_mean


def _adata(coords, sample_id, **obs):
    n = len(coords)
    a = AnnData(np.zeros((n, 2), dtype=float),
                obs=pd.DataFrame({"sample_id": sample_id, **obs}, index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=["G0", "G1"]))
    a.obsm["spatial"] = np.asarray(coords, dtype=float)
    return a


def test_assign_tiles_grid_ids_and_per_sample_origin():
    coords = [[0, 0], [40, 0], [120, 0], [0, 250]]
    a = _adata(coords, ["A", "A", "A", "A"])
    assign_tiles(a, tile_size=100.0)
    # tiles are floor((x-xmin)/100), floor((y-ymin)/100)
    assert list(a.obs["tile"]) == ["A:0:0", "A:0:0", "A:1:0", "A:0:2"]
    assert a.uns["tile_size"] == 100.0


def test_assign_tiles_separates_samples():
    coords = [[0, 0], [0, 0]]
    a = _adata(coords, ["A", "B"])
    assign_tiles(a, tile_size=100.0)
    assert a.obs["tile"].iloc[0] != a.obs["tile"].iloc[1]  # same xy, different section


def test_aggregate_tiles_means_fractions_and_min_cells():
    coords = [[0, 0], [10, 10], [20, 20], [500, 500]]  # first 3 in one tile, last alone
    a = _adata(coords, ["A"] * 4, is_dividing=[True, False, True, True])
    a.obsm["phi"] = np.array([[1.0, 0], [3.0, 0], [2.0, 0], [9.0, 0]])
    assign_tiles(a, tile_size=100.0)
    tiles = aggregate_tiles(a, obs_means=["is_dividing"], phi_key="phi", min_cells=2)

    assert len(tiles) == 1                       # the singleton tile is dropped
    row = tiles.iloc[0]
    assert row["n_cells"] == 3
    assert row["is_dividing_mean"] == pytest.approx(2 / 3)   # fraction dividing
    assert row["phi_mean_0"] == pytest.approx(2.0)           # mean of 1,3,2


def test_aggregate_tiles_gene_sums_and_offset():
    coords = [[0, 0], [10, 10]]
    a = _adata(coords, ["A", "A"])
    a.X = np.array([[2.0, 5.0], [3.0, 1.0]])     # cols = G0, G1
    assign_tiles(a, tile_size=100.0)
    tiles = aggregate_tiles(a, phi_key=None, gene_counts=["G0"], min_cells=1)
    assert tiles["G0_sum"].iloc[0] == pytest.approx(5.0)          # 2 + 3
    assert tiles["total_counts_sum"].iloc[0] == pytest.approx(11.0)  # (2+5)+(3+1)


def test_binned_conditional_mean_recovers_monotone_relationship():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(x - 0.5)))       # increasing P(y=1) in x
    y = (rng.random(n) < p).astype(float)

    binned = binned_conditional_mean(x, y, n_bins=8)
    assert len(binned) == 8
    # observed fraction should increase across bins
    assert binned["y_mean"].iloc[-1] > binned["y_mean"].iloc[0]
    assert (binned["y_se"] > 0).all()
    assert np.all(np.diff(binned["y_mean"]) > -0.05)   # roughly monotone


def test_binned_conditional_mean_unknown_binning_raises():
    with pytest.raises(ValueError):
        binned_conditional_mean(np.arange(10), np.arange(10), binning="nonsense")
