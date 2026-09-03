import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.covariates import (
    build_covariate_matrix,
    compute_ambient_background,
    compute_library_size,
    compute_local_density,
)


def _adata(coords, sample_id=None):
    n = len(coords)
    X = np.arange(1, n * 2 + 1, dtype=float).reshape(n, 2)  # nonzero counts
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    if sample_id is not None:
        obs["sample_id"] = sample_id
    a = AnnData(X, obs=obs, var=pd.DataFrame(index=["G0", "G1"]))
    a.obsm["spatial"] = np.asarray(coords, dtype=float)
    return a


def test_compute_library_size_totals_and_log():
    a = _adata([[0, 0], [0, 0]])
    a.X = np.array([[2.0, 3.0], [10.0, 0.0]])
    compute_library_size(a)
    np.testing.assert_allclose(a.obs["total_counts"], [5.0, 10.0])
    np.testing.assert_allclose(a.obs["log_total_counts"], np.log([5.0, 10.0]))


def test_compute_library_size_prefers_counts_layer_and_warns_on_zero():
    a = _adata([[0, 0], [0, 0]])
    a.X = np.array([[0.0, 0.0], [1.0, 1.0]])
    a.layers["counts"] = a.X.copy()
    with pytest.warns(UserWarning, match="zero library size"):
        compute_library_size(a)
    assert a.obs["total_counts"].iloc[0] == 0.0
    assert np.isfinite(a.obs["log_total_counts"].iloc[0])   # clipped, not -inf


def test_compute_local_density_counts_neighbors_within_radius():
    # 3 cells clustered within 20um, one far away
    a = _adata([[0, 0], [10, 0], [20, 0], [500, 0]], sample_id=["A"] * 4)
    compute_local_density(a, radius=30.0)
    d = a.obs["local_density"].to_numpy()
    # the clustered cells see more neighbours than the isolated one
    assert d[0] > d[3] and d[1] > d[3]
    assert a.uns["local_density_radius"] == 30.0
    # isolated cell: only itself -> density = 1 / (pi r^2 / 1e4)
    assert d[3] == pytest.approx(1.0 / (np.pi * 30.0 ** 2 / 1e4))


def test_compute_local_density_is_per_sample():
    # same coords in two sections must not count each other
    a = _adata([[0, 0], [0, 0]], sample_id=["A", "B"])
    compute_local_density(a, radius=30.0)
    # each cell alone in its section -> density = 1/area for both
    assert a.obs["local_density"].nunique() == 1


def test_build_covariate_matrix_standardizes_and_stores():
    a = _adata([[0, 0], [1, 0], [2, 0]], sample_id=["A"] * 3)
    a.obs["log_total_counts"] = [1.0, 2.0, 3.0]
    a.obs["local_density"] = [10.0, 20.0, 30.0]
    Z, names = build_covariate_matrix(a, covariate_keys=("log_total_counts", "local_density"))
    assert Z.shape == (3, 2)
    assert names == ["log_total_counts", "local_density"]
    np.testing.assert_allclose(Z.mean(axis=0), [0, 0], atol=1e-12)   # standardized
    np.testing.assert_allclose(Z.std(axis=0), [1, 1], atol=1e-12)
    assert "covariates" in a.obsm
    assert set(a.uns["covariates_stats"]) == {"mean", "std"}


def test_build_covariate_matrix_missing_column_raises():
    a = _adata([[0, 0]], sample_id=["A"])
    with pytest.raises(ValueError, match="not found"):
        build_covariate_matrix(a, covariate_keys=("does_not_exist",))


def _grid_adata(side=40, spacing=50.0):
    """A dense square grid of cells so 300um neighbourhoods contain many points."""
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    a = _adata(coords, sample_id=["A"] * coords.shape[0])
    return a


def test_compute_ambient_background_uniform_ratio():
    a = _grid_adata()
    a.obs["control_counts"] = np.ones(a.n_obs)             # u = 1
    a.obs["gene_counts"] = np.full(a.n_obs, 99.0)          # m = 99  -> pi = 1/100 everywhere
    compute_ambient_background(a, radius=300.0, bin_size=50.0)
    pi = a.obs["ambient_pi"].to_numpy()
    assert np.nanmedian(pi) == pytest.approx(0.01, rel=0.05)
    np.testing.assert_allclose(a.obs["log_ambient_pi"], np.log(a.obs["ambient_pi"]), atol=1e-9)
    assert a.uns["ambient_pi_radius"] == 300.0


def test_compute_ambient_background_tracks_local_control_fraction():
    a = _grid_adata()
    x = a.obsm["spatial"][:, 0]
    a.obs["gene_counts"] = np.full(a.n_obs, 90.0)
    a.obs["control_counts"] = np.where(x < x.mean(), 10.0, 1.0)   # left noisy, right clean
    compute_ambient_background(a, radius=300.0, bin_size=50.0)
    pi = a.obs["ambient_pi"].to_numpy()
    left = pi[x < x.min() + 300]                            # interior-left vs interior-right
    right = pi[x > x.max() - 300]
    assert np.nanmean(left) > 3 * np.nanmean(right)         # ~0.10 vs ~0.011
    assert np.nanmean(right) > 0


def test_compute_ambient_background_is_per_sample():
    # two overlapping sections must not pool each other's counts
    a = _grid_adata(side=20)
    n = a.n_obs
    a.obs["sample_id"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    a.obs["control_counts"] = np.r_[np.full(n // 2, 5.0), np.full(n - n // 2, 1.0)]
    a.obs["gene_counts"] = np.full(n, 95.0)
    compute_ambient_background(a, radius=300.0, bin_size=50.0)
    # section A (u=5) should read a higher ambient fraction than section B (u=1)
    pi = a.obs["ambient_pi"].to_numpy()
    assert np.nanmean(pi[:n // 2]) > np.nanmean(pi[n // 2:])
