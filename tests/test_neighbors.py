import numpy as np

from tissue_dynamics.neighbors import average_neighbors, build_adjacency


def test_build_adjacency_never_crosses_sections(small_adata):
    build_adjacency(small_adata, radius=80.0)
    A = small_adata.obsp["spatial_connectivity"]
    sample_id = small_adata.obs["sample_id"].to_numpy()

    rows, cols = A.nonzero()
    assert np.all(sample_id[rows] == sample_id[cols])


def test_build_adjacency_expected_neighbor_counts(small_adata):
    # Within each section, cells sit at x = 0, 30, 60, 200 (y=0): the first
    # three are pairwise within 80um (max pairwise distance 60), the fourth
    # (distance 140 from its nearest neighbor) is isolated.
    build_adjacency(small_adata, radius=80.0)

    degree = small_adata.obs["n_spatial_neighbors"].to_numpy()
    np.testing.assert_array_equal(degree, [2, 2, 2, 0, 2, 2, 2, 0])
    np.testing.assert_array_equal(
        small_adata.obs["is_isolated"].to_numpy(), [False, False, False, True] * 2,
    )
    assert small_adata.uns["spatial_connectivity_params"]["n_isolated"] == 2


def test_build_adjacency_is_symmetric_with_zero_diagonal(small_adata):
    build_adjacency(small_adata, radius=80.0)
    A = small_adata.obsp["spatial_connectivity"]
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0


def test_build_adjacency_is_order_independent(small_adata):
    perm = np.random.default_rng(0).permutation(small_adata.n_obs)
    shuffled = small_adata[perm].copy()

    build_adjacency(small_adata, radius=80.0)
    build_adjacency(shuffled, radius=80.0)

    A = small_adata.obsp["spatial_connectivity"].toarray()
    A_shuffled = shuffled.obsp["spatial_connectivity"].toarray()
    np.testing.assert_array_equal(A_shuffled, A[np.ix_(perm, perm)])


def test_average_neighbors_isolated_cells_get_zero_vector(small_adata):
    rng = np.random.default_rng(0)
    small_adata.obsm["phi"] = rng.normal(size=(small_adata.n_obs, 3))

    build_adjacency(small_adata, radius=80.0)
    average_neighbors(small_adata)

    isolated = small_adata.obs["is_isolated"].to_numpy()
    np.testing.assert_array_equal(small_adata.obsm["phi_bar"][isolated], np.zeros((isolated.sum(), 3)))


def test_average_neighbors_is_mean_not_sum_of_neighbor_phi(small_adata):
    small_adata.obsm["phi"] = np.repeat(np.arange(small_adata.n_obs, dtype=float)[:, None], 2, axis=1)

    build_adjacency(small_adata, radius=80.0)
    average_neighbors(small_adata)

    # cell index 1 has neighbors 0 and 2 (phi = 0 and 2) -> mean = 1, not sum = 2
    np.testing.assert_allclose(small_adata.obsm["phi_bar"][1], [1.0, 1.0])
