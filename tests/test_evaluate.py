import numpy as np
import pytest

from tissue_dynamics.evaluate import morans_i


def _grid(side=20):
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_morans_i_detects_spatial_gradient():
    coords = _grid(20)
    values = coords[:, 0] + coords[:, 1]                  # smooth gradient -> strong clustering
    res = morans_i(values, coords=coords, n_neighbors=8, permutations=199, seed=0)
    assert res["I"] > 0.5
    assert res["I"] > res["expected_I"]
    assert res["p_value"] < 0.01                          # far from the permutation null


def test_morans_i_random_field_near_zero():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 50, size=(400, 2))
    values = rng.normal(size=400)                         # no spatial structure
    res = morans_i(values, coords=coords, n_neighbors=8, permutations=199, seed=1)
    assert abs(res["I"] - res["expected_I"]) < 0.1
    assert res["p_value"] > 0.05


def test_morans_i_drops_nonfinite_and_respects_permutations_zero():
    coords = _grid(8)
    values = np.arange(coords.shape[0], dtype=float)
    values[::7] = np.nan
    res = morans_i(values, coords=coords, permutations=0)
    assert res["n"] == coords.shape[0] - len(range(0, coords.shape[0], 7))
    assert "p_value" not in res                           # skipped when permutations=0


def test_morans_i_accepts_precomputed_weight_matrix():
    from scipy import sparse
    coords = _grid(10)
    values = coords[:, 0].copy()                          # gradient along x
    from sklearn.neighbors import kneighbors_graph
    W = kneighbors_graph(coords, n_neighbors=4, mode="connectivity", include_self=False)
    res_W = morans_i(values, W=sparse.csr_matrix(W), permutations=0)
    res_c = morans_i(values, coords=coords, n_neighbors=4, permutations=0)
    assert res_W["I"] == pytest.approx(res_c["I"])
    assert res_W["I"] > 0.5


def test_morans_i_requires_coords_or_weights():
    with pytest.raises(ValueError, match="coords"):
        morans_i(np.arange(10.0))
