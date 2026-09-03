import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.annulus import (
    DEFAULT_RINGS,
    annulus_neighbor_stats,
    annulus_scan,
    spillover_chi2,
)


def _grid_adata(side=25, spacing=20.0, phi_dim=2, seed=0, smooth=None):
    """Cells on a regular grid so ring membership is hand-checkable.

    `smooth=None` gives white-noise phi (zero spatial autocorrelation);
    a float gives a Gaussian-smoothed field with that length scale.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    n = coords.shape[0]

    phi = rng.normal(size=(n, phi_dim))
    if smooth is not None:
        from scipy.ndimage import gaussian_filter
        sig = smooth / spacing
        phi = np.column_stack([
            gaussian_filter(phi[:, d].reshape(side, side), sig, mode="wrap").ravel()
            for d in range(phi_dim)
        ])

    a = AnnData(np.ones((n, 2)), obs=pd.DataFrame({"sample_id": ["A"] * n},
                                                  index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=["G0", "G1"]))
    a.obsm["spatial"] = coords
    a.obsm["phi"] = phi
    a.uns["phi_names"] = [f"axis{d}" for d in range(phi_dim)]
    return a


def test_annulus_neighbor_stats_selects_only_the_ring():
    a = _grid_adata(side=9, spacing=20.0)
    # on a 20um grid: 4 neighbours at 20um, 4 at 28.28um, 4 at 40um
    inner = annulus_neighbor_stats(a, 0.0, 25.0)          # only the 20um ring
    mid = annulus_neighbor_stats(a, 25.0, 30.0)           # only the 28.28um diagonals
    interior = (a.obsm["spatial"][:, 0] > 20) & (a.obsm["spatial"][:, 0] < 140) & \
               (a.obsm["spatial"][:, 1] > 20) & (a.obsm["spatial"][:, 1] < 140)
    assert set(np.unique(inner["counts"][interior])) == {4}
    assert set(np.unique(mid["counts"][interior])) == {4}
    # the cell itself is never its own neighbour (d == 0 is excluded)
    assert inner["counts"].min() >= 2                      # corners still see 2


def test_annulus_neighbor_stats_phi_bar_is_the_ring_mean():
    a = _grid_adata(side=7, spacing=20.0)
    st = annulus_neighbor_stats(a, 0.0, 25.0)
    coords, phi = a.obsm["spatial"], a.obsm["phi"]
    i = 24                                                 # an interior cell
    d = np.linalg.norm(coords - coords[i], axis=1)
    ring = (d > 0) & (d <= 25.0)
    np.testing.assert_allclose(st["phi_bar"][i], phi[ring].mean(0), rtol=1e-10)


def test_correlogram_is_zero_for_white_noise_and_high_for_smooth_field():
    white = annulus_neighbor_stats(_grid_adata(side=30, smooth=None), 0.0, 25.0)
    assert np.abs(white["rho_pair"]).max() < 0.1           # no spatial structure

    a = _grid_adata(side=30, smooth=60.0)                  # 60um correlation length
    near = annulus_neighbor_stats(a, 0.0, 25.0)["rho_pair"].mean()
    far = annulus_neighbor_stats(a, 120.0, 200.0)["rho_pair"].mean()
    assert near > 0.5                                      # neighbours look alike
    assert near > far                                      # and the resemblance decays


def test_annulus_neighbor_stats_max_neighbors_caps_and_stays_unbiased():
    a = _grid_adata(side=20, spacing=20.0, smooth=60.0)
    full = annulus_neighbor_stats(a, 0.0, 60.0)
    capped = annulus_neighbor_stats(a, 0.0, 60.0, max_neighbors=4, seed=0)
    assert full["counts"].max() > 4
    assert capped["counts"].max() == 4                     # thinned to the cap
    # random thinning is unbiased: the correlogram is unchanged up to noise
    assert np.allclose(capped["rho_pair"], full["rho_pair"], atol=0.1)


def test_annulus_neighbor_stats_is_per_sample():
    a = _grid_adata(side=6, spacing=20.0)
    a.obs["sample_id"] = ["A"] * (a.n_obs // 2) + ["B"] * (a.n_obs - a.n_obs // 2)
    both = annulus_neighbor_stats(a, 0.0, 25.0)["counts"]
    a.obs["sample_id"] = "A"
    pooled = annulus_neighbor_stats(a, 0.0, 25.0)["counts"]
    assert both.sum() < pooled.sum()                       # sections don't see each other


def _count_adata_for_scan(side=22, spacing=20.0, seed=0):
    """Grid adata with a division-gene count response, ready for fit_division_nb."""
    a = _grid_adata(side=side, spacing=spacing, seed=seed, smooth=40.0)
    rng = np.random.default_rng(seed)
    n = a.n_obs
    log_mu = -1.0 + 0.8 * a.obsm["phi"][:, 0]
    div = rng.poisson(np.exp(log_mu)[:, None] * np.ones((1, 2))).astype(float)
    filler = rng.poisson(20.0, size=(n, 3)).astype(float)
    X = np.column_stack([div, filler])
    out = AnnData(X, obs=a.obs.copy(), var=pd.DataFrame(index=["D0", "D1", "F0", "F1", "F2"]))
    out.obsm["spatial"] = a.obsm["spatial"]
    out.obsm["phi"] = a.obsm["phi"]
    out.uns["phi_names"] = a.uns["phi_names"]
    out.obs["log_total_counts"] = np.log(X.sum(1))
    out.obs["cell_area"] = rng.uniform(20.0, 200.0, size=n)
    return out, ["D0", "D1"]


def test_annulus_scan_shapes_and_columns():
    pytest.importorskip("statsmodels")
    a, genes = _count_adata_for_scan()
    rings = ((0.0, 25.0), (25.0, 45.0), (45.0, 90.0))
    coef, summary = annulus_scan(a, rings=rings, interaction=True,
                                 genes=genes, covariate_keys=("cell_area",),
                                 offset_key="log_total_counts", fitter="irls")
    assert len(summary) == len(rings)
    assert len(coef) == len(rings) * 2                     # rings x phi dims
    assert {"c", "se", "z", "rho_pair", "rho_own_bar", "dim"} <= set(coef.columns)
    assert {"c_mean_abs", "rho_phi", "B_fro", "lr_interaction"} <= set(summary.columns)
    assert (summary["mean_neighbors"].diff().dropna() > 0).all()   # wider rings, more cells
    assert summary["rho_phi"].iloc[0] > summary["rho_phi"].iloc[-1]  # autocorrelation decays


def test_annulus_scan_standardizes_phi_bar():
    pytest.importorskip("statsmodels")
    a, genes = _count_adata_for_scan(side=14)
    annulus_scan(a, rings=((0.0, 25.0),), interaction=False, genes=genes,
                 covariate_keys=None, fitter="irls")
    pb = a.obsm["phi_bar"]
    np.testing.assert_allclose(pb.mean(0), 0.0, atol=1e-8)
    np.testing.assert_allclose(pb.std(0), 1.0, atol=1e-8)


def _coef_frame(c_by_ring, rho_by_ring, se=0.01, dims=("axis0", "axis1")):
    rows = []
    for dim in dims:
        for (ri, ro), c, rho in zip([(0, 15), (15, 30), (30, 50)], c_by_ring, rho_by_ring):
            rows.append({"r_inner": ri, "r_outer": ro, "dim": dim,
                         "c": c, "se": se, "z": c / se, "rho_pair": rho, "rho_own_bar": rho})
    return pd.DataFrame(rows)


def test_spillover_chi2_accepts_exact_spillover():
    rho = [0.8, 0.4, 0.1]
    coef = _coef_frame([2.0 * r for r in rho], rho)        # c = k * rho exactly
    res = spillover_chi2(coef)
    assert res["chi2"] < 1e-9
    assert res["p_value"] > 0.99
    assert all(abs(k - 2.0) < 1e-9 for k in res["k"].values())
    assert len(res["deviations"]) == 6


def test_spillover_chi2_rejects_effect_that_outlives_autocorrelation():
    rho = [0.8, 0.4, 0.1]
    coef = _coef_frame([0.5, 0.5, 0.5], rho)               # effect flat while rho decays
    res = spillover_chi2(coef)
    assert res["p_value"] < 1e-6                            # not spillover-shaped
    assert res["df"] == 2 * (3 - 1)
    # the far ring is where the profiles part company
    far = res["deviations"].query("r_inner == 30")["std_deviation"].abs().mean()
    near = res["deviations"].query("r_inner == 0")["std_deviation"].abs().mean()
    assert far > near


def test_default_rings_are_contiguous_and_expanding():
    assert DEFAULT_RINGS[0][0] == 0.0
    for (a_in, a_out), (b_in, b_out) in zip(DEFAULT_RINGS, DEFAULT_RINGS[1:]):
        assert a_out == b_in                                # contiguous, no gaps/overlap
        assert b_out - b_in >= a_out - a_in                 # widening
