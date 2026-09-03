"""Validation suite for the oxygen field (spec section 7).

The data-dependent checks (7.7 signal share, 7.8 density decoupling, 7.9 HIF
targets) live in the real-data run; everything here is on synthetic geometry
where the analytic answer is known.
"""

import numpy as np
import pytest
from scipy.special import k0

from tissue_dynamics.oxygen import (
    assemble_robin,
    build_consumption,
    build_oxygen_field,
    build_tissue_mask,
    calibrate_cutoff,
    deposit,
    drop_singular_components,
    sample_at_cells,
    solve_oxygen,
    source_weights,
)


def _homogeneous(N, L, h, beta_src=None):
    tissue = np.ones((N, N), dtype=bool)
    kappa = np.full((N, N), 1.0 / L ** 2)
    Phi = np.zeros((N, N)) if beta_src is None else beta_src
    return tissue, Phi, kappa


# --- 7.1 Green's function -------------------------------------------------

def test_71_greens_function_reproduces_k0_kernel():
    L, h, half = 100.0, 2.0, 700.0
    N = int(2 * half / h) + 1
    ctr = (N - 1) // 2
    ax = (np.arange(N) - ctr) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(X, Y)

    tissue, Phi, kappa = _homogeneous(N, L, h)
    Phi[ctr, ctr] = 1.0
    u, info = solve_oxygen(tissue, Phi, kappa, h, beta=1e-4)   # weak -> linear regime

    band = (r > 8 * h) & (r < 4.5 * L)
    shape = u[band] / k0(r[band] / L)
    cv = shape.std() / shape.mean()
    assert info["rel_residual"] < 1e-8
    assert cv < 1e-3, f"u/K0(r/L) not constant: CV={cv:.2e}"


# --- 7.2 maximum principle ------------------------------------------------

def test_72_maximum_principle():
    N, L, h = 160, 100.0, 5.0
    tissue, Phi, kappa = _homogeneous(N, L, h)
    Phi[40, 40] = 5.0
    Phi[120, 90] = 2.0
    u, _ = solve_oxygen(tissue, Phi, kappa, h, beta=10.0)
    assert np.nanmin(u) >= -1e-12
    assert np.nanmax(u) <= 1.0 + 1e-12


# --- 7.3 harmonicity inside necrosis --------------------------------------

def test_73_field_is_harmonic_where_kappa_is_zero():
    N, L, h = 160, 100.0, 5.0
    tissue, Phi, kappa = _homogeneous(N, L, h)
    Phi[30, 30] = 1.0
    kappa[70:110, 70:110] = 0.0                       # necrotic patch: no consumption
    u, _ = solve_oxygen(tissue, Phi, kappa, h, beta=1.0)
    i, j = 90, 90                                      # interior of the patch
    nb_mean = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]) / 4.0
    assert abs(u[i, j] - nb_mean) < 1e-9


# --- 7.4 no-flux boundary -------------------------------------------------

def test_74_no_flux_at_the_exterior_edge():
    """The wall reflects: the field flattens into it and matches an image charge.

    With the mirrored-ghost stencil the zero flux is exact at the boundary *face*
    (half a pixel past the last centre), so `u[-1] - u[-2]` is not itself zero --
    it is the residual decay. The two things that must hold are that the normal
    difference collapses as the wall is approached (free decay would shrink it by
    only exp(-h/L) ~ 0.95 per step), and that the solution equals the free-space
    kernel plus its mirror image, which is what a reflecting boundary means.
    """
    N, L, h = 160, 100.0, 5.0
    tissue, Phi, kappa = _homogeneous(N, L, h)
    Phi[80, 80] = 1.0
    u, _ = solve_oxygen(tissue, Phi, kappa, h, beta=1.0)

    col = u[:, 80]
    d = np.abs(np.diff(col))
    assert (np.diff(d[-5:]) < 0).all()                     # differences shrink into the wall
    assert d[-1] / d[-2] < 0.6                             # ~0.5: derivative -> 0 at the face

    # image charge: no-flux at the wall == source + its mirror in free space
    y = np.arange(N) * h
    r_src = np.abs(y - 80 * h)
    r_img = np.abs(y - (2 * (N - 0.5) * h - 80 * h))
    pred = k0(np.maximum(r_src, 1e-9) / L) + k0(np.maximum(r_img, 1e-9) / L)
    band = slice(120, 158)
    ratio = col[band] / pred[band]
    assert ratio.std() / ratio.mean() < 1e-2


# --- 7.5 grid independence of source strength -----------------------------

def test_75_source_strength_is_grid_independent():
    """Phi is a SUM, so halving h must not change the field away from sources."""
    from scipy.stats import spearmanr

    L, half, beta = 100.0, 400.0, 1.0
    rng = np.random.default_rng(0)
    xy = rng.uniform(-half + 60, half - 60, size=(300, 2))
    w = rng.random(300)
    probe = rng.uniform(-half + 120, half - 120, size=(500, 2))

    fields = {}
    for h in (8.0, 4.0):
        N = int(2 * half / h) + 1
        origin = np.array([-half, -half])
        tissue = np.ones((N, N), dtype=bool)
        Phi, _ = deposit(xy, w, origin, (N, N), h)
        kappa = np.full((N, N), 1.0 / L ** 2)
        u, _ = solve_oxygen(tissue, Phi, kappa, h, beta=beta)
        fields[h], _, _ = sample_at_cells(u, probe, origin, h)

    rho = spearmanr(fields[8.0], fields[4.0]).statistic
    assert rho > 0.99, f"grid-dependent source strength: spearman={rho:.4f}"


# --- 7.6 beta limit -> Dirichlet ------------------------------------------

def test_76_large_beta_approaches_the_dirichlet_limit():
    N, L, h = 200, 100.0, 5.0
    ctr = (N - 1) // 2
    ax = (np.arange(N) - ctr) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    disc = np.hypot(X, Y) <= 25.0

    tissue = np.ones((N, N), dtype=bool)
    kappa = np.full((N, N), 1.0 / L ** 2)
    Phi = disc.astype(float)

    u_big, _ = solve_oxygen(tissue, Phi, kappa, h, beta=1e4)
    u_bigger, _ = solve_oxygen(tissue, Phi, kappa, h, beta=1e6)
    assert u_big.max() > 0.99                                     # pinned to ~1 on source
    assert np.abs(u_big - u_bigger).max() < 1e-4                  # converged in beta


# --- assembly / structure --------------------------------------------------

def test_matrix_is_symmetric_positive_definite():
    N, L, h = 40, 100.0, 5.0
    tissue, Phi, kappa = _homogeneous(N, L, h)
    tissue[:8, :8] = False                                        # a notch
    Phi[20, 20] = 1.0
    A, b, idx = assemble_robin(tissue, Phi, kappa, h, beta=1.0)
    assert abs(A - A.T).max() < 1e-12
    assert (A.diagonal() > 0).all()
    off = np.abs(A).sum(1).A1 - A.diagonal()
    assert (A.diagonal() >= off - 1e-12).all()                    # diagonally dominant
    assert idx[~tissue].max() == -1


def test_no_flux_drops_outside_neighbours_from_the_diagonal():
    """The whole boundary condition is n_valid counting in-tissue neighbours only."""
    tissue = np.zeros((5, 5), dtype=bool)
    tissue[1:4, 1:4] = True
    Phi = np.zeros((5, 5))
    kappa = np.zeros((5, 5))
    kappa[tissue] = 1.0
    A, _, idx = assemble_robin(tissue, Phi, kappa, h=1.0, beta=1.0)
    centre = A[idx[2, 2], idx[2, 2]]                              # 4 in-tissue neighbours
    corner = A[idx[1, 1], idx[1, 1]]                              # only 2
    assert centre == pytest.approx(4 + 1.0)
    assert corner == pytest.approx(2 + 1.0)


def test_drop_singular_components_removes_source_free_fragments():
    tissue = np.zeros((20, 20), dtype=bool)
    tissue[2:6, 2:6] = True                                       # fragment: no source, no kappa
    tissue[12:18, 12:18] = True                                   # fragment with consumption
    Phi = np.zeros((20, 20))
    kappa = np.zeros((20, 20))
    kappa[12:18, 12:18] = 1e-4
    keep, dropped, n_comp = drop_singular_components(tissue, Phi, kappa)
    assert n_comp == 2 and dropped == 1
    assert not keep[2:6, 2:6].any() and keep[12:18, 12:18].all()


# --- source weights / cutoff ----------------------------------------------

def test_source_weights_clip_and_soft_threshold():
    phi = np.array([-0.5, 0.0, 0.10, 0.15, 0.40])
    np.testing.assert_allclose(source_weights(phi, 0.15), [0.0, 0.0, 0.0, 0.0, 0.25])


def test_calibrate_cutoff_reports_signal_share_and_retention():
    rng = np.random.default_rng(0)
    n = 20000
    endo = rng.random(n) < 0.02
    phi = np.where(endo, rng.normal(0.8, 0.1, n), np.abs(rng.normal(0.0, 0.05, n)))
    raw = calibrate_cutoff(phi, 0.0)
    cut = calibrate_cutoff(phi, 0.15)
    assert cut["signal_share"] > raw["signal_share"]              # thresholding buys signal
    assert cut["signal_share"] > 0.9
    assert 0.005 < cut["retained_frac"] < 0.05                    # plausible endothelial fraction


def test_deposit_is_a_sum_not_a_mean():
    xy = np.array([[0.5, 0.5], [0.6, 0.6], [5.5, 5.5]])
    grid, _ = deposit(xy, np.array([1.0, 2.0, 4.0]), np.zeros(2), (8, 8), 1.0)
    assert grid[0, 0] == 3.0                                      # both cells summed
    assert grid[5, 5] == 4.0
    assert grid.sum() == 7.0


# --- mask / consumption ----------------------------------------------------

def test_build_tissue_mask_fills_interior_holes_but_not_exterior():
    occ = np.zeros((60, 60), dtype=bool)
    occ[10:50, 10:50] = True
    occ[25:35, 25:35] = False                                     # interior acellular hole
    tissue = build_tissue_mask(occ, h=5.0, close_um=10.0)
    assert tissue[30, 30]                                         # hole is INSIDE the tissue
    assert not tissue[2, 2]                                       # exterior stays outside


def test_build_consumption_scales_with_density_and_zeroes_necrosis():
    counts = np.zeros((40, 40))
    counts[:20, :] = 1.0
    counts[20:, :] = 4.0
    tissue = np.ones((40, 40), dtype=bool)
    necrosis = np.zeros((40, 40), dtype=bool)
    necrosis[30:35, 30:35] = True
    kappa, n0 = build_consumption(counts, tissue, h=5.0, L0=100.0,
                                  sigma_density=5.0, necrosis=necrosis)
    assert kappa[30, 5] > kappa[5, 5]                             # denser -> more consumption
    assert (kappa[30:35, 30:35] == 0).all()                       # necrosis consumes nothing
    assert n0 > 0


# --- end to end ------------------------------------------------------------

def test_build_oxygen_field_end_to_end():
    rng = np.random.default_rng(1)
    n = 4000
    xy = rng.uniform(0, 800, size=(n, 2))
    endo = rng.random(n) < 0.02
    phi = np.where(endo, rng.normal(0.8, 0.1, n), np.abs(rng.normal(0.0, 0.05, n)))

    res = build_oxygen_field(xy, phi, h=10.0, L0=100.0, beta=1.0, c=0.15)
    assert res.u_cells.shape == (n,) and res.grad_u_cells.shape == (n,)
    fin = np.isfinite(res.u_cells)
    assert fin.mean() > 0.95
    assert res.u_cells[fin].min() >= -1e-12 and res.u_cells[fin].max() <= 1 + 1e-12
    assert res.diagnostics["rel_residual"] < 1e-8
    assert res.diagnostics["signal_share"] > 0.9
    # cells nearer endothelium should be better oxygenated
    assert np.nanmean(res.u_cells[endo]) > np.nanmean(res.u_cells[~endo])
    assert "field_is_2d" in res.metadata and "necrosis_handling" in res.metadata


def _cells_adata(n=3000, seed=2):
    import pandas as pd
    from anndata import AnnData
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 700, size=(n, 2))
    endo = rng.random(n) < 0.02
    phi = np.where(endo, rng.normal(0.8, 0.1, n), np.abs(rng.normal(0.0, 0.05, n)))
    a = AnnData(np.ones((n, 2)), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=["G0", "G1"]))
    a.obsm["spatial"] = xy
    return a, phi, endo


def test_attach_oxygen_field_writes_covariate_columns():
    from tissue_dynamics.oxygen import attach_oxygen_field
    a, phi, endo = _cells_adata()
    res = attach_oxygen_field(a, phi, h=10.0, L0=100.0, beta=1.0, c=0.15)
    for col in ("oxygen_u", "oxygen_grad_u", "oxygen_valid"):
        assert col in a.obs.columns
    np.testing.assert_allclose(a.obs["oxygen_u"].to_numpy(), res.u_cells, equal_nan=True)
    fin = np.isfinite(a.obs["oxygen_u"].to_numpy())
    assert a.obs["oxygen_u"].to_numpy()[fin].max() <= 1 + 1e-12
    # endothelial cells sit in better-oxygenated tissue
    assert np.nanmean(a.obs["oxygen_u"].to_numpy()[endo]) > np.nanmean(a.obs["oxygen_u"].to_numpy()[~endo])
    # provenance and caveats travel with the covariate
    u = a.uns["oxygen_field"]
    assert u["params"]["c"] == 0.15 and u["solved_on"] == "adata"
    assert "field_is_2d" in u["metadata"] and u["diagnostics"]["rel_residual"] < 1e-8


def test_attach_oxygen_field_can_solve_on_a_larger_source_set():
    """Vessels outside the modelled subset still supply oxygen to it."""
    from tissue_dynamics.oxygen import attach_oxygen_field
    a, phi, _ = _cells_adata(n=3000)
    sub = a[:1500].copy()                                  # model only half the cells
    res = attach_oxygen_field(sub, None, source_xy=a.obsm["spatial"], source_phi_endo=phi,
                              h=10.0, L0=100.0, beta=1.0, c=0.15)
    assert sub.obs["oxygen_u"].shape[0] == 1500
    assert res.u_cells.shape[0] == 3000                    # solved on the full source set
    assert sub.uns["oxygen_field"]["solved_on"] == "source_xy"
    assert np.isfinite(sub.obs["oxygen_u"].to_numpy()).mean() > 0.9


def test_attach_oxygen_field_requires_source_phi_with_source_xy():
    from tissue_dynamics.oxygen import attach_oxygen_field
    a, phi, _ = _cells_adata(n=500)
    with pytest.raises(ValueError, match="source_phi_endo"):
        attach_oxygen_field(a, phi, source_xy=a.obsm["spatial"], h=20.0)
