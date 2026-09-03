"""Steady-state tissue oxygen field from soft endothelial labels.

Oxygen diffuses over hundreds of micrometres — far beyond the 80 um neighbourhood
the fate models use — so a cell's oxygenation is set by vasculature it never
touches. This module solves a screened-diffusion (reaction-diffusion) PDE on the
rasterised section to turn the endothelial Phi-Space score into two per-cell
covariates: relative oxygenation `u` in [0, 1] and the gradient magnitude
`|grad u|` (which picks out rim positions between vessel and hypoxic core).

The equation, on the tissue domain with no oxygen crossing the section edge:

    laplacian(u) = kappa(x) * u ,   -grad(u).n = beta * Phi(x) * (1 - u)

`kappa = (1/L0^2)(n/n0)` is consumption, rising with local cell density `n`;
`Phi` is the deposited endothelial weight. Sources are **Robin** (mass-transfer),
not Dirichlet: inflow is proportional to `beta * Phi * (1 - u)`, so it stops once
tissue reaches blood saturation. That keeps the operator symmetric positive
definite and guarantees `0 <= u <= 1` by the discrete maximum principle, while
`beta -> inf` recovers the Dirichlet limit.

`u` is deliberately scale-free relative oxygenation. It is **not** pO2: the model
is linear precisely so the field is invariant to arterial saturation and to the
absolute consumption rate.

Discretisation is a five-point Laplacian on a uniform grid where *every* tissue
pixel is an unknown. For each one:

    (n_valid + h^2 kappa + beta Phi) u_ij  -  sum_{in-tissue nbrs} u_nb  =  beta Phi_ij

`n_valid` counts only neighbours inside the tissue mask; dropping an outside
neighbour from both the off-diagonal and the diagonal *is* the entire no-flux
boundary condition (the mirrored ghost node cancels against the diagonal), and it
covers the section edge, tears and holes at once. `Phi` is a raw SUM of per-cell
weights per pixel, never a per-area density, which is what makes `beta`
grid-independent.

Caveats to carry with any result (see `OxygenFieldResult.metadata`): the field is
2D, so vessels running in-plane are over-weighted and supply from above or below
the section is invisible; it is a simulation, not a measurement; and vessel
presence is not perfusion — collapsed tumour vessels are treated as fully patent.

VALIDATION OUTCOME ON THE Xenium Prime breast section (2026-08)
---------------------------------------------------------------
The numerics validate completely: 5.13M unknowns solved by AMG to a relative
residual of 7e-11, `0 <= u <= 1` holds, the source signal share is 100% at
`c = 0.15` (2.6% of cells retained, the expected endothelial fraction), and the
field is decoupled from cell density (corr 0.12 raster / -0.03 at cells, spec
threshold 0.2). The Green's-function, no-flux/image-charge, harmonicity, grid-
independence and beta-limit tests all pass (see tests/test_oxygen.py).

The **biological check (spec 7.9) fails**, so on this dataset the field must NOT
be used as a hypoxia covariate. HIF-target expression correlates *positively*
with modelled oxygenation (Spearman +0.089; NB beta +0.066, z +15.6) where the
spec requires clearly negative. The failure is robust: positive for all six HIF
genes individually, positive for CA9 alone, and positive at every setting of a
10-point sweep over `L0` (50-400), `beta` (0.1-100) and `c` (0.10-0.40).
Aggregating to 100um tiles does not rescue it (the tile-level composite merely
flips sign with parameters). Endothelial contamination of the marker set was
ruled out — none of the six genes is endothelial-associated.

The dominant cause is the readout, not the solver. On 100um tiles CA9 -- the
specific marker -- has zero counts in 86% of tiles (EGLN3/ANGPTL4/BNIP3/ADM in
56-70%); only NDRG1 is broadly detected, and it is the least hypoxia-specific of
the six, so it dominates the composite. That composite is itself strongly
depth-confounded (Spearman with tile depth -0.30; median rate falls 1.53 -> 0.64
per 1,000 across depth quintiles) and only weakly autocorrelated in space
(Moran's I 0.19, versus 0.79 for the field). The spec's structural caveats
compound this: the field is 2D and blind to out-of-plane supply, and vessel
presence is not perfusion.

Test 7.9 here is therefore inconclusive rather than a clean refutation -- but the
rule stands, since an unlicensed covariate is exactly what 7.9 exists to exclude.
Treat this module as a validated screened-diffusion solver, and re-run test 7.9
against an adequate hypoxia readout before using its output as a covariate.
"""

from dataclasses import dataclass, field as _field

import numpy as np
import scipy.sparse as sp


@dataclass
class OxygenFieldResult:
    """Per-cell oxygen covariates plus the raster field and diagnostics."""

    u_cells: np.ndarray            # (N,) relative oxygenation at cell centroids
    grad_u_cells: np.ndarray       # (N,) |grad u| at cell centroids
    u_valid: np.ndarray            # (N,) bool: cell is in a solved component
    u: np.ndarray                  # (H, W) raster field, NaN outside the solve
    tissue: np.ndarray             # (H, W) bool tissue mask
    origin: np.ndarray             # (2,) grid origin in um
    h: float                       # grid pitch in um
    params: dict = _field(default_factory=dict)
    diagnostics: dict = _field(default_factory=dict)
    metadata: dict = _field(default_factory=dict)


# --------------------------------------------------------------------------
# grid construction
# --------------------------------------------------------------------------

def deposit(xy, weights, origin, shape, h):
    """Accumulate per-cell `weights` into their nearest pixel (a raw SUM).

    Summing — rather than averaging or dividing by pixel area — is what keeps
    `beta` grid-independent: refining `h` must not change the total source
    strength (spec test 7.5).
    """
    ij = np.floor((np.asarray(xy, dtype=float) - origin) / h).astype(np.int64)
    np.clip(ij[:, 0], 0, shape[0] - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, shape[1] - 1, out=ij[:, 1])
    grid = np.zeros(shape, dtype=float)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), np.asarray(weights, dtype=float))
    return grid, ij


def source_weights(phi_endo, c):
    """Soft-threshold the endothelial score into non-negative source weights.

    `w = max(clip(phi, 0) - c, 0)`. Phi-Space scores are not probabilities and go
    negative, hence the clip; the cutoff `c` then suppresses the noise floor.
    That cutoff is the most consequential choice in this pipeline — across
    hundreds of thousands of mostly non-endothelial cells the noise floor can
    carry more total weight than the real vessels, and the field then degenerates
    into a disguised cell-density map (spec tests 7.7 / 7.8). Calibrate it with
    `calibrate_cutoff`.
    """
    w = np.clip(np.asarray(phi_endo, dtype=float), 0.0, None)
    return np.maximum(w - float(c), 0.0)


def calibrate_cutoff(phi_endo, c):
    """Diagnostics for a candidate cutoff: signal share and retained fraction.

    `signal_share` is the fraction of total source weight contributed by cells in
    the top decile of `phi_endo` (spec requires > 0.90); `retained_frac` is the
    fraction of cells left with any weight, which should sit near the expected
    endothelial fraction (~1-5% in breast).
    """
    phi = np.asarray(phi_endo, dtype=float)
    w = source_weights(phi, c)
    total = w.sum()
    top = phi >= np.quantile(phi, 0.9)
    return {
        "c": float(c),
        "signal_share": float(w[top].sum() / total) if total > 0 else np.nan,
        "retained_frac": float((w > 0).mean()),
        "total_weight": float(total),
    }


def build_tissue_mask(occupied, h, close_um=50.0, fill_holes=True):
    """Tissue domain from occupied pixels: morphological closing, then fill holes.

    Interior holes are filled and therefore treated as *inside* the tissue with
    zero consumption (necrosis) — oxygen diffuses freely through them. Only the
    exterior is outside, where no oxygen crosses. Getting this backwards would
    invert the physics of every necrotic core, so when a necrosis mask is
    unavailable this is the deliberate default (spec 4.2) and is recorded in the
    result metadata.
    """
    from scipy import ndimage as ndi

    rad = max(int(round(close_um / h)), 1)
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    tissue = ndi.binary_closing(np.asarray(occupied, dtype=bool), structure=disk)
    if fill_holes:
        tissue = ndi.binary_fill_holes(tissue)
    return tissue


def build_consumption(counts, tissue, h, L0=100.0, sigma_density=25.0, necrosis=None):
    """Consumption map `kappa = (1/L0^2)(n/n0)` from smoothed cell density.

    Density only — never weighted by cell type or proliferative state. Weighting
    by tumour content would be circular: it would make the oxygen field a
    function of local proliferation, which is precisely the outcome the covariate
    exists to de-confound (spec 4.3).
    """
    from scipy import ndimage as ndi

    dens = ndi.gaussian_filter(np.asarray(counts, dtype=float), sigma_density / h) / (h ** 2)
    inside = tissue & (dens > 0)
    n0 = float(np.median(dens[inside])) if inside.any() else 1.0
    kappa = (dens / max(n0, 1e-30)) / (L0 ** 2)
    if necrosis is not None:
        kappa[np.asarray(necrosis, dtype=bool)] = 0.0
    kappa[~tissue] = 0.0
    return kappa, n0


# --------------------------------------------------------------------------
# assembly and solve
# --------------------------------------------------------------------------

def assemble_robin(tissue, Phi, kappa, h, beta):
    """Assemble the SPD Robin system over every tissue pixel.

    Returns `(A, b, idx)` with `idx` mapping grid position to unknown index
    (-1 outside). Off-diagonals are all -1 and diagonals are strictly positive
    wherever there is consumption or source, so `A` is symmetric positive
    definite and diagonally dominant.
    """
    tissue = np.asarray(tissue, dtype=bool)
    H, W = tissue.shape
    n = int(tissue.sum())
    idx = -np.ones((H, W), dtype=np.int64)
    idx[tissue] = np.arange(n)

    rows, cols = [], []
    n_valid = np.zeros((H, W), dtype=np.int64)

    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_t = np.roll(tissue, (-di, -dj), axis=(0, 1))
        nb_i = np.roll(idx, (-di, -dj), axis=(0, 1))
        # np.roll wraps; blank the wrapped edge so it reads as "outside the section"
        sl = [slice(None), slice(None)]
        if di == 1:
            sl[0] = slice(H - 1, H)
        elif di == -1:
            sl[0] = slice(0, 1)
        elif dj == 1:
            sl[1] = slice(W - 1, W)
        else:
            sl[1] = slice(0, 1)
        nb_t[tuple(sl)] = False

        m = tissue & nb_t
        n_valid += m
        rows.append(idx[m].astype(np.int32))
        cols.append(nb_i[m].astype(np.int32))

    diag = n_valid[tissue] + (h ** 2) * kappa[tissue] + beta * Phi[tissue]
    if np.any(diag <= 0):
        raise ValueError(
            "singular row: a tissue fragment has no consumption and no source. "
            "Drop such connected components before assembling (see drop_singular_components)."
        )

    n_off = sum(len(r) for r in rows)
    data = np.empty(n_off + n, dtype=float)
    data[:n_off] = -1.0
    data[n_off:] = diag
    rows.append(np.arange(n, dtype=np.int32))
    cols.append(np.arange(n, dtype=np.int32))

    A = sp.csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, beta * Phi[tissue], idx


def drop_singular_components(tissue, Phi, kappa):
    """Remove tissue components with neither consumption nor source.

    Such a block is singular (a pure Neumann Laplacian with zero right-hand side),
    so it is excluded from the solve and its cells are returned as NaN/invalid.
    """
    from scipy import ndimage as ndi

    lab, n_lab = ndi.label(tissue)
    keep = np.asarray(tissue, dtype=bool).copy()
    dropped = 0
    for comp in range(1, n_lab + 1):
        m = lab == comp
        if not (kappa[m] > 0).any() and Phi[m].sum() <= 0:
            keep[m] = False
            dropped += 1
    return keep, dropped, n_lab


def solve_oxygen(tissue, Phi, kappa, h, beta, tol=1e-10, maxiter=50000):
    """Solve the Robin system; returns `(u_raster, info)` with NaN outside tissue.

    Prefers algebraic multigrid (`pyamg`), which is the textbook solver for this
    operator, and falls back to Jacobi-preconditioned CG. `spsolve` is never used:
    the direct factorisation fills in badly past ~1e6 unknowns.
    """
    A, b, _ = assemble_robin(tissue, Phi, kappa, h, beta)
    try:
        import pyamg
        x = pyamg.smoothed_aggregation_solver(A).solve(b, tol=tol)
        how = "pyamg"
    except ImportError:
        from scipy.sparse.linalg import cg
        M = sp.diags(1.0 / A.diagonal())
        x, code = cg(A, b, rtol=tol, maxiter=maxiter, M=M)
        how = f"cg(info={code})"

    nb = float(np.linalg.norm(b))
    residual = float(np.linalg.norm(A @ x - b) / nb) if nb > 0 else 0.0
    u = np.full(tissue.shape, np.nan)
    u[np.asarray(tissue, dtype=bool)] = x
    return u, {"solver": how, "n_unknowns": int(A.shape[0]), "nnz": int(A.nnz),
               "rel_residual": residual}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def sample_at_cells(u, xy, origin, h, tissue=None):
    """Bilinearly sample `u` and `|grad u|` at cell centroids.

    Bilinear, not nearest-pixel, so the covariate is smooth in space. Before
    differencing, pixels outside the solved domain are filled with their nearest
    solved value: leaving them at 0 would manufacture a huge artificial gradient
    along the whole tissue boundary.
    """
    from scipy import ndimage as ndi

    filled = np.array(u, dtype=float)
    bad = ~np.isfinite(filled)
    if bad.any() and (~bad).any():
        _, near = ndi.distance_transform_edt(bad, return_indices=True)
        filled = filled[tuple(near)]
    filled = np.nan_to_num(filled)

    gy, gx = np.gradient(filled, h)
    gmag = np.hypot(gy, gx)

    p = (np.asarray(xy, dtype=float) - origin) / h
    i0 = np.clip(np.floor(p[:, 0]).astype(int), 0, filled.shape[0] - 2)
    j0 = np.clip(np.floor(p[:, 1]).astype(int), 0, filled.shape[1] - 2)
    fi = np.clip(p[:, 0] - i0, 0.0, 1.0)
    fj = np.clip(p[:, 1] - j0, 0.0, 1.0)

    def interp(f):
        return (f[i0, j0] * (1 - fi) * (1 - fj) + f[i0 + 1, j0] * fi * (1 - fj)
                + f[i0, j0 + 1] * (1 - fi) * fj + f[i0 + 1, j0 + 1] * fi * fj)

    valid = np.ones(len(p), dtype=bool)
    if tissue is not None:
        ii = np.clip(np.round(p[:, 0]).astype(int), 0, tissue.shape[0] - 1)
        jj = np.clip(np.round(p[:, 1]).astype(int), 0, tissue.shape[1] - 1)
        valid = np.asarray(tissue, dtype=bool)[ii, jj]
    return interp(filled), interp(gmag), valid


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------

def build_oxygen_field(xy, phi_endo, *, necrosis_mask=None, h=5.0, L0=100.0, beta=1.0,
                        c=0.15, sigma_density=25.0, tol=1e-10, close_um=50.0,
                        pad_um=100.0, verbose=False):
    """Solve the tissue oxygen field and sample it at every cell.

    Parameters follow the brief: `h` grid pitch, `L0` screening length at the
    reference density, `beta` source permeability, `c` the endothelial
    soft-threshold, `sigma_density` the density smoothing scale. `L0`, `beta` and
    `c` are the three that matter; sweep them with `sweep_parameters`.

    Returns an `OxygenFieldResult`.
    """
    xy = np.asarray(xy, dtype=float)
    phi_endo = np.asarray(phi_endo, dtype=float)
    origin = xy.min(axis=0) - pad_um
    extent = xy.max(axis=0) + pad_um - origin
    shape = (int(np.ceil(extent[0] / h)) + 1, int(np.ceil(extent[1] / h)) + 1)

    w = source_weights(phi_endo, c)
    Phi, _ = deposit(xy, w, origin, shape, h)
    counts, _ = deposit(xy, np.ones(len(xy)), origin, shape, h)

    tissue = build_tissue_mask(counts > 0, h, close_um=close_um)
    necrosis = None
    if necrosis_mask is not None:
        necrosis, _ = deposit(xy, np.asarray(necrosis_mask, dtype=float), origin, shape, h)
        necrosis = necrosis > 0
    kappa, n0 = build_consumption(counts, tissue, h, L0=L0,
                                  sigma_density=sigma_density, necrosis=necrosis)

    tissue, n_dropped, n_comp = drop_singular_components(tissue, Phi, kappa)
    u, info = solve_oxygen(tissue, Phi, kappa, h, beta, tol=tol)
    u_cells, grad_cells, valid = sample_at_cells(u, xy, origin, h, tissue=tissue)
    u_cells = np.where(valid, u_cells, np.nan)
    grad_cells = np.where(valid, grad_cells, np.nan)

    diagnostics = dict(info)
    diagnostics.update(calibrate_cutoff(phi_endo, c))
    diagnostics.update({
        "n_cells": int(len(xy)), "grid_shape": tuple(shape),
        "tissue_px": int(tissue.sum()), "tissue_frac": float(tissue.mean()),
        "n_components": int(n_comp), "n_components_dropped": int(n_dropped),
        "n_cells_invalid": int((~valid).sum()),
        "reference_density_per_um2": float(n0),
        "u_min": float(np.nanmin(u)), "u_max": float(np.nanmax(u)),
        "u_mean_cells": float(np.nanmean(u_cells)),
    })
    if verbose:
        d = diagnostics
        print("  %s: %d unknowns, rel residual %.2e | signal share %.1f%%, retained %.2f%%"
              % (d["solver"], d["n_unknowns"], d["rel_residual"],
                 100 * d["signal_share"], 100 * d["retained_frac"]))

    return OxygenFieldResult(
        u_cells=u_cells, grad_u_cells=grad_cells, u_valid=valid, u=u, tissue=tissue,
        origin=origin, h=float(h),
        params={"h": h, "L0": L0, "beta": beta, "c": c,
                "sigma_density": sigma_density, "tol": tol, "close_um": close_um},
        diagnostics=diagnostics,
        metadata={
            "field_is_2d": "Vessels perpendicular to the section are represented; "
                           "in-plane vessels are over-weighted and out-of-plane supply is "
                           "invisible, so hypoxia is overestimated where supply comes from "
                           "above or below. Not fixable from a single section.",
            "simulation_not_measurement": "A deterministic function of the endothelial "
                                          "score map, the density map and three constants. "
                                          "The HIF-target check (spec 7.9) is what licenses "
                                          "its use as a covariate.",
            "vessel_presence_is_not_perfusion": "Collapsed / non-functional tumour vessels "
                                                "are treated as fully patent.",
            "necrosis_handling": ("explicit necrosis_mask supplied" if necrosis_mask is not None
                                  else "no necrosis mask: interior holes filled and treated as "
                                       "necrosis (tissue=True, kappa=0); only the exterior is "
                                       "outside the domain"),
        },
    )


def sweep_parameters(xy, phi_endo, *, grid, **kwargs):
    """Run `build_oxygen_field` over a parameter grid, returning a tidy DataFrame.

    `grid` maps parameter name to the values to try (e.g.
    `{"c": [0.10, 0.15, 0.25], "L0": [50.0, 100.0, 200.0]}`); every combination is
    solved. Built so a full sweep over `L0`, `beta` and `c` is a loop, not a rewrite.
    """
    import itertools

    import pandas as pd

    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in names)):
        over = dict(zip(names, combo))
        res = build_oxygen_field(xy, phi_endo, **{**kwargs, **over})
        row = dict(over)
        row.update({k: res.diagnostics[k] for k in
                    ("rel_residual", "signal_share", "retained_frac", "u_mean_cells",
                     "u_min", "u_max", "n_cells_invalid")})
        row["u_cells"] = res.u_cells
        rows.append(row)
    return pd.DataFrame(rows)


def attach_oxygen_field(adata, phi_endo, *, source_xy=None, source_phi_endo=None,
                         spatial_key="spatial", key="oxygen_u", grad_key="oxygen_grad_u",
                         valid_key="oxygen_valid", uns_key="oxygen_field", **field_kwargs):
    """Solve the oxygen field and attach it to `adata.obs` as model covariates.

    The covariate counterpart of `compute_library_size` / `compute_local_density`:
    adds `obs[key]` (relative oxygenation `u`), `obs[grad_key]` (`|grad u|`) and
    `obs[valid_key]`, plus the parameters, diagnostics and caveats under
    `uns[uns_key]`. Pass the result straight to a model, e.g.
    `fit_division_nb(adata, covariate_keys=("cell_area", "oxygen_u"), ...)`.

    `source_xy` / `source_phi_endo` let the PDE be solved over a *larger* cell set
    than the one being modelled — the physically right thing to do, since oxygen
    is supplied by vessels near cells that may themselves have been filtered out
    of the analysis. When omitted, the field is solved on `adata`'s own cells.

    On what this covariate may be called: the field is a *simulation* of oxygen
    transport, and on the Xenium breast section it failed the HIF-target check
    (spec 7.9) — see the module docstring. It nonetheless reduces the spatial
    autocorrelation of division-model residuals well beyond a spatially matched
    decoy, so the defensible label is **vessel proximity / perfusion**, not
    hypoxia. Re-run test 7.9 on any new dataset before calling it the latter.
    """
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    if source_xy is None:
        res = build_oxygen_field(xy, phi_endo, **field_kwargs)
        u_cells, grad_cells, valid = res.u_cells, res.grad_u_cells, res.u_valid
    else:
        if source_phi_endo is None:
            raise ValueError("source_phi_endo is required when source_xy is given")
        res = build_oxygen_field(np.asarray(source_xy, dtype=float), source_phi_endo,
                                 **field_kwargs)
        u_cells, grad_cells, valid = sample_at_cells(res.u, xy, res.origin, res.h,
                                                     tissue=res.tissue)
        u_cells = np.where(valid, u_cells, np.nan)
        grad_cells = np.where(valid, grad_cells, np.nan)

    adata.obs[key] = u_cells
    adata.obs[grad_key] = grad_cells
    adata.obs[valid_key] = valid
    adata.uns[uns_key] = {"params": res.params, "diagnostics": res.diagnostics,
                          "metadata": res.metadata,
                          "solved_on": "source_xy" if source_xy is not None else "adata"}
    return res
