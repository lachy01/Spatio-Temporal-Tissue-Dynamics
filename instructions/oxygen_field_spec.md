# Implementation brief: tissue oxygen field via sparse PDE solve with soft endothelial labels

## Objective

Produce two per-cell covariates for a spatial transcriptomics regression model:

- `u_i` — relative oxygenation at cell *i*, dimensionless, in [0, 1]
- `grad_u_i` — magnitude of the oxygen gradient at cell *i* (identifies rim positions)

These are computed by solving a steady-state reaction–diffusion PDE on a rasterised
tissue domain, where oxygen sources are **soft-labelled** endothelial cells (a
continuous score, not a binary call) and consumption is proportional to local cell
density.

Deliverable: a tested Python module plus a script that runs the pipeline end to end
and writes the two covariates back to the cell table. Do not report the task complete
until every validation test in §7 passes.

---

## 1. The equation

Solve on the tissue domain Ω:

    ∇²u = κ(x) · u

with a Robin (mass-transfer) source at endothelial cells and no-flux at the tissue
boundary:

    -∇u · n̂ = β · Φ(x) · (1 - u)      at source pixels
     ∇u · n̂ = 0                        on ∂Ω

- `κ(x) = 1 / L(x)²` is the consumption rate, units µm⁻²
- `L(x) = L₀ · sqrt(n₀ / n(x))` is the local screening length, where `n(x)` is local
  cell density and `n₀` a reference density. Equivalently `κ(x) = (1/L₀²)·(n(x)/n₀)`.
- `Φ(x)` is the deposited endothelial source weight
- `β` is a dimensionless permeability parameter

`u` is scale-free relative oxygenation. Do **not** attempt to convert to absolute pO₂ —
the model is deliberately linear so that the field is invariant to arterial saturation
and absolute consumption rate.

---

## 2. Discretisation — read this section carefully

Uniform square grid, pitch `h`. Five-point Laplacian. **Every tissue pixel is an
unknown. There are no Dirichlet pixels.** For each tissue pixel (i,j):

    (n_valid + h²·κ_ij + β·Φ_ij) · u_ij  -  Σ_{in-tissue nbrs} u_nb  =  β·Φ_ij

Three details that are easy to get wrong:

**(a) `n_valid` counts only neighbours inside the tissue mask.** The default would be
4. If a neighbour pixel is outside the tissue, you drop its off-diagonal term *and*
decrement the diagonal. That single omission is the entire no-flux boundary condition
— it is the mirrored ghost node `u_ghost = u_ij` cancelling against the diagonal. Do
not add any other boundary handling. This automatically covers the section edge,
tissue tears, and holes.

**(b) `Φ_ij` is a raw SUM of per-cell weights in that pixel, not a per-area density
and not a mean.** This makes `β` grid-independent: refining `h` must not change the
total source strength. Verify this explicitly (test 7.5).

**(c) The matrix is symmetric positive definite.** Off-diagonals are all `-1`;
diagonals are `n_valid + h²κ + βΦ > 0`. Diagonal dominance is strict wherever there is
consumption or source. If your assembled matrix is not symmetric, you have a bug —
check it with `abs(A - A.T).max()`.

---

## 3. Inputs

From the cell table (AnnData / SpatialData / DataFrame):

| input | description |
|---|---|
| `xy` | (N, 2) cell centroid coordinates in **micrometres** |
| `phi_endo` | (N,) endothelial PhiSpace score |
| `necrosis_mask` | optional (N,) bool or a polygon set marking acellular/necrotic regions |

Confirm coordinate units before anything else. If the source is Xenium, centroids are
already in µm; if they are in pixels, convert.

---

## 4. Building the grids

Process in this order.

### 4.1 Source weight `Φ`

1. Clip: `w = np.clip(phi_endo, 0, None)`. PhiSpace scores are not probabilities and
   can be negative.
2. Soft threshold: `w = np.maximum(w - c, 0.0)` with cutoff `c`.
3. Deposit: `np.add.at(Phi, (ix, iy), w)` where `(ix, iy)` are nearest-pixel indices.

**Choosing `c` is the most important judgement call in this pipeline.** The noise floor
across hundreds of thousands of non-endothelial cells can exceed the true endothelial
signal — at a 2% endothelial fraction with a 0.05 noise floor, raw scores put ~70% of
total source strength in non-endothelial cells, and the resulting field correlates
~0.4 with a plain cell-density map. Calibrate `c` as follows:

- Plot the `phi_endo` histogram on a log-count axis. There is usually a noise mode near
  zero and a shoulder or second mode for real endothelium. Put `c` at the trough.
- Then compute the **signal share**: the fraction of `Φ.sum()` contributed by cells in
  the top decile of `phi_endo`. Require > 90%. Report the value.
- Report the retained fraction of cells with `w > 0` and sanity-check it against the
  expected endothelial fraction for breast tissue (roughly 1–5%).
- Start at `c = 0.15` and adjust. Expose `c` as a parameter and re-run the whole
  pipeline at `c ∈ {0.10, 0.15, 0.25}` as a sensitivity check.

### 4.2 Tissue mask

`tissue[i,j] = True` where tissue is present. Build it from occupied pixels via binary
closing (`scipy.ndimage.binary_closing`) with a structuring element of roughly 50 µm,
then fill interior holes.

**Critical distinction:**
- **Necrotic / acellular regions are INSIDE the tissue** (`tissue = True`, `κ = 0`).
  Oxygen diffuses freely through them.
- **Tears, folds, detachment, and the section exterior are OUTSIDE** (`tissue = False`).
  No oxygen crosses these.

Getting these backwards inverts the physics of every necrotic core in the section. If
you only have a low-density mask and cannot distinguish the two, default to treating
interior holes as necrosis (tissue = True, κ = 0) and only the exterior as non-tissue,
and flag this in the output metadata.

### 4.3 Consumption map `κ`

1. Deposit cell counts to a grid.
2. Smooth with a Gaussian, σ ≈ 25 µm, to get density `n(x)` in cells per µm².
3. `kappa = (1 / L0**2) * (n / n0)` with `n0 = np.median(n[tissue & (n > 0)])`.
4. Set `kappa[necrosis] = 0`.
5. Set `kappa = 0` outside the tissue (irrelevant, but keeps arrays clean).

**Do not weight consumption by cell type or proliferative state.** It is tempting,
since tumour cells consume more oxygen than stroma. It is also circular: it would make
the oxygen field a function of local proliferation, which is the outcome variable this
covariate exists to de-confound. Cell density only.

### 4.4 Connected components

Label connected components of `tissue`. Any component where `(kappa > 0).sum() == 0`
and `Phi.sum() == 0` yields a singular block. Drop those components from the solve and
return `NaN` for cells inside them. Report how many cells this affects.

---

## 5. Solving

Assemble as CSR. Solve with, in order of preference:

1. `pyamg.smoothed_aggregation_solver(A).solve(b, tol=1e-10)` — this problem is the
   textbook target for algebraic multigrid
2. `scipy.sparse.linalg.cg` with a Jacobi preconditioner `sp.diags(1/A.diagonal())`,
   `rtol=1e-10`, generous `maxiter`

Do **not** use `spsolve` — the direct factorisation fills in badly past ~10⁶ unknowns.

Use float64 throughout the solve. At 150 mm² and `h = 5` µm you have ~6M unknowns and
~30M nonzeros, roughly 400 MB for the matrix plus the AMG hierarchy. That is fine on a
workstation; if memory is tight, raise `h` before switching to float32.

Always report the relative residual `||Au - b|| / ||b||` and assert it is below 1e-8.

---

## 6. Extraction

- Sample `u` at cell centroids by **bilinear** interpolation, not nearest-pixel.
- Compute `|∇u|` from `np.gradient(u, h)` on the filled field, then sample the same way.
- Return both as columns on the cell table, plus a `u_valid` boolean flagging cells in
  dropped components or outside the mask.

---

## 7. Validation tests — all must pass

Write these as a test suite. Report every number.

**7.1 Green's function.** Homogeneous `κ = 1/L²`, no boundaries within range, a single
point source, small `β` (e.g. 1e-4) so the source stays in the linear regime. Then
`u(r) / K₀(r/L)` must be constant. Require CV < 1e-3 over `8h < r < 4.5L`. This
confirms the discretisation reproduces the analytic screened-diffusion kernel.

**7.2 Maximum principle.** `0 ≤ u ≤ 1` everywhere, to within 1e-12.

**7.3 Harmonicity in necrosis.** Pick an interior pixel of a `κ = 0` region. Its value
must equal the mean of its four neighbours to ~1e-9.

**7.4 No-flux boundary.** Along a straight exterior edge, `|u[-1,:] - u[-2,:]|` must be
< 1e-5 relative to the field scale.

**7.5 Grid independence of source strength.** Solve at `h` and `h/2` on the same
geometry with the same `β`. Away from sources, fields must agree closely; Spearman
correlation of `u` at cell positions > 0.99. If they disagree systematically, `Φ` is
being treated as a density rather than a sum.

**7.6 β limit.** As `β → ∞` with a binary source mask, the field must converge to the
Dirichlet solution (`u = 1` pinned on source pixels). Check that `max(u) → 1` and that
increasing `β` by 100× changes the field by < 1e-4.

**7.7 Signal share.** As described in §4.1: > 90% of `Φ.sum()` from the top decile of
`phi_endo`.

**7.8 Density decoupling.** Re-solve using cell *counts* as the source instead of `Φ`.
Correlation between the two resulting fields must be < 0.2. If it is high, the oxygen
field is a disguised density map and `c` is too low.

**7.9 Biological check.** Score each cell for HIF targets — `CA9`, `NDRG1`, `ADM`,
`EGLN3`, `BNIP3`, `ANGPTL4` — from raw counts with a library-size offset. Regress on
`u_i`. Report the Spearman correlation; it should be clearly negative. Deliberately
exclude the glycolytic block (`LDHA`, `PGK1`, `ALDOA`, `ENO1`, `PKM`, `TPI1`) from this
score: those genes track total transcriptional output and proliferative metabolism, so
they would make the check pass for the wrong reason. This is a reportable result, not
just a test — if it fails, the field is not measuring hypoxia and must not enter the
main model.

---

## 8. Parameters

Expose all of these; do not hard-code them inside functions.

| parameter | default | notes |
|---|---|---|
| `h` | 5.0 µm | grid pitch; ~20 px per screening length |
| `L0` | 100.0 µm | screening length at reference density; profile this |
| `beta` | 1.0 | permeability; profile over 0.1 – 100 |
| `c` | 0.15 | soft-threshold cutoff; calibrate per §4.1 |
| `sigma_density` | 25.0 µm | Gaussian smoothing for the density map |
| `tol` | 1e-10 | solver tolerance |

`L0`, `beta` and `c` are the three that matter. Build the pipeline so a full sweep over
them is a loop, not a rewrite.

---

## 9. Known limitations to record in the output metadata

- **The field is 2D.** Vessels running perpendicular to the section are correctly
  represented; vessels running in-plane are over-weighted, and out-of-plane supply is
  invisible. The field therefore overestimates hypoxia in regions fed from above or
  below, with spatially structured error. This cannot be fixed with a single section.
- **The output is a simulation, not a measurement.** It is a deterministic function of
  the endothelial score map, the density map, and three chosen constants. Test 7.9 is
  what licenses its use as a covariate; without it the field is decoration.
- **Vessel presence is not perfusion.** Tumour vasculature includes collapsed and
  non-functional vessels that this model treats as fully patent.

---

## 10. Suggested API

```python
def build_oxygen_field(
    xy, phi_endo, *,
    necrosis_mask=None, h=5.0, L0=100.0, beta=1.0, c=0.15,
    sigma_density=25.0, tol=1e-10,
) -> OxygenFieldResult
```

where `OxygenFieldResult` carries `u_cells`, `grad_u_cells`, `u_valid`, the raster `u`,
the grid origin and pitch, the parameter set, and a diagnostics dict holding the
residual, signal share, retained-cell fraction, dropped-component count, and every
test statistic from §7.

Keep assembly, solving and extraction in separate functions so the validation suite can
call them on synthetic geometries without going through the data loader.
