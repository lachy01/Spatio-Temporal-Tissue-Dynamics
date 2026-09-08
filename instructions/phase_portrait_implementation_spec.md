# Phase Portrait Construction — Implementation Spec

**Module:** `osdr_portrait/`
**Depends on:** fitted Stage B (or later) concomitant-variable Poisson mixture model
**Language:** Python (numpy / scipy / matplotlib). No new model fitting is required by this module.

---

## 0. Scope and what this module is *not*

This module converts an already-fitted mixture model into a deterministic vector field over
archetype-density state space, locates and classifies fixed points, and renders pairwise phase
portraits with uncertainty.

**It does not run an agent-based or stochastic simulation.** The phase portrait is the
right-hand side of the ODE evaluated on a grid. Simulation of tissue trajectories from an
initial spatial condition is a separate, complementary output and is explicitly out of scope
here. Do not implement neighbourhood updating, cell placement, or time-stepping.

Reference construction: Hartmann et al. OSDR, Methods §"Population versus neighbourhood
dynamics" and Supplementary Fig. S3D (vector field on a finite grid, bootstrap arrows overlaid).

---

## 1. Inputs

### 1.1 Fitted model coefficients

Extract from the Stage B/C fit object into a plain dataclass so the portrait code has no
dependency on the fitting framework:

```python
@dataclass(frozen=True)
class MixtureCoefficients:
    beta0: float                 # intercept of the mixing-weight linear predictor
    b_own: np.ndarray            # (29,)   own-state phi coefficients
    c_nbr: np.ndarray            # (29,)   neighbourhood-state phibar coefficients
    U: np.ndarray                # (29, r) bilinear left factor
    V: np.ndarray                # (29, r) bilinear right factor
    density_basis: SplineBasis   # basis object for s(n); must expose .design(n) and .deriv(n)
    density_coef: np.ndarray     # (n_basis,)
    gamma_oxygen: float          # coefficient on the vascular proximity field
    other_coefs: dict[str, float]  # any remaining weight-submodel covariates
```

`B = U @ V.T`. The linear predictor for the mixing weight of a cell with own state `phi`,
neighbourhood state `phibar`, local density `n`, oxygen `O`:

```
eta = beta0 + b_own @ phi + c_nbr @ phibar + phi @ B @ phibar
      + density_coef @ density_basis.design(n) + gamma_oxygen * O + <other covariates>
pi  = expit(eta)
```

**Only the mixing-weight submodel is used.** The component rates (library offset, area spline,
proportional-rates restriction) do not enter the portrait — the state variable is P(cycling),
not expected counts. Assert at load time that no component-rate coefficients have been
mistakenly passed in.

### 1.2 Per-cell arrays

From the fitted model's design matrices, for all `N ≈ 700,000` cells:

| Array | Shape | Meaning |
|---|---|---|
| `phi` | (N, 29) | own PhiSpace state |
| `phibar` | (N, 29) | neighbourhood-averaged state (80 µm disc) |
| `n_local` | (N,) | local cell count in the same neighbourhood |
| `oxygen` | (N,) | vascular proximity field value |
| `pi_hat` | (N,) | fitted mixing weight per cell |
| `block_id` | (N,) | contiguous tissue block label (from the spatial CV partition) |

### 1.3 Archetype definition

`A` is a `(K, 29)` matrix whose rows are archetype states in PhiSpace. Support two providers
behind a common interface, selected by config:

- `"archetypal"` — PCHA archetypal analysis fitted on the observed `phibar` matrix. Returns
  both `A` and per-cell simplex loadings `W` (N, K), `W >= 0`, rows summing to 1.
- `"centroid"` — named reference centroids (e.g. Cancer Epithelial, CAF, PVL, Myeloid,
  T-cell, B/Plasma, Endothelial) taken as the mean `phi` of cells with the corresponding
  dominant PhiSpace label. Loadings are then obtained by projecting `phibar` onto the simplex
  spanned by `A` via non-negative least squares with a sum-to-one constraint.

Default `K = 6`. The `"centroid"` provider is the default for the headline figure, because
named archetypes are what make the result directly comparable to OSDR's fibroblast–macrophage
portrait.

**Required diagnostic (blocking):** report the reconstruction quality of the archetype
representation,

```
R2_recon = 1 - ||phibar - W @ A||_F^2 / ||phibar - mean(phibar)||_F^2
```

per archetype and overall. The portrait discards the residual `phibar - W @ A`. If
`R2_recon < 0.8`, emit a loud warning in the run log and record the value in the output
manifest — it is a stated limitation, not a failure condition.

---

## 2. Coordinate map

State vector `X ∈ R_+^K`, archetype densities:

```
X_k = n * wbar_k
```

with inverse

```
n     = sum(X)
wbar  = X / n
phibar(X) = wbar @ A          # (29,) lift back into PhiSpace
```

Implement as two pure functions with a round-trip unit test:

```python
def state_from_cells(n_local, W) -> np.ndarray:      # (N, K)
def covariates_from_state(X, A) -> tuple[np.ndarray, float]:   # returns (phibar, n)
```

**Own state at archetype k** is `phi = A[k]`. Provide a config flag
`own_state_mode ∈ {"vertex", "loading_weighted"}`; `"loading_weighted"` uses
`sum_i W[i,k] * phi[i] / sum_i W[i,k]` instead, and should be run as a sensitivity variant.

---

## 3. Rates and flow

### 3.1 Reference values for frozen covariates

Oxygen, cell area, library size and any other technical covariates are **not** state variables.
Freeze them:

```python
@dataclass(frozen=True)
class ReferenceCovariates:
    oxygen: float          # default: median(oxygen)
    others: dict[str, float]   # default: median of each
```

Oxygen is a sensitivity axis (see §8), not a fixed constant across all runs.

### 3.2 Division and death

```
h_plus[k](X)  = pi_k(X) / tau[k]
h_minus[k]    = pi_bar[k] / tau[k]
```

where `pi_k(X) = expit(eta_k(X))` assembled per §1.1 with `phi = A[k]`,
`phibar = phibar(X)`, `n = sum(X)`, and

```
pi_bar[k] = sum_i W[i,k] * pi_hat[i] / sum_i W[i,k]
```

This loading-weighted mean is the continuous-state generalisation of OSDR's
`p_t^- = mean(y_t)` (division indicator averaged over cells of type `t`).

`tau[k]` defaults to `1.0` for all `k`, matching OSDR's `dt+ = dt- = 1` convention. Rates are
then in units of one marker-persistence interval.

### 3.3 Vector field

```
F_k(X) = X_k * (pi_k(X) - pi_bar[k]) / tau[k]
```

Implement vectorised over a grid of states:

```python
def velocity(X: np.ndarray,        # (M, K) batch of states
             coef, A, pi_bar, tau, ref) -> np.ndarray:   # (M, K)
```

**Critical implementation note.** When sweeping two coordinates on a slice, `n = sum(X)`
**must be recomputed at every grid point**. Do not freeze `n` at a reference value. The
density spline is the channel through which density-mediated competition enters; freezing `n`
silently deletes the confinement term and will produce a portrait with no bounded attractor.

---

## 4. Analytic Jacobian

Let `g_k = c_nbr + B.T @ A[k]` (the gradient of `eta_k` with respect to `phibar`), and
`s'(n) = density_coef @ density_basis.deriv(n)`. Using `d phibar / d X_j = (A[j] - phibar) / n`
and `d n / d X_j = 1`:

```
d eta_k / d X_j = g_k @ (A[j] - phibar(X)) / n + s_prime(n)

J[k, j] = delta_kj * (pi_k - pi_bar[k]) / tau[k]
        + X_k * pi_k * (1 - pi_k) / tau[k] * ( g_k @ (A[j] - phibar) / n + s_prime(n) )
```

At a fixed point the first term vanishes and the community matrix reduces to a row-centred
version of the archetype interaction matrix plus a rank-one all-ones contribution from the
density spline:

```
Atilde = A @ B @ A.T                     # (K, K) generalised Lotka–Volterra interaction matrix
g_k @ A[j]     = c_nbr @ A[j] + Atilde[k, j]
g_k @ phibar   = c_nbr @ phibar + (Atilde @ wbar)[k]
```

Expose `Atilde` as a first-class output (heatmap figure + CSV) — it is the object the portrait
is a geometric rendering of.

**Blocking unit test:** verify `J` against `scipy.optimize.approx_fprime` on at least 20 random
interior states, tolerance `1e-6` relative. Do not proceed if this fails.

---

## 5. Support masking

The linear predictor contains a bilinear term under an `expit` link; extrapolation outside the
observed data hull is not meaningful. Mask before plotting, not after.

For each grid point `X`:

1. **Per-axis cap.** Reject if any `X_k` exceeds the 95th percentile of the observed
   `X[:, k]` (OSDR's rule).
2. **Local support.** Reject if fewer than `min_cells = 20` observed cells fall within the
   grid cell (for the two swept coordinates) *and* within a tolerance band
   `|X_m - X_m^ref| < tol_m` for each held coordinate `m`, with `tol_m` set to half the
   interquartile range of `X[:, m]`.
3. **Convex hull.** Reject if outside the convex hull of the observed `(X_k, X_l)` projection.

Record `masked_fraction` in the manifest. Plot masked regions in neutral grey with no arrows —
do not silently crop the axes, since the reader needs to see where support ends.

**Extrapolation check (report, do not auto-correct).** At the 95th-percentile boundary of each
axis, evaluate `sign(F_k)`. OSDR bolted on `p_correction = c_k * d_k^n` because their logistic
model produced unbounded positive flux at high density. Your model has an estimated density
spline, so check first: report `s'(n)` at the 90th percentile of `n` and the sign of net flux
at the boundary. If flux is confining, no correction is needed and this should be stated as an
advantage over OSDR. If it is not, implement the OSDR correction as a **sensitivity variant
only**, never as default, and flag it prominently.

---

## 6. Fixed points

### 6.1 Interior fixed points

1. Evaluate `F` on a dense grid over the slice (default `200 x 200`).
2. Seed multi-start root finding from every grid point where `||F|| ` is below the 5th
   percentile of grid `||F||` values.
3. Solve with `scipy.optimize.root(method="hybr")` on the reduced 2D system (held coordinates
   fixed at reference), with the constraint `X > 0` enforced by solving in `log X`.
4. Deduplicate roots within `1e-3` relative distance; discard any outside the support mask.

### 6.2 Boundary fixed points

OSDR reports single-type equilibria (the "cold fibrosis" fibroblast-only point). These live on
faces `X_k = 0`. Because `F_k = X_k * (...)`, every face is invariant, so search them
explicitly: for each subset `S` of swept coordinates, set `X_S = 0` and solve the reduced
system in the remaining coordinates. Include the origin.

### 6.3 Classification

Compute eigenvalues of the **full** `K x K` Jacobian at each fixed point (not the reduced 2D
one — the slice can hide instability in a held direction; report both, and label any point that
is stable within the slice but unstable in a held direction).

- all `Re(lambda) < 0` → stable
- all `Re(lambda) > 0` → unstable
- mixed → saddle / semi-stable

Emit a fixed-point table: coordinates, `n` at the point, `wbar`, full and reduced eigenvalues,
classification, and whether inside the support mask.

### 6.4 Two invariance properties to assert in tests

- **Nullcline location is independent of `tau`.** The nullcline is `{X : pi_k(X) = pi_bar[k]}`,
  in which `tau[k]` cancels. Test: perturb `tau` to random positive values and assert fixed
  point coordinates are unchanged to `1e-8`. Only arrow magnitudes should change.
- **Stability is *not* automatically `tau`-invariant.** The Jacobian is premultiplied by
  `diag(1/tau)`, so sign preservation requires D-stability. Do not assert invariance of
  eigenvalue signs; instead, *report* whether classification changes across the `tau` sweep in
  §8, as a numerical result.

---

## 7. Pairwise slices and plotting

A pairwise portrait is a **slice**, not a projection. For archetype pair `(k, l)`, hold the
other `K - 2` coordinates at reference values `X_m^ref` and sweep `(X_k, X_l)` on a grid.

- Reference levels: run all three of `q25`, `q50`, `q75` of the observed `X[:, m]`. The `q50`
  slice is the headline; the other two are the topology-robustness check.
- Grid uniform in `log2 X` to match OSDR's axes, with lower bound at the 1st percentile of
  nonzero observed values.
- Plot velocity components in log space: `d(log2 X_k)/dt = (pi_k - pi_bar[k]) / (tau[k] * ln 2)`.
  These are per-capita rates and are better conditioned for the quiver than raw `F_k`. Fixed
  point locations are unchanged by the transform.
- Overlay: nullclines (`pi_k = pi_bar[k]` contours, one per swept archetype, distinct colours),
  fixed point markers coded by classification, support mask, and a faint scatter or 2D density
  of the observed cells.

Priority pairs: `(CAF, Myeloid)` first — this is the direct OSDR replication target. Then
`(Cancer Epithelial, CAF)`, `(Cancer Epithelial, Myeloid)`, `(Cancer Epithelial, T-cell)`.
Generate all `K choose 2` into a contact sheet as a secondary output.

---

## 8. Uncertainty

**Do not bootstrap individual cells.** With 80 µm neighbourhoods and a spatially smoothed
vascular field the effective sample size is orders of magnitude below `N`; a cell bootstrap
will produce absurdly tight intervals.

Two levels, both required:

### 8.1 Parametric bootstrap (default, cheap)

Draw `theta^(b) ~ MVN(theta_hat, V_hat)` for `B = 500`, using the Conley HAC-corrected
covariance matrix already produced by the Stage B fit. Recompute the full portrait per draw.
Cheap because no refitting is involved.

### 8.2 Spatial block bootstrap (confirmatory, expensive)

Resample contiguous tissue blocks with replacement (`block_id`), refit the mixture model,
recompute the portrait. `B = 50` on HPC is sufficient. This is the analogue of OSDR
Supplementary Fig. S3b–d.

### 8.3 Reported quantities

- Scatter of fixed-point locations across replicates, per fixed point (matched by nearest
  neighbour to the point estimate).
- **Topology-preservation fraction**: proportion of replicates yielding the same count and
  classification of fixed points inside the support mask. This is the headline robustness
  number.
- Per-grid-cell circular standard deviation of arrow direction; shade the quiver by it.
- Grey bootstrap arrows overlaid on the point-estimate quiver at a coarser grid, following
  OSDR S3D.

---

## 9. Sensitivity grid

Run as a factorial sweep, one portrait per cell, with topology-preservation as the summary
statistic. Sensitivity is a core output, not a supplementary check.

| Axis | Values |
|---|---|
| Death anchor multiplier on `pi_bar` | 0.7, 1.0, 1.3 (OSDR S3G used doubling/halving-time-equivalent perturbations) |
| Calibration `alpha` | point estimate ± 1 SD |
| Calibration `beta` | point estimate ± 1 SD |
| Proportional-rates offset `delta` | point estimate ± 1 SD |
| Neighbourhood radius | 60, 80, 100 µm (requires refit) |
| Vascular length scale `L0` | 100, 150, 200 µm (requires refit) |
| Oxygen reference | q25, q50, q75 |
| `K` (archetypes) | 5, 6, 7 |
| `own_state_mode` | vertex, loading_weighted |
| `tau` | uniform 1.0; per-archetype draws from a plausible S/G2M residence range |

Output a single summary table: axis, value, number of fixed points, classification string,
location of the principal stable point, topology matches baseline (bool).

---

## 10. Outputs

```
outputs/portrait/
  manifest.json                  # config, seeds, R2_recon, masked_fraction, package versions
  Atilde.csv                     # (K, K) interaction matrix
  Atilde_heatmap.pdf
  grid_velocities_<k>_<l>_<ref>.parquet   # X_k, X_l, held coords, n, pi_k, pi_l, F_k, F_l, in_mask
  fixed_points.csv
  portrait_<k>_<l>_<ref>.pdf
  portrait_contact_sheet.pdf
  bootstrap_fixed_points.parquet
  sensitivity_summary.csv
  run.log
```

`grid_velocities_*.parquet` is the analogue of OSDR's
`change_in_FM_across_all_neighborhoods.csv` and should be treated as a publishable
supplementary table.

---

## 11. Validation tests (all blocking)

1. **Coordinate round trip.** `covariates_from_state(state_from_cells(n, W), A)` recovers
   `phibar_recon` and `n` to `1e-10`.
2. **Analytic Jacobian** vs finite differences, 20 random states, `1e-6` relative (§4).
3. **Face invariance.** `F_k(X) == 0` exactly whenever `X_k == 0`, for random `X`.
4. **Death anchoring.** At the loading-weighted mean state of archetype `k`, `|F_k|` is small
   relative to the scale of `F_k` over the grid. This should hold by construction (mean value
   theorem); if it does not, `pi_bar` has been computed against the wrong weighting.
5. **`tau` invariance of nullclines** (§6.4).
6. **Known-system recovery.** Construct a synthetic `B`, `c_nbr` and density spline encoding a
   two-archetype system with a prescribed topology (one stable interior point; a stable
   single-type point; a saddle). Assert the module recovers the prescribed fixed points and
   classifications. Cover at least the four two-type topologies enumerated in OSDR Fig. 2d.

---

## 12. Documented assumptions (write into the module docstring and the manifest)

These must appear in the run log so they propagate into the thesis text:

1. **Mean-field / homogeneous-neighbourhood closure.** The portrait evaluates
   `E[dX/dt | all cells share neighbourhood N(x)]`. Spatial correlation structure is not
   represented. Testing this closure requires a spatial simulation and is out of scope.
2. **Existence of an interior fixed point is partly assumed.** Anchoring `h_minus[k] = pi_bar[k]`
   forces each nullcline through the observed data cloud. The informative content is the
   location, multiplicity and stability of fixed points, driven by `Atilde` and the density
   spline — not the existence of equilibrium.
3. **Archetype identity is fixed.** Cells cannot move in PhiSpace without dividing or dying.
   OSDR restricts to non-transdifferentiating types; with a continuous state this is a sharper
   limitation and corresponds to the unmodelled velocity term `u(x)` in their Fokker–Planck
   analysis.
4. **The archetype residual is discarded.** Only the component of `phibar` in the span of `A`
   participates in the dynamics; report `R2_recon`.
5. **Oxygen and endothelial density are coupled in tissue but decoupled here.** The vascular
   field is frozen at a reference value and swept as a sensitivity axis.
6. **Technical covariates are held at reference values** and are not on the causal path of the
   dynamics.
