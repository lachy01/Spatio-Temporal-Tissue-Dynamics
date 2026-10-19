# Phase portrait — preliminary findings

> **Read §12 first (2026-10-20).** The death anchor `π̄_k` is now each type's own mean
> cycling probability (OSDR's definition), not spec §3.2's neighbourhood-loading average.
> The anchor sets where every type grows or shrinks, so the numbers in §3–§6, §10 and §11
> are superseded: §12 gives the current ones. §1, §1a, §2 and §7 do not involve the anchor
> and stand. The output files in this directory hold the corrected runs; the spec-anchor
> runs are in git at `80dac2c` and can be regenerated with `--anchor nbhd_loading`.

**Status: preliminary, and deliberately UNBOUNDED.** OSDR's `p_correction = c_k · d_k^n`
is not applied. Spec §5 says to check confinement first and add the correction only as a
flagged sensitivity variant; the check (§2 below) fails for one of nine archetypes. The
correction is run as that variant in §10: it bounds every type, but where it puts the
tumour's ceiling is set by the correction's power, not by the fit.

Rewritten 2026-10-13 for the production model: MisTIC technical covariates in place of
cell area, log1p PhiSpace states, the contact/regional neighbourhood split, λ = 300,
rank 3, θ = 1, no oxygen term. The 2026-10-03 version described a SpaNorm-era fit
(678,885 cells, rank 2, oxygen in the weight); it is in git history at `c833585`, and
almost none of its readings survive (§3–§5).

Every number below comes from three scripts run on `outputs/hpc/point_estimate.json`:
`run/portrait_preliminary.py` (portraits, fixed points, confinement, `portrait_manifest.json`),
`run/portrait_bifurcation.py` (§4, `composition_bifurcation.txt`) and
`run/portrait_dynamics_summary.py` (§3, §5, §7, `dynamics_summary.txt`). No refitting: the
portrait is the ODE's right-hand side evaluated on a grid.

---

## 1. What the portrait is

The fitted mixture gives P(cycling) for a cell of state `φ` in a neighbourhood of
composition `φ̄` at density `n`. Spec §2–3 turn that into a deterministic field over
archetype densities `X ∈ R^9`, `X_k` = density of archetype *k*:

```
F_k(X) = X_k · (π_k(X) − π̄_k) / τ_k ,     π_k(X) = expit(η(A_k, φ̄(X), n(X)))
φ̄(X) = (X/n)·A ,   n(X) = Σ_j X_j ,   τ_k = 1
```

`π̄_k` is the death anchor, spec §3.2's loading-weighted mean of the fitted `π̂`. It is
**not estimated from death data**: it assumes division balances death on average.
Everything below is conditional on it.

| archetype | π̄ | archetype | π̄ |
|---|---|---|---|
| Endothelial | 0.057 | Myeloid | 0.079 |
| CAFs | 0.066 | Normal Epithelial | 0.105 |
| PVL | 0.078 | Plasmablasts | 0.121 |
| B-cells | 0.074 | Cancer Epithelial | 0.159 |
| T-cells | 0.088 | | |

- 561,897 cells, 9 `celltype_major` archetypes. The weight model has no oxygen term.
- Archetype reconstruction **R² = 1.000000**, exact by construction (9 archetypes in an
  8-dimensional space plus sum-to-one is a square system).
- `Ã = A B Aᵀ` has singular values **[4.08, 3.84, 2.46, 0, …]**: rank 3, inherited from
  `rank(B) = 3`.

**The state is a signed pseudo-density.** The exact reconstruction comes from a square
linear system, not a non-negative fit: **19.4%** of loading entries fall outside [0, 1],
and only **15.8%** of cells have all nine `X_k > 0`. "Density of archetype k" is literal
only where the loadings are positive; the 9-dimensional state is not a simplex. §1a
explains why; §11 rebuilds the portrait from cell-type counts (literal densities) and finds
the same results.

---

## 1a. How the state and the interaction matrix are built

Written 2026-10-20 in answer to three questions: what `A` is, what a pseudo-density is,
and how the contact and regional neighbourhoods enter the portrait. Every number is
recomputed from `outputs/hpc/point_estimate.json`
(`src/tissue_dynamics/portrait/core.py`, `mixture/basis.py`).

### What `A` is

`A` is the **9 × 8 archetype matrix**. Row `k` is the centroid of archetype *k* in the
model's own-state coordinates:

```
A_k = mean of φ_i over cells i typed as archetype k
φ_i = (s_i − m) Q_own      (s_i: the cell's nine PhiSpace scores; m: their section mean)
```

`Q_own` projects the nine compositional scores onto the 8-dimensional sum-to-zero subspace
and whitens them. These are the coordinates the weight model's `a`, `c` and `B` were
fitted in. Archetypes are the nine `celltype_major` types, and a cell's type is the argmax
of its column-standardised scores.

`A` does three jobs:
1. **It turns a composition into a state.** A neighbourhood made of a fraction `w_j` of
   each archetype has mean state `φ̄ = w A` (with `Σ_j w_j = 1`). The portrait's state
   `X = n · w` therefore maps to the model's covariates through `A`
   (`covariates_from_state`).
2. **It supplies each archetype's own state.** The growth rate of archetype *k* is the
   cycling probability of a cell sitting exactly at its centroid, `π_k(X) = π(φ = A_k, φ̄(X), n(X))`.
3. **It translates the interaction into archetype terms.** `Ã = A B̃ Aᵀ` re-expresses
   the fitted state-by-neighbourhood interaction `B̃` (8 × 8, below) as a 9 × 9 matrix of
   archetype pairs. That is the generalised Lotka–Volterra form of spec §4.

### What a pseudo-density is, and why loadings leave [0, 1]

Each cell's **loadings** `w_i` solve `φ̄_i = w_i A` with `Σ_k w_ik = 1`. Here `φ̄_i` is the
mean PhiSpace state of the cell's 80 µm disc, in the same coordinates. Nine unknowns,
eight coordinates plus the sum-to-one constraint: the system is square, so the solution
is unique and exact (reconstruction R² = 1). These are **barycentric coordinates** of the
neighbourhood relative to the nine centroids. The state is `X_i = n_i · w_i`, with `n_i`
the local density (cells per (100 µm)² in the disc). So `X_ik` is a **pseudo-density**:
"archetype-k cells per unit area" when the neighbourhood lies inside the simplex the
centroids span, and a signed coordinate when it does not.

Loadings leave [0, 1] whenever a neighbourhood's mean state lies **outside the simplex of
the nine centroids**. Here that is almost always on the negative side: 19.4% of entries
are below 0, and fewer than 0.1% are above 1. It happens because the centroids are not
pure types:
- Each centroid averages every cell of its type across the whole section. That includes
  cells in mixed regions whose PhiSpace scores carry other types' signal, from
  segmentation spillover and from the scores themselves being mixtures. A typical stromal
  centroid, for example, carries a little Cancer Epithelial signal.
- A neighbourhood cleaner than that average sits beyond the simplex. A stromal
  neighbourhood far from tumour has less Cancer Epithelial signal than the stromal
  centroids, so it needs a negative Cancer Epithelial weight to reproduce it. Cancer
  Epithelial has the most negative loadings, 33.5% of cells.
- Individual cells scatter around their centroid much more than neighbourhood means do.
  Loading each cell's own state on the same centroids puts 53% of entries outside
  [0, 1], and the median own-type loading is 0.83–0.95, not 1. Averaging over the disc
  removes most of that scatter, but not all.

| archetype | loadings < 0 | 1st–99th percentile | min | max |
|---|---|---|---|---|
| Endothelial | 17.2% | −0.06 to 0.48 | −0.84 | 2.30 |
| CAFs | 15.2% | −0.03 to 0.60 | −0.55 | 2.52 |
| PVL | 15.1% | −0.10 to 0.37 | −0.97 | 2.48 |
| B-cells | 25.7% | −0.11 to 0.63 | −0.84 | 0.94 |
| T-cells | 22.7% | −0.10 to 0.68 | −0.62 | 1.84 |
| Myeloid | 10.1% | −0.06 to 0.45 | −0.38 | 2.47 |
| Normal Epithelial | 18.1% | −0.15 to 0.50 | −2.55 | 4.70 |
| Plasmablasts | 16.6% | −0.12 to 0.37 | −0.56 | 4.28 |
| Cancer Epithelial | 33.5% | −0.19 to 0.86 | −2.41 | 1.49 |

Most negative loadings are small (1st percentile −0.03 to −0.19). The death anchors `π̄_k`
use the signed loadings, as spec §3.2 states. Clipping them at 0 moves no anchor by more
than 0.017.

### How the contact and regional neighbourhoods are combined

**In the fitted model the two blocks are separate.** The weight's neighbourhood terms are

```
η ⊃ φ̄_cᵀ c_c + φ̄_rᵀ c_r + φᵀ U (V_cᵀ φ̄_c + V_rᵀ φ̄_r)
```

`φ̄_c` (contact, 0–15 µm) and `φ̄_r` (regional, 15–80 µm) are each centred, projected and
whitened on their own. The interaction is one rank-3 matrix over the stacked
neighbourhood, `B = U Vᵀ` (8 × 16), with `V = [V_c; V_r]`. So
`B_c = U V_cᵀ` and `B_r = U V_rᵀ` **share the own-state factor `U`**. The low-rank
constraint is on the joint matrix, not on each block separately.

**In the portrait they share one composition.** The portrait has one neighbourhood
composition per state, `φ̄(X) = w A`, the 80 µm disc's. It uses a *mean-field closure*:
the contact and regional blocks are both given that composition. `basis.nbr_from_own()`
gives each block's design coordinates as an exact affine function of own-state
coordinates, `φ̄_b = o_b + φ̄ M_b`. Substituting folds the two blocks into one own-state
model (`PortraitModel.from_fit`):

```
c̃ = M_c c_c + M_r c_r
B̃ = B_c M_cᵀ + B_r M_rᵀ = U (M_c V_c + M_r V_r)ᵀ        (8 × 8)
the offsets o_b are absorbed into a and a0
```

**Why `Ã` has rank 3.** Your intuition is right that two separate rank-3 matrices can
sum to rank 6. That would happen here if the blocks had their own own-state factors:
replacing `U` in the regional term with a different 8 × 3 matrix gives rank 6. But
because both blocks share `U`, every column of `B̃` lies in the 3-dimensional span of
`U`. So `B̃` has rank ≤ 3, and so does `Ã = A B̃ Aᵀ`.

| matrix | leading singular values (rest 0) |
|---|---|
| `B_c` (contact half of `B`) | 0.372, 0.133, 0.061 |
| `B_r` (regional half) | 0.177, 0.152, 0.092 |
| contact term `B_c M_cᵀ` | 0.441, 0.202, 0.106 |
| regional term `B_r M_rᵀ` | 0.577, 0.430, 0.282 |
| `B̃` (their sum) | 0.801, 0.692, 0.434 |
| `Ã = A B̃ Aᵀ` | 4.08, 3.84, 2.46 |

Mapped into own-state units, the regional term is the larger of the two (Frobenius norm
0.77 against 0.50).

**What the closure costs.** The portrait cannot represent a cell whose contacts differ
from its region, such as a stromal cell touching tumour inside a stromal region. Every
portrait statement is about shifting the whole 0–80 µm neighbourhood together.
Boundary-specific effects are in the fit but not in the portrait.

---

## 2. Confinement: every archetype but CAFs self-confines

Spec §5's check, `dη/d(log n)` at the 90th percentile of density:

| archetype | slope | | archetype | slope |
|---|---|---|---|---|
| Endothelial | −0.064 | | Myeloid | −0.025 |
| **CAFs** | **+0.831 outward** | | Normal Epithelial | −0.438 |
| PVL | −0.349 | | Plasmablasts | −0.407 |
| B-cells | −0.446 | | Cancer Epithelial | −0.233 |
| T-cells | −0.635 | | | |

Eight of nine archetypes are confining at high density. In the October fit it was the two
epithelial/secretory archetypes, Cancer Epithelial (+1.20) and Plasmablasts (+0.91), that
escaped; both now confine. CAFs are the exception now. Endothelial and Myeloid confine
only weakly, so the OSDR bound would still change the high-density branch for three types.

---

## 3. Density thresholds at each archetype's own neighbourhood

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

Total density swept at each archetype's own mean neighbourhood composition, with each
crossing of `π_k = π̄_k`:

| archetype | crossings | structure |
|---|---|---|
| Endothelial, CAFs, PVL, Myeloid | none | below the anchor at every density: shrinks |
| B-cells | n = 205 (p98) stable | grows up to a ceiling at the edge of the data |
| T-cells | n = 200 (p97) stable | the same |
| Plasmablasts | n = 201 (p97) stable | the same |
| Cancer Epithelial | n = 203 (p97) stable | the same |
| Normal Epithelial | n = 68 (p31) unstable; n = 76 (p42) stable | a narrow Allee window |

The October fit had a textbook Allee structure (extinction below n ≈ 80, a ceiling near
n ≈ 150) in five archetypes, and an escape threshold with no ceiling in Cancer Epithelial
and Plasmablasts. Neither survives. Now the cycling-rich types (lymphocytes, plasmablasts,
tumour) grow at every observed density below about the 97th percentile, where a ceiling
sits; the stromal and myeloid types sit below their anchors throughout. Only Normal
Epithelial keeps an Allee window, between the 31st and 42nd percentiles.

"Shrinks at every density" is relative to `π̄`: the type's cycling at its own mean
composition is below its loading-weighted average. It is a statement about the anchor as
much as about the fit.

---

## 4. The tumour's ceiling holds at every composition, at the edge of the data

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

Blending the neighbourhood from the q50 reference (tumour fraction w̄_CE = 0.139) to a
purely tumour neighbourhood, and sweeping total density at each blend:

- one crossing at every blend, always a **stable ceiling**; no escape threshold and no fold
  anywhere on the path;
- the ceiling sits at **n = 207–214** cells per (100 µm)², the **97.6th–98.2nd percentile**
  of observed density;
- 45.3% of observed neighbourhoods lie on the blended range; the other 54.7% hold less
  tumour than the reference, below where the sweep starts.

The October fit's headline, a ceiling only inside a window of tumour fraction 0.27–0.49
with saddle-node bifurcations at both ends, is gone. Tumour growth is bounded in every
tumour-rich neighbourhood, but only at the top 2–2.5% of density. Below that the tumour's
cycling stays above its anchor. Because the ceiling sits where data are sparse, it is the
result most exposed to the missing bound and to the absent uncertainty (§9).

`composition_bifurcation.png` draws the curve; the crossings per blend are in
`composition_bifurcation.csv`.

---

## 5. Nullclines in the priority slices: mostly outside the window

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

At the q50 reference, almost no nullcline crosses the plotted window:

| slice | first archetype | second archetype |
|---|---|---|
| CAFs × Myeloid | shrinks throughout | shrinks throughout |
| Cancer Epithelial × CAFs | grows throughout | shrinks throughout |
| Cancer Epithelial × Myeloid | grows throughout | nullcline at mean n 70.9 (cv 0.21) |
| Cancer Epithelial × T-cells | grows throughout | grows throughout |

The October reading, that nullclines were near-degenerate total-density contours (cv
0.06–0.20, the CAF × Myeloid angle 1.3°), no longer holds. The angle between the two
nullcline gradients at the 80th percentile is now 26° (CAFs × Myeloid), 62° (CE × CAFs),
71° (CE × Myeloid) and 54° (CE × T-cells). The CAF/Myeloid bistability OSDR reports is still
not recovered, now because both archetypes shrink throughout that slice, not because the
nullclines coincide.

---

## 6. Fixed points

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

Across the 12 masked panels (4 priority pairs × q25/q50/q75), **no fixed point lies inside
the support mask**. The interior points found are stable within their slice but unstable in
the held directions, and all sit far beyond support: Cancer Epithelial × CAFs at q25, q50 and
q75, and Cancer Epithelial × Myeloid at q50 and q75, each at n ≈ 202–221. Boundary points lie
on the Cancer Epithelial face (X_CE = 112–198). Full table in `fixed_points.csv`.

**Support.** The masked fraction is 0.67–0.96 at q25, 0.73–0.81 at q50 and **1.00 at q75**,
where only 123–177 cells survive the held-coordinate band. So spec §7's robustness check
across reference levels cannot run at q75; q25 and q50 agree (no supported fixed point).

Two deviations from spec §5 stand: the `min_cells = 20` rule is counted on a coarse 30×30
support grid rather than the 150×150 plotting grid, and conditioning on seven held
coordinates at once is what empties q75, a limit of slice-based support, not of the model.

---

## 7. The interaction matrix `Ã = A B Aᵀ`

Rank 3 (why: §1a), mostly reciprocal (symmetric / antisymmetric Frobenius norms 5.93 /
1.50). Every self-term of `Ã` is negative, but `Ã` leaves out the neighbourhood main
effect; with it, the reading changes for Cancer Epithelial and T-cells (caution below). Largest entries, in log-odds of cycling per unit neighbourhood fraction (row ←
column):

| effect | value |
|---|---|
| PVL ← PVL | −2.12 |
| Endothelial ← PVL | −1.54 |
| Plasmablasts ← Plasmablasts | −1.49 |
| Myeloid ← Myeloid | −1.44 |
| Myeloid ← Plasmablasts | +1.31 |
| Plasmablasts ← Myeloid | +1.27 |
| Endothelial ← Endothelial | −1.17 |
| Cancer Epithelial ← Normal Epithelial | −1.17 |
| Cancer Epithelial ← Cancer Epithelial | −1.08 |

**Cancer Epithelial and T-cells are still mutually stimulating**, more strongly than in
October: +0.66 (Cancer Epithelial ← T-cells) and +0.71 (T-cells ← Cancer Epithelial),
against +0.45 and +0.48. Myeloid and Plasmablasts form a second reciprocal pair (+1.31,
+1.27). One October entry reversed: Plasmablasts ← Endothelial is now +0.90, against −0.70.

**Caution (added 2026-10-20): `Ã` is only the interaction part of the neighbourhood
effect.** The full effect on archetype *k*'s log-odds of a unit of neighbourhood fraction
of archetype *j* is

```
∂η_k / ∂w_j = Ã_kj + c̃ · A_j
```

The main-effect term `c̃ · A_j` is the same for every row. It is as large as `Ã`
(Frobenius norm 7.2 against 6.1): Cancer Epithelial +1.00, Plasmablasts +0.70,
T-cells +0.46, Endothelial −1.78, CAFs −0.72. And because the loadings sum to one,
only differences within a row are meaningful: raising one archetype's fraction lowers
the others'. Read that way (each row of `Ã + c̃ · A_j` centred), most self-terms are
still at or near the bottom of their row. Endothelial, CAFs, PVL, B-cells, Myeloid and
Normal Epithelial rank first or second most negative. Two do not: Cancer Epithelial's
self-effect is about 0 (−0.02, fifth of nine) and T-cells' is +0.07 (fifth). "Crowding
by the same archetype lowers its cycling odds" therefore holds for stroma and most
immune types, but not for Cancer Epithelial or T-cells. Their neighbourhood response is
dominated by the main effect: a Cancer Epithelial and T-cell neighbourhood raises every
archetype's cycling odds.

The portrait's dynamics (§2–§6) use the full model, `c̃` and `B̃` together, so only this
reading of `Ã` changes. The row-centred full matrix:

| own ↓ / neighbour → | Endo | CAFs | PVL | B | T | Myel | NormEpi | Plasma | CancerEpi |
|---|---|---|---|---|---|---|---|---|---|
| Endothelial | −2.68 | −0.65 | −1.28 | 0.37 | 0.69 | −0.18 | 0.53 | 1.36 | 1.85 |
| CAFs | −1.26 | −0.95 | −0.31 | −0.15 | 0.52 | 0.39 | 0.50 | −0.12 | 1.39 |
| PVL | −2.33 | −0.93 | −1.80 | 0.74 | 0.78 | 0.61 | 0.61 | 0.48 | 1.84 |
| B-cells | −1.47 | −0.74 | 0.74 | −0.96 | 0.10 | −0.90 | 0.81 | 0.77 | 1.66 |
| T-cells | −1.63 | −0.81 | 0.42 | −0.80 | 0.07 | −0.86 | 0.99 | 0.74 | 1.87 |
| Myeloid | −2.22 | −0.34 | 0.93 | −1.11 | 0.16 | −1.73 | 0.56 | 2.09 | 1.65 |
| Normal Epithelial | −1.37 | −0.20 | 0.51 | −0.39 | 1.11 | 0.17 | −0.99 | 1.03 | 0.13 |
| Plasmablasts | −0.70 | −0.87 | −0.12 | −0.08 | 0.85 | 1.06 | −0.18 | −0.62 | 0.66 |
| Cancer Epithelial | −1.30 | −0.17 | 0.57 | −0.39 | 1.18 | 0.25 | −1.14 | 1.01 | −0.02 |

---

## 8. Validation

All 13 spec §11 tests pass (`tests/test_portrait.py`, rerun 2026-10-13). They test the code
on synthetic systems, so the model change does not touch them:

- **§11.2 analytic Jacobian** passes at 2.7e-08, against a probe taking the best of
  finite-difference steps 1e-4/1e-5/1e-6.
- **§11.6 known-system recovery** recovers all five planar topologies (stable node, stable
  spiral, saddle, unstable node, unstable spiral). With **equal** death anchors an unstable
  interior point is impossible in this model class (`det > 0` needs `μ_a < μ_b`, while
  `trace > 0` needs `γ_a μ_a > γ_b μ_b`, `γ_k = π̄_k(1−π̄_k)`), confirmed by a sweep of 20,480
  parameter sets.
- **§11.4 is implemented in a corrected form.** As written it asks for `|F_k|` to be small
  at archetype *k*'s loading-weighted mean state, which Jensen's inequality breaks; the test
  asserts instead that `π̄_k` lies strictly inside the observed range of `π_k`, so each
  nullcline passes through the data cloud.

Until 2026-10-20 `π̄` used spec §3.2's signed-loading formula; clipping the negative
loadings moved every anchor by at most 0.010. The anchor is now each type's own mean
cycling (§12), which uses no loadings, so the clipped variant no longer applies.

---

## 9. Not yet done

- Spec §8 and §9: run 2026-10-20 (§12). Not done: spec §8.3's arrow-direction SD and overlaid
  bootstrap arrows; the calibration α/β, oxygen, `L0` and `own_state_mode` axes do not apply.
- ~~The OSDR maximal-density bound~~: run as a sensitivity variant (§10, §12). Its fixed
  points and nullclines are not reported as findings.
- Spec §7's q25/q50/q75 robustness check is only partly possible (§6): q75's mask is still
  empty in every priority slice.

## 10. Sensitivity: OSDR's maximal-density correction

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

**A sensitivity variant only, never the default (spec §5).** `portrait.core.fit_osdr_bound`
transcribes OSDR's `CellTypeSpecificDensityEnforcer` (`tdm.model.maximal_density_enforcer`,
commit 7d04970) with the settings of OSDR's published analyses: power p = 8 and the division
rate truncated. Each script takes `--bound osdr [--power p] [--target-at-max]` and writes to
`outputs/portrait/osdr_p<p>[_target_at_max]/`, with every figure flagged.

Each archetype's net rate becomes

```
min(π_k(X), cap_k) − π̄_k + max(c_k · X_k^p, −0.5),     c_k = −max(0, r_k) / x_max,k^p
```

in its own density `X_k`. Here `x_max,k` is the 95th percentile of `X_k`, `cap_k` the largest
`π_k` over the observed neighbourhoods of type-*k* cells, and `r_k` the largest capped net rate
over 10,000 draws of the state, every density uniform on log2(1 + x) up to its `x_max`. OSDR's
code draws the target type too; its docstring says the target is held at its maximum. The
docstring version (`--target-at-max`) gives smaller corrections and the same conclusions.

| archetype | x_max (q95) | π cap | r_k, OSDR's code | r_k, target at max |
|---|---|---|---|---|
| Endothelial | 21.1 | 0.298 | 0.119 | 0.033 |
| CAFs | 23.2 | 0.090 | 0.004 | 0 |
| PVL | 19.0 | 0.310 | 0.135 | 0.041 |
| B-cells | 61.4 | 0.491 | 0.303 | 0.086 |
| T-cells | 88.0 | 0.571 | 0.324 | 0.154 |
| Myeloid | 27.2 | 0.376 | 0.112 | 0.036 |
| Normal Epithelial | 25.4 | 0.281 | 0.143 | 0.081 |
| Plasmablasts | 21.4 | 0.242 | 0.118 | 0.087 |
| Cancer Epithelial | 55.9 | 0.448 | 0.285 | 0.172 |

Every archetype grows somewhere in the observed range, so every one is corrected; `r_k` is
also the correction at `x_max,k`. CAFs barely: they shrink almost everywhere, so the one type
that fails §2's confinement check is the one the bound hardly touches.

**What the bound changes, at p = 8:**

- **The tumour's ceiling moves into the data.** At its own composition it falls from n = 203
  (97th percentile) to n = 85 (52nd). On the composition path (§4) it runs from n = 213 at the
  reference blend to n = 46 in a pure tumour neighbourhood, against 214 and 207 unbounded.
- **Fixed points appear inside support.** Cancer Epithelial × Myeloid has a point stable within
  its slice at q25 (n = 73) and q50 (n = 100). Both are unstable in the held directions (largest
  real eigenvalue 0.20 and 0.14), so the full system does not settle there. The tumour-face
  equilibria move from X_CE = 112–198 to about 45.
- **The tumour's nullcline becomes a wall.** In every tumour slice it runs near-vertically at
  X_CE ≈ 44–45, whatever the other type's density (`osdr_p8/portrait_q50.png`).
- **CAF × Myeloid is unchanged:** both shrink throughout the slice, so OSDR's fibroblast–
  macrophage bistability is not recovered with OSDR's own correction either.

**Those changes are the correction's, not the fit's.** Every tumour equilibrium sits at
X_CE = 44–49, 0.8–0.9 of `x_max` (55.9). Where the correction dominates, the equilibrium solves
`rate = r_k (X_k / x_max)^p`, so its position is set by p. The ceilings at each archetype's own
composition (`osdr_p<p>/dynamics_summary.txt`; n, with its density percentile):

| archetype | p = 2 | p = 4 | p = 8 (OSDR) | p = 16 | unbounded |
|---|---|---|---|---|---|
| Cancer Epithelial | 46 (11th) | 70 (33rd) | 85 (52nd) | 95 (61st) | 203 (97th) |
| Plasmablasts | 47 (12th) | 81 (47th) | 125 (78th) | 140 (84th) | 201 (97th) |
| B-cells | 60 (21st) | 115 (74th) | 156 (89th) | 183 (95th) | 205 (98th) |
| T-cells | 68 (31st) | 133 (82nd) | 171 (92nd) | 192 (96th) | 200 (97th) |

The pure-tumour ceiling moves the same way: n = 25, 37, 46 and 50 for p = 2, 4, 8 and 16.
Endothelial, CAFs, PVL and Myeloid shrink at every density under every power, as unbounded.

**Reading.** OSDR's correction does what it is built to do: no type grows at its maximal
density. But it gives the tumour a ceiling anywhere from the 11th to the 61st percentile of
density depending on a power OSDR chose by hand, and the in-support fixed points sit on the
wall that power builds. They are a property of the bound, not an estimate, and are not
reported as findings. The fit's own density response confines eight of nine archetypes
(§2) and puts the tumour's ceiling at the 97th–98th percentile (§3, §4); that, with its
caveat about sparse data, is the result.

## 11. Variant: the state from cell-type counts (no negative densities)

*Computed with spec §3.2's neighbourhood-loading anchor; superseded by §12.*

**A sensitivity variant (user, 2026-10-20).** Asked whether one-hot archetypes would remove
the signed pseudo-densities of §1a. PhiSpace's pure vertices (+1 for the type, −1
elsewhere) would: only 0.2% of loadings fall outside [0, 1]. But each archetype's growth rate
would then be evaluated 5–13 SD from the section mean, where almost no cell lies. The most
tumour-rich observed neighbourhood is only 42% of the way to the pure Cancer Epithelial vertex.
So that variant was not built. This one keeps the centroids `A` as the archetype states and
changes only how a neighbourhood becomes a state:

```
w_ik = share of the analysis cells within 80 µm of cell i (itself excluded) whose type is k
X_i  = n_i · w_i                 (n_i: the model's density covariate, as before)
φ̄(X) = (X / n) A                 (unchanged)
```

`X_k` is now a literal density of type-*k* cells: never negative, and OSDR's state. What it
gives up is the neighbours' within-type variation. `w A` reproduces the disc's actual mean
state with R² 0.84 (0.53–0.93 across the eight coordinates), against 1 for the barycentric
loadings. Type fractions correlate 0.79–0.98 with the barycentric loadings. The pool is the
analysis cells; the table's neighbourhood means also include the 0.4% of cells outside the
oxygen mask. `Ã` and the confinement slopes (§2) do not depend on the state and are unchanged.
Run with `--state counts` on all three portrait scripts; outputs are in
`outputs/portrait/counts/` (and `counts/osdr_p8/`).

| result | barycentric (§1–§6, §10) | type counts |
|---|---|---|
| loadings outside [0, 1]; cells with every `X_k > 0` | 19.4%; 15.8% | 0%; 75.6% |
| death anchors `π̄_k` | 0.057–0.159 | 0.071–0.144 (each within 0.015) |
| confinement (§2) | all but CAFs confine | identical |
| ceilings at own composition (§3) | B-cells 205, T-cells 200, Plasmablasts 201, Cancer Epithelial 203 (p97–98) | 189, 200, 207, 217 (p96–98) |
| Normal Epithelial (§3) | Allee window, n = 68 unstable / 76 stable (p31–42) | no crossing: shrinks at every density |
| tumour ceiling on the composition path (§4) | n = 207–214 (p97.6–98.2), one stable crossing per blend, no folds | n = 220–226 (p98.5–98.7), the same structure |
| nullclines at q50 (§5) | only Myeloid's crosses (CE × Myeloid, mean n 70.9, cv 0.21) | the same one (mean n 79.5, cv 0.13) |
| fixed points (§6) | none inside support; interior points in CE × CAFs and CE × Myeloid, stable in slice and unstable in the held directions, n ≈ 202–221 | the same pattern, n ≈ 216–232 |
| OSDR p = 8 (§10) | tumour wall at X_CE ≈ 44–45; in-support CE × Myeloid points at n = 73 and 100; pure-tumour ceiling n = 46 | wall at X_CE ≈ 39–40; points at n = 73 and 93; ceiling n = 40 |

**Reading.** Every qualitative result of §2–§6 and §10 holds with literal densities. Eight
archetypes confine; the lymphocyte, plasmablast and tumour ceilings sit at the edge of the
data; there is no supported fixed point unbounded; and OSDR's bound builds the same
power-set wall. The ceilings move by 0–14 cells. The one change is Normal Epithelial's Allee
window, which disappears. It was the narrowest structure in §3 (between the 31st and 42nd
percentiles of density), so it should not be reported as a finding. The signed
pseudo-densities were cosmetic for these conclusions.

One visual difference: at low density the count-based state sits on a lattice
(`portrait_q50.png`), because type counts are integers.

## 12. The death anchor corrected: each type's own mean cycling (2026-10-20)

### What changed and why

Each archetype's net rate is `π_k(X) − π̄_k`: its cycling probability in a given
neighbourhood minus a constant death rate. OSDR's anchor, and the intent spec §3.2 states,
is a population steady state: in aggregate each type divides as often as it dies, so `π̄_k`
is the type's **own mean cycling probability**. The formula spec §3.2 writes is a different
quantity:

```
spec §3.2:   π̄_k = Σ_i W_ik π_i / Σ_i W_ik        W_ik: cell i's neighbourhood loading on k
OSDR:        π̄_k = mean of π_i over the cells i whose type is k
```

The spec version averages the cycling of every cell, of any type, by how much of archetype
*k* surrounds it. A CAF-rich neighbourhood also holds tumour and immune cells, so the CAF
anchor picked up their cycling: 0.066, three times the CAFs' own 0.022. This pulled every
anchor toward the section mean (0.10). Stromal and myeloid anchors were too high, so those
types appeared to shrink at every density (§3). Lymphocyte and tumour anchors were too low,
so those types appeared to grow almost to the edge of the data (§3, §4).

The own-type anchor is now the default (`--anchor own_type`, `portrait.core.build_archetypes`).
The spec formula remains as `--anchor nbhd_loading`. The new anchor does not use the
loadings, so it is the same in both states:

| archetype | spec §3.2 | own type |
|---|---|---|
| Endothelial | 0.057 | **0.037** |
| CAFs | 0.066 | **0.022** |
| PVL | 0.078 | **0.056** |
| B-cells | 0.074 | **0.090** |
| T-cells | 0.088 | **0.110** |
| Myeloid | 0.079 | **0.042** |
| Normal Epithelial | 0.105 | **0.092** |
| Plasmablasts | 0.121 | **0.114** |
| Cancer Epithelial | 0.159 | **0.203** |

Not affected:

- confinement (§2), a slope of `η`;
- `Ã` (§7);
- the archetypes, the states and the support masks;
- the spec §11 tests (17 pass).

### Thresholds at each archetype's own composition (replaces §3)

| archetype | unbounded | OSDR p = 8 |
|---|---|---|
| Endothelial | n = 129 (p80) **unstable**; n = 189 (p95) stable | shrinks |
| CAFs | n = 71 (p34) **unstable**, no ceiling | n = 75 (p39) unstable; 83 (p49) stable |
| PVL | shrinks | shrinks |
| B-cells | n = 132 (p81) stable | 120 (p76) |
| T-cells | n = 154 (p88) stable | 141 (p85) |
| Myeloid | n = 83 (p50) **unstable**, no ceiling | shrinks |
| Normal Epithelial | n = 185 (p95) stable | 113 (p73) |
| Plasmablasts | n = 211 (p98) stable | 130 (p80) |
| **Cancer Epithelial** | **n = 79 (p45) stable** | 65 (p27) |

Unbounded, these are identical in the two states. The sweep evaluates `π_k` at the type's
observed mean neighbourhood state, which does not depend on how a neighbourhood becomes `X`.
Under the bound the type's own density `X_k` enters. Since 2026-10-20 it takes the mean
loadings in the portrait's state (type fractions for counts); before that, both states used
the barycentric loadings.

Three structures are new:

- **The tumour's ceiling is inside the data without any bound.** It is n = 79, the 45th
  percentile of density, against the 97th under the spec anchor. This is the largest change.
- **CAFs and Myeloid have escape thresholds.** Above n = 71 (CAFs) and n = 83 (Myeloid) at
  their own composition they grow, with no ceiling in the fit's own response. For CAFs this
  is §2's confinement failure, now visible. Myeloid confine only weakly (slope −0.025).
- **Endothelial has an Allee window high in the density range**, between the 80th and 95th
  percentiles.

Before, PVL was the only stromal type that shrinks everywhere; that is still true of PVL alone.

### The tumour along the composition path (replaces §4)

Blending from the q50 reference (tumour fraction 0.139) to a pure tumour neighbourhood, and
sweeping total density at each blend (`composition_bifurcation.csv`):

| tumour fraction | lower ceiling | threshold | upper ceiling |
|---|---|---|---|
| 0.139 (reference) | n = 98 (p64) | 134 (p82) | 174 (p93) |
| 0.40 | 94 | 140 | 169 |
| 0.66 | 91 | 149 (p87) | 162 |
| ≥ 0.72 | 89 → 86 (p53) | — | — |

**Every blend has a stable tumour ceiling at the 53rd–64th percentile of density.** Where
the neighbourhood is less than about 72% tumour there is a second, high-density stable state
at n ≈ 156–174 (p89–93), separated from the first by an unstable threshold. The two
high-density branches meet in a fold (saddle-node) at tumour fraction 0.723. Beyond that,
only the lower ceiling remains. So mixed neighbourhoods are bistable in tumour density and
tumour-dominated ones are not.

The count-based state gives the same structure: lower ceiling 96 → 86, upper 171 → 157, fold
at 0.664.

### Nullclines and fixed points (replaces §5, §6)

Every nullcline in the four priority slices now crosses the q50 window. Several are close
to total-density contours (cv of n along the contour 0.004–0.12, except Myeloid against
tumour, 0.23, and the tumour against T-cells, 0.34):

| slice (q50) | first archetype's nullcline, mean n | second's |
|---|---|---|
| CAFs × Myeloid | 58.4 | 56.0 |
| Cancer Epithelial × CAFs | 92.7 | 56.3 |
| Cancer Epithelial × Myeloid | 91.4 | 47.0 |
| Cancer Epithelial × T-cells | 60.0 | 39.9 |

The 12 masked panels have **one interior fixed point inside support**. It is a **saddle in
CAFs × Myeloid at q50**: X_CAF = 14.6, X_Myeloid = 11.2, n = 66 (p29). The rest lie outside
support:

- a CE × CAFs point at q50, n = 84, stable in its slice but unstable in the held directions;
- a CE × Myeloid point at q75, n = 193, also stable in slice and unstable held;
- saddles at very low or high tumour density.

A face point sits at X_CAF = 0, X_Myeloid = 10.2 (a Myeloid-only state).

**Reading of the CAF × Myeloid slice** (`portrait_q50.png`, top left; signs checked on
`grid_velocities_CAFs_Myeloid_q50.parquet`).

- **Myeloid settles at X_Myeloid ≈ 10–12** whatever the CAF density. They grow below their
  nullcline and shrink above it, so that nullcline attracts.
- **CAFs grow only where Myeloid is plentiful.** They shrink below their nullcline and grow
  above it, so that nullcline repels.
- **Left of the saddle** (X_CAF < 14.6), the Myeloid nullcline lies below the CAFs'. Once
  Myeloid has settled, CAFs therefore shrink and the flow runs to the face point
  X_CAF = 0, X_Myeloid = 10.2: a Myeloid-only state.
- **Right of the saddle** the order flips, and CAFs grow with no ceiling.

So the saddle is a CAF threshold set by Myeloid. Below it the tissue loses its CAFs; above it
they expand. The Myeloid nullcline rises slightly with CAF density, so CAFs support Myeloid
weakly. This is half of OSDR's fibroblast–macrophage picture: one stable state (Myeloid
without CAFs) and a threshold. Where OSDR has a second stable state there is a runaway,
because CAFs fail confinement (§2).

**It is fragile.** The two nullclines are near-parallel density contours 2.4 cells apart
(angle 26° at the 80th percentile). In the count-based state they are 5.6 apart (59.2 and
64.8) and do not cross inside the window, so the counts portrait has no supported fixed point.
Whether the saddle exists is what the uncertainty runs (below) have to decide.

### Under OSDR's bound (replaces §10's results)

The bound's form and settings are as in §10. Its `r_k` depends on the anchor and so changed.
With OSDR's code, `r_k` is the largest capped net rate:

| | Endo | CAFs | PVL | B | T | Myel | NormEpi | Plasma | CancerEpi |
|---|---|---|---|---|---|---|---|---|---|
| `r_k` | 0.140 | 0.047 | 0.157 | 0.287 | 0.303 | 0.149 | 0.156 | 0.125 | 0.241 |

The own-composition ceilings by power (n, with its density percentile):

| archetype | p = 2 | p = 4 | p = 8 (OSDR) | p = 16 | unbounded |
|---|---|---|---|---|---|
| Cancer Epithelial | 21 (4th) | 44 (10th) | 65 (27th) | 77 (42nd) | 79 (45th) |
| B-cells | 28 (5th) | 80 (46th) | 120 (76th) | 132 (81st) | 132 (81st) |
| T-cells | 38 (8th) | 72 (37th) | 141 (85th) | 154 (88th) | 154 (88th) |
| Normal Epithelial | 45 (10th) | 85 (52nd) | 113 (73rd) | 135 (82nd) | 185 (95th) |
| Plasmablasts | 57 (19th) | 97 (63rd) | 130 (80th) | 142 (85th) | 211 (98th) |

The pure-tumour ceiling is n = 14, 27, 38 and 46 for p = 2, 4, 8 and 16, against 86
unbounded. The bound still places its ceilings where p puts them, as §10 found, but it now
sits on top of a fit whose own ceilings are inside the data. At p = 16 the bound barely
moves the tumour, B-cells or T-cells.

At p = 8, the in-support fixed points in the barycentric state are:

- **CE × Myeloid at q50**: n = 94, stable in its slice, unstable in the held directions
  (largest real eigenvalue 0.087).
- **CAFs × Myeloid**: the q50 saddle is gone. CAFs now have a ceiling (n = 83), and the slice
  has a stable point only at q75 (n = 164), outside support.

The count-based state at p = 8 has stable-in-slice points in support for CAFs × Myeloid
(q50, n = 73) and CE × Myeloid (q50, n = 92). Both are unstable in the held directions.

OSDR's docstring version of the bound (`--target-at-max`, smaller corrections) falls between
p = 8 and the unbounded fit:

- tumour ceiling at own composition n = 69 (p33); pure-tumour ceiling n = 41;
- CAFs: unstable at n = 73, stable at 86;
- Myeloid: unstable at 87, stable at 112;
- no fixed point inside support.

### Reading

The corrected anchor turns the portrait from "most types shrink, the rest grow to the edge
of the data" into one with structure inside the data, without any bound:

- a tumour ceiling near the median density;
- bistable tumour density in mixed neighbourhoods, lost through a fold as the neighbourhood
  becomes tumour-dominated;
- escape thresholds for CAFs and Myeloid.

None of this has an interval yet. The sensitivity grid below shows that the tumour's
structure hangs on the anchor's level within about 5%. The CAF × Myeloid saddle depends on the
state, the bound and the anchor. The Spartan uncertainty runs (spec §8) will give the
parameter intervals.

### Sensitivity (spec §9, 2026-10-20)

**The anchor's level decides the tumour's structure, and the margin is about 5%.** The
results come from `sensitivity/grid.csv` (the spec's axes) and `sensitivity/anchor_sweep.txt`
(`run/portrait_anchor_sweep.py`, multiplier 0.7–1.3, barycentric). Every anchor is scaled by
the multiplier:

| multiplier (CE anchor) | CE at own composition | CE path ceiling | CAFs | Myeloid | T-cells |
|---|---|---|---|---|---|
| 0.80 (0.163) | stable 200 (p97) | 204–212 | unstable 59 | unstable 44 | stable 201 |
| 0.90 (0.183) | stable 106, unstable 124, stable 180 | 185–195 | unstable 65 | unstable 64 | stable 180 |
| 0.95 (0.193) | stable 92, unstable 145, stable 165 | 97–185, one fold | unstable 68 | unstable 74 | stable 168 |
| **1.00 (0.203)** | **stable 79 (p45)** | **86–174, one fold** | **unstable 71** | **unstable 83** | **stable 154** |
| 1.05 (0.213) | shrinks | 55–87 | unstable 74 | unstable 93 | stable 48 |
| 1.10 (0.224) | shrinks | 14–66 | unstable 77, stable 100, unstable 130 | unstable 104 | stable 15 |
| 1.20 (0.244) | shrinks | none | unstable 144 | unstable 144, stable 172 | shrinks |

The spec's own points behave the same way:

- **×0.7:** the tumour ceiling is at n = 218 (p98) and no fixed point lies inside support.
- **×1.3:** the tumour shrinks everywhere and the only supported point is a CE × T-cells
  saddle.

A 5% lower anchor gives the tumour a second, high-density state. A 5% higher one makes the
tumour shrink at its own composition. The spec anchor's tumour value, 0.159, is ×0.78. That
alone explains why §3–§4 put the ceiling at the edge of the data.

**Why:** the tumour's cycling barely changes with density at its own composition. Where a
nearly flat curve crosses a constant, the crossing is ill-conditioned. The CAFs threshold
(n = 65–77 over ×0.9–1.1) and the Myeloid one (64–104) are firmer. T-cells and B-cells are
as fragile as the tumour.

**What is robust:**

- **δ ± 1 SD:** thresholds move by at most 3 cells and the in-support topology is unchanged,
  in both states.
- **τ (20 draws):** every fixed point keeps its classification.
- **K = 7 and K = 6 merges (counts state):** the tumour's ceiling (n = 77–79) and the Myeloid
  threshold are unchanged. The merged CAFs + PVL group's threshold rises to n = 187 because
  PVL shrinks everywhere.

**Neighbourhood radius (refits at 60 and 100 µm).** Both were refitted from 10 restarts
(10/10 converged, 90% agreement), with δ unchanged (e^δ 1.902). Densities are per
(100 µm)², so they compare directly across radii.

| radius | CAFs | Myeloid | tumour (own composition) | tumour path ceiling | in-support fixed points (barycentric) |
|---|---|---|---|---|---|
| 60 µm | stable 40, unstable 108, stable 145, unstable 198 | unstable 106 | unstable 70, stable 126 | 130–153 | CAFs × Myeloid stable; CE × CAFs and CE × Myeloid saddles (q50) |
| **80 µm** | **unstable 71** | **unstable 83** | **stable 79** | **86–174** | **CAFs × Myeloid saddle (q50)** |
| 100 µm | unstable 69 | unstable 81 | unstable 14, stable 74 | 93–99 | none |

At 100 µm the thresholds reproduce production's within 5 cells, but the CAF × Myeloid saddle
is gone. At 60 µm the picture changes:

- the CAFs gain a low-density stable state at n = 40;
- the tumour gains an escape threshold at n = 70, with its ceiling moving up to 126;
- the CAF × Myeloid point becomes stable in its slice.

So the 80 µm thresholds are stable to widening the disc but not to narrowing it. Across all
three radii, two things hold: the tumour has a ceiling inside the data, and Myeloid has an
escape threshold. The CAF × Myeloid fixed point is not robust to the radius.

The parameter uncertainty in the anchor is small. One MVN draw and one block resample both
gave a tumour anchor of 0.202–0.203. The Spartan runs will give its interval. The open risk
is the steady-state assumption itself. A tumour that grows on net has death below division,
so its true anchor sits below its mean π, pushing toward the ×0.9–0.95 rows. The finding to
carry forward is therefore conditional. If the tumour is near steady state (within ~5%), its
ceiling sits near the median density. A modest net growth gives bistability in tumour
density, with a high-density state near the 90th percentile.

### Uncertainty (spec §8, Spartan 2026-10-20)

The runs: **500 parametric draws** (MVN from the block-cluster sandwich) and **50 spatial
block-bootstrap refits** (all converged). The full tables are in `uncertainty/report.md`. The
two sources agree closely, so the numbers below give the parametric run with the block run in
brackets.

**Anchors are tight.** The tumour's is 0.199–0.207 (0.196–0.210), about ±2–3%. The other
types' intervals are within about ±12% of their point estimates. Even so, ±3% on the tumour anchor is
about half the ±5% margin found in the sensitivity sweep, so it carries into the tumour
results.

| feature | point estimate | share of replicates | 95% range of n |
|---|---|---|---|
| CAFs escape threshold (own composition) | n = 71 | **100%** (100%) | 66–77 (67–76) |
| Myeloid escape threshold | n = 83 | **100%** (100%) | 71–100 (72–97) |
| tumour ceiling at own composition | n = 79 | **95%** (96%) | 53–111 (50–152) |
| B-cells ceiling | n = 132 | 100% (96%) | 58–159 (23–166) |
| T-cells ceiling | n = 154 | 100% (100%) | 45–164 (38–158) |
| tumour bistability on the path (a fold) | one fold | **57%** (40%) | — |
| CAF × Myeloid saddle at q50 (barycentric) | n = 66, in support | found 86% (80%), still a saddle 71% (74%) | 55–91 (54–85) |

**Topology preservation** (spec §8.3's headline: the same set of in-support fixed points)
is low:

- **Barycentric state:** 36% (38%).
- **Counts state**, where the point estimate has no supported fixed point: 50% (46%).

The CAF × Myeloid saddle itself recurs in most replicates. The headline fails because other
in-support points come and go near the edge of support.

**What survives:**

- **CAFs and Myeloid escape thresholds:** in every replicate, at a narrow density (CAFs
  n ≈ 66–77, the 30th–40th percentiles). This is the firmest result of the portrait.
- **A tumour ceiling inside the data at its own composition:** 95–96% of replicates. Its
  position is loose, from about the 20th to the 70th percentile (block bootstrap: to the
  90th).
- **The CAF × Myeloid saddle:** present as a saddle in about three replicates in four.

**What does not survive:** the tumour's bistability along the composition path. It appears
in only 40–57% of replicates, so it should not be reported as a finding. This matches the
anchor sweep: the fold appears only between ×0.95 and ×1.0.

Not done from spec §8.3: the per-cell circular SD of arrow direction and the overlaid
bootstrap arrows (the replicates did not store velocity grids).

### Uncertainty and sensitivity runs (spec §8–§9)

**Covariance for spec §8.1** (`run/portrait_uncertainty_prepare.py`, written to
`uncertainty/param_cov.npz` and `uncertainty/se.csv`). The 145 parameters of the production
mixture get a block-cluster sandwich over the 250 spatial blocks. It took 0.5 min locally.
Three directions are flat (the rank-3 U/V rotation) and are dropped. Two checks:

- **Against the 300-replicate spatial block bootstrap**, for the parameters that job kept
  (a, b0, c, δ, η_m, h): the median ratio of sandwich SE to bootstrap SD is 0.96–1.07 per
  block. For δ it is 0.0061 against 0.0059. The parametric draws therefore have the right
  spread.
- **Against a cell-level sandwich**, block clustering inflates the SEs 1.1–1.5× for most
  blocks and 2.3–2.4× for b0 and δ.

**Spartan jobs:**

| job | spec | what it does |
|---|---|---|
| `portrait_param_boot` | §8.1 | 50 tasks × 10 draws from MVN(θ̂, V) |
| `portrait_block_boot` | §8.2 | 50 spatial-block resamples, each refitted from the point estimate and kept with full parameters |

Each replicate recomputes its anchors from its own posterior π and summarises both states on
the full-data geometry (`portrait.summary.summarise`). `run/portrait_uncertainty_gather.py`
reports:

- topology preservation, the share of replicates whose in-support (slice, class) set matches;
- each fixed point's recovery and class;
- the share of replicates with each threshold or ceiling, and their ranges;
- the tumour-path ceiling and folds.

**Sensitivity grid for spec §9** (`run/portrait_sensitivity.py`, written to
`sensitivity/grid.csv`). Axes:

- anchor × 0.7 and × 1.3;
- δ ± 1 bootstrap SD;
- 20 draws of per-archetype τ (log-uniform on [0.5, 2]), which reclassifies the fixed points
  because their locations do not move;
- K = 7 and 6 (B-cells + Plasmablasts and CAFs + PVL merged, then Normal into Cancer
  Epithelial), on the counts state;
- neighbourhood radius 60 and 100 µm. These are refits (`config_r60.yaml`, `config_r100.yaml`,
  tables from `run/portrait_radius_tables.py`, Spartan `multistart`). The grid shows them as
  pending until their point estimates exist.

## Files

```
outputs/portrait/
  FINDINGS.md                                  this document
  dynamics_summary.txt                         sections 3, 5, 7 (run/portrait_dynamics_summary.py)
  composition_bifurcation.txt/.png/.csv        section 4 (run/portrait_bifurcation.py)
  portrait_manifest.json                       config, anchors, confinement, masked fractions
  Atilde.csv, Atilde_heatmap.png               the rank-3 interaction matrix
  fixed_points.csv                             interior + boundary, both eigenvalue sets
  portrait_q25.png, portrait_q50.png, portrait_q75.png    priority pairs, masked
  portrait_contact_sheet.png                   all 36 pairs at q50, unmasked
  grid_velocities_<a>_<b>_q50.parquet          150x150 grids: n, pi, F, in_mask
  osdr_p8/, osdr_p8_target_at_max/             sections 10, 12: the same set under OSDR's correction
  osdr_p2/, osdr_p4/, osdr_p16/                sections 10, 12: the power sweep (summary, bifurcation)
  counts/, counts/osdr_p8/                     sections 11, 12: the count-based state
  uncertainty/param_cov.npz, se.csv            section 12: block-cluster sandwich (portrait_uncertainty_prepare.py)
  uncertainty/report.md, summary.json          spec 8, after the Spartan jobs (portrait_uncertainty_gather.py)
  sensitivity/grid.csv, grid.json              spec 9 (portrait_sensitivity.py)
  sensitivity/anchor_sweep.csv/.txt            anchor multiplier 0.7-1.3 (portrait_anchor_sweep.py)
```

Every portrait output in this directory was regenerated on 2026-10-20 with the own-type
anchor (§12). The spec-anchor versions that §3–§6, §10 and §11 describe are in git at
`80dac2c`; `--anchor nbhd_loading` regenerates them into `anchor_nbhd/`.
