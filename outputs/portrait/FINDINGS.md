# Phase portrait — preliminary findings

**Status: preliminary, and deliberately UNBOUNDED.** OSDR's `p_correction = c_k · d_k^n`
is not applied. Spec §5 says to check confinement first and add the correction only as a
flagged sensitivity variant; the check (§2 below) fails for one of nine archetypes.

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
only where the loadings are positive; the 9-dimensional state is not a simplex.

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

Rank 3, mostly reciprocal (symmetric / antisymmetric Frobenius norms 5.93 / 1.50). Every
self-term is negative: compositional crowding by the same archetype lowers its cycling
odds. Largest entries, in log-odds of cycling per unit neighbourhood fraction (row ←
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

`π̄` uses spec §3.2's signed-loading formula. Clipping the negative loadings moves every
anchor by at most 0.010 (`pi_bar_clipped` in the manifest); its effect on the dynamics has
not been recomputed under this fit.

---

## 9. Not yet done

- **Spec §8, uncertainty: none of it.** No parametric or spatial block bootstrap, no
  topology-preservation fraction. Every number above is a point estimate, and the §4 ceiling
  at the 98th percentile of density is exactly the kind of claim that needs an interval.
- **Spec §9, sensitivity grid: none of it** (the `π̄` multiplier, calibration and `δ`
  perturbations, radius/`L0`, `K`, `own_state_mode`, `τ`, and the clipped anchor).
- **The OSDR maximal-density bound**, needed for CAFs (§2).
- Spec §7's q25/q50/q75 robustness check is only partly possible (§6).

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
```
