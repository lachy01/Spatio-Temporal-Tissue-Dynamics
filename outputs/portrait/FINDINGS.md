# Phase portrait — preliminary findings

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
- ~~The OSDR maximal-density bound~~: run as a sensitivity variant (§10). Its fixed points and
  nullclines are not reported as findings.
- Spec §7's q25/q50/q75 robustness check is only partly possible (§6).

## 10. Sensitivity: OSDR's maximal-density correction

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
  osdr_p8/, osdr_p8_target_at_max/             section 10: the same set under OSDR's correction
  osdr_p2/, osdr_p4/, osdr_p16/                section 10: the power sweep (summary, bifurcation)
```
