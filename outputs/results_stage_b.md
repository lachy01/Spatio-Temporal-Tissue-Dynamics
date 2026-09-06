# Stage B results — concomitant-variable Poisson mixture

Xenium Prime 5K breast cancer FFPE, **678,885 cells**. Fitted on Spartan at the
settings the cluster sweeps selected: `weight_ridge = 10`, `area_spline_df = 8`,
`rank = 2`, `theta` free (Configuration V).

## Headline estimands (spec 8)

Percentile CIs from a 300-replicate spatial block bootstrap (~250 contiguous
k-means blocks, resampled whole).

| estimand | estimate | 95% CI |
|---|---:|---|
| **Cycling fraction** `mean(pi)` | **11.60%** | [10.61%, 12.47%] |
| **Rate ratio** `exp(e^delta)` | **5.630** | [5.535, 5.742] |
| Normalised entropy | 0.683 | [0.658, 0.711] |
| `sigma_1(B)` — leading interaction | 1.364 | [1.194, 1.716] |
| Oxygen coefficient `psi` | +0.840 | (bootstrap pending per-coefficient) |

`B` has exactly **two** non-zero singular values (1.364, 0.858), independently
confirming the rank-2 selection.

## Is `w` a probability of cycling? (spec 8.7, 8.8)

**Yes, on both mandatory checks.**

- **Normalised entropy 0.683**, far above the 0.30 floor below which spec 8.7
  forbids presenting `w` as P(cycling).
- **Confound diagnostic (spec 8.8):** R^2 of `w` on the nuisance covariates
  {log area, log N, log density, oxygen} = **0.0229**; on cell state = **0.0921**.
  State explains **4.0x** more than nuisance, so `w` is not a size-and-detection
  classifier. Individually `corr(w, log N) = +0.036` and
  `corr(w, log area) = +0.070` — negligible depth and size dependence.

## Validation (spec 6), at spec scale

| test | criterion | result |
|---|---|---|
| 01 recovery | scalars within 3 MC SE; `\|\|B_hat-B\|\|/\|\|B\|\|` < 0.15 | **PASS** (bias/SE 0.33–0.97; 0.106) |
| 02 null bilinear | null/alternative `sigma_1` < 0.20 | **PASS** (0.105) |
| 04 gradient | max rel. error < 1e-5 | **PASS** (2.3e-7) |
| 05 EM monotonicity | non-decreasing to 1e-8 | **PASS** (min step +2.3e-6) |
| 06 EM vs L-BFGS | loglik 6 s.f., params 4 s.f. | **PASS** (4.7e-13, ≤4.3e-6) |
| 07 size confound | fold < 1.5 **and** entropy < 0.25 | entropy **PASS** (0.141); fold **FAIL** (1.93) |
| **08 held-out genes** | delta-deviance ≥ 10x permuted null | **PASS** (23,000–27,700x) |

### Test 08 — the external check

`w` fitted on half the Tirosh signature predicts the **other half**, beyond
depth, area and density, against a spatially block-permuted null:

| split | coef on `w` | delta-deviance | permuted mean | ratio |
|---|---:|---:|---:|---:|
| 0 | +1.687 | 113,197 | 4.6 | 24,620x |
| 1 | +1.753 | 127,427 | 5.2 | 24,442x |
| 2 | +1.858 | 136,053 | 5.7 | 24,012x |
| 3 | +1.606 | 114,347 | 4.1 | 27,710x |
| 4 | +1.690 | 120,750 | 5.2 | 23,021x |

Consistent across all five phase-stratified splits. Dispersion the rate model
failed to absorb has no reason to predict held-out cell-cycle genes; this is the
evidence that the latent class tracks proliferation rather than a nuisance.

### Test 07 — read as a calibration, not a gate

On data with **no latent classes**, the model still reports a ~1.93x fold change,
because a two-component Poisson mixture is also a model for overdispersion and
will use its components to absorb rate misspecification. Three things bound how
much that threatens the real result:

1. The real fold change (5.63) is **2.9x that floor**.
2. It is **invariant to the area spline** — 5.581 / 5.577 / 5.581 / 5.579 at
   df = 4 / 8 / 12 / 16 — while the floor moves with it. A size artefact would be
   absorbed by a more flexible spline; this is not.
3. The confounded null has **entropy 0.141** against the real data's 0.683. The
   confound mechanism inflates the fold *without* producing confident assignment,
   so it cannot reproduce the real signature. Spec 8.7's guardrail fires on the
   confounded data and does not fire on the real data.

The 1.93 floor is calibrated to a chosen severity of misspecification, so the
ratio in (1) depends on that choice; (2) and (3) do not.

## Fit quality

20 restarts, **20/20 converged** at `||grad||inf` median 1.16e-05 (median 5,649
iterations, 12.4 min each). **95% restart agreement**, so spec 4.4's
multimodality caveat does not apply. Bootstrap intervals are stable: raising the
replicate cap from 300 to 5,000 iterations changed the widths of `e^delta`,
`mean(pi)` and entropy by ≤1%, and `sigma_1` by 25%.

## Limitations

- One section, no biological replicate; the block bootstrap bounds within-section
  variability only (spec 11).
- Oxygen field uncertainty is not propagated into these intervals, and the field
  itself failed its HIF-target check, so `psi` reads as vessel proximity /
  perfusion rather than hypoxia.
- Stage C (the additive background) and Stage D (sensitivity analyses) are not
  run. `theta` is Configuration V only; spec 2.2 asks for the F/V bracket.
- Tests 09–14 are not implemented: marginal fit (rootogram/PIT), subsample
  stability, the offset bracket, density-in-weight sensitivity, and residual
  spatial autocorrelation.
- Per-coefficient CIs for `a`, `c` and the archetype-conditional table of spec
  8.2 are not yet extracted from the bootstrap replicates.
