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

---

# Configuration bracket, Stage C and Stage D

## The theta bracket (spec 2.2 / 6.12) — the two configurations agree

Reported side by side, never selected between (spec 9.11). 10/10 converged.

| | log-likelihood | theta | fold change | cycling fraction | entropy | sigma_1 |
|---|---:|---:|---:|---:|---:|---:|
| **F** (theta fixed at 1) | −625,150.8 | 1.0000 | 5.668 | 10.50% | 0.712 | 1.329 |
| **V** (theta free) | −625,110.4 | **1.0362** | 5.630 | 11.60% | 0.683 | 1.364 |

**The bracket is narrow.** The fold change differs by 0.7% and the cycling
fraction by 9.5%, so the causal ambiguity spec 2.2 raises — library size being
simultaneously mediator and confounder — does not materially change the answer.
Configuration F works essentially as well as V, so the fixed offset is a
defensible simplification here.

The reason is visible in `theta` itself: freed, it estimates **1.0362**, almost
exactly the 1.0 the offset assumes. Note this differs sharply from the NB
division model of notebook section 21, where the implied depth exponent was
**0.70**. The difference is `N`: here it excludes the full blocklist, so the
outcome genes and everything correlated with them are outside the offset. In
section 21 `log_total_counts` included them, which is itself a route by which
the outcome inflates its own offset — the definitional dependence spec 1.2
item 2 exists to prevent. The sub-proportional scaling found there was at least
partly that artefact.

## Stage C — the free background fails spec 3.4's own check

Adding the additive background gains **+475 log-likelihood** for 3 parameters,
and the estimands barely move (fold 5.630 → 5.683, cycling fraction 11.60% →
11.63%). That looks like an improvement. It is not one that can be interpreted.

**The spec 3.4 stability test fails.** The fitted `h(density)` curve in the
mixing weight, measured across the density interquartile range (61 to 122 cells
per (100um)^2), has its swing cut from **+0.147 to +0.062 log-odds, a factor of
2.35**. Spec 3.4's condition — sign change, or a change of more than a factor of
two — is met, so the biological and technical density terms are trading off
rather than being separately identified.

**And the control probes say the background should be negligible.** Anchoring it
externally as spec 3.4 prescribes, `rho_i = ctrl_count_i * (53 Tirosh / 649
control features)`:

| | rho per cell | share of the mean count |
|---|---:|---:|
| freely estimated | 0.0211 | 2.4% |
| control-anchored | 0.000119 | 0.014% |

The free background is **177x larger than the controls justify**. Control counts
are zero in 99.9% of cells, so there is almost no probe contamination in this
sample to model. The +475 log-likelihood is therefore not contamination being
captured; it is the background absorbing density signal from the mixing weight.

**Conclusion: report Stage B.** Anchored to the controls the background is
0.014% of the signal, which makes Stage C numerically indistinguishable from
Stage B; freely estimated it is not identifiable as contamination and degrades
the density term we care about.

## Stage D — sensitivity

| variant | log-likelihood | fold change | sigma_1 | vs Stage B |
|---|---:|---:|---:|---|
| rank 1 | −625,379.9 | 5.634 | 1.259 | worse by 269 |
| **rank 2 (reference)** | −625,110.4 | 5.630 | 1.364 | — |
| rank 3 | −625,110.4 | 5.630 | 1.364 | identical |
| rank 5 | −625,110.4 | 5.630 | 1.364 | identical |

Ranks 3 and 5 are numerically identical to rank 2 — the extra components carry
no weight — which confirms the rank-2 selection for a third time, independently
of the cross-validation and of the point estimate's singular values.

**The density definition does not matter.** The fixed-k kNN density spec 1.3
prefers, re-run after the no-op was fixed, moves nothing that matters:

| | radius 50 um (used) | fixed-k kNN (k=30) | change |
|---|---:|---:|---:|
| log-likelihood | −625,110.4 | −625,109.9 | +0.5 |
| fold change | 5.6295 | 5.6302 | +0.0% |
| cycling fraction | 11.60% | 11.56% | −0.3% |
| normalised entropy | 0.6832 | 0.6842 | +0.2% |
| `psi` (oxygen) | +0.840 | +0.862 | +2.7% |
| `sigma_1(B)` | 1.364 | 1.406 | +3.1% |

The density covariate itself changes substantially — the weight spline goes from
`h = [−1.566, +0.086]` to `[−0.949, +0.502]` — so this is a real perturbation
that the estimands simply absorb. The deviation taken in the data contract
(radius-based density, for comparability with the notebook's NB models) is
therefore not load-bearing for any conclusion.

**One Stage D variant is deliberately unavailable.** Relaxing the
proportional-rates restriction (`theta_1 != theta_0`) raises
`NotImplementedError` rather than silently doing something else: giving each
component its own theta needs a second parameter in the model, and spec 2.2
makes that restriction load-bearing for identifiability at this count depth, so
it is a modelling change rather than a config flag. It remains the one assumption
in the specification that has not been tested.

---

## Spec 8.2 — archetype-conditional neighbourhood effects (the headline estimand)

Odds ratio of cycling per IQR of each neighbourhood component, for a cell whose
own state sits at the given archetype centroid: `exp((c + B^T phi_a)_k * IQR_k)`,
mapped back from the projected basis to cell types. Full table with CIs in
`outputs/estimands_8_2.md`; **42 of 81 entries have a bootstrap CI excluding 1**.

| own archetype | Endo | CAFs | PVL | B | T | Mye | NormEp | Plasma | **CancEp** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Endothelial | 0.85 | 0.71 | 0.33 | 0.87 | 1.14 | 1.15 | 1.18 | 0.02 | **11.44** |
| CAFs | 0.59 | 0.80 | 0.61 | 0.81 | 1.18 | 1.04 | 1.03 | 0.03 | **7.58** |
| T-cells | 0.55 | 0.80 | 0.65 | 0.80 | 1.17 | 1.01 | 1.01 | 0.04 | **7.38** |
| Plasmablasts | 0.24 | 1.07 | **2.82** | 0.67 | 1.27 | 0.81 | 0.73 | 0.18 | 2.73 |
| **Cancer Epithelial** | 0.18 | 1.23 | **5.70** | 0.63 | 1.35 | 0.76 | 0.62 | 0.23 | **1.62** |

Two patterns stand out.

**A tumour neighbourhood raises cycling odds for everything except tumour.** The
`Cancer Epithelial` column runs 7-11x for stromal and immune own-states but only
**1.62x** for a cancer cell itself. Being surrounded by tumour matters far more
if you are not tumour — the interaction is not a uniform "tumour is
proliferative" effect.

**Cancer and plasmablast cells respond to PVL where others do not.** The `PVL`
column is 0.33-0.65 (suppressive) for most own-states but **5.70x** for Cancer
Epithelial and **2.82x** for Plasmablasts. This is exactly the context-dependence
`B` exists to capture, and it is invisible to any model with main effects only.

## Spatial fit (`outputs/figures/spatial_residuals.png`)

Observed and fitted mean counts track closely across the section; the Pearson
residual map shows speckle but no coherent regions, consistent with Moran's I =
0.0142 and a correlation range under 25 um.

`pi_hat` shows a bright band along one tissue margin. **Checked and it is not an
edge artefact**: binning by neighbour count, `pi_hat` is *lowest* in the sparsest
decile (0.075 at 39 neighbours) and peaks at middle density (0.133 at 155),
tracking the observed count in the same direction throughout. `h(log density)` is
hump-shaped rather than monotone, despite its negative leading spline
coefficient — the coefficient's sign does not give the direction of a spline.

---

## Test 10 — subsample stability (spec 6.10)

| n | fold change | mean(pi) | sigma_1 | within the full-sample CI? |
|---|---:|---:|---:|---|
| 50,000 | 5.8685 | 0.0828 | 0.916 | fold no, mean(pi) no |
| 200,000 | 5.6526 | 0.0962 | 0.000 | fold yes, mean(pi) no |
| **678,885** | **5.6295** | **0.1160** | **1.364** | — |

**The rate ratio is stable** (5.87 / 5.65 / 5.63; only the 50,000-cell value falls
outside the full-sample bootstrap CI, and modestly). **`mean(pi)` drifts
monotonically with n** — correlation +0.983 — which is exactly what spec 6.10
says must not happen.

**The drift is an artefact of the penalty parameterisation, not of the model.**
`weight_ridge` was an *absolute* penalty while the log-likelihood grows with n,
so a subsample was shrunk far harder: at n = 50,000 the ridge is 13.6x stronger
per observation than at the full sample. Scaling it by n removes the drift:

| n | lambda | mean(pi) | sigma_1 |
|---|---|---:|---:|
| 50,000 | fixed 10 | 0.0865 | 1.06 |
| 50,000 | **scaled 0.74** | **0.1141** | 2.14 |
| 200,000 | fixed 10 | 0.1024 | 1.07 |
| 200,000 | **scaled 2.95** | **0.1094** | 1.56 |
| 678,885 | 10 (= scaled) | 0.1160 | 1.36 |

Scaled, `mean(pi)` reads 0.1141 / 0.1094 / 0.1160 — flat within noise.

The penalty is now scaled by `n / weight_ridge_n_reference`. At the full sample
the factor is exactly 1, so **every result reported above is unchanged** (verified:
the stored point estimate's log-likelihood reproduces to 4 decimal places). The
fix matters for fitting at any other n, and Test 10 should be re-run against it.

## Test 13 — density-in-weight sensitivity (spec 6.13)

| `h()` | log-likelihood | fold | sigma_1 | psi |
|---|---:|---:|---:|---:|
| dropped | −625,220.0 | 5.6328 | 1.301 | 0.775 |
| **df = 3 (fitted)** | **−625,110.4** | **5.6295** | **1.364** | **0.840** |
| df = 5 | −625,098.0 | 5.6294 | 1.368 | 0.840 |

Change in each neighbourhood coefficient when `h()` is removed entirely:

| cell type | with `h` | no `h` | change | |
|---|---:|---:|---:|---|
| **Cancer Epithelial** | **+3.1168** | **+3.1166** | **−0.0%** | |
| Plasmablasts | −0.9744 | −0.9565 | +1.8% | |
| Endothelial | −0.7559 | −0.9201 | −21.7% | |
| Normal Epithelial | −0.6801 | −0.8618 | −26.7% | |
| PVL | +0.1384 | +0.0824 | −40.4% | |
| CAFs | −0.5957 | −0.8796 | −47.7% | |
| B-cells | −0.5592 | −0.2743 | +50.9% | flagged |
| T-cells | +0.3798 | +0.8371 | +120.4% | flagged |
| Myeloid | −0.0696 | −0.1437 | −106.6% | flagged |

**The headline neighbourhood effect is untouched.** `nbr[Cancer Epithelial]`
moves by 0.0% — the "being surrounded by tumour raises cycling odds" result does
not depend on the density term at all. `sigma_1` moves −4.6% and `psi` −7.7%.

**Three small immune coefficients are density-sensitive**: B-cells, T-cells and
Myeloid, all flagged at >50% change. These are exactly the coefficients whose
magnitudes are small enough for spatial confounding to matter, and spec 6.13
predicts it: density and `phi_bar` are both spatially smooth on similar scales,
so a flexible `h()` can absorb genuine neighbourhood signal. Report the immune
neighbourhood effects with that caveat; the tumour one is safe.

Raising `df(h)` from 3 to 5 changes nothing (log-likelihood +12, all estimands to
four figures), so the spec's cap of 3 is not binding. `df = 2` could not be run:
patsy's `cr()` cannot build a 2-df natural cubic basis.

### Test 10, final (penalty scaled, 20 restarts per size)

| n | restart agreement | `sigma_1` | `mean(pi)` | fold change |
|---|---:|---:|---:|---:|
| 50,000 | **5%** | 1.902 | 0.1403 | 5.594 |
| 200,000 | **5%** | 1.463 | **0.1160** | 5.531 |
| **678,885** | **95%** | 1.364 | **0.1160** | 5.630 |

Full-sample bootstrap CIs: fold [5.535, 5.742], `mean(pi)` [0.1061, 0.1247].

**The fold change is stable at every size** — 5.594 / 5.531 / 5.630, with only the
200,000 value outside the CI and by 0.004. This is the headline estimand and it
does not depend on sample size.

**`mean(pi)` and `sigma_1` are inflated at 50,000 but converged by 200,000.**
At 200,000 `mean(pi)` is 0.1160, matching the full sample exactly; at 50,000 it
is 0.1403, above the CI. `sigma_1` runs 1.902 / 1.463 / 1.364, decreasing
monotonically — the classic signature of a variance-dominated estimate at small
n, and it is still shrinking between 200,000 and the full sample. The practical
reading is that this model needs on the order of 200,000 cells; at 50,000 it
overfits.

**Both subsamples are severely multimodal**: only 1 of 20 restarts finds the best
optimum (5% agreement), against 95% at the full sample. Spec 4.4's criterion
therefore applies to any subsample fit — best-of-N with an explicit caveat, not a
silent pick — while the full-sample fit is exempt.

An earlier version of this section, run with 5 restarts, reported `sigma_1`
collapsing to zero at 200,000 and concluded that `B` is identified only at the
full sample. **That was wrong.** The zeros were poor local optima found by too
few starts; with 20 restarts `sigma_1` is non-zero at every size. `B` is
identified at 200,000 — it is merely biased upward there.
