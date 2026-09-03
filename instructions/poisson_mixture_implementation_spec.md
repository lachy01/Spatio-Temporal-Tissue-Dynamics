# Implementation spec: concomitant-variable Poisson mixture for cell-cycle transcript counts

**Audience:** a coding agent with no prior conversational context. Read this document in full before writing code. Where the spec says MUST, deviation is a bug. Where it says SHOULD, deviation requires a written justification in the run log.

**Deliverable:** a Python package that fits a two-component Poisson mixture with covariate-dependent mixing weights to per-cell cell-cycle transcript counts from a Xenium Prime 5K breast cancer section (n ≈ 700,000 cells), plus a validation suite of 14 numbered tests.

---

## 0. Objective and why the model has this shape

The scientific target is **P(cell i is in S/G2M)** as a function of the cell's own transcriptional state, the state of its spatial neighbourhood, local oxygen, and local density.

The observed data are counts of transcripts from a cell-cycle gene set. These counts are a noisy, depth-dependent readout of the latent cycling state. A single-equation negative binomial regression on these counts estimates `E[count]`, which then requires an arbitrary external calibration to become a probability. The mixture removes that step: the mixing weight *is* the probability, and it carries the covariates of interest directly.

The core structural decision, which MUST NOT be altered without explicit instruction:

- **State covariates go in the mixing weight, never in the component rates.** If own-state, neighbourhood-state and their bilinear interaction are placed in the Poisson rate, the estimand silently becomes "expected transcripts given the cell is already cycling", which is not the quantity of interest.
- **Detection covariates (library size, cell area) go in the component rates, never in the mixing weight.** These affect how many transcripts are *observed* given the latent state, not the latent state itself.
- **Density and oxygen go in the mixing weight** (biological: contact inhibition, hypoxic quiescence), and density optionally *also* in an additive background term (technical: spillover). See §3.4 for the identifiability hazard this creates.

---

## 1. Data contract

### 1.1 Required inputs

A single table (parquet), one row per cell, with these columns. The agent MUST validate every item in §1.2 before fitting and MUST abort with a clear error on failure.

| Column | Type | Description |
|---|---|---|
| `cell_id` | str | Unique cell identifier |
| `x`, `y` | float64 | Centroid coordinates, µm |
| `y_count` | int64 | Outcome. Sum of counts over the Tirosh S + G2M gene set. |
| `N` | int64 | Library size **excluding** the Tirosh set and the ~280-gene blocklist |
| `area` | float64 | Segmented cell area, µm² |
| `nuc_area` | float64 | Segmented nucleus area, µm² (used in robustness checks) |
| `density` | float64 | Local cell density (see §1.3) |
| `oxygen` | float64 | Oxygen field value `u_i` from the PDE solve |
| `phi_00` … `phi_28` | float64 | PhiSpace own-state, 29 components |
| `nbr_phi_00` … `nbr_phi_28` | float64 | kNN-mean neighbourhood state, 29 components |
| `archetype` | category | argmax cell-type label. **Stratification and reporting only. MUST NOT enter the likelihood.** |
| `ctrl_count` | int64 | *Optional.* Negative control probe + blank codeword counts per cell. Used in Stage C. |

### 1.2 Validation assertions (all MUST pass)

1. `y_count >= 0`, integer dtype, no NaN.
2. **`N` must not contain any Tirosh or blocklisted gene.** If the pipeline that produced `N` cannot be verified, abort. A library size that includes the outcome genes creates definitional dependence between offset and outcome and invalidates `θ` (§2.2).
3. `N > 0` for all rows; drop rows with `N == 0` and log the count.
4. `area > 0`, `density > 0`.
5. No NaN in any `phi_*` or `nbr_phi_*` column.
6. Check `rowsum(phi)`. Record whether it is (a) constant at 1.0 within 1e-6, or (b) not constant. This determines the basis construction in §3.1 and MUST be logged.
7. Report `corr(log N, log area)`. If `> 0.9`, apply the orthogonalisation in §2.2 and log that it was applied.
8. Report the marginal distribution of `y_count` (counts 0–20 plus tail mass) and the fraction of exact zeros.

### 1.3 Density

Use the same density definition already used in the existing single-equation model. If it must be recomputed: fixed-k kNN density, `density_i = k / (π * r_k(i)^2)` with `k = 30`, where `r_k(i)` is the distance to the k-th nearest neighbour. Fixed-k is required rather than fixed-radius so that measurement error in the neighbourhood average is not heteroskedastic in density. Log the value of `k` used.

---

## 2. Model specification

### 2.1 Generative model

For cell `i`, latent class `Z_i ∈ {0, 1}` with `Z_i = 1` meaning "cycling":

```
P(Z_i = 1) = π_i
y_i | Z_i = k  ~  Poisson( ρ_i + λ_ki )
```

Poisson, not negative binomial. The negative binomial is itself a Poisson–gamma mixture; a two-component NB mixture is a mixture of mixtures in which continuous gamma heterogeneity within each component competes with the discrete two-point structure for the same variance. At this count depth that competition is weakly identified. Forcing all overdispersion through the class structure is both better identified and mechanistically cleaner. **Do not add NB dispersion in Stages A–C.** Stage D tests whether it is needed.

### 2.2 Component rates

```
log λ_0i = b_0     + θ · logN_i + s(log area_i)
log λ_1i = b_0 + e^δ + θ · logN_i + s(log area_i)
```

Note what is shared: `b_0`, `θ`, and the entire spline `s(·)`. The components differ only by the additive constant `e^δ`, so

```
λ_1i / λ_0i = exp(e^δ)     for all i
```

This is a **proportional-rates restriction**. It is what makes the model identifiable at this count depth: the classes are separated by one number (the fold change) rather than by two independently-shaped regression surfaces. Giving each component its own `θ` and spline lets the mixture separate cells on size rather than biology, which is the failure mode this design exists to prevent. Stage D tests the relaxation; Stages A–C MUST impose it.

The `e^δ` parameterisation (not a free additive constant) enforces `λ_1 > λ_0` strictly, which pins component identity and eliminates label switching. Do not replace it with an unconstrained parameter and post-hoc relabelling.

**Spline:** `s(·)` is a fixed-df natural cubic regression spline on `log(area)`, `df = 4`, knots at quantiles. Use `patsy.dmatrix("cr(x, df=4) - 1", ...)` or an equivalent explicit basis constructed **once** on the full data and reused across all fits and bootstrap replicates. Do **not** use a penalised smooth: inside an EM loop, per-iteration penalty selection breaks the monotone-likelihood guarantee and is slow.

**Offset coefficient `θ`:** fit in two configurations and report both.

- **Configuration F (fixed):** `θ ≡ 1`. This is a pure rate model. `log N` is treated as a blocked path.
- **Configuration V (free):** `θ` estimated.

These are not competing models to be selected between by fit. Library size sits on the causal path from cycling to count — a cycling cell genuinely contains more RNA — while also being a common cause of the measured count through capture efficiency. It is simultaneously mediator and confounder. Configuration F blocks the path and attenuates; Configuration V partially unblocks it by an amount chosen by likelihood rather than by causal argument. **Report both as a bracket.** A likelihood-ratio test between them is not a valid basis for choosing one, and the agent MUST NOT present it as such.

**Orthogonalisation (conditional):** if `corr(log N, log area) > 0.9`, replace `logN_i` with the residual from regressing `log N` on the area spline basis. This makes `θ` interpretable as an effect of per-area detection efficiency and stabilises the fit. Log the R² of that auxiliary regression.

### 2.3 Background term

```
log ρ_i = r_0 + g(log density_i)          [Stage C onward]
ρ_i ≡ 0                                    [Stages A and B]
```

`g(·)` is a natural cubic spline, `df = 3`.

The background is **additive**, not multiplicative, because contamination is additive: a cell's observed count is the sum of its own transcripts and any misassigned ones. Poisson sums are Poisson, so `y_i ~ Poisson(ρ_i + λ_ki)` is the mechanistically faithful form. This makes the link non-canonical, which is why Stage C requires direct optimisation rather than EM with GLM M-steps (§4).

### 2.4 Mixing weight

```
logit(π_i) = a_0 + φ̃_i^T a + φ̄̃_i^T c + φ̃_i^T B φ̄̃_i + h(log density_i) + ψ · oxygen_i
```

with `B = U V^T`, `U` and `V` each `28 × r`, rank `r` selected per §6.11.

`h(·)`: natural cubic spline, `df = 3`. Low df is deliberate — see §3.4.

`φ̃` and `φ̄̃` are centred and basis-projected per §3.1.

---

## 3. Identifiability — read before writing any model code

Four separate hazards. Each has produced silently wrong results in models of this shape.

### 3.1 Simplex collinearity of the PhiSpace vectors

If `rowsum(phi) == 1` (check recorded in §1.2 item 6), then the 29 columns are linearly dependent with the intercept and the design matrix is rank-deficient. Centring alone does not fix this — centred rows sum to zero, which is still a rank-28 subspace.

**Required construction:**

1. Compute global means `m = mean(phi, axis=0)` and `m̄ = mean(nbr_phi, axis=0)`. Centre: `phi_c = phi - m`, `nbr_c = nbr_phi - m̄`.
2. If rowsums were constant, build an orthonormal basis `Q` (29 × 28) for the sum-to-zero subspace (Helmert contrasts, or the last 28 columns of a QR of `I - (1/29)11^T`). Project: `φ̃ = phi_c @ Q`, `φ̄̃ = nbr_c @ Q`. All downstream dimensions are 28.
3. If rowsums were **not** constant, still centre, and still use dimension 29. Check the condition number of `[1, φ̃, φ̄̃]`; if `> 1e6`, project onto the top principal directions capturing 99.9% of variance and log the reduction.

Centring is not cosmetic: it makes `a` and `c` interpretable as effects **at the mean state**, and it removes the aliasing between the bilinear term and the main effects that would otherwise occur when both `φ` and `φ̄` live on a simplex.

**Compute `Q`, `m`, `m̄` once on the full dataset and freeze them.** Recomputing per bootstrap replicate changes the parameterisation between replicates and produces meaningless confidence intervals.

### 3.2 The factorisation `B = UV^T` is not identified

For any invertible `r × r` matrix `M`, `(UM)(VM^{-T})^T = UV^T`. `U` and `V` individually are not estimable.

**Consequences the agent MUST respect:**

- Never report, plot, or interpret `U` or `V` directly.
- Report `B̂ = ÛV̂^T` and its SVD. The singular values and singular vectors of `B̂` are identified and are the interpretable objects.
- Bootstrap `B̂` (the product), never `Û` or `V̂`.
- For numerical stability during optimisation, add a small ridge penalty `τ(‖U‖_F² + ‖V‖_F²)` with `τ = 1e-4`. This does not resolve non-identifiability of the factors but conditions the surface. Report sensitivity to `τ ∈ {0, 1e-4, 1e-3}`.

### 3.3 Label switching and degenerate components

Handled by the `e^δ` order constraint, but two residual checks are required:

- If `ê^δ` drifts below `1e-3` (fold change → 1), the components have collapsed and the model has not found two classes. Flag prominently. Do not report `π̂` as a probability of cycling in this case.
- If `mean(π̂_i)` is within `1e-3` of 0 or 1, the mixture has degenerated to a single component. Same treatment.

### 3.4 Density enters in two places

Density appears in the mixing weight (`h`, biological) and in the background (`g`, technical). These are formally distinguishable because they enter through different channels — logistic on class membership versus additive on rate — but the separation is fragile at realistic signal levels.

**Mitigations, in order of preference:**

1. **Staging (required).** Stage B fits density in the weight only. Stage C adds the background. Compare `ĥ(·)` between stages. If the fitted curve changes sign or changes by more than a factor of two over the interquartile range of density, the two terms are trading off and the Stage C background MUST be anchored externally rather than freely estimated.
2. **External anchoring (Stage C fallback).** If `ctrl_count` is available, fix `r_0` by matching `exp(r_0)` to the per-cell control-probe rate scaled by the ratio of panel sizes (number of Tirosh genes in panel / number of control probes), and estimate only `g(·)`'s shape with a strong prior toward flatness — or fix `ρ_i` entirely as a known offset and estimate nothing in it.

**A separate, more serious version of the same problem:** the density spline in the *weight* competes with `c` and `B`. Local density and `φ̄` are both spatially smooth on similar scales, so a flexible `h(·)` can absorb genuine neighbourhood signal. This is spatial confounding acting directly on the primary estimand, not on a nuisance parameter.

- Keep `df(h) = 3`. Do not increase without explicit instruction.
- Test 13 (§6.13) is mandatory: refit with `h(·)` dropped entirely and tabulate the change in every element of `c` and in the singular values of `B̂`.

**Related note on oxygen:** `oxygen_i` is itself the output of a PDE solve driven partly by cell density, so `ψ` and `h(·)` also overlap. Additionally, uncertainty in the oxygen field is not propagated into these standard errors. Both facts MUST appear in the limitations section of any output report the agent generates. Do not interpret `ĥ(·)` as contact inhibition when oxygen is in the model.

---

## 4. Estimation

### 4.1 Marginal likelihood

```
ℓ_i(Θ) = logsumexp[ log(1-π_i) + logPois(y_i; ρ_i + λ_0i),
                    log(π_i)   + logPois(y_i; ρ_i + λ_1i) ]
ℓ(Θ) = Σ_i ℓ_i(Θ) - τ(‖U‖_F² + ‖V‖_F²)
```

`logsumexp` is mandatory. Naive computation of the mixture density underflows for cells where one component has near-zero probability, which at n = 700,000 will occur.

Use `log1p`/`expit` carefully: compute `log π_i` and `log(1-π_i)` from the logit directly via `-log1p(exp(-η))` and `-log1p(exp(η))` with the standard sign-stable branch, not by exponentiating then logging.

### 4.2 Primary optimiser

**Direct maximisation of the marginal likelihood by quasi-Newton with automatic differentiation.** Implement in JAX.

- Full-batch gradient; the data fit in memory as float32 (700K × ~200 design columns ≈ 560 MB). Use float64 for the accumulated log-likelihood.
- Optimiser: L-BFGS (`optax.lbfgs` or `jaxopt.LBFGS`), `maxiter = 2000`.
- Convergence: relative change in `ℓ` below `1e-9` **and** `‖∇ℓ‖_∞ < 1e-5`. Both, not either.
- Obtain the observed information from `jax.hessian` at the optimum for use as the bread of the sandwich estimator (§7). Do not use its inverse as a variance estimate on its own.

### 4.3 EM (Stages A–B only, cross-check and starting values)

EM is retained because it is monotone and its M-steps are ordinary weighted GLMs, which makes it a reliable independent check on the direct optimiser. It applies only when `ρ_i ≡ 0`; with the additive background the M-step is no longer a standard GLM.

**E-step:**
```
log w_i = log π_i + logPois(y_i; λ_1i) - ℓ_i
w_i = exp(log w_i)
```

**M-step:**
1. Component parameters: a single Poisson GLM on the *stacked* data — each cell appears twice, once with weight `1 - w_i` and a class indicator of 0, once with weight `w_i` and indicator 1. The indicator's coefficient is `e^δ`; enforce positivity by refitting with `δ` on the log scale, or fit unconstrained and check the sign each iteration.
2. Weight parameters: logistic regression of the fractional response `w_i` on `[1, φ̃, φ̄̃, bilinear features, h-basis, oxygen]`. The bilinear term is not linear in `U, V`, so alternate: hold `V` fixed and update `(a, c, U, ...)` — linear in `U` given `V` — then hold `U` fixed and update `V`. Two inner alternations per M-step is sufficient.

**Assert monotonicity every iteration.** If `ℓ` decreases by more than `1e-8`, abort with an error. This is the single most valuable bug detector in the implementation.

### 4.4 Starting values and multi-start

1. Fit a single-component Poisson GLM with the full component design (`b_0`, `θ`, area spline). Use its coefficients for the component block.
2. `δ`: set `e^δ = log(3)`, i.e. `δ = log(log 3) ≈ 0.0940`. This corresponds to cycling cells having roughly a threefold higher rate.
3. `a_0 = logit(0.10)`. A ~10% cycling fraction is a reasonable prior for this tissue.
4. `a`, `c`, `U`, `V` ~ `N(0, 0.01²)`.
5. `ψ = 0`, spline coefficients `= 0`, `r_0` from `mean(ctrl_count)` if used.

**Run 20 random restarts** (perturbing steps 2–5; keep step 1 fixed). Record the log-likelihood of every restart. Report the fraction reaching within `1e-6` of the best value. If fewer than 60% converge to the same optimum, the surface is multimodal — report this prominently and use the best-of-20 with an explicit caveat, do not silently take the first.

### 4.5 Scale

Prototype on a 50,000-cell stratified subsample of Cancer Epithelial cells only. Only proceed to the full section after Tests 1–7 pass on the prototype.

---

## 5. Staged build

Each stage MUST pass its gate before the next begins. Write a run log entry per stage.

| Stage | Model | Gate |
|---|---|---|
| **A** | Two-component Poisson. Weight: intercept only. Components: `b_0`, `e^δ`, `θ`, area spline. No background. | Tests 1, 4, 5, 6. Cross-check against `flexmix` (R) on 20K cells — log-likelihoods must agree to 4 s.f. |
| **B** | Add to weight: `φ̃`, `φ̄̃`, bilinear `B` (r = 2), density spline, oxygen. Still no background. | Tests 2, 3, 7, 8, 9, 10. |
| **C** | Add additive background `ρ_i`. EM no longer applies; direct optimisation only. | Tests 9, 13, plus the §3.4 stability comparison against Stage B. |
| **D** | Sensitivity only. NB component dispersion; relaxed proportional rates (`θ_1 ≠ θ_0`); rank sweep; `θ` bracket. | Tests 11, 12, 14. Report as sensitivity analyses, not as the primary specification. |

`flexmix` with `FLXPmultinom` will handle Stage A and a reduced Stage B (linear weight covariates, no bilinear, no background). It **cannot** fit the bilinear term or the additive background. Use it as a correctness oracle for Stage A, then set it aside. Do not attempt to force the full model into it.

---

## 6. Validation suite

Fourteen tests. Each is a separate file `tests/test_NN_name.py`, each writes a pass/fail line plus its numbers to `outputs/validation_report.md`.

### 6.1 Test 01 — Simulation recovery
Simulate n = 100,000 from the Stage B model with known parameters, moderate separation (`exp(e^δ) = 4`), `mean(π) = 0.15`, `r = 2`, `‖B‖` giving an odds-ratio modification of ~1.3 across the archetype range. Fit. **Pass:** every scalar parameter within 3 Monte Carlo standard errors of truth across 20 simulation replicates; `‖B̂ - B‖_F / ‖B‖_F < 0.15`.

### 6.2 Test 02 — Null bilinear
Simulate with `B = 0`, everything else as Test 01. **Pass:** block-bootstrap CIs for the leading singular value of `B̂` cover 0 in ≥ 90 of 100 replicates; median `σ_1(B̂)` less than 20% of the Test 01 value.

### 6.3 Test 03 — Single-component null
Simulate with `π_i ≡ 0` (all cells non-cycling, one Poisson rate). Fit the two-component model. **Pass:** `ê^δ < 0.05` or `mean(π̂)` within `1e-2` of a boundary; normalised entropy `< 0.10`; the §3.3 degeneracy flags fire. This test verifies the model *reports* absence of structure rather than manufacturing it.

### 6.4 Test 04 — Gradient check
Compare the JAX gradient against central finite differences at three random parameter vectors. **Pass:** max relative error `< 1e-5` on every component.

### 6.5 Test 05 — EM monotonicity
Run 200 EM iterations on real Stage A data. **Pass:** `ℓ` non-decreasing at every iteration to within `1e-8`.

### 6.6 Test 06 — Optimiser agreement
EM to convergence versus L-BFGS on the same Stage A problem. **Pass:** log-likelihoods agree to 6 s.f.; parameters agree to 4 s.f.

### 6.7 Test 07 — Adversarial size-confound (the critical test)
Simulate data with **no latent class structure at all** (`π_i ≡ 0`), but where the count depends on `log area` through a *wiggly* function the fitted spline cannot represent — e.g. simulate with `df = 10` and fit with `df = 4`. Fit the full Stage B model.

**Pass:** normalised entropy `< 0.25`, `σ_1(B̂)` not distinguishable from zero, and the fold change `exp(ê^δ) < 1.5`.

**If this test fails, the mixture is separating cells on size and segmentation quality rather than on biology, and the entire approach is compromised.** This is the same failure that produced the earlier binary GMM death label. A failure here is not a tuning problem — it means the area spline must be made flexible enough (raise df, refit, re-run) before any real-data result is reported.

### 6.8 Test 08 — Held-out gene validation (real data, primary external check)
Split the Tirosh gene set into halves A and B, stratified by phase (S versus G2M) so both halves contain both phases. Fit the full model with `y^A` as the outcome. Extract `ŵ_i`.

Then fit two auxiliary Poisson GLMs for `y^B`:
- Nuisance model: `y^B ~ offset(log N) + s(log area) + s(log density)`
- Full model: nuisance `+ ŵ_i`

**Pass:** coefficient on `ŵ` positive and large relative to its block-bootstrap SE; Δdeviance between the two models at least 10× the Δdeviance obtained when `ŵ` is replaced by a spatially-block-permuted version of itself (100 permutations, report the null distribution).

The permutation control is the point: `ŵ` will predict `y^B` trivially through library size and size. What matters is predictive content **beyond** the nuisance covariates. Report both raw and permutation-calibrated results.

Repeat with 5 independent A/B splits and report the spread.

### 6.9 Test 09 — Marginal fit
Rootogram and randomised PIT computed from the **mixture** marginal PMF, `P(y) = (1-π_i)Pois(y; ρ_i+λ_0i) + π_i Pois(y; ρ_i+λ_1i)`, aggregated over cells. Produce overall plus per-archetype panels.

**Pass:** no bin with hanging discrepancy exceeding the Stage-A-equivalent single-component fit; specifically the counts 1–2 region must improve. Report RMS hanging discrepancy per archetype, single-component versus mixture, as a table. Cancer Epithelial is the stratum that matters — improvement in low-count strata at the expense of Cancer Epithelial is a failure, not a success.

Randomised PIT (not the non-randomised version) is required here because with a discrete mixture the non-randomised PIT is conservative and will look artificially good.

### 6.10 Test 10 — Subsample stability
Fit on 50K, 200K, and the full set. **Pass:** rate ratios (§8) stable to within their block-bootstrap CIs; no systematic drift in `mean(π̂)` with n.

### 6.11 Test 11 — Rank selection
Sweep `r ∈ {1, 2, 3, 5}`. Select by **held-out log-likelihood on spatial blocks** (5-fold, contiguous tissue blocks), not by AIC or BIC. Spatial dependence means the effective sample size is far below 700,000, so information criteria that assume independent observations will systematically over-select rank. **Pass:** a clear maximum or plateau in held-out log-likelihood; report the full curve.

### 6.12 Test 12 — Offset bracket
Fit Configuration F (`θ = 1`) and Configuration V (`θ` free). **Output:** a side-by-side table of every rate ratio and the singular values of `B̂`. There is no pass/fail. If the two configurations disagree materially, that instability is itself a reportable finding and MUST be surfaced, not resolved by picking the better-fitting one.

### 6.13 Test 13 — Density-in-weight sensitivity
Refit Stage B with `h(·)` removed from the mixing weight. **Output:** change in every element of `c`, in `σ_1(B̂)`, and in `ψ`. Flag any element of `c` that changes by more than 50% or changes sign. Also fit with `df(h) ∈ {2, 3, 5}` and report the trajectory.

### 6.14 Test 14 — Residual spatial autocorrelation
Compute Pearson residuals from the marginal mean `E[y_i] = ρ_i + (1-π_i)λ_0i + π_i λ_1i`. Moran's I with the same weight matrix used in the existing single-equation model, plus the full correlogram out to 500 µm. **Output:** comparison against the existing NB model's residual Moran's I. The bandwidth for Conley SEs (§7) is read off this correlogram.

---

## 7. Inference

Model-based standard errors from the inverse Hessian are **invalid** here and MUST NOT be reported. Cells are spatially dependent; the independence likelihood understates variance severely at n = 700,000.

### 7.1 Primary: spatial block bootstrap

1. Partition the section into contiguous blocks by k-means on `(x, y)`, with block side length ≥ 5× the correlation range read from the Test 14 correlogram. Target 150–400 blocks.
2. Resample blocks with replacement to total approximately n cells.
3. Refit, warm-starting from the full-data estimate. Cap at 300 L-BFGS iterations per replicate.
4. 300 replicates minimum.
5. Percentile CIs on all reported estimands (§8).

Freeze `Q`, `m`, `m̄`, and all spline bases at their full-data values across replicates (§3.1).

Some replicates will fail to converge. Record the failure rate; if above 5%, investigate rather than silently dropping.

### 7.2 Secondary: Conley HAC

Extract per-cell score contributions `g_i = ∂ℓ_i/∂Θ` via `jax.jacrev` in chunks. Sandwich:

```
V = H^{-1} ( Σ_i Σ_{j: d_ij < B} k(d_ij) g_i g_j^T ) H^{-1}
```

Bartlett kernel `k(d) = 1 - d/B`, bandwidth `B` from the Test 14 correlogram. Build the neighbour list with `scipy.spatial.cKDTree.query_pairs`. Accumulate in chunks over `i`; do not materialise the full pair list.

If block bootstrap and Conley disagree by more than a factor of 1.5 on any headline SE, report both and note the discrepancy.

---

## 8. Estimands to report

The primary results are **not** coefficients and **not** p-values. At n = 700,000 with spatial dependence, statistical significance is near-uninformative.

1. **Odds ratio per IQR of each neighbourhood component:** `exp(c_k · Δ_k)` where `Δ_k = IQR(φ̄̃_k)`. 28 (or 29) values with bootstrap CIs.
2. **Archetype-conditional neighbourhood effects.** For each reference own-state archetype `a` (tumour, stromal, immune centroids in the projected basis), the modified neighbourhood coefficient vector is `c + B^T φ̃_a`. Report `exp((c + B^T φ̃_a)_k · Δ_k)` as a **component × archetype table** with CIs. This is the headline result and the direct generalisation of the discrete OSDR interaction matrix.
3. **`B̂` and its SVD.** Leading singular values with CIs; leading left and right singular vectors annotated with the PhiSpace components that load most heavily.
4. **Fold change `exp(ê^δ)`** with CI. Sanity-check against the reference scRNA-seq: the ratio of mean Tirosh-set expression in S/G2M-called cells versus G1-called cells, restricted to panel genes and depth-matched. These should be the same order of magnitude. A large discrepancy means the components are not what they are being called.
5. **`mean(π̂)` overall and per archetype**, with the caveat that this is a snapshot prevalence, not a rate.
6. **Oxygen coefficient `ψ`** as an odds ratio per IQR of the oxygen field.
7. **Normalised entropy** of the responsibilities:
   `E = 1 - (1/(n log 2)) Σ_i [w_i log w_i + (1-w_i) log(1-w_i)]`
   Report alongside a histogram of `ŵ_i`. **If `E < 0.30`, the components are not separated and `ŵ` MUST NOT be presented as P(cycling).**
8. **Confound diagnostic.** Regress `ŵ_i` on `{log area, log N, log density, oxygen}` and separately on `φ̃_i`; report both R². If the nuisance R² substantially exceeds the state R², the model is a size-and-detection classifier. Report this number in every output, prominently, regardless of what it shows.

### 8.1 Rate-to-probability interpretation

For slowly varying rates, `P(S/G2M at snapshot) ≈ λ · T_{S/G2M}`, so a division rate can be recovered as `λ_i ≈ π̂_i / T_{S/G2M}` with `T_{S/G2M}` taken from literature for the relevant cell type. Report `π̂` as the primary quantity and the rate conversion as a clearly-labelled downstream transformation with its assumption stated. Do not bury the assumption.

---

## 9. Prohibited actions

The agent MUST NOT do any of the following without explicit new instruction:

1. Add zero inflation or a hurdle component. The residual pattern that motivated this model (over-prediction at 0, under-prediction at 1–2) is the *opposite* of zero inflation, and both mixture components already place mass at zero.
2. Use a likelihood-ratio test to choose the number of components. The null is on the parameter-space boundary and the LRT has a non-standard distribution.
3. Use AIC or BIC for rank or component selection. See §6.11.
4. Report inverse-Hessian standard errors.
5. Put `φ` or `φ̄` into the component rates.
6. Put `log N` or `area` into the mixing weight.
7. Apply SpaNorm, or any spatially-smooth normalisation, to `y_count`. Proliferating cells genuinely produce more RNA; a spatially smooth library-size component would absorb the signal being estimated.
8. Include Tirosh or blocklisted genes in `N`.
9. Use `archetype` as a covariate in the likelihood.
10. Increase `df(h)` in the mixing weight beyond 3.
11. Select between Configuration F and Configuration V by fit.
12. Report `U` or `V` rather than `B`.
13. Drop non-converged bootstrap replicates without recording the rate.
14. Present Stage D results as the primary specification.

---

## 10. Deliverables

```
mixture/
  config.yaml            # all hyperparameters; no magic numbers in code
  data.py                # loading + §1.2 assertions
  basis.py               # Q, centring, splines — built once, frozen, serialised
  model.py               # log-likelihood, JAX
  em.py                  # Stage A/B EM, monotonicity assertion
  fit.py                 # L-BFGS, multi-start, convergence reporting
  inference.py           # block bootstrap, Conley
  diagnostics.py         # entropy, rootogram, randomised PIT, Moran's I
  simulate.py            # generators for Tests 01, 02, 03, 07
  estimands.py           # §8 tables
tests/
  test_01_recovery.py ... test_14_spatial_residual.py
run/
  stage_a.py  stage_b.py  stage_c.py  stage_d.py
outputs/
  validation_report.md   # all 14 tests, numbers not just pass/fail
  results_stage_b.md     # §8 estimands
  fitted_w.parquet       # cell_id, w_hat, pi_hat, lambda_0, lambda_1, rho
  figures/
```

Every fit writes: git commit hash, config hash, log-likelihood, gradient norm, iteration count, multi-start agreement fraction, wall time, and the §8.7 entropy and §8.8 confound R² values. A result reported without those last two is not usable.

---

## 11. Known limitations to carry into the write-up

- No biological replicate. Spatial block CV bounds within-section variability only.
- Oxygen field uncertainty is not propagated.
- Segmentation quality is not directly measured in this build (no MisTIC score). The additive background with a density spline is a coarse proxy. If a per-cell segmentation quality score becomes available, it enters `log ρ_i` linearly and Test 13 should be re-run.
- The latent class has no external anchor within this dataset. Test 08 provides internal cross-validation; genuine external validation requires the planned orthogonal protein readout (PCNA/Ki67 IF on the CytAssist section).
- The proportional-rates restriction is an assumption, tested only in Stage D.
