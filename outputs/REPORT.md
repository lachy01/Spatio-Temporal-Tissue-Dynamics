# A concomitant-variable Poisson mixture for cell-cycle state

**Xenium Prime 5K breast cancer FFPE — 678,885 cells.**

The scientific target is `P(cell i is in S/G2M)` as a function of the cell's own
transcriptional state, its spatial neighbourhood, local oxygen and local density.

A negative-binomial regression on cell-cycle transcript counts estimates
`E[count]`, which then needs an arbitrary threshold to become a probability. This
model removes that step: the mixing weight **is** the probability, and it carries
the covariates of interest directly.

Everything below is reproduced in `notebooks/mixture_model.ipynb`, which loads the
fitted results rather than refitting and shows the code behind each number.

---

## 1. Headline results

| estimand | estimate | 95% CI |
|---|---:|---|
| **Cycling fraction** `mean(pi)` | **11.60%** | [10.61%, 12.47%] |
| **Rate ratio** `exp(e^delta)` | **5.63** | [5.54, 5.74] |
| Normalised entropy | 0.683 | [0.658, 0.711] |
| `sigma_1(B)` — leading interaction | 1.364 | [1.194, 1.716] |
| `theta` — log library size | 1.036 | [1.014, 1.057] |
| `psi` — oxygen (perfusion) | +0.840 | [+0.511, +1.131] |

CIs are percentile intervals from a 300-replicate **spatial block bootstrap**:
the section is cut into ~250 contiguous tiles and replicates draw tiles with
replacement. Inverse-Hessian errors are forbidden here (spec 7) and would be far
too small — neighbouring cells share a microenvironment, so 678,885 cells carry
much less information than 678,885 independent ones.

### Which cell types cycle, and next to what

Log-odds of cycling at the mean state. **These sum to zero by construction** — the
PhiSpace scores are compositional, so only contrasts are identified.

| cell type | own `a` | neighbour `c` |
|---|---:|---:|
| Cancer Epithelial | **+2.98** | **+3.12** |
| T-cells | **+2.88** | +0.38 |
| B-cells | **+2.34** | −0.56 |
| PVL | −0.42 | +0.14 |
| Normal Epithelial | −0.57 | −0.68 |
| Plasmablasts | −0.63 | −0.97 |
| Endothelial | −1.07 | −0.76 |
| Myeloid | −1.47 | −0.07 |
| CAFs | **−4.05** | −0.60 |

Every own-state interval excludes zero. In the neighbourhood, only
`nbr[Cancer Epithelial]` is large: **being surrounded by tumour raises a cell's
odds of cycling beyond its own state.**

### The interaction — spec 8.2's headline estimand

Odds ratio per IQR of each neighbourhood component, conditional on own archetype.
42 of 81 entries have a CI excluding 1.

| own archetype | PVL | NormEp | **CancEp** |
|---|---:|---:|---:|
| Endothelial | 0.33 | 1.18 | **11.44** |
| CAFs | 0.61 | 1.03 | **7.58** |
| T-cells | 0.65 | 1.01 | **7.38** |
| Plasmablasts | **2.82** | 0.73 | 2.73 |
| Cancer Epithelial | **5.70** | 0.62 | **1.62** |

Two patterns invisible to a main-effects model:

**A tumour neighbourhood raises the odds for everything except tumour** — 7–11×
for stromal and immune own-states, but only 1.62× for a cancer cell itself.

**Cancer and plasmablast cells respond to PVL where others do not** — that column
is suppressive (0.33–0.65) for most own-states but 5.70× for Cancer Epithelial.

---

## 2. Validation

| test | criterion | result |
|---|---|---|
| 01 recovery | scalars within 3 MC SE; `‖B̂−B‖/‖B‖` < 0.15 | **PASS** — bias/SE 0.33–0.97; 0.106 |
| 02 null bilinear | null/alt `sigma_1` < 0.20 | **PASS** — 0.105 |
| 03 single-component null | components collapse | **PASS** at λ≥10 (entropy 0.084) |
| 04 gradient | max rel. error < 1e-5 | **PASS** — 2.3e-7 |
| 05 EM monotonicity | non-decreasing to 1e-8 | **PASS** — min step +2.3e-6 |
| 06 EM vs L-BFGS | loglik 6 s.f., params 4 s.f. | **PASS** — 4.7e-13, ≤4.3e-6 |
| 07 size confound | fold < 1.5 **and** entropy < 0.25 | entropy **PASS** (0.141); fold **FAIL** (1.93) |
| **08 held-out genes** | Δdeviance ≥ 10× permuted null | **PASS** — 23,000–27,700× |
| 09 marginal fit | mixture beats one component | **PASS** — RMS 13.5 → 2.4 |
| 10 subsample stability | rate ratios stable; no drift in `mean(pi)` | fold **PASS**; `mean(pi)`/`sigma_1` inflated below 200k |
| 11 rank selection | clear maximum or plateau | **rank 2**, confirmed three ways |
| 12 theta bracket | report F and V side by side | agree to 0.7% on fold |
| 13 density-in-weight | change in `c` when `h()` dropped | tumour effect **0.0%**; three immune coefficients flagged |
| 14 spatial residual | Moran's I vs the NB model | **0.0142**, correlogram <0.01 by 25 µm |

### Test 08 — the only external check on what `w` means

Everything else establishes the model is *internally* sound. This asks whether the
latent class is **proliferation**: `w` fitted on half the Tirosh signature predicts
the **other half**, beyond depth/area/density, at **23,000–27,700×** a spatially
block-permuted null. Coefficient +1.61 to +1.86, positive on all five
phase-stratified splits.

Dispersion absorbed from misspecification has no reason to predict held-out
cell-cycle genes. This is the evidence the latent class is what we call it.

### Is `w` a size-and-detection classifier? (spec 8.8)

No. R² of `w` on the nuisance covariates {log area, log N, log density, oxygen}
is **0.023**; on cell state, **0.092**. State explains **4.0×** more.
`corr(w, log N) = +0.036`.

### Test 07 fails its fold criterion — how to read that

A two-component Poisson mixture is also a **model for overdispersion**. When the
rate model cannot capture part of the true rate, two components absorb the
leftover variance. So the 1.93× fold on class-free data is the width the model
needed, not a biological difference.

Three things bound the threat to the real result:

1. The real fold (5.63) is **2.9× that floor**.
2. It is **invariant to the area spline** — 5.581 / 5.577 / 5.581 / 5.579 at
   df = 4/8/12/16 — while the floor moves (2.84 / 1.84 / 1.89 / 5.28). A size
   artefact would be absorbed by a more flexible spline; this is not.
3. The confound produces a high fold **without confident assignment**: entropy
   0.141 against the real data's 0.683. Spec 8.7's guardrail fires on the
   confounded data and not on the real data.

Argument 1 depends on how severe I made the simulated misspecification; 2 and 3
do not, and are the ones to rely on.

---

## 3. Model selection and sensitivity

**Penalty.** Spec 3.2's ridge (1e-4 on `U`,`V` only) is far too weak: on
class-free data the fit reaches `‖a‖ = 57.5` and `sd(eta) = 203`, pushing `pi` to
0 or 1 for 98.6% of cells. The penalty now covers all six weight blocks.
Held-out log-likelihood **cannot select it** — `pi` is a prior on a latent class,
so errors are damped in the marginal density and CV prefers λ=0 monotonically
(paired t −3.1 to −17.3), the setting where the weight model diverges. Selected
instead as the smallest λ that converges and puts `sigma_1` on its plateau: **10**.

**Rank.** 2. Confirmed three ways: held-out log-likelihood (rank 2 beats rank 1 at
paired t = +4.9; ranks 3 and 5 are numerically identical to 2), and the fitted `B`
has exactly two non-zero singular values.

**Area spline df = 8**, raised from the spec's 4 on evidence: the Test 07 confound
floor falls from 2.84 to 1.84 while the real fold change is unmoved.

**Library size (spec 2.2's bracket).** F (θ ≡ 1) and V (θ free) agree to 0.7% on
fold and 9.5% on cycling fraction. Freed, θ = **1.036** — essentially the offset's
assumed 1.0, so the fixed offset is defensible. This differs from the NB division
model's 0.70 exponent because `N` here excludes the blocklist: there,
`log_total_counts` included the outcome genes, letting the outcome inflate its own
offset.

**Stage C (additive background) — rejected.** It gains +475 log-likelihood but
fails spec 3.4's own stability test: `h()`'s swing across the density IQR is cut
by a factor of 2.35, so the biological and technical density terms trade off. And
anchored to the control probes the background should be **0.014%** of the mean
count against the free estimate's **2.4%** — 177× smaller. Control counts are zero
in 99.9% of cells. The +475 is the background eating density signal, not
contamination being captured.

**Stage D.** Density definition immaterial: fixed-k kNN moves the fold change by
0.0%, cycling fraction by 0.3%, `sigma_1` by 3.1%, despite the weight spline
changing substantially. The proportional-rates relaxation is **not implemented** —
it needs a second θ in the model and spec 2.2 makes that restriction load-bearing
for identifiability.

---

## 4. Limitations

- **One section, no biological replicate.** The block bootstrap bounds
  *within*-section variability only.
- **Test 07's fold criterion fails**; carry the 1.93 floor alongside the 5.63.
- **Test 09 is not uniform** — Cancer Epithelial improves 77%, but T-cells worsen
  46% and B-cells 69%. Spec 6.9 only guards Cancer Epithelial.
- **Three immune neighbourhood coefficients are density-sensitive** (B-cells,
  T-cells, Myeloid; Test 13). The tumour neighbourhood effect is not.
- **`B` is biased upward below ~200,000 cells** (`sigma_1` 1.90 / 1.46 / 1.36),
  and subsample fits are severely multimodal (1 of 20 restarts finds the best
  optimum, against 95% at the full sample).
- **The proportional-rates restriction is untested** — the one assumption in the
  specification that has not been challenged.
- **The oxygen field failed its HIF-target check**, so `psi` reads as vessel
  proximity / perfusion, not hypoxia; its uncertainty is not propagated.
- **`pi` is a snapshot prevalence, not a rate.** Converting needs
  `lambda_i ≈ pi_i / T_S/G2M` with the phase duration from literature.

---

## 5. Where things live

| | |
|---|---|
| `notebooks/mixture_model.ipynb` | every result, with the code that produced it |
| `src/tissue_dynamics/mixture/` | the package |
| `hpc/` | Spartan submission (`README.md` has the run order) |
| `outputs/generated/` | machine-written reports and CSVs |
| `outputs/hpc/` | raw per-task JSON from every SLURM array |
| `outputs/figures/` | figures |
| `outputs/archive/` | superseded reports, including retracted conclusions |
