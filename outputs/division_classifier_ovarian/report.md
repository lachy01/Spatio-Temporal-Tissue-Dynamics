> **Ovarian section (`MIX_DATASET=ovarian`), run 2026-10-15/16.** Stages 1-4 of the
> division classifier comparison on the ovarian Xenium section, with the breast
> pipeline unchanged. The interpretive rule, pre-stated expectations and decisions
> below were written for the breast section and are copied as the pipeline writes
> them; X1-X3 are judged here as written. Stage 2's and Stage 3's V5 compare the
> fold sizes with those stored with the fold map (no ovarian mixture CV exists);
> Stage 3's held-out "Spartan CV" column is empty for the same reason. Stages 5
> (Wu breast reference, receptor biology) and 6-7 were not run.

# Division classifier comparison

Spec: `instructions/division_classifier_comparison_spec.md`. This file is the run
log: pre-stated rules and expectations first, then one status note per stage.

## Interpretive rule (written before any results)

- Each LOG arm is trained directly on its own label, so it is expected to win
  label-scale scores on that label.
- A label-scale win for a LOG arm is not evidence that it infers division better.
- NB and MIX can beat a LOG arm on the label scale only if their count likelihood
  carries information that binarisation discards.
- Section 7 decides which approach recovers the cycling state, including whether
  the OSDR pipeline (LOG-O) or a score pipeline (LOG-S) matches MIX.
- Label scale, P(L = 1 | x), and state scale, P(Z = 1 | x), are never compared
  with each other.

## Pre-stated expectations (spec section 11; recorded before Stage 4, not edited)

- **X1.** Each LOG arm has the best label-scale log loss on its own label.
- **X2.** NB-c's calibration-in-the-large changes sign across the threshold grids:
  over-prediction where c_i falls in 5 to 10, under-prediction where c_i falls in
  15 to 30. This follows the NB rootogram wave.
- **X3.** MIX is better calibrated than NB-c across the grids, but under-predicts
  at the strictest thresholds (its 30+ tail shortfall).
- **X4.** NB-d moves predictions mainly in Cancer Epithelial.
- **X5.** Disagreement between NB and MIX concentrates in Cancer Epithelial and
  Plasmablasts.
- **X6.** BASE-O's null call rate at Td = 0.5 is several percent, not near zero.
  Where mu0 is below 1, a count of 1 or 2 already clears the threshold, so OSDR's
  [0, 1] robustness range will not transfer.
- **X7.** Among cells with y = 0, the SCORE value has non-trivial spread, coming
  entirely from the control-gene term.
- **X8.** Under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from
  1 than MIX, with the largest distortion in small, low-depth cell types.

## Decisions and deviations from the spec

| item | spec | adopted | why |
| --- | --- | --- | --- |
| NB arms' depth term | fixed offset, theta = 1 | fixed offset (theta = 1) in every count arm, MIX included, since 2026-10-06 (theta free before, for parity with the mixture of that time) | parity with MIX, so arms differ only in likelihood. Decided by the user. |
| technical null depth term | fixed offset | fixed offset, by `model.technical`: log Nt (MisTIC-weighted N) plus theta_m m under `mistic` (production since 2026-10-12, decided by the user), raw log N under `plain`; the Stage 1 section gives the fitted form | labels are definitions, rebuilt whenever the null changes |
| NB-d dispersion | type indicators | **log alpha_i = gamma0 + phi_i^T gamma**, 8-dim projected state | matches the earlier dispersion submodel the spec refers to. Decided by the user. |
| Wu et al. reference; nucleus-only matrix | used in 5.2 and stage 7 | code paths built; **runs wait for the user to supply the files** | neither is on this machine. Decided by the user. |
| stage gating | stop after every stage | review after stages 1, 3 and 6; stop on any V-test failure | decided by the user |
| scanpy `score_genes` pool | pool excludes Tirosh and blocklist | pool = panel minus blocklist, plus the Tirosh genes for binning only, `ctrl_as_ref=False` | scanpy 1.12 raises a KeyError if scored genes are absent from the pool; `ctrl_as_ref=False` still keeps them out of the control draw |
| OSDR's SD (BASE-O) | check Supp. Fig. 1D | **per-type s+ primary**; global s+ as the sensitivity BASE-O-global | Supp. Fig. 1D was not supplied, but OSDR's code settles it: `tdm.preprocess.ki67.is_dividing` (github.com/JonathanSomer/osdr, commit 7d04970) sets one cutoff per cell type, `Td * sigma_type + Tn`, with `sigma_type = np.std(x[x > Tn] - Tn)` (ddof 0), and the Danenberg labels are built through it (Tn = 0.5, Td = 0.5). Consistent with the Methods ("to correct for different scaling and division rates between cell lines") and with the function's docstring ("The transformed values should have similar distributions accross different cell types"). |
| primary alpha_B (BASE-T) | 0.001 | **0.05** | at 0.001 the label called 0.18% of cells (9 positives among CAFs); 0.05 has PPV 0.86 under MIX and sits in OSDR's range of label rates. Full grid still reported. Decided by the user after Stage 1. |
| SCORE threshold | robust half-normal, p < 0.001 | unchanged, literal | its zero-count dependence is reported as a finding. Decided by the user. |
| state covariates | -- | **log1p-normalised PhiSpace** (`phispace_scores_major_lognorm.csv`), cells with N >= 10 | the first run (SpaNorm-based scores) stopped in Stage 2: those scores were ~86% depth, an artefact shown by a marker-gene test. That run's report is archived in `archive_spanorm/report.md`. Shallow cells (N < 10, 17.4%) are dropped as analysis cells and left out of neighbour averages; their labels defaulted to Endothelial. Decided by the user. |
| oxygen field source | -- | new major Endothelial score, typed cells | the old source (SpaNorm minor run) gave a field with AUROC 0.515 for marker endothelium against marker-negative cells; the new field 0.788 |
| LOG-O regression | "OSDR's pipeline" | matched covariates, pooled over types | OSDR fits a separate logistic regression per cell type on neighbour counts within 80 um (`tdm.model.Model`, "two models per cell_type"). LOG-O reproduces OSDR's **label** exactly in translation, but its regression is the shared weight block, for covariate parity. It is OSDR's labelling with our regression, not OSDR end to end. |


## Stage 1 -- labels (built and frozen 2026-10-15 23:47)

395,522 analysis cells. Cell type is the argmax of the raw PhiSpace scores, with Granulosa Cells and Fallopian Tube Epithelium folded into Other, agreeing with the stored label for 100.00% of cells.

**Outcome and library size.** y recomputed from the count matrix with the
mixture's 53-gene Tirosh list equals the table's `y_count` for every
cell; N (excluding the Tirosh genes and the 530-gene blocklist) equals the
table's `N` for every cell.

### Technical null (spec 2.1)

mu0 = Nt e^b0 + theta_m m (MisTIC; `model.technical: mistic`): Nt the MisTIC-weighted N (Nt_nonB) as a fixed offset, m the expected count of misassigned Tirosh transcripts (m_cyc), theta_m = exp(eta) estimated (**1.7912**, SE of eta 0.0197); no area term. NB negative log-likelihood minimised directly (scipy L-BFGS-B). Robust fit by iterative trimming at P(Y >= y) < 0.001; stopped on
**tolerance** after 2 fits. Final alpha0 = **0.9851**.

| iteration | n_fit | alpha0 | flagged | flagged_frac | changed | converged |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 395522 | 0.9923 | 175 | 0.0004425 | 175 | True |
| 2 | 395347 | 0.9851 | 184 | 0.0004652 | 9 | True |

### Label rates

| cell type | cells | BASE-T 0.05 | BASE-T 0.01 | BASE-T 0.001 | BASE-T 0.0001 | BASE-O 0.5 | BASE-O-global 0.5 | BASE-O-neg 0.5 | SCORE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 395522 | 0.0537 | 0.008331 | 0.0004652 | 3.287e-05 | 0.1951 | 0.1839 | 0.2621 | 0.2263 |
| Tumour | 181539 | 0.1125 | 0.01758 | 0.000975 | 7.161e-05 | 0.3226 | 0.3277 | 0.5239 | 0.3829 |
| Fibroblasts | 86585 | 0.002622 | 0.0003003 | 1.155e-05 | 0 | 0.07668 | 0.05157 | 0.05149 | 0.08272 |
| Endothelial Cells | 21595 | 0.002639 | 0.0002778 | 0 | 0 | 0.07057 | 0.04797 | 0.02505 | 0.07103 |
| Pericytes | 13280 | 0.005648 | 0.0006024 | 7.53e-05 | 0 | 0.0869 | 0.07154 | 0.0314 | 0.0991 |
| Smooth Muscle Cells | 47877 | 0.0008981 | 8.355e-05 | 0 | 0 | 0.08388 | 0.05144 | 0.00142 | 0.07655 |
| Macrophages | 32672 | 0.008692 | 0.0007346 | 6.121e-05 | 0 | 0.1192 | 0.09559 | 0.07891 | 0.1413 |
| T & NK Cells | 8421 | 0.01591 | 0.004156 | 0.0003563 | 0 | 0.1393 | 0.1279 | 0.04441 | 0.1793 |
| Other | 3553 | 0.001407 | 0.0002815 | 0 | 0 | 0.05263 | 0.03406 | 0.03349 | 0.05629 |

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

| N decile | N median | mu0 median | c^T median @0.05 | mean alpha_i @0.05 | c^T median @0.01 | mean alpha_i @0.01 | c^T median @0.001 | mean alpha_i @0.001 | c^T median @0.0001 | mean alpha_i @0.0001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21 | 0.2937 | 3 | 0.02496 | 4 | 0.004964 | 5 | 0.0004956 | 7 | 5.055e-05 |
| 2 | 40 | 0.562 | 3 | 0.03183 | 5 | 0.006417 | 7 | 0.0006341 | 9 | 6.369e-05 |
| 3 | 62 | 0.8668 | 4 | 0.03479 | 6 | 0.0069 | 9 | 0.0007021 | 12 | 6.979e-05 |
| 4 | 91 | 1.281 | 6 | 0.03882 | 8 | 0.007561 | 12 | 0.0007577 | 16 | 7.627e-05 |
| 5 | 132 | 1.869 | 7 | 0.04064 | 11 | 0.008084 | 17 | 0.0008113 | 22 | 8.081e-05 |
| 6 | 185 | 2.639 | 10 | 0.04232 | 15 | 0.008573 | 22 | 0.0008561 | 29 | 8.57e-05 |
| 7 | 249 | 3.56 | 13 | 0.0443 | 19 | 0.008811 | 28 | 0.0008833 | 37 | 8.84e-05 |
| 8 | 324 | 4.644 | 16 | 0.04547 | 24 | 0.009087 | 36 | 0.0009079 | 47 | 9.06e-05 |
| 9 | 422 | 6.066 | 20 | 0.04632 | 30 | 0.009263 | 45 | 0.0009268 | 60 | 9.272e-05 |
| 10 | 612 | 8.884 | 28 | 0.04746 | 43 | 0.009486 | 65 | 0.0009493 | 86 | 9.492e-05 |

### BASE-O (spec 2.3)

Primary BASE-O scales the residual within each cell type, as OSDR's own code
does (`is_dividing`: one cutoff per type, `Td * sigma_type + Tn`, ddof 0).
Per-type s+: Tumour 1.132, Fibroblasts 0.615, Endothelial Cells 0.630, Pericytes 0.767, Smooth Muscle Cells 0.533, Macrophages 0.746, T & NK Cells 0.942, Other 0.626.
BASE-O-global uses one s+ = 1.0763 for all cells.
Negative-codeword floor (BASE-O-neg, decoding noise only, no spillover):
rho = 1.455e-05 per um^2 over 609 negative-control codewords,
|T| = 53, mean b_i = 8.85e-05 against mean mu0 = 3.1652;
s+ = 776.0469.

Caveat: per-type scaling raises the effective threshold in types with more
cycling cells, so it partly removes between-type proliferation differences.
That is a property of OSDR's construction, inherited here.

### SCORE (spec 2.4)

scanpy `score_genes`, 53 genes, n_bins 25, ctrl_size 50, control pool
4,571 genes (panel minus blocklist), log1p(X s / N) with s = median
N = 157. Robust null: KDE mode -0.0207, left-half
RMS sigma 0.0091, threshold 0.0074 (upper-tail p <
0.001).

| stratum | n | kappa vs BASE-T | kappa vs BASE-O | SCORE rate | BASE-T rate | BASE-O rate |
| --- | --- | --- | --- | --- | --- | --- |
| all | 395522 | 0.325 | 0.8749 | 0.2263 | 0.0537 | 0.1951 |
| Tumour | 181539 | 0.3392 | 0.8656 | 0.3829 | 0.1125 | 0.3226 |
| Fibroblasts | 86585 | 0.05665 | 0.8456 | 0.08272 | 0.002622 | 0.07668 |
| Endothelial Cells | 21595 | 0.0669 | 0.8613 | 0.07103 | 0.002639 | 0.07057 |
| Pericytes | 13280 | 0.0982 | 0.8661 | 0.0991 | 0.005648 | 0.0869 |
| Smooth Muscle Cells | 47877 | 0.02146 | 0.8598 | 0.07655 | 0.0008981 | 0.08388 |
| Macrophages | 32672 | 0.1012 | 0.8406 | 0.1413 | 0.008692 | 0.1192 |
| T & NK Cells | 8421 | 0.1378 | 0.8166 | 0.1793 | 0.01591 | 0.1393 |
| Other | 3553 | 0.04616 | 0.8497 | 0.05629 | 0.001407 | 0.05263 |

**Zero-count diagnostic.**

| cells with y = 0 | score sd, y = 0 | score sd, all cells | score IQR, y = 0 | score 1st-99th pct, y = 0 | SCORE call rate among y = 0 | share of SCORE calls with y <= 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 160753 | 0.00887 | 0.03341 | 0.01303 | -0.041 to -0.003 | 0 | 0.11 |

### Validation

- **V3** survivor functions vs scipy: max |diff| NB 1.63e-14, Poisson
  0.00e+00 (< 1e-12): **pass**.
- **V10** BASE-O discrete equivalence, every cell, every Td, all three
  variants: **pass**.
- **V2** null calibration, end to end (simulate from the null, refit the robust
  null, rebuild the labels): **pass**.

| alpha_B | label rate (refit) | mean alpha_i | relative error | pass (+/-30%) | label rate (frozen c) | mean alpha_i (frozen) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.04072 | 0.03956 | 0.02936 | pass | 0.03944 | 0.03965 |
| 0.01 | 0.008475 | 0.007873 | 0.07642 | pass | 0.00807 | 0.007907 |
| 0.001 | 0.0009532 | 0.0007894 | 0.2074 | pass | 0.0008874 | 0.0007916 |
| 0.0001 | 9.608e-05 | 7.878e-05 | 0.2196 | pass | 8.596e-05 | 7.927e-05 |

BASE-O call rate on null data (reported; no pass criterion):

| Td | null call rate (rebuilt) | null call rate (frozen c) | observed call rate |
| --- | --- | --- | --- |
| 0 | 0.3562 | 0.3541 | 0.2956 |
| 0.25 | 0.2824 | 0.29 | 0.2377 |
| 0.5 | 0.221 | 0.2383 | 0.1951 |
| 0.75 | 0.1722 | 0.1949 | 0.1622 |
| 1 | 0.1337 | 0.1599 | 0.1344 |

Stage 1 complete in 0.3 min. **Review point** (stages 1,
3 and 6 pause for review).


## Stage 2 -- NB refits (2026-10-16 00:00)

**The shared model** (decisions recorded in the mixture's specification doc,
2026-10-06 to 10-07, and 2026-10-12 for the MisTIC technical terms; all by the user): log N is an offset
(theta = 1) in every count arm, MIX included; the oxygen term is dropped; the
neighbour block is split into contact (0-15 um) and regional (15-80 um) means;
both state blocks are whitened; B is penalised by lambda ||UV'||_F^2 with the
balance term tau ||U'U - V'V||_F^2; lambda was selected on null-calibrated
Test 03 and the rank by spatial CV. With theta = 1 in MIX too, the spec's offset
mismatch (section 1) no longer arises. The labels are unaffected: they depend
only on y, the library size and the technical null.

NB-c and NB-d, matched to the mixture's covariates and penalty: the shared weight
block (8 own and 16 neighbour coordinates,
rank 3), log N as an offset, the MisTIC-weighted library size and theta_m m in the mean (no area term), weight_ridge 300 x n /
678,885 on every eta coefficient (frobenius form on B) and
the balance term at tau = 1. NB-d adds log alpha_i = gamma0 +
phi_i' gamma on the projected own state (gamma not penalised). Optimiser:
L-BFGS, then Newton on the exact Hessian with absolute-value eigenvalues and a
backtracking line search.

### V5 -- fold identity

250 k-means blocks (MiniBatchKMeans, seed 0) dealt into 5 folds (seed 0); map
loaded from folds.parquet, sha256 `8ad3a96ff738c24d...`. The mixture CV did not store its
cell-to-fold map, so identity is checked on the fold sizes it recorded: all five
match to the cell, which a different partition would not do by chance.
**pass**

| fold | n_train here | n_test here | n_test, mixture CV (Spartan) | match |
| --- | --- | --- | --- | --- |
| 0 | 315560 | 79962 | 79962 | yes |
| 1 | 318459 | 77063 | 77063 | yes |
| 2 | 313974 | 81548 | 81548 | yes |
| 3 | 315785 | 79737 | 79737 | yes |
| 4 | 318310 | 77212 | 77212 | yes |

### V6 -- parity

All 6 arms (MIX, NB-c, NB-d, LOG-T, LOG-O, LOG-S) share the same
141 eta columns in the same order. The shared `eta_state`
reproduces the production mixture's logit to 3.6e-15 on every cell. **pass**

### V1 -- convergence

Criterion: |f_t - f_(t-1)| < 1e-9 |f_(t-1)| and ||grad||_inf < 1e-5, on the
summed penalised objective. **pass**: 14
of 14 fits converged. sd(eta) > 20 flagged: **none**.

| arm | data | seed | n | loglik | objective | grad_inf | L-BFGS iters | Newton iters | converged (V1) | attempts | theta | alpha | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-c | all cells | 0 | 395522 | -7.044e+05 | 7.045e+05 | 5.973e-09 | 503 | 16 | pass | 1 | 1 | 0.5793 | 0.3937 | 0.3365 | 0.2464 | 0.7595 | no | 1.083 |
| NB-c | all cells | 1 | 395522 | -7.044e+05 | 7.045e+05 | 9.202e-08 | 495 | 15 | pass | 1 | 1 | 0.5793 | 0.3937 | 0.3365 | 0.2464 | 0.7595 | no | 1.021 |
| NB-d | all cells | 0 | 395522 | -7.039e+05 | 7.04e+05 | 4.224e-10 | 484 | 16 | pass | 1 | 1 | 0.384-0.829 | 0.3948 | 0.3393 | 0.2514 | 0.7605 | no | 1.204 |
| NB-d | all cells | 1 | 395522 | -7.039e+05 | 7.04e+05 | 9.333e-08 | 686 | 10 | pass | 1 | 1 | 0.384-0.829 | 0.3948 | 0.3393 | 0.2514 | 0.7605 | no | 1.078 |
| NB-c | train fold 0 | 0 | 315560 | -5.631e+05 | 5.631e+05 | 2.455e-08 | 689 | 10 | pass | 1 | 1 | 0.5797 | 0.404 | 0.3298 | 0.2469 | 0.7595 | no | 0.7575 |
| NB-c | train fold 1 | 0 | 318459 | -5.704e+05 | 5.705e+05 | 1.185e-06 | 494 | 13 | pass | 1 | 1 | 0.5848 | 0.3923 | 0.3472 | 0.2526 | 0.7486 | no | 0.7782 |
| NB-c | train fold 2 | 0 | 313974 | -5.48e+05 | 5.48e+05 | 3.231e-08 | 644 | 8 | pass | 1 | 1 | 0.5826 | 0.3874 | 0.343 | 0.2409 | 0.7591 | no | 0.675 |
| NB-c | train fold 3 | 0 | 315785 | -5.543e+05 | 5.544e+05 | 1.005e-07 | 668 | 14 | pass | 1 | 1 | 0.5695 | 0.3886 | 0.333 | 0.2504 | 0.7661 | no | 0.8765 |
| NB-c | train fold 4 | 0 | 318310 | -5.814e+05 | 5.815e+05 | 2.506e-08 | 501 | 10 | pass | 1 | 1 | 0.5760 | 0.3937 | 0.3352 | 0.2463 | 0.7655 | no | 0.6865 |
| NB-d | train fold 0 | 0 | 315560 | -5.627e+05 | 5.628e+05 | 5.045e-11 | 665 | 9 | pass | 1 | 1 | 0.388-0.831 | 0.4046 | 0.3329 | 0.2523 | 0.7604 | no | 0.8182 |
| NB-d | train fold 1 | 0 | 318459 | -5.7e+05 | 5.701e+05 | 1.624e-07 | 487 | 14 | pass | 1 | 1 | 0.408-0.841 | 0.3916 | 0.3512 | 0.257 | 0.7495 | no | 0.9481 |
| NB-d | train fold 2 | 0 | 313974 | -5.476e+05 | 5.477e+05 | 2.704e-07 | 652 | 7 | pass | 1 | 1 | 0.375-0.815 | 0.3898 | 0.3453 | 0.2463 | 0.76 | no | 0.7471 |
| NB-d | train fold 3 | 0 | 315785 | -5.539e+05 | 5.54e+05 | 3.785e-08 | 652 | 8 | pass | 1 | 1 | 0.370-0.819 | 0.3907 | 0.3348 | 0.2542 | 0.767 | no | 0.807 |
| NB-d | train fold 4 | 0 | 318310 | -5.81e+05 | 5.811e+05 | 6.657e-08 | 628 | 10 | pass | 1 | 1 | 0.374-0.832 | 0.3946 | 0.3382 | 0.2518 | 0.7667 | no | 0.8589 |

`run/compare_nb_mixture.py`'s matched NB is unpenalised, so it is not the
same model as NB-c, which matches the mixture's penalty.

**Uniqueness.** Two seeds per arm on all cells:

| arm | |objective diff| | max |B diff| | max |a diff| | |theta diff| |
| --- | --- | --- | --- | --- |
| NB-c | 3.49e-10 | 2.49e-13 | 6.83e-15 | 0 |
| NB-d | 1.16e-10 | 2.68e-13 | 5.05e-15 | 0 |

**NB-d against NB-c.** Log-likelihood -703,936.16 against -704,418.48, a gain of
482.3 for 8 parameters (LR statistic 965). The
dispersion depends on state.

For reference, the mixture's data log-likelihood on the same cells is
-741,977.75. As in the main write-up, the NB fits the per-cell counts better;
the label-scale comparison begins in Stage 3.

Out-of-fold NB predictions (log mu and log alpha, per held-out cell) are saved
in `oof_nb.parquet` for Stage 3. Stage 2 complete in 12 min.


## Stage 3 -- out-of-fold fits (2026-10-16 00:16)

Every arm fitted on four folds and predicted on the fifth, on the stored
partition. NB-c and NB-d are Stage 2's fold fits. MIX is refitted per training
fold (penalty scaled to training size) from the full-data point estimate, to
the mixture's convergence criteria. LOG-T, LOG-O and LOG-S regress the primary
labels (LT_a05, LO_td050, LS) on the shared weight
block plus b0 + theta log N (log N the MisTIC-weighted Nt under `technical: mistic`; plus s(log A)
under `technical: area`), with the mixture's penalty term for
term. Count arms are pushed through every threshold on the alpha_B and Td
grids: NB p = P(Y >= c); MIX p = pi P1(Y >= c) + (1 - pi) P0(Y >= c), with the
prior pi.

### V5 -- fold identity

Map sha256 `8ad3a96ff738c24d...`, equal to the stored map's; Stage 2's
out-of-fold NB predictions use the identical assignment; fold sizes match the
Spartan rank CV on every fold. **pass**

### MIX fold fits

All five converged: **yes**. Each fold's held-out
log-likelihood per cell agrees with the Spartan CV fit for the same fold (rank
3, lambda 300) to nan, so starting
from the full-data estimate did not change where the fits end.

| data | converged | iterations | grad_inf | held-out loglik / cell | Spartan CV, same fold | e^delta | mean pi (test) | sd(eta) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train fold 0 | pass | 2305 | 2.684e-05 | -1.854 | nan | 1.745 | 0.1791 | 2.896 | 1.45 |
| train fold 1 | pass | 1829 | 2.279e-05 | -1.837 | nan | 1.742 | 0.1622 | 2.794 | 1.241 |
| train fold 2 | pass | 1917 | 3.18e-05 | -2.024 | nan | 1.755 | 0.2086 | 2.978 | 1.189 |
| train fold 3 | pass | 3318 | 3.647e-05 | -1.981 | nan | 1.747 | 0.1996 | 2.923 | 2.01 |
| train fold 4 | pass | 1492 | 1.163e-05 | -1.671 | nan | 1.736 | 0.1483 | 3.008 | 1.165 |

### LOG fits and V7 -- separation

Criterion: sd(eta) <= 20 at the production ridge for every LOG fit.
**pass** (largest sd(eta) 1.869);
all converged: **yes**.

| arm | data | converged | iterations | grad_inf | theta (log N) | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 (V7) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOG-T | all cells | pass | 577 | 4.802e-10 | 0.8554 | 0.7489 | 0.8185 | 0.4806 | 1.835 | no | 0.4167 |
| LOG-T | train fold 0 | pass | 527 | 4.914e-09 | 0.8757 | 0.7712 | 0.7808 | 0.468 | 1.854 | no | 0.353 |
| LOG-T | train fold 1 | pass | 517 | 6.219e-11 | 0.8359 | 0.744 | 0.8279 | 0.4853 | 1.796 | no | 0.317 |
| LOG-T | train fold 2 | pass | 562 | 1.312e-08 | 0.8488 | 0.7523 | 0.852 | 0.478 | 1.869 | no | 0.3283 |
| LOG-T | train fold 3 | pass | 644 | 5.894e-08 | 0.84 | 0.7188 | 0.8163 | 0.4818 | 1.83 | no | 0.3658 |
| LOG-T | train fold 4 | pass | 558 | 2.045e-10 | 0.8789 | 0.7576 | 0.8257 | 0.5039 | 1.86 | no | 0.3332 |
| LOG-O | all cells | pass | 545 | 2.159e-06 | 0.2279 | 0.4916 | 0.4694 | 0.4748 | 0.9886 | no | 0.4808 |
| LOG-O | train fold 0 | pass | 635 | 5.942e-08 | 0.2356 | 0.5007 | 0.4574 | 0.463 | 0.9885 | no | 0.599 |
| LOG-O | train fold 1 | pass | 904 | 7.086e-06 | 0.2297 | 0.4946 | 0.4805 | 0.4933 | 0.9801 | no | 1.128 |
| LOG-O | train fold 2 | pass | 879 | 1.364e-08 | 0.2015 | 0.4743 | 0.4925 | 0.4818 | 0.9914 | no | 0.4586 |
| LOG-O | train fold 3 | pass | 568 | 1.951e-06 | 0.2164 | 0.4846 | 0.4725 | 0.4833 | 1.003 | no | 0.3394 |
| LOG-O | train fold 4 | pass | 657 | 2.754e-09 | 0.2559 | 0.5009 | 0.4536 | 0.467 | 0.9889 | no | 0.4438 |
| LOG-S | all cells | pass | 609 | 1.245e-07 | 0.2315 | 0.5153 | 0.5276 | 0.4941 | 1.091 | no | 0.4762 |
| LOG-S | train fold 0 | pass | 658 | 2.523e-07 | 0.2411 | 0.521 | 0.5161 | 0.4796 | 1.088 | no | 0.4047 |
| LOG-S | train fold 1 | pass | 583 | 1.7e-07 | 0.2279 | 0.5204 | 0.5446 | 0.5187 | 1.086 | no | 0.3785 |
| LOG-S | train fold 2 | pass | 852 | 5.712e-08 | 0.211 | 0.4951 | 0.5457 | 0.4963 | 1.091 | no | 0.4524 |
| LOG-S | train fold 3 | pass | 643 | 1.864e-06 | 0.2233 | 0.5128 | 0.5252 | 0.4989 | 1.109 | no | 0.4048 |
| LOG-S | train fold 4 | pass | 653 | 8.144e-06 | 0.2526 | 0.5242 | 0.5141 | 0.4912 | 1.091 | no | 0.3668 |

### V4 -- push-through identity

50,000 cells drawn at random; y simulated from each arm's out-of-fold
predictive distribution; the simulated label rate compared with the mean
pushed-through p. Formal criterion on the primary thresholds (cT_a05,
cO_td050): within 3 Monte Carlo SE. **pass**.
The other grid thresholds are shown for information; with
21 further comparisons, one beyond 3 SE would be
expected about 0.06 times by chance.

| model | threshold | mean p | simulated rate | MC SE | z | within 3 SE |
| --- | --- | --- | --- | --- | --- | --- |
| NB-c | LT_a05 | 0.042584 | 0.04314 | 0.00085465 | 0.65063 | pass |
| NB-c | LT_a01 | 0.011039 | 0.01086 | 0.00045565 | -0.39253 | pass |
| NB-c | LT_a001 | 0.0018264 | 0.00174 | 0.00018923 | -0.45653 | pass |
| NB-c | LT_a0001 | 0.00034142 | 0.00036 | 8.2326e-05 | 0.22574 | pass |
| NB-c | LO_td000 | 0.32148 | 0.32412 | 0.0018494 | 1.4259 | pass |
| NB-c | LO_td025 | 0.25565 | 0.2573 | 0.0017692 | 0.9349 | pass |
| NB-c | LO_td050 | 0.20185 | 0.20238 | 0.001654 | 0.31982 | pass |
| NB-c | LO_td075 | 0.15865 | 0.15976 | 0.0015233 | 0.72874 | pass |
| NB-c | LO_td100 | 0.12432 | 0.12524 | 0.0013883 | 0.66482 | pass |
| NB-d | LT_a05 | 0.042967 | 0.04372 | 0.00085835 | 0.87674 | pass |
| NB-d | LT_a01 | 0.011217 | 0.01196 | 0.0004592 | 1.6183 | pass |
| NB-d | LT_a001 | 0.0018821 | 0.00194 | 0.00019202 | 0.30172 | pass |
| NB-d | LT_a0001 | 0.00036008 | 0.00034 | 8.4513e-05 | -0.23759 | pass |
| NB-d | LO_td000 | 0.32057 | 0.31932 | 0.0018461 | -0.67505 | pass |
| NB-d | LO_td025 | 0.25499 | 0.25344 | 0.001766 | -0.87555 | pass |
| NB-d | LO_td050 | 0.20146 | 0.20018 | 0.0016514 | -0.77515 | pass |
| NB-d | LO_td075 | 0.1585 | 0.15716 | 0.0015216 | -0.88062 | pass |
| NB-d | LO_td100 | 0.12436 | 0.12292 | 0.0013876 | -1.0371 | pass |
| MIX | LT_a05 | 0.049878 | 0.04926 | 0.0009286 | -0.6655 | pass |
| MIX | LT_a01 | 0.0015323 | 0.00166 | 0.00017455 | 0.73134 | pass |
| MIX | LT_a001 | 1.1721e-05 | 4e-05 | 1.5309e-05 | 1.8472 | pass |
| MIX | LT_a0001 | 2.4722e-07 | 0 | 2.2236e-06 | -0.11118 | pass |
| MIX | LO_td000 | 0.31042 | 0.31264 | 0.0019327 | 1.1507 | pass |
| MIX | LO_td025 | 0.26142 | 0.26344 | 0.0018199 | 1.1117 | pass |
| MIX | LO_td050 | 0.22857 | 0.22922 | 0.0017207 | 0.37603 | pass |
| MIX | LO_td075 | 0.20043 | 0.2004 | 0.0016312 | -0.018494 | pass |
| MIX | LO_td100 | 0.17138 | 0.171 | 0.0015408 | -0.24854 | pass |

### V8 -- determinism

Refitting LOG-T on training fold 0 reproduces its parameters to 0.0e+00;
refitting NB-c on training fold 0 reproduces Stage 2's saved fit to 0.0e+00;
rerunning the V4 simulation with its seed reproduces it to 0.0e+00.
**pass**

### Sanity check, not scoring

Overall mean out-of-fold p against each label's rate. Scoring (log loss,
Brier, AUPRC, calibration by type, intervals) is Stage 4.

| label | label rate | NB-c | NB-d | MIX | LOG-T | LOG-O | LOG-S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BASE-T (LT_a05) | 0.05370 | 0.04246 | 0.04288 | 0.04985 | 0.05382 | nan | nan |
| BASE-O (LO_td050) | 0.19508 | 0.20184 | 0.20145 | 0.22825 | nan | 0.19545 | nan |
| SCORE (LS) | 0.22632 | nan | nan | nan | nan | nan | 0.22670 |

Out-of-fold predictions for every arm and threshold are in
`oof_predictions.parquet`. Stage 3 complete in 17 min.
**Review point** (stages 1, 3 and 6 pause for review).


## Stage 4 -- label-scale scoring (2026-10-16 00:18)

Every score is out of fold (Stage 3's predictions; nothing refitted). Intervals
are 95% spatial block bootstrap: the 250 stored blocks resampled with
replacement, 300 replicates, every arm scored on the same resampled cells,
so a paired difference is taken within each replicate. Per the interpretive
rule, each LOG arm was trained on the label it is scored against, and a
label-scale win for it is not evidence that it infers division better.
Calibration-in-the-large is reported as mean p / label rate (1 = calibrated).
Cell types are Stage 1's.

### BASE-T (LT_a05, label rate 0.0537)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.1662 [0.1475, 0.1818] | 0.0463 [0.0407, 0.0514] | 0.8501 [0.8376, 0.8647] | 0.1933 [0.1814, 0.2037] | 0.791 [0.743, 0.849] |
| NB-d | 0.1661 [0.1474, 0.1817] | 0.0463 [0.0407, 0.0514] | 0.8499 [0.8369, 0.8642] | 0.1926 [0.1810, 0.2029] | 0.799 [0.751, 0.858] |
| MIX | 0.1641 [0.1463, 0.1795] | 0.0459 [0.0404, 0.0509] | 0.8483 [0.8356, 0.8627] | 0.2008 [0.1892, 0.2129] | 0.928 [0.878, 0.998] |
| LOG-T | 0.1602 [0.1429, 0.1754] | 0.0451 [0.0397, 0.0500] | 0.8609 [0.8493, 0.8741] | 0.2257 [0.2141, 0.2368] | 1.002 [0.950, 1.076] |

Paired differences (first minus second; for log loss and Brier, negative means
the first arm is better; for AUROC and AUPRC, positive):

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | -0.0020 [-0.0031, -0.0009] | -0.0004 [-0.0005, -0.0002] | -0.0018 [-0.0031, -0.0002] | 0.0075 [0.0051, 0.0105] |
| NB-d - NB-c | -0.0001 [-0.0003, 0.0001] | 0.0000 [0.0000, 0.0000] | -0.0002 [-0.0007, 0.0002] | -0.0007 [-0.0011, -0.0004] |
| LOG-T - MIX | -0.0040 [-0.0046, -0.0032] | -0.0008 [-0.0010, -0.0007] | 0.0126 [0.0102, 0.0145] | 0.0249 [0.0215, 0.0286] |
| LOG-T - NB-c | -0.0060 [-0.0074, -0.0044] | -0.0012 [-0.0014, -0.0009] | 0.0108 [0.0088, 0.0128] | 0.0323 [0.0291, 0.0357] |

### BASE-O (LO_td050, label rate 0.1951)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.4117 [0.3928, 0.4282] | 0.1313 [0.1234, 0.1379] | 0.7814 [0.7735, 0.7903] | 0.4454 [0.4208, 0.4674] | 1.035 [0.997, 1.083] |
| NB-d | 0.4115 [0.3924, 0.4281] | 0.1312 [0.1233, 0.1379] | 0.7817 [0.7738, 0.7906] | 0.4463 [0.4217, 0.4682] | 1.033 [0.995, 1.080] |
| MIX | 0.4186 [0.4004, 0.4342] | 0.1324 [0.1249, 0.1387] | 0.7764 [0.7682, 0.7853] | 0.4526 [0.4277, 0.4754] | 1.170 [1.123, 1.228] |
| LOG-O | 0.4181 [0.3999, 0.4337] | 0.1317 [0.1241, 0.1381] | 0.7683 [0.7594, 0.7770] | 0.4514 [0.4256, 0.4743] | 1.002 [0.966, 1.050] |

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | 0.0069 [0.0052, 0.0087] | 0.0011 [0.0005, 0.0017] | -0.0049 [-0.0072, -0.0027] | 0.0073 [0.0043, 0.0101] |
| NB-d - NB-c | -0.0003 [-0.0004, -0.0001] | -0.0001 [-0.0001, -0.0001] | 0.0003 [0.0002, 0.0005] | 0.0010 [0.0007, 0.0013] |
| LOG-O - MIX | -0.0005 [-0.0020, 0.0008] | -0.0006 [-0.0011, -0.0002] | -0.0081 [-0.0121, -0.0048] | -0.0012 [-0.0034, 0.0009] |
| LOG-O - NB-c | 0.0063 [0.0051, 0.0077] | 0.0005 [0.0001, 0.0008] | -0.0131 [-0.0176, -0.0097] | 0.0060 [0.0030, 0.0090] |

### SCORE (LOG-S only; not comparable with the tables above)

Label rate 0.2263. Log loss 0.4382 [0.4196, 0.4549];
Brier 0.1405 [0.1327, 0.1474];
AUROC 0.7876 [0.7790, 0.7963];
AUPRC 0.5164 [0.4900, 0.5399].

### Murphy decomposition of the Brier score

Brier = reliability - resolution + uncertainty + within-bin, on 20
equal-count bins of each arm's own p. Lower reliability is better calibration;
higher resolution is better discrimination.

**BASE-T**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.046265 | 0.000352 | 0.004918 | 0.050817 | 0.000013 |
| NB-d | 0.046285 | 0.000343 | 0.004888 | 0.050817 | 0.000013 |
| MIX | 0.045914 | 0.000045 | 0.004884 | 0.050817 | -0.000064 |
| LOG-T | 0.045087 | 0.000005 | 0.005629 | 0.050817 | -0.000106 |

**BASE-O**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.131274 | 0.000173 | 0.025782 | 0.157023 | -0.000140 |
| NB-d | 0.131182 | 0.000156 | 0.025851 | 0.157023 | -0.000146 |
| MIX | 0.132358 | 0.001301 | 0.025770 | 0.157023 | -0.000196 |
| LOG-O | 0.131729 | 0.000090 | 0.025161 | 0.157023 | -0.000223 |

### Noise ceilings (spec 4.4)

What each arm would score if it were the true model (labels drawn as
L ~ Bernoulli(p); for a count arm, the same as simulating y and applying c_i).
A gap near zero means the arm's errors are what its own uncertainty predicts.
The ceiling is computed from the arm's own p, so it moves with the arm's
calibration: an arm that over-predicts a rare label raises its own expected log
loss and shows a negative gap (MIX on BASE-T, mean p / rate 1.12), and one that
under-predicts lowers it. Read a gap together with the arm's calibration; only
for a calibrated arm does a log-loss gap above zero, or an AUPRC gap below,
mean the label carries structure the arm does not.

| label | arm | log loss observed | log loss ceiling | log loss gap | AUPRC observed | AUPRC ceiling | AUPRC gap | AUROC observed | AUROC ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE-T | NB-c | 0.1662 | 0.1331 | 0.0331 | 0.1933 | 0.2054 | -0.0121 | 0.8501 | 0.8715 |
| BASE-T | NB-d | 0.1661 | 0.1343 | 0.0317 | 0.1926 | 0.2066 | -0.0140 | 0.8499 | 0.8703 |
| BASE-T | MIX | 0.1641 | 0.1565 | 0.0076 | 0.2008 | 0.1844 | 0.0164 | 0.8483 | 0.8465 |
| BASE-T | LOG-T | 0.1602 | 0.1610 | -0.0008 | 0.2257 | 0.2281 | -0.0024 | 0.8609 | 0.8582 |
| BASE-O | NB-c | 0.4117 | 0.4261 | -0.0144 | 0.4454 | 0.4363 | 0.0091 | 0.7814 | 0.7690 |
| BASE-O | NB-d | 0.4115 | 0.4247 | -0.0132 | 0.4463 | 0.4385 | 0.0078 | 0.7817 | 0.7707 |
| BASE-O | MIX | 0.4186 | 0.4581 | -0.0395 | 0.4526 | 0.4875 | -0.0348 | 0.7764 | 0.7598 |
| BASE-O | LOG-O | 0.4181 | 0.4170 | 0.0010 | 0.4514 | 0.4497 | 0.0017 | 0.7683 | 0.7713 |
| SCORE | LOG-S | 0.4382 | 0.4371 | 0.0012 | 0.5164 | 0.5130 | 0.0034 | 0.7876 | 0.7895 |

### Calibration-in-the-large by cell type (mean p / rate; * = 95% interval excludes 1)

**BASE-T**

| stratum | NB-c | NB-d | MIX | LOG-T |
| --- | --- | --- | --- | --- |
| all | 0.791* | 0.799* | 0.928* | 1.002 |
| Tumour | 0.784* | 0.785* | 0.912* | 0.981 |
| Fibroblasts | 0.864 | 0.884 | 1.830* | 1.774* |
| Endothelial Cells | 0.749 | 0.808 | 1.419* | 1.459* |
| Pericytes | 0.624* | 0.872 | 1.091 | 1.192 |
| Smooth Muscle Cells | 1.327* | 1.166 | 1.925* | 1.414* |
| Macrophages | 1.114* | 1.346* | 1.135* | 1.502* |
| T & NK Cells | 0.860 | 1.267 | 0.826 | 1.345* |
| Other | 3.040* | 3.920* | 1.364 | 1.372 |

**BASE-O**

| stratum | NB-c | NB-d | MIX | LOG-O |
| --- | --- | --- | --- | --- |
| all | 1.035 | 1.033 | 1.170* | 1.002 |
| Tumour | 1.006 | 1.005 | 1.066* | 1.000 |
| Fibroblasts | 1.114* | 1.098* | 1.496* | 0.949* |
| Endothelial Cells | 1.161* | 1.152* | 1.664* | 1.067* |
| Pericytes | 1.089* | 1.120* | 1.426* | 1.149* |
| Smooth Muscle Cells | 1.053* | 1.041* | 1.798* | 0.967 |
| Macrophages | 1.178* | 1.185* | 1.269* | 1.027 |
| T & NK Cells | 1.177* | 1.174* | 1.106* | 1.238* |
| Other | 1.550* | 1.601* | 1.512* | 0.880 |

### Threshold sweep (count arms, all cells)

**BASE-T**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| alpha_B 0.05 (primary) | 0.0537 | 0.791 [0.743, 0.849] | 0.799 [0.751, 0.858] | 0.928 [0.878, 0.998] |
| alpha_B 0.01 | 0.008331 | 1.323 [1.186, 1.488] | 1.345 [1.208, 1.510] | 0.184 [0.170, 0.200] |
| alpha_B 0.001 | 0.0004652 | 3.933 [3.127, 4.988] | 4.048 [3.247, 5.080] | 0.026 [0.021, 0.033] |
| alpha_B 0.0001 | 3.287e-05 | 10.443 [6.265, 22.189] | 10.939 [6.614, 23.435] | 0.008 [0.005, 0.017] |

**BASE-O**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| Td 0 | 0.2956 | 1.089 [1.058, 1.127] | 1.086 [1.055, 1.123] | 1.051 [1.014, 1.095] |
| Td 0.25 | 0.2377 | 1.077 [1.040, 1.122] | 1.074 [1.038, 1.119] | 1.099 [1.057, 1.151] |
| Td 0.5 (primary) | 0.1951 | 1.035 [0.997, 1.083] | 1.033 [0.995, 1.080] | 1.170 [1.123, 1.228] |
| Td 0.75 | 0.1622 | 0.978 [0.940, 1.026] | 0.977 [0.939, 1.025] | 1.234 [1.183, 1.297] |
| Td 1 | 0.1344 | 0.925 [0.886, 0.972] | 0.925 [0.887, 0.973] | 1.275 [1.222, 1.346] |

By cell type, in `calibration_sweep_BASE-T.png` and `calibration_sweep_BASE-O.png`;
by the cell's own threshold c_i, in `calibration_by_threshold.csv`.

### Pre-stated expectations X1-X3

- **X1** (each LOG arm has the best log loss on its own label): BASE-T
  **holds**; BASE-O **does not hold**
  (every paired log-loss difference against the LOG arm, interval excluding 0).
- **X2** (NB-c over-predicts where c_i is 5-10 and under-predicts where it is
  15-30, pooled over both grids): mean p - rate +0.01543 at c_i 5-10 and
  -0.02968 at 15-30: **holds**.
- **X3** (MIX better calibrated than NB-c across the grids, but under-predicting
  at the strictest thresholds). The two clauses are judged separately, since
  the second concerns the strictest thresholds. Mean |log(p / rate)| over the
  seven grid thresholds other than alpha_B 0.001 and 0.0001: 0.361 for
  MIX against 0.116 for NB-c (**not better**).
  At alpha_B 0.001 and 0.0001 MIX's p / rate is
  0.03, 0.01 (**under-predicts**),
  far below NB-c's, so over all nine thresholds MIX is the worse calibrated
  (1.225 against 0.503). Overall: **partly holds**.

Figures: `reliability_BASE-T.png`, `reliability_BASE-O.png`,
`reliability_SCORE.png`, `calibration_sweep_BASE-T.png`,
`calibration_sweep_BASE-O.png`. Every score and interval is in `scores.csv`
(including the full per-type metrics and the Murphy components for every
stratum); ceilings in `ceilings.csv`. Stage 4 complete in
2 min.
