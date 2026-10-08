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
| NB arms' depth term | fixed offset, theta = 1 | **theta free**, as MIX | parity with MIX, so arms differ only in likelihood; fixing the offset cost ~880 log-likelihood on the division NB. Fixed-offset NB and MIX with theta = 1 are sensitivity rows. Decided by the user. |
| technical null depth term | fixed offset | fixed offset, raw log N | unchanged; labels are definitions |
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


## Stage 1 -- labels (built and frozen 2026-10-06 20:42)

561,897 analysis cells. Cell type is the argmax of column-standardised PhiSpace
scores, agreeing with the stored label for 97.99% of cells.

**Outcome and library size.** y recomputed from the count matrix with the
mixture's 53-gene Tirosh list equals the table's `y_count` for every
cell; N (excluding the Tirosh genes and the 530-gene blocklist) equals the
table's `N` for every cell.

### Technical null (spec 2.1)

log mu0 = b0 + log N + s(log A), area spline as the NB arms (df 8), raw log N as
a fixed offset. Robust fit by iterative trimming at P(Y >= y) < 0.001; stopped on
**tolerance** after 2 fits. Final alpha0 = **0.8284**.

| iteration | n_fit | alpha0 | flagged | flagged_frac | changed | converged |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 561897 | 0.8877 | 920 | 0.001637 | 920 | True |
| 2 | 560977 | 0.8284 | 1123 | 0.001999 | 203 | True |

### Label rates

| cell type | cells | BASE-T 0.05 | BASE-T 0.01 | BASE-T 0.001 | BASE-T 0.0001 | BASE-O 0.5 | BASE-O-global 0.5 | BASE-O-neg 0.5 | SCORE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 561897 | 0.03437 | 0.01096 | 0.001999 | 0.0003559 | 0.1828 | 0.1803 | 0.2216 | 0.2342 |
| Endothelial | 48729 | 0.01155 | 0.003201 | 0.0003694 | 4.104e-05 | 0.1215 | 0.1152 | 0.1013 | 0.1408 |
| CAFs | 52079 | 0.009697 | 0.00192 | 0.0002688 | 1.92e-05 | 0.1288 | 0.1084 | 0.1077 | 0.153 |
| PVL | 42834 | 0.01868 | 0.00537 | 0.001097 | 0.0001868 | 0.1476 | 0.1432 | 0.1291 | 0.1819 |
| B-cells | 59119 | 0.02434 | 0.00614 | 0.000998 | 0.0001353 | 0.2112 | 0.2028 | 0.1643 | 0.2267 |
| T-cells | 87004 | 0.03195 | 0.009804 | 0.002103 | 0.0002988 | 0.2384 | 0.2302 | 0.1926 | 0.2662 |
| Myeloid | 69669 | 0.01455 | 0.003947 | 0.0005024 | 8.612e-05 | 0.1372 | 0.1278 | 0.1172 | 0.1598 |
| Normal Epithelial | 54508 | 0.03368 | 0.01115 | 0.002 | 0.0002935 | 0.167 | 0.1701 | 0.2637 | 0.2368 |
| Plasmablasts | 42523 | 0.04271 | 0.01416 | 0.002187 | 0.0003528 | 0.2007 | 0.204 | 0.2359 | 0.2724 |
| Cancer Epithelial | 105432 | 0.08116 | 0.02817 | 0.005359 | 0.001119 | 0.2215 | 0.2373 | 0.4686 | 0.3491 |

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

| N decile | N median | mu0 median | c^T median @0.05 | mean alpha_i @0.05 | c^T median @0.01 | mean alpha_i @0.01 | c^T median @0.001 | mean alpha_i @0.001 | c^T median @0.0001 | mean alpha_i @0.0001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0.09621 | 2 | 0.007597 | 2 | 0.00574 | 3 | 0.0005205 | 4 | 4.494e-05 |
| 2 | 21 | 0.1572 | 2 | 0.01794 | 3 | 0.002294 | 4 | 0.0002845 | 5 | 3.459e-05 |
| 3 | 30 | 0.2268 | 2 | 0.03216 | 3 | 0.00546 | 4 | 0.00053 | 6 | 4.286e-05 |
| 4 | 40 | 0.3022 | 3 | 0.02679 | 4 | 0.005007 | 5 | 0.0004918 | 6 | 5.192e-05 |
| 5 | 51 | 0.3906 | 3 | 0.0206 | 4 | 0.005198 | 6 | 0.0005065 | 7 | 5.634e-05 |
| 6 | 65 | 0.5029 | 3 | 0.03381 | 5 | 0.00603 | 6 | 0.0005992 | 8 | 5.721e-05 |
| 7 | 86 | 0.6791 | 4 | 0.02997 | 5 | 0.006442 | 8 | 0.0006529 | 10 | 6.413e-05 |
| 8 | 125 | 1.017 | 5 | 0.03646 | 7 | 0.00709 | 10 | 0.0006959 | 13 | 7.02e-05 |
| 9 | 206 | 1.807 | 7 | 0.0394 | 10 | 0.007902 | 15 | 0.0007877 | 20 | 7.851e-05 |
| 10 | 385 | 3.997 | 13 | 0.04425 | 20 | 0.008835 | 29 | 0.0008825 | 37 | 8.819e-05 |

### BASE-O (spec 2.3)

Primary BASE-O scales the residual within each cell type, as OSDR's own code
does (`is_dividing`: one cutoff per type, `Td * sigma_type + Tn`, ddof 0).
Per-type s+: Endothelial 1.133, CAFs 0.957, PVL 1.207, B-cells 1.142, T-cells 1.188, Myeloid 1.116, Normal Epithelial 1.318, Plasmablasts 1.321, Cancer Epithelial 1.479.
BASE-O-global uses one s+ = 1.2771 for all cells.
Negative-codeword floor (BASE-O-neg, decoding noise only, no spillover):
rho = 2.795e-06 per um^2 over 609 negative-control codewords,
|T| = 53, mean b_i = 1.73e-05 against mean mu0 = 0.9798;
s+ = 625.9145.

Caveat: per-type scaling raises the effective threshold in types with more
cycling cells, so it partly removes between-type proliferation differences.
That is a property of OSDR's construction, inherited here.

### SCORE (spec 2.4)

scanpy `score_genes`, 53 genes, n_bins 25, ctrl_size 50, control pool
4,571 genes (panel minus blocklist), log1p(X s / N) with s = median
N = 57. Robust null: KDE mode -0.0065, left-half
RMS sigma 0.0027, threshold 0.0020 (upper-tail p <
0.001).

| stratum | n | kappa vs BASE-T | kappa vs BASE-O | SCORE rate | BASE-T rate | BASE-O rate |
| --- | --- | --- | --- | --- | --- | --- |
| all | 561897 | 0.2084 | 0.8293 | 0.2342 | 0.03437 | 0.1828 |
| Endothelial | 48729 | 0.1331 | 0.8852 | 0.1408 | 0.01155 | 0.1215 |
| CAFs | 52079 | 0.1028 | 0.8638 | 0.153 | 0.009697 | 0.1288 |
| PVL | 42834 | 0.1577 | 0.8632 | 0.1819 | 0.01868 | 0.1476 |
| B-cells | 59119 | 0.1569 | 0.9304 | 0.2267 | 0.02434 | 0.2112 |
| T-cells | 87004 | 0.1668 | 0.9043 | 0.2662 | 0.03195 | 0.2384 |
| Myeloid | 69669 | 0.1441 | 0.8801 | 0.1598 | 0.01455 | 0.1372 |
| Normal Epithelial | 54508 | 0.202 | 0.7795 | 0.2368 | 0.03368 | 0.167 |
| Plasmablasts | 42523 | 0.2129 | 0.7987 | 0.2724 | 0.04271 | 0.2007 |
| Cancer Epithelial | 105432 | 0.2828 | 0.6924 | 0.3491 | 0.08116 | 0.2215 |

**Zero-count diagnostic.**

| cells with y = 0 | score sd, y = 0 | score sd, all cells | score IQR, y = 0 | score 1st-99th pct, y = 0 | SCORE call rate among y = 0 | share of SCORE calls with y <= 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 349427 | 0.002895 | 0.01079 | 0.004012 | -0.013 to 0.000 | 0 | 0.4752 |

### Validation

- **V3** survivor functions vs scipy: max |diff| NB 1.63e-14, Poisson
  0.00e+00 (< 1e-12): **pass**.
- **V10** BASE-O discrete equivalence, every cell, every Td, all three
  variants: **pass**.
- **V2** null calibration, end to end (simulate from the null, refit the robust
  null, rebuild the labels): **pass**.

| alpha_B | label rate (refit) | mean alpha_i | relative error | pass (+/-30%) | label rate (frozen c) | mean alpha_i (frozen) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.02933 | 0.02857 | 0.02661 | pass | 0.02864 | 0.02878 |
| 0.01 | 0.006272 | 0.005942 | 0.05544 | pass | 0.005969 | 0.005974 |
| 0.001 | 0.0006158 | 0.0005894 | 0.04474 | pass | 0.0005606 | 0.0005931 |
| 0.0001 | 4.983e-05 | 5.807e-05 | -0.1418 | pass | 4.805e-05 | 5.872e-05 |

BASE-O call rate on null data (reported; no pass criterion):

| Td | null call rate (rebuilt) | null call rate (frozen c) | observed call rate |
| --- | --- | --- | --- |
| 0 | 0.2834 | 0.2827 | 0.2737 |
| 0.25 | 0.2445 | 0.2366 | 0.2266 |
| 0.5 | 0.2036 | 0.1909 | 0.1828 |
| 0.75 | 0.1639 | 0.1492 | 0.1437 |
| 1 | 0.13 | 0.115 | 0.1115 |

Stage 1 complete in 1.5 min. **Review point** (stages 1,
3 and 6 pause for review).


## Stage 2 -- NB refits (2026-10-08 08:32)

**The model changed after Stage 1** (decisions recorded in the mixture's
specification doc, 2026-10-06 to 10-07, all by the user): log N is an offset
(theta = 1) in every count arm, MIX included; the oxygen term is dropped; the
neighbour block is split into contact (0-15 um) and regional (15-80 um) means;
both state blocks are whitened; B is penalised by lambda ||UV'||_F^2 with the
balance term tau ||U'U - V'V||_F^2; lambda was selected on null-calibrated
Test 03 and the rank by spatial CV. With theta = 1 in MIX too, the spec's offset
mismatch (section 1) no longer arises. The labels are unaffected: they depend
only on y, N, A and the technical null.

NB-c and NB-d, matched to the mixture's covariates and penalty: the shared weight
block (8 own and 16 neighbour coordinates,
rank 3), log N as an offset, the same
df-8 area spline, weight_ridge 300 x n /
678,885 on every eta coefficient (frobenius form on B) and
the balance term at tau = 1. NB-d adds log alpha_i = gamma0 +
phi_i' gamma on the projected own state (gamma not penalised). Optimiser:
L-BFGS, then Newton on the exact Hessian with absolute-value eigenvalues and a
backtracking line search.

### V5 -- fold identity

250 k-means blocks (MiniBatchKMeans, seed 0) dealt into 5 folds (seed 0); map
loaded from folds.parquet, sha256 `32abc1e083fa4d69...`. The mixture CV did not store its
cell-to-fold map, so identity is checked on the fold sizes it recorded: all five
match to the cell, which a different partition would not do by chance.
**pass**

| fold | n_train here | n_test here | n_test, mixture CV (Spartan) | match |
| --- | --- | --- | --- | --- |
| 0 | 443599 | 118298 | 118298 | yes |
| 1 | 455993 | 105904 | 105904 | yes |
| 2 | 444258 | 117639 | 117639 | yes |
| 3 | 449795 | 112102 | 112102 | yes |
| 4 | 453943 | 107954 | 107954 | yes |

### V6 -- parity

All 6 arms (MIX, NB-c, NB-d, LOG-T, LOG-O, LOG-S) share the same
141 eta columns in the same order. The shared `eta_state`
reproduces the production mixture's logit to 7.1e-15 on every cell. **pass**

### V1 -- convergence

Criterion: |f_t - f_(t-1)| < 1e-9 |f_(t-1)| and ||grad||_inf < 1e-5, on the
summed penalised objective. **pass**: 14
of 14 fits converged. sd(eta) > 20 flagged: **none**.

| arm | data | seed | n | loglik | objective | grad_inf | L-BFGS iters | Newton iters | converged (V1) | attempts | theta | alpha | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-c | all cells | 0 | 561897 | -5.964e+05 | 5.964e+05 | 8.717e-06 | 1212 | 21 | pass | 1 | 1 | 0.7449 | 0.3296 | 0.1737 | 0.1787 | 0.3906 | no | 4.651 |
| NB-c | all cells | 1 | 561897 | -5.964e+05 | 5.964e+05 | 6.333e-09 | 1319 | 8 | pass | 1 | 1 | 0.7449 | 0.3296 | 0.1737 | 0.1787 | 0.3906 | no | 6.468 |
| NB-d | all cells | 0 | 561897 | -5.962e+05 | 5.963e+05 | 1.07e-06 | 1331 | 18 | pass | 1 | 1 | 0.544-0.866 | 0.3299 | 0.1739 | 0.1783 | 0.391 | no | 7.844 |
| NB-d | all cells | 1 | 561897 | -5.962e+05 | 5.963e+05 | 1.904e-07 | 1374 | 10 | pass | 1 | 1 | 0.544-0.866 | 0.3299 | 0.1739 | 0.1783 | 0.391 | no | 5.19 |
| NB-c | train fold 0 | 0 | 443599 | -4.706e+05 | 4.707e+05 | 1.559e-06 | 1452 | 7 | pass | 1 | 1 | 0.7454 | 0.3293 | 0.1737 | 0.177 | 0.3934 | no | 3.765 |
| NB-c | train fold 1 | 0 | 455993 | -4.791e+05 | 4.792e+05 | 2.039e-07 | 1436 | 7 | pass | 1 | 1 | 0.7495 | 0.3299 | 0.1721 | 0.1791 | 0.3907 | no | 3.512 |
| NB-c | train fold 2 | 0 | 444258 | -4.762e+05 | 4.763e+05 | 2.996e-08 | 1219 | 9 | pass | 1 | 1 | 0.7431 | 0.3284 | 0.1758 | 0.1813 | 0.388 | no | 3.683 |
| NB-c | train fold 3 | 0 | 449795 | -4.849e+05 | 4.85e+05 | 1.98e-06 | 1562 | 14 | pass | 1 | 1 | 0.7443 | 0.3287 | 0.1763 | 0.1802 | 0.3902 | no | 4.231 |
| NB-c | train fold 4 | 0 | 453943 | -4.744e+05 | 4.745e+05 | 2.018e-10 | 1195 | 13 | pass | 1 | 1 | 0.7407 | 0.3309 | 0.1734 | 0.1791 | 0.3925 | no | 4.184 |
| NB-d | train fold 0 | 0 | 443599 | -4.705e+05 | 4.706e+05 | 5.947e-07 | 1359 | 9 | pass | 1 | 1 | 0.559-0.867 | 0.3293 | 0.1742 | 0.1765 | 0.3938 | no | 4.023 |
| NB-d | train fold 1 | 0 | 455993 | -4.79e+05 | 4.791e+05 | 9.231e-07 | 1498 | 12 | pass | 1 | 1 | 0.548-0.865 | 0.3302 | 0.1718 | 0.1785 | 0.391 | no | 3.502 |
| NB-d | train fold 2 | 0 | 444258 | -4.761e+05 | 4.762e+05 | 1.135e-08 | 1010 | 18 | pass | 1 | 1 | 0.551-0.856 | 0.329 | 0.1757 | 0.1808 | 0.3884 | no | 3.24 |
| NB-d | train fold 3 | 0 | 449795 | -4.848e+05 | 4.848e+05 | 1.561e-08 | 1260 | 7 | pass | 1 | 1 | 0.523-0.890 | 0.3288 | 0.1774 | 0.18 | 0.3909 | no | 2.397 |
| NB-d | train fold 4 | 0 | 453943 | -4.743e+05 | 4.744e+05 | 2.219e-08 | 1229 | 7 | pass | 1 | 1 | 0.536-0.854 | 0.3315 | 0.173 | 0.1786 | 0.3929 | no | 2.383 |

`run/compare_nb_mixture.py`'s matched NB is unpenalised, so it is not the
same model as NB-c, which matches the mixture's penalty.

**Uniqueness.** Two seeds per arm on all cells:

| arm | |objective diff| | max |B diff| | max |a diff| | |theta diff| |
| --- | --- | --- | --- | --- |
| NB-c | 2.33e-10 | 3.21e-11 | 1.27e-14 | 0 |
| NB-d | 1.16e-10 | 4.43e-12 | 1.78e-15 | 0 |

**NB-d against NB-c.** Log-likelihood -596,235.56 against -596,377.03, a gain of
141.5 for 8 parameters (LR statistic 283). The
dispersion depends on state.

For reference, the mixture's data log-likelihood on the same cells is
-603,106.13. As in the main write-up, the NB fits the per-cell counts better;
the label-scale comparison begins in Stage 3.

Out-of-fold NB predictions (log mu and log alpha, per held-out cell) are saved
in `oof_nb.parquet` for Stage 3. Stage 2 complete in 59 min.


## Stage 3 -- out-of-fold fits (2026-10-08 09:23)

Every arm fitted on four folds and predicted on the fifth, on the stored
partition. NB-c and NB-d are Stage 2's fold fits. MIX is refitted per training
fold (penalty scaled to training size) from the full-data point estimate, to
the mixture's convergence criteria. LOG-T, LOG-O and LOG-S regress the primary
labels (LT_a05, LO_td050, LS) on the shared weight
block plus b0 + theta log N + s(log A), with the mixture's penalty term for
term. Count arms are pushed through every threshold on the alpha_B and Td
grids: NB p = P(Y >= c); MIX p = pi P1(Y >= c) + (1 - pi) P0(Y >= c), with the
prior pi.

### V5 -- fold identity

Map sha256 `32abc1e083fa4d69...`, equal to the stored map's; Stage 2's
out-of-fold NB predictions use the identical assignment; fold sizes match the
Spartan rank CV on every fold. **pass**

### MIX fold fits

All five converged: **yes**. Each fold's held-out
log-likelihood per cell agrees with the Spartan CV fit for the same fold (rank
3, lambda 300) to 4.5e-11, so starting
from the full-data estimate did not change where the fits end.

| data | converged | iterations | grad_inf | held-out loglik / cell | Spartan CV, same fold | e^delta | mean pi (test) | sd(eta) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train fold 0 | pass | 1453 | 1.661e-05 | -1.074 | -1.074 | 1.721 | 0.1262 | 1.339 | 1.827 |
| train fold 1 | pass | 1689 | 8.483e-05 | -1.12 | -1.12 | 1.727 | 0.1249 | 1.345 | 2.401 |
| train fold 2 | pass | 1926 | 2.055e-05 | -1.032 | -1.032 | 1.721 | 0.1237 | 1.315 | 2.25 |
| train fold 3 | pass | 2174 | 1.096e-05 | -1.005 | -1.005 | 1.725 | 0.1252 | 1.319 | 2.679 |
| train fold 4 | pass | 1108 | 3.113e-05 | -1.142 | -1.142 | 1.721 | 0.1305 | 1.328 | 1.211 |

### LOG fits and V7 -- separation

Criterion: sd(eta) <= 20 at the production ridge for every LOG fit.
**pass** (largest sd(eta) 0.876);
all converged: **yes**.

| arm | data | converged | iterations | grad_inf | theta (log N) | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 (V7) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOG-T | all cells | pass | 1095 | 1.334e-07 | 0.5468 | 0.5877 | 0.3794 | 0.3793 | 0.8564 | no | 1.136 |
| LOG-T | train fold 0 | pass | 1111 | 3.375e-07 | 0.551 | 0.5846 | 0.3788 | 0.3801 | 0.8641 | no | 0.8316 |
| LOG-T | train fold 1 | pass | 1596 | 1.29e-08 | 0.5632 | 0.5933 | 0.3663 | 0.3804 | 0.8646 | no | 1.186 |
| LOG-T | train fold 2 | pass | 1136 | 1.203e-06 | 0.5416 | 0.5819 | 0.3871 | 0.384 | 0.8497 | no | 0.848 |
| LOG-T | train fold 3 | pass | 1111 | 2.724e-11 | 0.5502 | 0.5895 | 0.3875 | 0.3812 | 0.8445 | no | 2.312 |
| LOG-T | train fold 4 | pass | 1232 | 1.386e-08 | 0.5292 | 0.5913 | 0.3809 | 0.3776 | 0.8757 | no | 0.9131 |
| LOG-O | all cells | pass | 1709 | 8.15e-07 | 0.1659 | 0.3346 | 0.1672 | 0.192 | 0.4152 | no | 2.095 |
| LOG-O | train fold 0 | pass | 1135 | 4.208e-11 | 0.1662 | 0.3375 | 0.1696 | 0.1942 | 0.4174 | no | 1.5 |
| LOG-O | train fold 1 | pass | 1672 | 8.199e-06 | 0.1742 | 0.3331 | 0.1684 | 0.1961 | 0.4146 | no | 2.53 |
| LOG-O | train fold 2 | pass | 1446 | 1.128e-10 | 0.1628 | 0.3391 | 0.1654 | 0.1953 | 0.4132 | no | 1.898 |
| LOG-O | train fold 3 | pass | 1938 | 1.181e-09 | 0.1599 | 0.3339 | 0.1689 | 0.1932 | 0.4173 | no | 2.545 |
| LOG-O | train fold 4 | pass | 1106 | 3.106e-06 | 0.1646 | 0.3313 | 0.1684 | 0.1885 | 0.4149 | no | 2.218 |
| LOG-S | all cells | pass | 2140 | 4.51e-07 | 0.2828 | 0.3637 | 0.1572 | 0.1601 | 0.4352 | no | 3.573 |
| LOG-S | train fold 0 | pass | 1592 | 1.496e-07 | 0.2837 | 0.3626 | 0.1588 | 0.1574 | 0.4391 | no | 2.233 |
| LOG-S | train fold 1 | pass | 1758 | 4.466e-06 | 0.293 | 0.3621 | 0.1588 | 0.1626 | 0.4335 | no | 2.302 |
| LOG-S | train fold 2 | pass | 1605 | 1.36e-07 | 0.2809 | 0.3686 | 0.1525 | 0.1609 | 0.4334 | no | 2.904 |
| LOG-S | train fold 3 | pass | 1464 | 2.316e-09 | 0.2715 | 0.3635 | 0.1571 | 0.1592 | 0.4396 | no | 1.76 |
| LOG-S | train fold 4 | pass | 2054 | 3.438e-07 | 0.2854 | 0.3616 | 0.1602 | 0.1597 | 0.4376 | no | 2.933 |

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
| NB-c | LT_a05 | 0.031675 | 0.03234 | 0.00077168 | 0.86147 | pass |
| NB-c | LT_a01 | 0.0076889 | 0.00796 | 0.0003879 | 0.69886 | pass |
| NB-c | LT_a001 | 0.0010861 | 0.00098 | 0.00014675 | -0.72288 | pass |
| NB-c | LT_a0001 | 0.00018154 | 0.0001 | 6.0048e-05 | -1.3579 | pass |
| NB-c | LO_td000 | 0.28023 | 0.28316 | 0.0019041 | 1.5372 | pass |
| NB-c | LO_td025 | 0.23484 | 0.2369 | 0.0018271 | 1.1254 | pass |
| NB-c | LO_td050 | 0.18938 | 0.1906 | 0.0017037 | 0.71891 | pass |
| NB-c | LO_td075 | 0.148 | 0.15082 | 0.001552 | 1.8162 | pass |
| NB-c | LO_td100 | 0.11445 | 0.11706 | 0.0013955 | 1.8682 | pass |
| NB-d | LT_a05 | 0.03158 | 0.03236 | 0.00077015 | 1.0124 | pass |
| NB-d | LT_a01 | 0.0077668 | 0.0073 | 0.00038964 | -1.198 | pass |
| NB-d | LT_a001 | 0.0011425 | 0.001 | 0.00015047 | -0.94735 | pass |
| NB-d | LT_a0001 | 0.00020246 | 0.00014 | 6.3416e-05 | -0.98492 | pass |
| NB-d | LO_td000 | 0.28028 | 0.2807 | 0.0019051 | 0.21828 | pass |
| NB-d | LO_td025 | 0.23478 | 0.23562 | 0.0018271 | 0.45991 | pass |
| NB-d | LO_td050 | 0.18921 | 0.18972 | 0.0017029 | 0.29856 | pass |
| NB-d | LO_td075 | 0.14778 | 0.14772 | 0.0015507 | -0.039315 | pass |
| NB-d | LO_td100 | 0.11422 | 0.11512 | 0.0013939 | 0.64842 | pass |
| MIX | LT_a05 | 0.038526 | 0.03818 | 0.00084427 | -0.41008 | pass |
| MIX | LT_a01 | 0.0081053 | 0.00794 | 0.00039942 | -0.41396 | pass |
| MIX | LT_a001 | 0.00057361 | 0.00068 | 0.00010704 | 0.99397 | pass |
| MIX | LT_a0001 | 3.6229e-05 | 2e-05 | 2.6917e-05 | -0.60293 | pass |
| MIX | LO_td000 | 0.27133 | 0.27268 | 0.0019076 | 0.70723 | pass |
| MIX | LO_td025 | 0.22466 | 0.22682 | 0.0018126 | 1.1925 | pass |
| MIX | LO_td050 | 0.18377 | 0.18376 | 0.001692 | -0.0029557 | pass |
| MIX | LO_td075 | 0.14789 | 0.1489 | 0.0015558 | 0.64738 | pass |
| MIX | LO_td100 | 0.11825 | 0.11878 | 0.0014178 | 0.37124 | pass |

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
| BASE-T (LT_a05) | 0.03437 | 0.03173 | 0.03163 | 0.03860 | 0.03442 | nan | nan |
| BASE-O (LO_td050) | 0.18282 | 0.18960 | 0.18943 | 0.18394 | nan | 0.18292 | nan |
| SCORE (LS) | 0.23423 | nan | nan | nan | nan | nan | 0.23438 |

Out-of-fold predictions for every arm and threshold are in
`oof_predictions.parquet`. Stage 3 complete in 51 min.
**Review point** (stages 1, 3 and 6 pause for review).


## Stage 4 -- label-scale scoring (2026-10-08 11:25)

Every score is out of fold (Stage 3's predictions; nothing refitted). Intervals
are 95% spatial block bootstrap: the 250 stored blocks resampled with
replacement, 300 replicates, every arm scored on the same resampled cells,
so a paired difference is taken within each replicate. Per the interpretive
rule, each LOG arm was trained on the label it is scored against, and a
label-scale win for it is not evidence that it infers division better.
Calibration-in-the-large is reported as mean p / label rate (1 = calibrated).
Cell types are the Stage 1 standardised argmax.

### BASE-T (LT_a05, label rate 0.0344)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.1358 [0.1291, 0.1429] | 0.0322 [0.0302, 0.0342] | 0.7544 [0.7484, 0.7607] | 0.0923 [0.0873, 0.0975] | 0.923 [0.895, 0.950] |
| NB-d | 0.1358 [0.1291, 0.1429] | 0.0322 [0.0302, 0.0342] | 0.7541 [0.7482, 0.7603] | 0.0923 [0.0873, 0.0975] | 0.920 [0.893, 0.947] |
| MIX | 0.1353 [0.1287, 0.1420] | 0.0320 [0.0301, 0.0340] | 0.7558 [0.7496, 0.7623] | 0.1023 [0.0963, 0.1086] | 1.123 [1.094, 1.152] |
| LOG-T | 0.1365 [0.1298, 0.1433] | 0.0321 [0.0302, 0.0341] | 0.7424 [0.7350, 0.7497] | 0.0987 [0.0923, 0.1053] | 1.002 [0.975, 1.028] |

Paired differences (first minus second; for log loss and Brier, negative means
the first arm is better; for AUROC and AUPRC, positive):

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | -0.0006 [-0.0009, -0.0002] | -0.0002 [-0.0002, -0.0001] | 0.0015 [0.0002, 0.0029] | 0.0100 [0.0083, 0.0114] |
| NB-d - NB-c | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0002 [-0.0005, 0.0001] | 0.0000 [-0.0001, 0.0002] |
| LOG-T - MIX | 0.0012 [0.0010, 0.0015] | 0.0001 [0.0000, 0.0001] | -0.0134 [-0.0159, -0.0112] | -0.0036 [-0.0046, -0.0027] |
| LOG-T - NB-c | 0.0006 [0.0003, 0.0010] | -0.0001 [-0.0001, -0.0001] | -0.0119 [-0.0150, -0.0093] | 0.0063 [0.0047, 0.0080] |

### BASE-O (LO_td050, label rate 0.1828)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.4501 [0.4443, 0.4564] | 0.1420 [0.1396, 0.1447] | 0.6658 [0.6631, 0.6685] | 0.2887 [0.2835, 0.2941] | 1.037 [1.024, 1.049] |
| NB-d | 0.4500 [0.4442, 0.4563] | 0.1420 [0.1396, 0.1446] | 0.6661 [0.6633, 0.6688] | 0.2890 [0.2837, 0.2943] | 1.036 [1.023, 1.048] |
| MIX | 0.4509 [0.4450, 0.4574] | 0.1422 [0.1397, 0.1448] | 0.6638 [0.6613, 0.6665] | 0.2876 [0.2820, 0.2926] | 1.006 [0.991, 1.020] |
| LOG-O | 0.4622 [0.4560, 0.4688] | 0.1455 [0.1430, 0.1483] | 0.6216 [0.6179, 0.6254] | 0.2497 [0.2445, 0.2554] | 1.001 [0.986, 1.014] |

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | 0.0008 [0.0004, 0.0012] | 0.0001 [0.0000, 0.0003] | -0.0020 [-0.0031, -0.0011] | -0.0012 [-0.0019, -0.0004] |
| NB-d - NB-c | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0000, -0.0000] | 0.0003 [0.0002, 0.0003] | 0.0002 [0.0001, 0.0003] |
| LOG-O - MIX | 0.0114 [0.0107, 0.0119] | 0.0034 [0.0032, 0.0036] | -0.0421 [-0.0445, -0.0394] | -0.0379 [-0.0399, -0.0353] |
| LOG-O - NB-c | 0.0121 [0.0115, 0.0126] | 0.0035 [0.0033, 0.0037] | -0.0441 [-0.0462, -0.0421] | -0.0390 [-0.0410, -0.0369] |

### SCORE (LOG-S only; not comparable with the tables above)

Label rate 0.2342. Log loss 0.5193 [0.5116, 0.5264];
Brier 0.1705 [0.1672, 0.1736];
AUROC 0.6502 [0.6459, 0.6548];
AUPRC 0.3450 [0.3347, 0.3554].

### Murphy decomposition of the Brier score

Brier = reliability - resolution + uncertainty + within-bin, on 20
equal-count bins of each arm's own p. Lower reliability is better calibration;
higher resolution is better discrimination.

**BASE-T**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.032200 | 0.000033 | 0.001083 | 0.033185 | 0.000065 |
| NB-d | 0.032194 | 0.000029 | 0.001083 | 0.033185 | 0.000064 |
| MIX | 0.032034 | 0.000043 | 0.001169 | 0.033185 | -0.000025 |
| LOG-T | 0.032105 | 0.000026 | 0.001111 | 0.033185 | 0.000007 |

**BASE-O**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.142018 | 0.000103 | 0.007522 | 0.149397 | 0.000040 |
| NB-d | 0.141981 | 0.000101 | 0.007548 | 0.149397 | 0.000031 |
| MIX | 0.142153 | 0.000045 | 0.007341 | 0.149397 | 0.000052 |
| LOG-O | 0.145549 | 0.000087 | 0.003965 | 0.149397 | 0.000030 |

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
| BASE-T | NB-c | 0.1358 | 0.1281 | 0.0077 | 0.0923 | 0.0905 | 0.0018 | 0.7544 | 0.7464 |
| BASE-T | NB-d | 0.1358 | 0.1274 | 0.0084 | 0.0923 | 0.0921 | 0.0003 | 0.7541 | 0.7508 |
| BASE-T | MIX | 0.1353 | 0.1470 | -0.0117 | 0.1023 | 0.1094 | -0.0070 | 0.7558 | 0.7579 |
| BASE-T | LOG-T | 0.1365 | 0.1369 | -0.0004 | 0.0987 | 0.0941 | 0.0045 | 0.7424 | 0.7415 |
| BASE-O | NB-c | 0.4501 | 0.4580 | -0.0079 | 0.2887 | 0.3070 | -0.0183 | 0.6658 | 0.6676 |
| BASE-O | NB-d | 0.4500 | 0.4576 | -0.0076 | 0.2890 | 0.3067 | -0.0177 | 0.6661 | 0.6679 |
| BASE-O | MIX | 0.4509 | 0.4546 | -0.0038 | 0.2876 | 0.2891 | -0.0015 | 0.6638 | 0.6544 |
| BASE-O | LOG-O | 0.4622 | 0.4621 | 0.0001 | 0.2497 | 0.2557 | -0.0060 | 0.6216 | 0.6190 |
| SCORE | LOG-S | 0.5193 | 0.5190 | 0.0003 | 0.3450 | 0.3485 | -0.0035 | 0.6502 | 0.6508 |

### Calibration-in-the-large by cell type (mean p / rate; * = 95% interval excludes 1)

**BASE-T**

| stratum | NB-c | NB-d | MIX | LOG-T |
| --- | --- | --- | --- | --- |
| all | 0.923* | 0.920* | 1.123* | 1.002 |
| Endothelial | 1.014 | 1.022 | 1.216* | 1.221* |
| CAFs | 0.957 | 0.827* | 1.192* | 1.151* |
| PVL | 0.998 | 0.993 | 1.112* | 1.050 |
| B-cells | 1.190* | 1.167* | 1.190* | 1.019 |
| T-cells | 1.185* | 1.184* | 1.168* | 0.984 |
| Myeloid | 1.127* | 1.181* | 1.277* | 1.272* |
| Normal Epithelial | 0.903* | 0.853* | 1.211* | 1.083* |
| Plasmablasts | 0.888* | 0.848* | 1.041 | 0.947* |
| Cancer Epithelial | 0.766* | 0.784* | 1.069* | 0.939* |

**BASE-O**

| stratum | NB-c | NB-d | MIX | LOG-O |
| --- | --- | --- | --- | --- |
| all | 1.037* | 1.036* | 1.006 | 1.001 |
| Endothelial | 1.043* | 1.044* | 1.172* | 1.037* |
| CAFs | 1.002 | 0.996 | 1.203* | 0.963* |
| PVL | 1.042* | 1.040* | 1.070* | 1.007 |
| B-cells | 0.980* | 0.984 | 0.929* | 0.989 |
| T-cells | 0.991 | 0.991 | 0.927* | 0.980* |
| Myeloid | 1.104* | 1.104* | 1.149* | 1.071* |
| Normal Epithelial | 1.079* | 1.075* | 0.982 | 1.032 |
| Plasmablasts | 1.053* | 1.055* | 0.956* | 0.982 |
| Cancer Epithelial | 1.066* | 1.062* | 0.971* | 0.990 |

### Threshold sweep (count arms, all cells)

**BASE-T**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| alpha_B 0.05 (primary) | 0.03437 | 0.923 [0.895, 0.950] | 0.920 [0.893, 0.947] | 1.123 [1.094, 1.152] |
| alpha_B 0.01 | 0.01096 | 0.702 [0.668, 0.731] | 0.709 [0.676, 0.737] | 0.743 [0.708, 0.780] |
| alpha_B 0.001 | 0.001999 | 0.544 [0.501, 0.594] | 0.572 [0.528, 0.626] | 0.288 [0.260, 0.324] |
| alpha_B 0.0001 | 0.0003559 | 0.508 [0.438, 0.594] | 0.569 [0.491, 0.664] | 0.101 [0.085, 0.120] |

**BASE-O**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| Td 0 | 0.2737 | 1.026 [1.016, 1.037] | 1.026 [1.016, 1.037] | 0.994 [0.980, 1.007] |
| Td 0.25 | 0.2266 | 1.037 [1.026, 1.049] | 1.037 [1.026, 1.049] | 0.992 [0.978, 1.006] |
| Td 0.5 (primary) | 0.1828 | 1.037 [1.024, 1.049] | 1.036 [1.023, 1.048] | 1.006 [0.991, 1.020] |
| Td 0.75 | 0.1437 | 1.032 [1.017, 1.046] | 1.031 [1.016, 1.044] | 1.031 [1.015, 1.045] |
| Td 1 | 0.1115 | 1.025 [1.008, 1.040] | 1.023 [1.006, 1.038] | 1.059 [1.040, 1.076] |

By cell type, in `calibration_sweep_BASE-T.png` and `calibration_sweep_BASE-O.png`;
by the cell's own threshold c_i, in `calibration_by_threshold.csv`.

### Pre-stated expectations X1-X3

- **X1** (each LOG arm has the best log loss on its own label): BASE-T
  **does not hold**; BASE-O **does not hold**
  (every paired log-loss difference against the LOG arm, interval excluding 0).
- **X2** (NB-c over-predicts where c_i is 5-10 and under-predicts where it is
  15-30, pooled over both grids): mean p - rate +0.00485 at c_i 5-10 and
  -0.01466 at 15-30: **holds**.
- **X3** (MIX better calibrated than NB-c across the grids, but under-predicting
  at the strictest thresholds). The two clauses are judged separately, since
  the second concerns the strictest thresholds. Mean |log(p / rate)| over the
  seven grid thresholds other than alpha_B 0.001 and 0.0001: 0.074 for
  MIX against 0.084 for NB-c (**better**).
  At alpha_B 0.001 and 0.0001 MIX's p / rate is
  0.29, 0.10 (**under-predicts**),
  far below NB-c's, so over all nine thresholds MIX is the worse calibrated
  (0.450 against 0.208). Overall: **holds**.

Figures: `reliability_BASE-T.png`, `reliability_BASE-O.png`,
`reliability_SCORE.png`, `calibration_sweep_BASE-T.png`,
`calibration_sweep_BASE-O.png`. Every score and interval is in `scores.csv`
(including the full per-type metrics and the Murphy components for every
stratum); ceilings in `ceilings.csv`. Stage 4 complete in
2 min.

### Stage 4 reading

**The count arms beat the label-trained logistic arms on their own labels (X1
fails on both).** On BASE-O the margin is wide: LOG-O's log loss is 0.012
worse than NB-c's and its AUPRC is 0.250 against 0.289. On BASE-T, LOG-T's log
loss is worse than both NB-c's and MIX's. Its AUPRC beats NB-c's (0.099
against 0.092) but not MIX's (0.102). By the interpretive rule, this means the
count likelihood carries information that binarisation discards. Two
mechanisms contribute:
- the push-through knows each cell's threshold c_i exactly, which steps with
  depth and area in a way a smooth g(log N, log A) cannot reproduce;
- the count likelihood uses the whole count, not only whether it crosses c_i.

One caveat: the LOG arms carry the mixture's penalty (lambda = 300 at the
mixture's scaling), which was chosen for the mixture's weight model. The
Stage 7 sensitivity with unpenalised NB and LOG arms tests whether the shared
penalty handicaps them.

**MIX against NB is close, and the sign depends on the label.**
- **BASE-T.** MIX is better: log loss -0.0006, AUPRC +0.010 [0.008, 0.011].
  But it over-predicts the label by 12%, where NB-c under-predicts by 8%.
- **BASE-O.** NB-c is marginally better on every score (log loss -0.0008,
  AUROC +0.002), while MIX is the better calibrated (1.006 against 1.037).

These differences are small beside the LOG gap. NB-d is indistinguishable from
NB-c on every score (at most 0.0003 apart): state-dependent dispersion does not
matter at the label scale.

**Calibration by type.**
- **BASE-T, NB-c.** Under-predicts Cancer Epithelial (0.77) and over-predicts
  B- and T-cells (1.19).
- **BASE-T, MIX.** Over-predicts most types by 4-28%. Plasmablasts are
  calibrated.
- **BASE-O.** Every count arm is within 0.93-1.20 by type.
- **Strict BASE-T thresholds.** Both count arms under-predict at alpha_B 0.01
  and below. MIX collapses at the two strictest (0.29, 0.10). This is X3's tail
  shortfall: the mixture's Poisson components are too light-tailed for counts
  far above the null. NB-c holds at about 0.5.

**Noise ceilings.**
- **LOG arms.** LOG-T, LOG-O and LOG-S sit at their own ceilings (|log-loss
  gap| <= 0.0004). They are calibrated, and extract what a model of their form
  can from their label.
- **Count arms.** Their gaps mostly track calibration: NB-c +0.008 on BASE-T,
  where it under-predicts, and MIX -0.012, where it over-predicts. On BASE-O,
  MIX sits close to its ceiling (AUPRC gap -0.0015). NB-c, which over-predicts
  BASE-O by 4%, has a ceiling AUPRC of 0.307 against an observed 0.289.

**X2 holds.** NB-c over-predicts at c_i 5-10 and under-predicts at 15-30.
**X3 holds**, narrowly on its first clause.

**Next: Stage 5** (biology, spec 5, and deviation, spec 6). Spec 5.2's Wu et
al. reference is still not on this machine; everything else in Stage 5 can run
without it.


## Stage 5 -- biology and deviation (2026-10-08 11:53)

Out-of-fold predictions from Stage 3; intervals are the Stage 4 block bootstrap
(250 blocks, 300 replicates). Label rates and mean p are label scale; MIX's
pi and the Rogan-Gladen fraction are state scale; the two are not compared
with each other.

### 5.1 Fractions per cell type

Rows ordered by MIX's mean pi. The Rogan-Gladen row corrects the BASE-T rate
for its false positives (mean alpha_i) and for its sensitivity, which it
borrows from MIX (mean P1(Y >= c^T_i)).

| type | BASE-T rate | BASE-O rate | SCORE rate | MIX mean pi | Rogan-Gladen BASE-T (borrows from MIX) | Wu S/G2M, all patients | Wu S/G2M, ER+ patients |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | 0.081 [0.076, 0.086] | 0.221 [0.214, 0.230] | 0.349 [0.339, 0.361] | 0.219 [0.214, 0.225] | 0.121 [0.109, 0.134] | 0.251 | 0.063 |
| T-cells | 0.032 [0.030, 0.034] | 0.238 [0.234, 0.243] | 0.266 [0.260, 0.273] | 0.162 [0.158, 0.167] | 0.037 [0.028, 0.045] | 0.297 | 0.330 |
| Plasmablasts | 0.043 [0.040, 0.046] | 0.201 [0.195, 0.207] | 0.272 [0.263, 0.281] | 0.149 [0.145, 0.153] | 0.055 [0.044, 0.066] | 0.087 | 0.089 |
| B-cells | 0.024 [0.022, 0.027] | 0.211 [0.206, 0.216] | 0.227 [0.221, 0.233] | 0.136 [0.132, 0.140] | 0.003 [-0.010, 0.018] | 0.219 | 0.277 |
| Normal Epithelial | 0.034 [0.031, 0.037] | 0.167 [0.160, 0.175] | 0.237 [0.228, 0.247] | 0.123 [0.119, 0.128] | 0.012 [0.003, 0.022] | 0.055 | 0.047 |
| PVL | 0.019 [0.017, 0.021] | 0.148 [0.143, 0.152] | 0.182 [0.176, 0.188] | 0.082 [0.078, 0.086] | -0.038 [-0.046, -0.028] | 0.068 | 0.038 |
| Myeloid | 0.015 [0.013, 0.016] | 0.137 [0.133, 0.141] | 0.160 [0.156, 0.164] | 0.066 [0.063, 0.068] | -0.060 [-0.065, -0.054] | 0.137 | 0.186 |
| Endothelial | 0.012 [0.011, 0.013] | 0.122 [0.118, 0.125] | 0.141 [0.137, 0.145] | 0.055 [0.052, 0.058] | -0.074 [-0.079, -0.068] | 0.055 | 0.035 |
| CAFs | 0.010 [0.009, 0.011] | 0.129 [0.124, 0.133] | 0.153 [0.148, 0.158] | 0.033 [0.031, 0.035] | -0.085 [-0.089, -0.080] | 0.035 | 0.026 |
| all | 0.034 [0.032, 0.037] | 0.183 [0.179, 0.187] | 0.234 [0.227, 0.241] | 0.126 [0.122, 0.130] | 0.024 [0.016, 0.032] |  |  |

The Rogan-Gladen fraction is negative, with its interval below 0, in
4 types (Endothelial, CAFs, PVL, Myeloid). There the BASE-T label
fires less often than its own mean alpha_i: the technical null, fitted across
all types with one dispersion (alpha0 = 0.83, widened by the cycling cells),
overstates those types' false-positive rate. The correction is not usable
there; it says the label's false positives are type-dependent, not that the
fraction is below zero.

Each arm's mean p against its label (label scale):

| type | BASE-T rate | NB-c p, BASE-T | MIX p, BASE-T | LOG-T p, BASE-T | BASE-O rate | NB-c p, BASE-O | MIX p, BASE-O | LOG-O p, BASE-O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | 0.0812 | 0.0622 | 0.0867 | 0.0762 | 0.2215 | 0.2361 | 0.2150 | 0.2193 |
| T-cells | 0.0320 | 0.0379 | 0.0373 | 0.0314 | 0.2384 | 0.2362 | 0.2210 | 0.2336 |
| Plasmablasts | 0.0427 | 0.0379 | 0.0445 | 0.0404 | 0.2007 | 0.2114 | 0.1919 | 0.1971 |
| B-cells | 0.0243 | 0.0290 | 0.0290 | 0.0248 | 0.2112 | 0.2069 | 0.1961 | 0.2089 |
| Normal Epithelial | 0.0337 | 0.0304 | 0.0408 | 0.0365 | 0.1670 | 0.1803 | 0.1640 | 0.1724 |
| PVL | 0.0187 | 0.0186 | 0.0208 | 0.0196 | 0.1476 | 0.1539 | 0.1580 | 0.1486 |
| Myeloid | 0.0146 | 0.0164 | 0.0186 | 0.0185 | 0.1372 | 0.1514 | 0.1577 | 0.1469 |
| Endothelial | 0.0116 | 0.0117 | 0.0141 | 0.0141 | 0.1215 | 0.1268 | 0.1425 | 0.1260 |
| CAFs | 0.0097 | 0.0093 | 0.0116 | 0.0112 | 0.1288 | 0.1290 | 0.1550 | 0.1241 |
| all | 0.0344 | 0.0317 | 0.0386 | 0.0344 | 0.1828 | 0.1896 | 0.1839 | 0.1829 |

### 5.2 Anchors

**The section, in situ.** Receptor genes in Cancer Epithelial cells, against a
per-gene technical null (log N offset, area spline, robustly trimmed, as the
labels' null). Descriptive only; no clinical subtype is assigned.

| gene | detected (y >= 1) | above null, p < 0.05 | above null, p < 0.001 | transcripts per 1,000 captured |
| --- | --- | --- | --- | --- |
| ESR1 | 0.4048 | 0.0573 | 0.0045 | 2.9009 |
| PGR | 0.0324 | 0.0215 | 0.0008 | 0.1426 |
| ERBB2 | 0.2335 | 0.0392 | 0.0018 | 1.2379 |

For reference, Normal Epithelial and all other types:

| gene | stratum | detected (y >= 1) | above null, p < 0.05 | transcripts per 1,000 captured |
| --- | --- | --- | --- | --- |
| ESR1 | Normal Epithelial | 0.1960 | 0.0363 | 2.3307 |
| ESR1 | all other types | 0.0274 | 0.0083 | 0.5853 |
| PGR | Normal Epithelial | 0.0149 | 0.0121 | 0.1178 |
| PGR | all other types | 0.0024 | 0.0022 | 0.0381 |
| ERBB2 | Normal Epithelial | 0.1190 | 0.0271 | 1.1226 |
| ERBB2 | all other types | 0.0148 | 0.0056 | 0.2631 |

**The closest Wu subtype.** log2 enrichment of each receptor gene in Cancer
Epithelial over Normal Epithelial (pooled transcripts per transcript
captured), here and in each Wu subtype; distance is Euclidean over the three
genes. Closest: **ER+**. Wu's subtype is the clinical label per patient
(its HER2+ group includes HER2+/ER+ tumours).

| profile | ESR1 | PGR | ERBB2 | distance |
| --- | --- | --- | --- | --- |
| this section (Xenium) | +0.316 | +0.275 | +0.141 | +0.000 |
| Wu ER+ | +1.030 | +1.044 | +0.272 | +1.058 |
| Wu HER2+ | -3.005 | -4.523 | +5.536 | +7.947 |
| Wu TNBC | -5.233 | -5.982 | -0.686 | +8.404 |

**Wu et al. 2021 (GSE176078).** 100,064 cells from 26
primary tumours (10x Chromium). Phase from Seurat's CellCycleScoring via
scanpy's port (43 S and 54 G2M
genes; MLF1IP as CENPU), on LogNormalize expression. S/G2M fraction per type,
pooled over cells and as the median and IQR over patients with at least 20
cells of the type:

| scope | stratum | cells | patients | S/G2M fraction (pooled) | patient median | patient IQR |
| --- | --- | --- | --- | --- | --- | --- |
| all patients | Endothelial | 7605 | 26 | 0.055 | 0.069 | 0.038-0.087 |
| all patients | CAFs | 6573 | 26 | 0.035 | 0.024 | 0.015-0.073 |
| all patients | PVL | 5423 | 26 | 0.068 | 0.096 | 0.036-0.171 |
| all patients | B-cells | 3206 | 20 | 0.219 | 0.212 | 0.158-0.340 |
| all patients | T-cells | 35214 | 26 | 0.297 | 0.329 | 0.263-0.379 |
| all patients | Myeloid | 9675 | 26 | 0.137 | 0.153 | 0.110-0.202 |
| all patients | Normal Epithelial | 4355 | 13 | 0.055 | 0.045 | 0.037-0.073 |
| all patients | Plasmablasts | 3524 | 14 | 0.087 | 0.078 | 0.047-0.137 |
| all patients | Cancer Epithelial | 24489 | 20 | 0.251 | 0.192 | 0.108-0.449 |
| ER+ patients | Endothelial | 5206 | 11 | 0.035 | 0.045 | 0.022-0.080 |
| ER+ patients | CAFs | 2573 | 11 | 0.026 | 0.039 | 0.015-0.073 |
| ER+ patients | PVL | 3265 | 11 | 0.038 | 0.113 | 0.027-0.149 |
| ER+ patients | B-cells | 606 | 8 | 0.277 | 0.274 | 0.169-0.379 |
| ER+ patients | T-cells | 9821 | 11 | 0.330 | 0.301 | 0.270-0.401 |
| ER+ patients | Myeloid | 1831 | 11 | 0.186 | 0.168 | 0.135-0.229 |
| ER+ patients | Normal Epithelial | 2430 | 5 | 0.047 | 0.045 | 0.044-0.046 |
| ER+ patients | Plasmablasts | 631 | 7 | 0.089 | 0.073 | 0.041-0.120 |
| ER+ patients | Cancer Epithelial | 11878 | 9 | 0.063 | 0.099 | 0.054-0.118 |

Platform gap: droplet scRNA-seq at much higher depth (median
3,573 UMIs per cell, against a median N of
57 here), and a score-based phase call that is relative
to control genes, so it calls many non-cycling cells S or G2M. Read the Wu
fractions as an ordering of types more than as absolute rates.

**Literature Ki67 index.** As a range only, and approximate (from memory; to
be checked against a cited source before use): roughly 10-20% in luminal-A-like
tumours, rising to 30-60% or more in HER2-positive and triple-negative ones.
Ki67 protein marks G1 as well as S/G2/M, so a transcript S/G2M fraction should
sit below it; it is an upper bound for Cancer Epithelial, not a target.

### 5.3 Qualitative checks

Rules, fixed before any result was seen: Cancer Epithelial highest -- pass if
ranked first, unclear if second. Plasmablasts and T-cells intermediate -- pass
if both are below Cancer Epithelial and above all four low types, unclear if
each is above at least two of them. CAFs, PVL, Endothelial and Normal
Epithelial low -- pass if all four are in the bottom five of nine, unclear if
three are. Consistency with Wu -- Spearman correlation across the nine types
with the closest-subtype Wu fractions: pass at 0.6 or above, unclear from 0.3.

| source | Cancer Epithelial highest | Plasmablasts, T-cells intermediate | CAFs, PVL, Endothelial, Normal Epi. low | Spearman with Wu ER+ |
| --- | --- | --- | --- | --- |
| MIX mean pi | pass (rank 1) | pass | pass (4/4 in bottom 5) | +0.67 (pass) |
| Rogan-Gladen BASE-T (borrows from MIX) | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.55 (unclear) |
| BASE-T rate | pass (rank 1) | unclear | unclear (3/4 in bottom 5) | +0.47 (unclear) |
| BASE-O rate | unclear (rank 2) | unclear | pass (4/4 in bottom 5) | +0.75 (pass) |
| SCORE rate | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.53 (unclear) |
| Wu S/G2M, all patients | unclear (rank 2) | unclear | pass (4/4 in bottom 5) |  |
| Wu S/G2M, ER+ patients | fail (rank 5) | unclear | pass (4/4 in bottom 5) |  |

### 6.1-6.2 Where the arms disagree

Mean out-of-fold difference by cell type (* = 95% interval excludes 0):

| type | MIX - NB-c, BASE-T | MIX - NB-c, BASE-O | MIX - LOG-T, BASE-T | MIX - LOG-O, BASE-O | NB-c - NB-d, BASE-T | NB-c - NB-d, BASE-O | LOG-O - LOG-S (percentile rank) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | +0.0023* | +0.0157* | -0.0000 | +0.0165* | -0.0001* | -0.0001* | +0.0167* |
| CAFs | +0.0023* | +0.0260* | +0.0004* | +0.0309* | +0.0013* | +0.0008* | -0.0156* |
| PVL | +0.0021* | +0.0041* | +0.0012* | +0.0093* | +0.0001* | +0.0003* | -0.0042 |
| B-cells | -0.0000 | -0.0108* | +0.0042* | -0.0128* | +0.0006* | -0.0008* | +0.1399* |
| T-cells | -0.0006* | -0.0153* | +0.0059* | -0.0126* | +0.0000 | -0.0001* | +0.1325* |
| Myeloid | +0.0022* | +0.0063* | +0.0001 | +0.0108* | -0.0008* | -0.0001* | +0.0254* |
| Normal Epithelial | +0.0104* | -0.0163* | +0.0043* | -0.0084* | +0.0017* | +0.0006* | -0.0824* |
| Plasmablasts | +0.0066* | -0.0195* | +0.0040* | -0.0052* | +0.0017* | -0.0003* | -0.0478* |
| Cancer Epithelial | +0.0246* | -0.0211* | +0.0105* | -0.0043* | -0.0015* | +0.0008* | -0.1410* |

Along the continuous strata, first level to last (decile 1 to 10; tumour
edge from 100 um or more inside to 300 um or more outside):

| difference | neighbourhood tumour fraction decile | density decile | vascular proximity decile (u) | distance to tumour edge (um; inside < 0) |
| --- | --- | --- | --- | --- |
| MIX - NB-c, BASE-T | +0.0009 to +0.0226 | +0.0014 to +0.0010 | +0.0089 to +0.0022 | +0.0238 to +0.0023 |
| MIX - NB-c, BASE-O | +0.0086 to -0.0238 | +0.0213 to -0.0072 | -0.0151 to +0.0162 | -0.0239 to +0.0033 |
| MIX - LOG-T, BASE-T | +0.0024 to +0.0092 | +0.0007 to +0.0041 | +0.0067 to +0.0006 | +0.0092 to +0.0024 |
| MIX - LOG-O, BASE-O | +0.0089 to -0.0035 | +0.0255 to -0.0106 | -0.0063 to +0.0191 | -0.0084 to +0.0067 |
| NB-c - NB-d, BASE-T | +0.0003 to -0.0006 | +0.0002 to +0.0003 | -0.0001 to +0.0002 | -0.0010 to +0.0003 |
| NB-c - NB-d, BASE-O | -0.0002 to +0.0006 | +0.0000 to -0.0004 | +0.0001 to -0.0000 | +0.0003 to +0.0000 |
| LOG-O - LOG-S (percentile rank) | +0.0609 to -0.1588 | -0.0017 to +0.1211 | -0.0093 to -0.0049 | -0.1619 to +0.0336 |

The tumour region (100 um bins with Cancer Epithelial share >= 0.5) holds
13.1% of cells. Every level, with intervals and mean
|difference|, is in `deviation.csv`; the trends are drawn in
`deviation_strata.png`.

**X4** (NB-d moves predictions mainly in Cancer Epithelial): largest mean
|NB-c - NB-d| in Cancer Epithelial (BASE-T) and CAFs (BASE-O) --
**does not hold**.
**X5** (NB-MIX disagreement concentrates in Cancer Epithelial and
Plasmablasts): the two largest mean |MIX - NB-c| are
Cancer Epithelial, Normal Epithelial (BASE-T) and
CAFs, Cancer Epithelial (BASE-O) --
**does not hold**.

### 6.3 Residual maps

Local Moran's I of the out-of-fold label residual L - p on 100 um bins, as
`run/spatial_analysis.py` computes it. HH bins are under-prediction (more
labelled cells than predicted), LL bins over-prediction. The reference is the
mixture's count-residual clustering from the spatial analysis: HH by quadrant
lower-right 46%, lower-left 34%, upper-left 15%, upper-right 5%; LL by quadrant lower-right 46%, upper-left 23%, upper-right 22%, lower-left 10% (quadrants in plotted
coordinates, y upward).

| label | arm | global I | HH bins (under-prediction) | LL bins (over-prediction) | HH overlap with count-residual HH (Jaccard) | LL overlap with count-residual LL (Jaccard) | HH by quadrant | LL by quadrant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE-T | NB-c | 0.116 | 62 | 30 | 0.261 | 0.072 | lower-left 58%, lower-right 42% | lower-left 63%, lower-right 17%, upper-right 17%, upper-left 3% |
| BASE-T | NB-d | 0.114 | 61 | 31 | 0.281 | 0.079 | lower-left 57%, lower-right 43% | lower-left 65%, lower-right 19%, upper-right 13%, upper-left 3% |
| BASE-T | MIX | 0.099 | 33 | 66 | 0.160 | 0.185 | lower-left 55%, lower-right 45% | lower-right 48%, lower-left 41%, upper-right 9%, upper-left 2% |
| BASE-T | LOG-T | 0.094 | 47 | 51 | 0.242 | 0.187 | lower-left 60%, lower-right 40% | lower-right 51%, lower-left 31%, upper-right 16%, upper-left 2% |
| BASE-O | NB-c | 0.157 | 60 | 138 | 0.303 | 0.335 | lower-right 47%, upper-left 27%, lower-left 17%, upper-right 10% | lower-right 43%, upper-right 33%, lower-left 20%, upper-left 4% |
| BASE-O | NB-d | 0.156 | 62 | 142 | 0.308 | 0.335 | lower-right 47%, upper-left 26%, lower-left 18%, upper-right 10% | lower-right 44%, upper-right 32%, lower-left 19%, upper-left 4% |
| BASE-O | MIX | 0.169 | 109 | 112 | 0.408 | 0.392 | lower-right 44%, lower-left 29%, upper-left 21%, upper-right 6% | upper-right 36%, lower-right 31%, lower-left 21%, upper-left 12% |
| BASE-O | LOG-O | 0.153 | 81 | 125 | 0.369 | 0.365 | lower-right 40%, upper-left 27%, lower-left 25%, upper-right 9% | lower-right 48%, upper-right 30%, lower-left 17%, upper-left 5% |
| SCORE | LOG-S | 0.179 | 106 | 154 | 0.267 | 0.380 | lower-right 33%, upper-left 32%, lower-left 31%, upper-right 4% | lower-right 54%, upper-right 23%, lower-left 14%, upper-left 10% |

### 6.4 Over-dispersed types

Stage 4's scores for Cancer Epithelial and Plasmablasts:

| label | type | arm | log loss | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- | --- |
| BASE-T | Plasmablasts | NB-c | 0.1684 [0.1600, 0.1779] | 0.6833 [0.6711, 0.6949] | 0.0812 [0.0748, 0.0895] | 0.8879 [0.8360, 0.9341] |
| BASE-T | Cancer Epithelial | NB-c | 0.2735 [0.2611, 0.2860] | 0.6554 [0.6469, 0.6631] | 0.1302 [0.1238, 0.1358] | 0.7658 [0.7363, 0.7979] |
| BASE-T | Plasmablasts | NB-d | 0.1687 [0.1601, 0.1782] | 0.6841 [0.6721, 0.6952] | 0.0814 [0.0749, 0.0896] | 0.8480 [0.7983, 0.8924] |
| BASE-T | Cancer Epithelial | NB-d | 0.2732 [0.2608, 0.2855] | 0.6541 [0.6456, 0.6614] | 0.1297 [0.1233, 0.1353] | 0.7841 [0.7536, 0.8171] |
| BASE-T | Plasmablasts | MIX | 0.1683 [0.1602, 0.1775] | 0.6801 [0.6674, 0.6922] | 0.0844 [0.0772, 0.0938] | 1.0415 [0.9856, 1.0988] |
| BASE-T | Cancer Epithelial | MIX | 0.2705 [0.2597, 0.2819] | 0.6597 [0.6515, 0.6665] | 0.1411 [0.1340, 0.1474] | 1.0688 [1.0227, 1.1100] |
| BASE-T | Plasmablasts | LOG-T | 0.1698 [0.1615, 0.1792] | 0.6615 [0.6474, 0.6770] | 0.0809 [0.0731, 0.0906] | 0.9468 [0.8938, 0.9998] |
| BASE-T | Cancer Epithelial | LOG-T | 0.2707 [0.2592, 0.2825] | 0.6607 [0.6529, 0.6671] | 0.1429 [0.1354, 0.1494] | 0.9390 [0.8986, 0.9745] |
| BASE-O | Plasmablasts | NB-c | 0.4847 [0.4767, 0.4923] | 0.6316 [0.6248, 0.6381] | 0.2894 [0.2792, 0.2991] | 1.0533 [1.0279, 1.0803] |
| BASE-O | Cancer Epithelial | NB-c | 0.5147 [0.5051, 0.5252] | 0.6183 [0.6126, 0.6245] | 0.2986 [0.2905, 0.3074] | 1.0660 [1.0366, 1.0945] |
| BASE-O | Plasmablasts | NB-d | 0.4846 [0.4766, 0.4922] | 0.6321 [0.6253, 0.6386] | 0.2893 [0.2791, 0.2990] | 1.0548 [1.0294, 1.0818] |
| BASE-O | Cancer Epithelial | NB-d | 0.5146 [0.5050, 0.5251] | 0.6185 [0.6127, 0.6245] | 0.2988 [0.2906, 0.3076] | 1.0624 [1.0328, 1.0909] |
| BASE-O | Plasmablasts | MIX | 0.4836 [0.4750, 0.4919] | 0.6346 [0.6279, 0.6411] | 0.2917 [0.2821, 0.3012] | 0.9560 [0.9331, 0.9787] |
| BASE-O | Cancer Epithelial | MIX | 0.5129 [0.5026, 0.5239] | 0.6253 [0.6193, 0.6314] | 0.3028 [0.2947, 0.3115] | 0.9709 [0.9448, 0.9983] |
| BASE-O | Plasmablasts | LOG-O | 0.4956 [0.4870, 0.5039] | 0.5772 [0.5702, 0.5846] | 0.2470 [0.2384, 0.2561] | 0.9821 [0.9566, 1.0088] |
| BASE-O | Cancer Epithelial | LOG-O | 0.5202 [0.5104, 0.5314] | 0.5936 [0.5866, 0.6009] | 0.2768 [0.2684, 0.2856] | 0.9903 [0.9621, 1.0187] |
| SCORE | Plasmablasts | LOG-S | 0.5741 [0.5652, 0.5824] | 0.5965 [0.5893, 0.6041] | 0.3348 [0.3223, 0.3485] | 0.9850 [0.9572, 1.0127] |
| SCORE | Cancer Epithelial | LOG-S | 0.6354 [0.6289, 0.6428] | 0.5886 [0.5821, 0.5948] | 0.4211 [0.4095, 0.4338] | 0.9890 [0.9671, 1.0125] |

Figures: `fraction_ladder.png`, `deviation_strata.png`, `residual_maps.png`.
Stage 5 complete in 3 min.

### Stage 5 reading

**Only the mixture's state scale passes every biological check.** MIX's mean pi
puts Cancer Epithelial first (0.219), then T-cells, Plasmablasts, B-cells and
Normal Epithelial (0.16-0.12), then PVL, Myeloid, Endothelial and CAFs
(0.08-0.03). That passes all three ordering checks and correlates +0.67 with
the Wu ER+ fractions.

The labels each miss a check:
- **BASE-O** ranks T-cells above Cancer Epithelial (0.238 against 0.221). This
  is the depth floor found in Stage 1: where mu0 is below 1, one or two counts
  clear the threshold.
- **BASE-T and SCORE** put Cancer Epithelial first but do not separate Normal
  Epithelial from the low group.

**The BASE-T label's false-positive rate depends on cell type.** In
Endothelial, CAFs, PVL and Myeloid cells the label fires less often than the
null's own alpha_i, so the Rogan-Gladen correction goes negative. The
technical null, with one dispersion fitted across all types, is too wide for
the quiet types.

**Anchors.** The section's receptor profile is closest to Wu's ER+ tumours,
and far from HER2+ and TNBC (distance 1.1 against about 8). This is
descriptive only:
- ESR1 is detected in 40% of Cancer Epithelial cells, against 20% of Normal
  Epithelial;
- ERBB2 is detected in 23%, against 12%;
- PGR is rare (3%).

In Wu's ER+ patients, Cancer Epithelial is only fifth by S/G2M fraction
(0.063). T-cells (0.33) and B-cells (0.28) top the list. That ordering is
partly the score-based phase call, which is generous to lymphocytes. Read the
Wu fractions as an ordering, and not on the same scale as pi (a mixture
weight, not an S/G2M fraction).

**Where the arms disagree.**
- **MIX against NB-c.** The difference is largest in Cancer Epithelial and in
  tumour-rich neighbourhoods, with opposite signs on the two labels. MIX
  predicts more BASE-T calls (+0.025 in Cancer Epithelial; +0.001 rising to
  +0.023 across neighbourhood tumour deciles) and fewer BASE-O calls (-0.021).
  On BASE-O it also predicts more calls in sparse tissue and fewer in dense
  (+0.021 falling to -0.007 across density deciles).
- **LOG-O against LOG-S.** Ranked against each other, LOG-O puts T- and B-cells
  higher (+0.13 to +0.14 percentile) and LOG-S puts Cancer Epithelial higher
  (-0.14). The OSDR-style label leans to lymphocytes, the module score to
  tumour.
- **X4 does not hold.** NB-d's largest shift is in Cancer Epithelial for BASE-T
  but in CAFs for BASE-O, and every NB-c - NB-d difference is at most 0.002.
- **X5 partly holds.** Cancer Epithelial is in the top two on both labels;
  Plasmablasts are not (Normal Epithelial for BASE-T, CAFs for BASE-O).

**Residual maps.** Every arm, on every label, shows the same two persistent
regions as the mixture's count residuals: under-prediction at the lower left
and over-prediction at the lower right. Overlap with the count-residual
clusters reaches Jaccard 0.41 (BASE-O, MIX). The clustering belongs to the
tissue, not to any one model. Global I runs from 0.09 to 0.18, lowest for
BASE-T under MIX and LOG-T.

**Over-dispersed types (6.4).** In Cancer Epithelial and Plasmablasts, MIX ties
or leads on log loss and AUPRC for both labels, and LOG-O is last.

**Next: Stage 6**, the known-truth simulation (spec 7). It needs the
allocation model and its test V9, then a Spartan array: 10 replicates per
generator (G1-G3) at n = 100,000, refitting NB-c, MIX, LOG-T, LOG-O and LOG-S
in each. Stage 6 is a review point.


## Stage 6 -- known-truth simulation (2026-10-08 12:03)

### Allocation model and V9

SCORE is rebuilt in the simulation by splitting each simulated total across
the 53 Tirosh genes in the panel: per cell type, the gene proportions
of the observed counts (pooled over cells with y >= 1) and one
Dirichlet-multinomial concentration, fitted so the per-gene detection rates
match (exact under the model: each gene's count is beta-binomial). Control
genes are held at their observed values, which holds each cell's control term
fixed: scanpy's score is the scored genes' mean log1p(X s / N) minus the
control genes', and N excludes the Tirosh genes. That decomposition
reproduces scanpy's own score, recomputed with the Tirosh columns permuted
among cells of equal N (which leaves every gene's mean, and so scanpy's control
draw, unchanged), to 1.8e-08.

| type | cells | cells with y >= 1 | concentration a | genes tested | max |rel. error| | within 10% |
| --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 48729 | 11797 | 17.38 | 12 | 0.1063 | 11/12 |
| CAFs | 52079 | 13730 | 13.58 | 13 | 0.07335 | 13/13 |
| PVL | 42834 | 11322 | 14.77 | 14 | 0.1156 | 13/14 |
| B-cells | 59119 | 15576 | 29.13 | 12 | 0.05726 | 12/12 |
| T-cells | 87004 | 26916 | 40.58 | 13 | 0.07692 | 13/13 |
| Myeloid | 69669 | 18058 | 18.69 | 12 | 0.1099 | 11/12 |
| Normal Epithelial | 54508 | 24705 | 28.63 | 38 | 0.1482 | 34/38 |
| Plasmablasts | 42523 | 17776 | 23.71 | 32 | 0.127 | 28/32 |
| Cancer Epithelial | 105432 | 72590 | 51.17 | 51 | 0.2108 | 45/51 |

**V9** (a simulated split of the observed totals reproduces every per-type,
per-gene detection rate within 10%, for genes detected in at least 1% of the
type's cells): **FAIL**, over 197 type-gene pairs.
The largest relative errors:

| type | gene | observed | simulated | rel. error (simulated) |
| --- | --- | --- | --- | --- |
| Cancer Epithelial | BIRC5 | 0.1561 | 0.1890 | 0.2108 |
| Cancer Epithelial | TOP2A | 0.0872 | 0.1012 | 0.1606 |
| Normal Epithelial | BIRC5 | 0.0592 | 0.0680 | 0.1482 |
| Normal Epithelial | TYMS | 0.0281 | 0.0321 | 0.1432 |
| Cancer Epithelial | TYMS | 0.0795 | 0.0906 | 0.1399 |
| Cancer Epithelial | CENPF | 0.0610 | 0.0692 | 0.1335 |
| Plasmablasts | BIRC5 | 0.0533 | 0.0601 | 0.1270 |
| Plasmablasts | TYMS | 0.0260 | 0.0293 | 0.1247 |

The observed variance-to-mean ratio of y in Cancer Epithelial, which G2
matches, is 9.205 (the spec's 7.29 was the SpaNorm-era table's).
Inputs for the Spartan array are in `sim_inputs.npz`.


### STOPPED in Stage 6: V9 (allocation model) failed

Per spec section 8, nothing further was run.

### V9 failure: diagnosis (2026-10-08)

**Two separate problems.**
- **Real misfit, in the canonical proliferation markers.** The single DM
  over-predicts their detection in Cancer Epithelial: BIRC5 +21%, TOP2A +16%,
  CENPF +13%, TYMS +14%. Smaller misses of the same kind appear in Normal
  Epithelial and Plasmablasts. In real cells these genes come together, in the
  minority that are cycling; one profile per type spreads them too thinly.
- **Simulation noise.** The rest of the 17 failures are at detection rates of
  1-3%. There one simulated draw has a relative error of about 4% (1 SD), so a
  10% tolerance on a single draw flags some of 197 pairs by chance.

**Alternatives tried** (`run/division_stage6_v9_diagnosis.py` holds the second):
- **A free two-component DM mixture per type,** with constant weights fitted
  by EM: 27/197 pairs beyond 10%. No better.
- **A state-conditional allocation:** one DM profile for cycling and one for
  non-cycling cells per type, weighted by the mixture's posterior w.
  - Expected (exact) detection rates are within 10% for **all 197 pairs**
    (largest 10.0%, Cancer Epithelial CENPF).
  - One simulated draw misses 5 of 197, all at low detection rates.
  - The cycling profile is markedly more concentrated in every type
    (concentration a1 about 2x a0).

**Decision needed** (the user's):
1. **Allocation model.** The state-conditional allocation would be adopted,
   as a deviation from spec 3.4, and V9 judged on the expected detection rates,
   or the average of many draws, instead of one draw.
2. **Generators.** G1 and G2 would allocate by the simulated Z. G3, with no Z,
   would allocate by the fitted mixture's posterior given the simulated count.

Until then, Stage 6 stops here.


### Allocation model as adopted, and V9 (2026-10-08 13:50)

**Decision (user, 2026-10-08):** the state-conditional allocation replaces
spec 3.4's single Dirichlet-multinomial. Each cell type gets one DM profile
for cycling and one for non-cycling cells, each fitted by maximum likelihood
with cells weighted by the mixture's posterior w (or 1 - w). The simulation
splits a cell's count by the profile of its simulated state (G1, G2), or, in
G3, of a state drawn from the fitted mixture's posterior given the simulated
count. The mixture never sees per-gene counts; the allocation borrows its
posterior only to describe how genes split. V9 is judged on the expected
detection rates (exact under the model), since a single draw's noise at 1-2%
detection is comparable to the 10% tolerance.

| type | cells | mean w | concentration, cycling | concentration, non-cycling | genes tested | max |rel. error|, expected | beyond 10%, one draw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 48729 | 0.0522 | 19.31 | 13.33 | 12 | 0.03992 | 0 |
| CAFs | 52079 | 0.03109 | 21.45 | 10.25 | 13 | 0.07015 | 0 |
| PVL | 42834 | 0.08033 | 18.84 | 9.424 | 14 | 0.06884 | 1 |
| B-cells | 59119 | 0.1362 | 24 | 8.772 | 12 | 0.04034 | 0 |
| T-cells | 87004 | 0.1636 | 31.68 | 11.58 | 13 | 0.04129 | 0 |
| Myeloid | 69669 | 0.05995 | 22.43 | 10.86 | 12 | 0.03803 | 0 |
| Normal Epithelial | 54508 | 0.1192 | 44.17 | 23.98 | 38 | 0.08419 | 0 |
| Plasmablasts | 42523 | 0.1511 | 38.73 | 18.56 | 32 | 0.09115 | 2 |
| Cancer Epithelial | 105432 | 0.2257 | 65.81 | 30.85 | 51 | 0.09963 | 2 |

**V9: pass**, every one of 197 tested type-gene pairs within
10% in expectation. The largest:

| type | gene | observed | expected | one draw | mean of 10 draws | rel. error (expected) |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | CENPF | 0.0610 | 0.0671 | 0.0674 | 0.0672 | 0.0996 |
| Cancer Epithelial | NEK2 | 0.0443 | 0.0486 | 0.0495 | 0.0486 | 0.0970 |
| Plasmablasts | BIRC5 | 0.0533 | 0.0582 | 0.0570 | 0.0579 | 0.0912 |
| Normal Epithelial | CENPF | 0.0215 | 0.0233 | 0.0226 | 0.0237 | 0.0842 |
| Plasmablasts | NEK2 | 0.0121 | 0.0131 | 0.0135 | 0.0135 | 0.0830 |
| Normal Epithelial | TYMS | 0.0281 | 0.0303 | 0.0296 | 0.0301 | 0.0786 |

The SCORE decomposition (gene term minus a fixed control term) reproduces
scanpy's own score, recomputed with the Tirosh columns permuted among cells of
equal N, to 1.8e-08; the stored control term (float32) differs from the
float64 one by at most 9.1e-10. The observed variance-to-mean ratio of y
in Cancer Epithelial, which G2 matches, is 9.205 (the spec's 7.29 was the
SpaNorm-era table's). The inputs for the Spartan array are in `sim_inputs.npz`.
