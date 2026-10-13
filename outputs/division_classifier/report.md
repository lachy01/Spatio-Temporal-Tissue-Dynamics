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


## Stage 1 -- labels (built and frozen 2026-10-13 01:22)

561,897 analysis cells. Cell type is the argmax of column-standardised PhiSpace
scores, agreeing with the stored label for 97.99% of cells.

**Outcome and library size.** y recomputed from the count matrix with the
mixture's 53-gene Tirosh list equals the table's `y_count` for every
cell; N (excluding the Tirosh genes and the 530-gene blocklist) equals the
table's `N` for every cell.

### Technical null (spec 2.1)

mu0 = Nt e^b0 + theta_m m (MisTIC; `model.technical: mistic`): Nt the MisTIC-weighted N (Nt_nonB) as a fixed offset, m the expected count of misassigned Tirosh transcripts (m_cyc), theta_m = exp(eta) estimated (**2.8185**, SE of eta 0.0098); no area term. NB negative log-likelihood minimised directly (scipy L-BFGS-B). Robust fit by iterative trimming at P(Y >= y) < 0.001; stopped on
**tolerance** after 2 fits. Final alpha0 = **0.7889**.

| iteration | n_fit | alpha0 | flagged | flagged_frac | changed | converged |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 561897 | 0.854 | 920 | 0.001637 | 920 | True |
| 2 | 560977 | 0.7889 | 1197 | 0.00213 | 277 | True |

### Label rates

| cell type | cells | BASE-T 0.05 | BASE-T 0.01 | BASE-T 0.001 | BASE-T 0.0001 | BASE-O 0.5 | BASE-O-global 0.5 | BASE-O-neg 0.5 | SCORE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 561897 | 0.03293 | 0.01138 | 0.00213 | 0.0003951 | 0.1818 | 0.1777 | 0.2216 | 0.2342 |
| Endothelial | 48729 | 0.01069 | 0.002627 | 0.0003694 | 4.104e-05 | 0.1242 | 0.1156 | 0.1013 | 0.1408 |
| CAFs | 52079 | 0.008679 | 0.001805 | 0.000192 | 1.92e-05 | 0.1318 | 0.1104 | 0.1077 | 0.153 |
| PVL | 42834 | 0.01464 | 0.003665 | 0.0003735 | 9.338e-05 | 0.1462 | 0.1363 | 0.1291 | 0.1819 |
| B-cells | 59119 | 0.01981 | 0.004398 | 0.0007273 | 0.0001692 | 0.2018 | 0.1907 | 0.1643 | 0.2267 |
| T-cells | 87004 | 0.02727 | 0.007643 | 0.00154 | 0.0001724 | 0.2286 | 0.2163 | 0.1926 | 0.2662 |
| Myeloid | 69669 | 0.0129 | 0.003431 | 0.0003732 | 0.0001148 | 0.1367 | 0.1256 | 0.1172 | 0.1598 |
| Normal Epithelial | 54508 | 0.02644 | 0.008054 | 0.001119 | 0.0001284 | 0.1632 | 0.1578 | 0.2637 | 0.2368 |
| Plasmablasts | 42523 | 0.03292 | 0.009595 | 0.001458 | 0.0001411 | 0.1906 | 0.185 | 0.2359 | 0.2724 |
| Cancer Epithelial | 105432 | 0.09122 | 0.03801 | 0.007844 | 0.001603 | 0.2338 | 0.2592 | 0.4686 | 0.3491 |

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

| N decile | N median | mu0 median | c^T median @0.05 | mean alpha_i @0.05 | c^T median @0.01 | mean alpha_i @0.01 | c^T median @0.001 | mean alpha_i @0.001 | c^T median @0.0001 | mean alpha_i @0.0001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0.1006 | 2 | 0.00943 | 2 | 0.005023 | 3 | 0.0004308 | 4 | 3.686e-05 |
| 2 | 21 | 0.1639 | 2 | 0.02003 | 3 | 0.003224 | 4 | 0.000373 | 5 | 4.051e-05 |
| 3 | 30 | 0.2356 | 2 | 0.02982 | 3 | 0.005126 | 4 | 0.0005071 | 6 | 4.763e-05 |
| 4 | 40 | 0.314 | 3 | 0.02714 | 4 | 0.005357 | 5 | 0.0004881 | 7 | 5.126e-05 |
| 5 | 51 | 0.4053 | 3 | 0.02311 | 4 | 0.005169 | 6 | 0.0005613 | 7 | 5.481e-05 |
| 6 | 65 | 0.5203 | 3 | 0.03133 | 5 | 0.006119 | 7 | 0.0005867 | 8 | 5.845e-05 |
| 7 | 86 | 0.6977 | 4 | 0.03188 | 5 | 0.00637 | 8 | 0.0006356 | 10 | 6.336e-05 |
| 8 | 125 | 1.033 | 5 | 0.03553 | 7 | 0.006999 | 10 | 0.0007011 | 13 | 6.972e-05 |
| 9 | 206 | 1.744 | 7 | 0.03924 | 10 | 0.007787 | 14 | 0.0007768 | 19 | 7.759e-05 |
| 10 | 385 | 3.318 | 11 | 0.04314 | 16 | 0.00862 | 24 | 0.00086 | 31 | 8.59e-05 |

### BASE-O (spec 2.3)

Primary BASE-O scales the residual within each cell type, as OSDR's own code
does (`is_dividing`: one cutoff per type, `Td * sigma_type + Tn`, ddof 0).
Per-type s+: Endothelial 1.087, CAFs 0.918, PVL 1.092, B-cells 1.082, T-cells 1.118, Myeloid 1.065, Normal Epithelial 1.175, Plasmablasts 1.184, Cancer Epithelial 1.540.
BASE-O-global uses one s+ = 1.2507 for all cells.
Negative-codeword floor (BASE-O-neg, decoding noise only, no spillover):
rho = 2.795e-06 per um^2 over 609 negative-control codewords,
|T| = 53, mean b_i = 1.73e-05 against mean mu0 = 0.9196;
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
| all | 561897 | 0.2003 | 0.8201 | 0.2342 | 0.03293 | 0.1818 |
| Endothelial | 48729 | 0.1237 | 0.8764 | 0.1408 | 0.01069 | 0.1242 |
| CAFs | 52079 | 0.09243 | 0.8585 | 0.153 | 0.008679 | 0.1318 |
| PVL | 42834 | 0.1253 | 0.8458 | 0.1819 | 0.01464 | 0.1462 |
| B-cells | 59119 | 0.129 | 0.9019 | 0.2267 | 0.01981 | 0.2018 |
| T-cells | 87004 | 0.1435 | 0.8749 | 0.2662 | 0.02727 | 0.2286 |
| Myeloid | 69669 | 0.1286 | 0.8632 | 0.1598 | 0.0129 | 0.1367 |
| Normal Epithelial | 54508 | 0.161 | 0.7615 | 0.2368 | 0.02644 | 0.1632 |
| Plasmablasts | 42523 | 0.1667 | 0.7627 | 0.2724 | 0.03292 | 0.1906 |
| Cancer Epithelial | 105432 | 0.3153 | 0.7242 | 0.3491 | 0.09122 | 0.2338 |

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
| 0.05 | 0.02973 | 0.02885 | 0.03064 | pass | 0.02919 | 0.02897 |
| 0.01 | 0.006282 | 0.005934 | 0.05866 | pass | 0.006049 | 0.005962 |
| 0.001 | 0.0006336 | 0.0005877 | 0.07809 | pass | 0.0005998 | 0.0005901 |
| 0.0001 | 5.873e-05 | 5.81e-05 | 0.01077 | pass | 5.339e-05 | 5.846e-05 |

BASE-O call rate on null data (reported; no pass criterion):

| Td | null call rate (rebuilt) | null call rate (frozen c) | observed call rate |
| --- | --- | --- | --- |
| 0 | 0.2926 | 0.292 | 0.2779 |
| 0.25 | 0.2509 | 0.2443 | 0.2269 |
| 0.5 | 0.2073 | 0.1978 | 0.1818 |
| 0.75 | 0.1666 | 0.1557 | 0.1423 |
| 1 | 0.132 | 0.1213 | 0.1107 |

Stage 1 complete in 0.3 min. **Review point** (stages 1,
3 and 6 pause for review).


## Stage 2 -- NB refits (2026-10-13 01:56)

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
reproduces the production mixture's logit to 3.6e-15 on every cell. **pass**

### V1 -- convergence

Criterion: |f_t - f_(t-1)| < 1e-9 |f_(t-1)| and ||grad||_inf < 1e-5, on the
summed penalised objective. **pass**: 14
of 14 fits converged. sd(eta) > 20 flagged: **none**.

| arm | data | seed | n | loglik | objective | grad_inf | L-BFGS iters | Newton iters | converged (V1) | attempts | theta | alpha | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-c | all cells | 0 | 561897 | -5.936e+05 | 5.937e+05 | 4.898e-07 | 361 | 11 | pass | 2 | 1 | 0.7139 | 0.3613 | 0.1725 | 0.1765 | 0.4067 | no | 1.429 |
| NB-c | all cells | 1 | 561897 | -5.936e+05 | 5.937e+05 | 3.261e-10 | 239 | 16 | pass | 1 | 1 | 0.7139 | 0.3613 | 0.1725 | 0.1765 | 0.4067 | no | 1.712 |
| NB-d | all cells | 0 | 561897 | -5.934e+05 | 5.935e+05 | 1.895e-07 | 443 | 13 | pass | 1 | 1 | 0.524-0.853 | 0.3617 | 0.173 | 0.1766 | 0.4073 | no | 1.953 |
| NB-d | all cells | 1 | 561897 | -5.934e+05 | 5.935e+05 | 6.997e-06 | 381 | 10 | pass | 1 | 1 | 0.524-0.853 | 0.3617 | 0.173 | 0.1766 | 0.4073 | no | 1.625 |
| NB-c | train fold 0 | 0 | 443599 | -4.685e+05 | 4.685e+05 | 2.599e-06 | 391 | 12 | pass | 1 | 1 | 0.7151 | 0.3599 | 0.1728 | 0.1751 | 0.4099 | no | 1.311 |
| NB-c | train fold 1 | 0 | 455993 | -4.769e+05 | 4.769e+05 | 6.336e-06 | 402 | 7 | pass | 1 | 1 | 0.7182 | 0.361 | 0.1701 | 0.1753 | 0.4063 | no | 0.9857 |
| NB-c | train fold 2 | 0 | 444258 | -4.741e+05 | 4.741e+05 | 1.156e-07 | 422 | 11 | pass | 1 | 1 | 0.7126 | 0.359 | 0.1758 | 0.1771 | 0.4051 | no | 1.22 |
| NB-c | train fold 3 | 0 | 449795 | -4.826e+05 | 4.826e+05 | 1.995e-08 | 370 | 10 | pass | 1 | 1 | 0.7127 | 0.3613 | 0.1756 | 0.1774 | 0.4073 | no | 1.104 |
| NB-c | train fold 4 | 0 | 453943 | -4.723e+05 | 4.723e+05 | 5.012e-06 | 406 | 13 | pass | 1 | 1 | 0.7102 | 0.3644 | 0.1704 | 0.1787 | 0.4075 | no | 1.334 |
| NB-d | train fold 0 | 0 | 443599 | -4.684e+05 | 4.684e+05 | 2.579e-06 | 408 | 10 | pass | 1 | 1 | 0.538-0.855 | 0.3601 | 0.1736 | 0.1754 | 0.4105 | no | 1.345 |
| NB-d | train fold 1 | 0 | 455993 | -4.768e+05 | 4.768e+05 | 4.095e-07 | 394 | 7 | pass | 1 | 1 | 0.531-0.851 | 0.3613 | 0.17 | 0.1753 | 0.4067 | no | 1.121 |
| NB-d | train fold 2 | 0 | 444258 | -4.739e+05 | 4.74e+05 | 6.859e-11 | 422 | 13 | pass | 1 | 1 | 0.525-0.848 | 0.3599 | 0.176 | 0.1775 | 0.4057 | no | 1.581 |
| NB-d | train fold 3 | 0 | 449795 | -4.824e+05 | 4.825e+05 | 8.458e-07 | 373 | 8 | pass | 1 | 1 | 0.503-0.877 | 0.3615 | 0.177 | 0.1779 | 0.4081 | no | 1.143 |
| NB-d | train fold 4 | 0 | 453943 | -4.722e+05 | 4.722e+05 | 4.864e-07 | 540 | 12 | pass | 1 | 1 | 0.518-0.840 | 0.365 | 0.1703 | 0.1783 | 0.4082 | no | 1.638 |

`run/compare_nb_mixture.py`'s matched NB is unpenalised, so it is not the
same model as NB-c, which matches the mixture's penalty.

**Uniqueness.** Two seeds per arm on all cells:

| arm | |objective diff| | max |B diff| | max |a diff| | |theta diff| |
| --- | --- | --- | --- | --- |
| NB-c | 1.16e-10 | 2.94e-12 | 4.27e-13 | 0 |
| NB-d | 0 | 1.22e-11 | 2.6e-15 | 0 |

**NB-d against NB-c.** Log-likelihood -593,444.29 against -593,607.98, a gain of
163.7 for 8 parameters (LR statistic 327). The
dispersion depends on state.

For reference, the mixture's data log-likelihood on the same cells is
-600,031.61. As in the main write-up, the NB fits the per-cell counts better;
the label-scale comparison begins in Stage 3.

Out-of-fold NB predictions (log mu and log alpha, per held-out cell) are saved
in `oof_nb.parquet` for Stage 3. Stage 2 complete in 24 min.


## Stage 3 -- out-of-fold fits (2026-10-13 02:19)

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

Map sha256 `32abc1e083fa4d69...`, equal to the stored map's; Stage 2's
out-of-fold NB predictions use the identical assignment; fold sizes match the
Spartan rank CV on every fold. **pass**

### MIX fold fits

All five converged: **yes**. Each fold's held-out
log-likelihood per cell agrees with the Spartan CV fit for the same fold (rank
3, lambda 300) to nan, so starting
from the full-data estimate did not change where the fits end.

_Added 2026-10-13: the MisTIC CV ran afterwards (job `cv_production`, rank 3, lambda 300);
every fold's held-out log-likelihood per cell agrees with it to 6.5e-12._

| data | converged | iterations | grad_inf | held-out loglik / cell | Spartan CV, same fold | e^delta | mean pi (test) | sd(eta) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train fold 0 | pass | 861 | 2.104e-05 | -1.068 | nan | 1.9 | 0.09681 | 1.323 | 1.138 |
| train fold 1 | pass | 1227 | 2.155e-05 | -1.116 | nan | 1.911 | 0.09925 | 1.319 | 1.522 |
| train fold 2 | pass | 988 | 1.213e-05 | -1.027 | nan | 1.895 | 0.09688 | 1.303 | 1.212 |
| train fold 3 | pass | 1166 | 1.549e-05 | -1.001 | nan | 1.901 | 0.09696 | 1.299 | 1.432 |
| train fold 4 | pass | 1160 | 1.283e-05 | -1.136 | nan | 1.902 | 0.1026 | 1.306 | 1.429 |

### LOG fits and V7 -- separation

Criterion: sd(eta) <= 20 at the production ridge for every LOG fit.
**pass** (largest sd(eta) 0.787);
all converged: **yes**.

| arm | data | converged | iterations | grad_inf | theta (log N) | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 (V7) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOG-T | all cells | pass | 350 | 1.341e-08 | 0.5997 | 0.5644 | 0.3347 | 0.3538 | 0.7745 | no | 0.5023 |
| LOG-T | train fold 0 | pass | 521 | 2.471e-08 | 0.6023 | 0.56 | 0.3269 | 0.3599 | 0.774 | no | 0.5234 |
| LOG-T | train fold 1 | pass | 528 | 2.152e-07 | 0.6112 | 0.5677 | 0.3216 | 0.348 | 0.775 | no | 0.6227 |
| LOG-T | train fold 2 | pass | 396 | 3.517e-09 | 0.5982 | 0.5533 | 0.3443 | 0.3637 | 0.7761 | no | 0.4407 |
| LOG-T | train fold 3 | pass | 460 | 2.38e-08 | 0.6088 | 0.5673 | 0.3487 | 0.3533 | 0.7723 | no | 0.4825 |
| LOG-T | train fold 4 | pass | 356 | 1.613e-09 | 0.5796 | 0.574 | 0.3353 | 0.3527 | 0.7871 | no | 0.421 |
| LOG-O | all cells | pass | 499 | 2.839e-07 | 0.1726 | 0.3255 | 0.1073 | 0.1609 | 0.3612 | no | 0.9301 |
| LOG-O | train fold 0 | pass | 483 | 6.635e-07 | 0.1748 | 0.3275 | 0.108 | 0.1652 | 0.362 | no | 0.7442 |
| LOG-O | train fold 1 | pass | 512 | 1.01e-08 | 0.1783 | 0.3256 | 0.1089 | 0.1644 | 0.36 | no | 0.7513 |
| LOG-O | train fold 2 | pass | 462 | 4.961e-07 | 0.1737 | 0.3303 | 0.1077 | 0.1636 | 0.3628 | no | 0.7727 |
| LOG-O | train fold 3 | pass | 491 | 4.077e-06 | 0.1675 | 0.3245 | 0.1098 | 0.1635 | 0.3638 | no | 0.8576 |
| LOG-O | train fold 4 | pass | 444 | 9.391e-11 | 0.169 | 0.3211 | 0.107 | 0.1566 | 0.3605 | no | 0.7558 |
| LOG-S | all cells | pass | 565 | 7.332e-06 | 0.2872 | 0.3623 | 0.155 | 0.172 | 0.4277 | no | 1.36 |
| LOG-S | train fold 0 | pass | 789 | 5.482e-09 | 0.2897 | 0.3613 | 0.1564 | 0.1692 | 0.4314 | no | 1.043 |
| LOG-S | train fold 1 | pass | 475 | 2.347e-09 | 0.2967 | 0.361 | 0.1561 | 0.1747 | 0.4268 | no | 2.224 |
| LOG-S | train fold 2 | pass | 422 | 5.766e-06 | 0.2854 | 0.3678 | 0.1503 | 0.1725 | 0.4264 | no | 0.6294 |
| LOG-S | train fold 3 | pass | 577 | 2.917e-06 | 0.2767 | 0.362 | 0.1549 | 0.1711 | 0.4313 | no | 0.7122 |
| LOG-S | train fold 4 | pass | 895 | 7.124e-11 | 0.2877 | 0.3597 | 0.1582 | 0.1724 | 0.4295 | no | 0.7538 |

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
| NB-c | LT_a05 | 0.031451 | 0.03252 | 0.00076711 | 1.3936 | pass |
| NB-c | LT_a01 | 0.0076342 | 0.00814 | 0.00038596 | 1.3104 | pass |
| NB-c | LT_a001 | 0.0011118 | 0.00104 | 0.00014848 | -0.48348 | pass |
| NB-c | LT_a0001 | 0.00019169 | 0.00012 | 6.1761e-05 | -1.1608 | pass |
| NB-c | LO_td000 | 0.28598 | 0.28822 | 0.0019136 | 1.1698 | pass |
| NB-c | LO_td025 | 0.23811 | 0.23986 | 0.0018384 | 0.94961 | pass |
| NB-c | LO_td050 | 0.19215 | 0.19408 | 0.0017169 | 1.1245 | pass |
| NB-c | LO_td075 | 0.15087 | 0.15262 | 0.0015677 | 1.1153 | pass |
| NB-c | LO_td100 | 0.11657 | 0.11874 | 0.0014099 | 1.5421 | pass |
| NB-d | LT_a05 | 0.031376 | 0.03008 | 0.0007656 | -1.6927 | pass |
| NB-d | LT_a01 | 0.0077889 | 0.00738 | 0.00038941 | -1.05 | pass |
| NB-d | LT_a001 | 0.0012153 | 0.0012 | 0.0001551 | -0.098352 | pass |
| NB-d | LT_a0001 | 0.00023242 | 0.00026 | 6.7969e-05 | 0.40573 | pass |
| NB-d | LO_td000 | 0.28573 | 0.28278 | 0.0019144 | -1.5416 | pass |
| NB-d | LO_td025 | 0.23772 | 0.23534 | 0.001838 | -1.2942 | pass |
| NB-d | LO_td050 | 0.1917 | 0.19036 | 0.0017154 | -0.78317 | pass |
| NB-d | LO_td075 | 0.15045 | 0.14876 | 0.0015656 | -1.0792 | pass |
| NB-d | LO_td100 | 0.11619 | 0.11508 | 0.0014075 | -0.78907 | pass |
| MIX | LT_a05 | 0.038737 | 0.03822 | 0.00083909 | -0.61582 | pass |
| MIX | LT_a01 | 0.011672 | 0.01168 | 0.00047337 | 0.01645 | pass |
| MIX | LT_a001 | 0.0010172 | 0.00106 | 0.00014237 | 0.30055 | pass |
| MIX | LT_a0001 | 6.6078e-05 | 0.00012 | 3.6349e-05 | 1.4835 | pass |
| MIX | LO_td000 | 0.28217 | 0.28312 | 0.0019312 | 0.4941 | pass |
| MIX | LO_td025 | 0.22775 | 0.22856 | 0.0018262 | 0.44219 | pass |
| MIX | LO_td050 | 0.18337 | 0.1823 | 0.0016946 | -0.63194 | pass |
| MIX | LO_td075 | 0.14681 | 0.14524 | 0.0015536 | -1.0132 | pass |
| MIX | LO_td100 | 0.11718 | 0.11582 | 0.0014135 | -0.96139 | pass |

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
| BASE-T (LT_a05) | 0.03293 | 0.03138 | 0.03129 | 0.03863 | 0.03298 | nan | nan |
| BASE-O (LO_td050) | 0.18184 | 0.19241 | 0.19197 | 0.18377 | nan | 0.18189 | nan |
| SCORE (LS) | 0.23423 | nan | nan | nan | nan | nan | 0.23436 |

Out-of-fold predictions for every arm and threshold are in
`oof_predictions.parquet`. Stage 3 complete in 23 min.
**Review point** (stages 1, 3 and 6 pause for review).


## Stage 4 -- label-scale scoring (2026-10-13 02:22)

Every score is out of fold (Stage 3's predictions; nothing refitted). Intervals
are 95% spatial block bootstrap: the 250 stored blocks resampled with
replacement, 300 replicates, every arm scored on the same resampled cells,
so a paired difference is taken within each replicate. Per the interpretive
rule, each LOG arm was trained on the label it is scored against, and a
label-scale win for it is not evidence that it infers division better.
Calibration-in-the-large is reported as mean p / label rate (1 = calibrated).
Cell types are the Stage 1 standardised argmax.

### BASE-T (LT_a05, label rate 0.0329)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.1294 [0.1231, 0.1354] | 0.0306 [0.0288, 0.0325] | 0.7667 [0.7602, 0.7730] | 0.1049 [0.0984, 0.1124] | 0.953 [0.926, 0.981] |
| NB-d | 0.1294 [0.1231, 0.1354] | 0.0306 [0.0288, 0.0324] | 0.7664 [0.7600, 0.7727] | 0.1048 [0.0982, 0.1123] | 0.950 [0.924, 0.978] |
| MIX | 0.1298 [0.1236, 0.1357] | 0.0306 [0.0288, 0.0324] | 0.7641 [0.7575, 0.7707] | 0.1141 [0.1061, 0.1230] | 1.173 [1.142, 1.206] |
| LOG-T | 0.1293 [0.1231, 0.1351] | 0.0304 [0.0285, 0.0321] | 0.7563 [0.7485, 0.7646] | 0.1250 [0.1164, 0.1337] | 1.002 [0.976, 1.025] |

Paired differences (first minus second; for log loss and Brier, negative means
the first arm is better; for AUROC and AUPRC, positive):

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | 0.0004 [0.0001, 0.0007] | -0.0000 [-0.0001, 0.0000] | -0.0026 [-0.0038, -0.0014] | 0.0092 [0.0074, 0.0108] |
| NB-d - NB-c | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0003 [-0.0006, 0.0001] | -0.0001 [-0.0003, 0.0001] |
| LOG-T - MIX | -0.0006 [-0.0009, -0.0002] | -0.0002 [-0.0003, -0.0002] | -0.0077 [-0.0112, -0.0045] | 0.0110 [0.0089, 0.0131] |
| LOG-T - NB-c | -0.0002 [-0.0005, 0.0002] | -0.0003 [-0.0003, -0.0002] | -0.0103 [-0.0141, -0.0069] | 0.0201 [0.0174, 0.0231] |

### BASE-O (LO_td050, label rate 0.1818)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.4528 [0.4479, 0.4581] | 0.1427 [0.1406, 0.1449] | 0.6535 [0.6506, 0.6561] | 0.2759 [0.2706, 0.2815] | 1.058 [1.045, 1.070] |
| NB-d | 0.4527 [0.4477, 0.4579] | 0.1426 [0.1406, 0.1449] | 0.6539 [0.6510, 0.6565] | 0.2763 [0.2710, 0.2818] | 1.056 [1.043, 1.068] |
| MIX | 0.4542 [0.4494, 0.4599] | 0.1430 [0.1409, 0.1453] | 0.6476 [0.6449, 0.6502] | 0.2715 [0.2668, 0.2757] | 1.011 [0.996, 1.024] |
| LOG-O | 0.4640 [0.4591, 0.4696] | 0.1458 [0.1437, 0.1481] | 0.6055 [0.6017, 0.6096] | 0.2423 [0.2368, 0.2489] | 1.000 [0.986, 1.012] |

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | 0.0014 [0.0008, 0.0020] | 0.0003 [0.0001, 0.0005] | -0.0060 [-0.0076, -0.0045] | -0.0044 [-0.0062, -0.0026] |
| NB-d - NB-c | -0.0002 [-0.0002, -0.0002] | -0.0001 [-0.0001, -0.0001] | 0.0004 [0.0003, 0.0004] | 0.0004 [0.0003, 0.0005] |
| LOG-O - MIX | 0.0098 [0.0090, 0.0106] | 0.0028 [0.0026, 0.0031] | -0.0421 [-0.0458, -0.0382] | -0.0293 [-0.0329, -0.0256] |
| LOG-O - NB-c | 0.0112 [0.0105, 0.0119] | 0.0031 [0.0029, 0.0033] | -0.0480 [-0.0507, -0.0450] | -0.0337 [-0.0362, -0.0310] |

### SCORE (LOG-S only; not comparable with the tables above)

Label rate 0.2342. Log loss 0.5204 [0.5129, 0.5275];
Brier 0.1709 [0.1676, 0.1740];
AUROC 0.6472 [0.6428, 0.6520];
AUPRC 0.3437 [0.3332, 0.3542].

### Murphy decomposition of the Brier score

Brier = reliability - resolution + uncertainty + within-bin, on 20
equal-count bins of each arm's own p. Lower reliability is better calibration;
higher resolution is better discrimination.

**BASE-T**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.030622 | 0.000022 | 0.001224 | 0.031842 | -0.000017 |
| NB-d | 0.030618 | 0.000016 | 0.001219 | 0.031842 | -0.000021 |
| MIX | 0.030606 | 0.000124 | 0.001279 | 0.031842 | -0.000081 |
| LOG-T | 0.030359 | 0.000060 | 0.001440 | 0.031842 | -0.000103 |

**BASE-O**

| arm | Brier | reliability | resolution | uncertainty | within-bin |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.142665 | 0.000194 | 0.006350 | 0.148774 | 0.000047 |
| NB-d | 0.142607 | 0.000177 | 0.006379 | 0.148774 | 0.000036 |
| MIX | 0.142979 | 0.000028 | 0.005873 | 0.148774 | 0.000051 |
| LOG-O | 0.145792 | 0.000012 | 0.003019 | 0.148774 | 0.000026 |

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
| BASE-T | NB-c | 0.1294 | 0.1259 | 0.0035 | 0.1049 | 0.0972 | 0.0077 | 0.7667 | 0.7559 |
| BASE-T | NB-d | 0.1294 | 0.1251 | 0.0043 | 0.1048 | 0.0997 | 0.0051 | 0.7664 | 0.7610 |
| BASE-T | MIX | 0.1298 | 0.1429 | -0.0130 | 0.1141 | 0.1353 | -0.0212 | 0.7641 | 0.7821 |
| BASE-T | LOG-T | 0.1293 | 0.1296 | -0.0004 | 0.1250 | 0.1085 | 0.0166 | 0.7563 | 0.7637 |
| BASE-O | NB-c | 0.4528 | 0.4643 | -0.0115 | 0.2759 | 0.3032 | -0.0273 | 0.6535 | 0.6606 |
| BASE-O | NB-d | 0.4527 | 0.4636 | -0.0110 | 0.2763 | 0.3021 | -0.0259 | 0.6539 | 0.6608 |
| BASE-O | MIX | 0.4542 | 0.4567 | -0.0024 | 0.2715 | 0.2810 | -0.0094 | 0.6476 | 0.6474 |
| BASE-O | LOG-O | 0.4640 | 0.4639 | 0.0002 | 0.2423 | 0.2447 | -0.0024 | 0.6055 | 0.6052 |
| SCORE | LOG-S | 0.5204 | 0.5202 | 0.0003 | 0.3437 | 0.3488 | -0.0051 | 0.6472 | 0.6469 |

### Calibration-in-the-large by cell type (mean p / rate; * = 95% interval excludes 1)

**BASE-T**

| stratum | NB-c | NB-d | MIX | LOG-T |
| --- | --- | --- | --- | --- |
| all | 0.953* | 0.950* | 1.173* | 1.002 |
| Endothelial | 1.062 | 1.088 | 1.219* | 1.189* |
| CAFs | 1.052 | 0.919 | 1.269* | 1.197* |
| PVL | 1.103* | 1.092* | 1.171* | 1.137* |
| B-cells | 1.234* | 1.216* | 1.187* | 1.032 |
| T-cells | 1.159* | 1.166* | 1.134* | 0.982 |
| Myeloid | 1.149* | 1.208* | 1.290* | 1.282* |
| Normal Epithelial | 1.053 | 0.965 | 1.371* | 1.211* |
| Plasmablasts | 0.971 | 0.912* | 1.104* | 1.016 |
| Cancer Epithelial | 0.811* | 0.829* | 1.144* | 0.915* |

**BASE-O**

| stratum | NB-c | NB-d | MIX | LOG-O |
| --- | --- | --- | --- | --- |
| all | 1.058* | 1.056* | 1.011 | 1.000 |
| Endothelial | 1.075* | 1.075* | 1.228* | 1.048* |
| CAFs | 1.032* | 1.026 | 1.272* | 0.970* |
| PVL | 1.070* | 1.067* | 1.124* | 1.010 |
| B-cells | 0.985 | 0.988 | 0.936* | 0.988 |
| T-cells | 0.991 | 0.990 | 0.919* | 0.972* |
| Myeloid | 1.118* | 1.118* | 1.198* | 1.073* |
| Normal Epithelial | 1.136* | 1.128* | 1.015 | 1.038* |
| Plasmablasts | 1.088* | 1.087* | 0.981 | 0.990 |
| Cancer Epithelial | 1.087* | 1.082* | 0.901* | 0.984 |

### Threshold sweep (count arms, all cells)

**BASE-T**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| alpha_B 0.05 (primary) | 0.03293 | 0.953 [0.926, 0.981] | 0.950 [0.924, 0.978] | 1.173 [1.142, 1.206] |
| alpha_B 0.01 | 0.01138 | 0.668 [0.641, 0.694] | 0.681 [0.654, 0.708] | 1.017 [0.976, 1.058] |
| alpha_B 0.001 | 0.00213 | 0.517 [0.479, 0.560] | 0.565 [0.524, 0.612] | 0.474 [0.436, 0.526] |
| alpha_B 0.0001 | 0.0003951 | 0.472 [0.410, 0.561] | 0.573 [0.500, 0.682] | 0.166 [0.142, 0.202] |

**BASE-O**

| threshold | label rate | NB-c p / rate | NB-d p / rate | MIX p / rate |
| --- | --- | --- | --- | --- |
| Td 0 | 0.2779 | 1.030 [1.021, 1.039] | 1.029 [1.020, 1.038] | 1.016 [1.004, 1.028] |
| Td 0.25 | 0.2269 | 1.052 [1.042, 1.062] | 1.050 [1.040, 1.061] | 1.007 [0.993, 1.019] |
| Td 0.5 (primary) | 0.1818 | 1.058 [1.045, 1.070] | 1.056 [1.043, 1.068] | 1.011 [0.996, 1.024] |
| Td 0.75 | 0.1423 | 1.060 [1.046, 1.073] | 1.057 [1.043, 1.070] | 1.032 [1.016, 1.044] |
| Td 1 | 0.1107 | 1.052 [1.037, 1.066] | 1.049 [1.033, 1.062] | 1.058 [1.041, 1.072] |

By cell type, in `calibration_sweep_BASE-T.png` and `calibration_sweep_BASE-O.png`;
by the cell's own threshold c_i, in `calibration_by_threshold.csv`.

### Pre-stated expectations X1-X3

- **X1** (each LOG arm has the best log loss on its own label): BASE-T
  **does not hold**; BASE-O **does not hold**
  (every paired log-loss difference against the LOG arm, interval excluding 0).
- **X2** (NB-c over-predicts where c_i is 5-10 and under-predicts where it is
  15-30, pooled over both grids): mean p - rate +0.00236 at c_i 5-10 and
  -0.01793 at 15-30: **holds**.
- **X3** (MIX better calibrated than NB-c across the grids, but under-predicting
  at the strictest thresholds). The two clauses are judged separately, since
  the second concerns the strictest thresholds. Mean |log(p / rate)| over the
  seven grid thresholds other than alpha_B 0.001 and 0.0001: 0.042 for
  MIX against 0.100 for NB-c (**better**).
  At alpha_B 0.001 and 0.0001 MIX's p / rate is
  0.47, 0.17 (**under-predicts**),
  far below NB-c's, so over all nine thresholds MIX is the worse calibrated
  (0.316 against 0.234). Overall: **holds**.

Figures: `reliability_BASE-T.png`, `reliability_BASE-O.png`,
`reliability_SCORE.png`, `calibration_sweep_BASE-T.png`,
`calibration_sweep_BASE-O.png`. Every score and interval is in `scores.csv`
(including the full per-type metrics and the Murphy components for every
stratum); ceilings in `ceilings.csv`. Stage 4 complete in
3 min.


## Stage 5 -- biology and deviation (2026-10-13 02:23)

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
| Cancer Epithelial | 0.091 [0.086, 0.097] | 0.234 [0.226, 0.242] | 0.349 [0.339, 0.361] | 0.197 [0.193, 0.202] | 0.110 [0.101, 0.121] | 0.251 | 0.063 |
| Plasmablasts | 0.033 [0.031, 0.035] | 0.191 [0.185, 0.196] | 0.272 [0.263, 0.281] | 0.117 [0.114, 0.121] | 0.011 [0.002, 0.020] | 0.087 | 0.089 |
| T-cells | 0.027 [0.026, 0.029] | 0.229 [0.225, 0.233] | 0.266 [0.260, 0.273] | 0.109 [0.106, 0.113] | 0.006 [-0.000, 0.012] | 0.297 | 0.330 |
| Normal Epithelial | 0.026 [0.025, 0.028] | 0.163 [0.157, 0.169] | 0.237 [0.228, 0.247] | 0.100 [0.096, 0.104] | -0.016 [-0.022, -0.010] | 0.055 | 0.047 |
| B-cells | 0.020 [0.018, 0.022] | 0.202 [0.197, 0.206] | 0.227 [0.221, 0.233] | 0.090 [0.087, 0.094] | -0.025 [-0.032, -0.016] | 0.219 | 0.277 |
| PVL | 0.015 [0.013, 0.016] | 0.146 [0.142, 0.150] | 0.182 [0.176, 0.188] | 0.062 [0.059, 0.065] | -0.053 [-0.058, -0.047] | 0.068 | 0.038 |
| Myeloid | 0.013 [0.012, 0.014] | 0.137 [0.133, 0.139] | 0.160 [0.156, 0.164] | 0.048 [0.046, 0.050] | -0.056 [-0.060, -0.052] | 0.137 | 0.186 |
| Endothelial | 0.011 [0.010, 0.012] | 0.124 [0.121, 0.128] | 0.141 [0.137, 0.145] | 0.041 [0.039, 0.043] | -0.062 [-0.066, -0.057] | 0.055 | 0.035 |
| CAFs | 0.009 [0.008, 0.010] | 0.132 [0.128, 0.136] | 0.153 [0.148, 0.158] | 0.024 [0.022, 0.025] | -0.068 [-0.071, -0.065] | 0.035 | 0.026 |
| all | 0.033 [0.031, 0.035] | 0.182 [0.178, 0.186] | 0.234 [0.227, 0.241] | 0.098 [0.094, 0.102] | 0.014 [0.008, 0.020] |  |  |

The Rogan-Gladen fraction is negative, with its interval below 0, in
6 types (Endothelial, CAFs, PVL, B-cells, Myeloid, Normal Epithelial). There the BASE-T label
fires less often than its own mean alpha_i: the technical null, fitted across
all types with one dispersion (widened by the cycling cells),
overstates those types' false-positive rate. The correction is not usable
there; it says the label's false positives are type-dependent, not that the
fraction is below zero.

Each arm's mean p against its label (label scale):

| type | BASE-T rate | NB-c p, BASE-T | MIX p, BASE-T | LOG-T p, BASE-T | BASE-O rate | NB-c p, BASE-O | MIX p, BASE-O | LOG-O p, BASE-O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | 0.0912 | 0.0740 | 0.1044 | 0.0835 | 0.2338 | 0.2542 | 0.2107 | 0.2302 |
| Plasmablasts | 0.0329 | 0.0320 | 0.0363 | 0.0335 | 0.1906 | 0.2073 | 0.1870 | 0.1887 |
| T-cells | 0.0273 | 0.0316 | 0.0309 | 0.0268 | 0.2286 | 0.2265 | 0.2102 | 0.2222 |
| Normal Epithelial | 0.0264 | 0.0278 | 0.0362 | 0.0320 | 0.1632 | 0.1854 | 0.1658 | 0.1695 |
| B-cells | 0.0198 | 0.0244 | 0.0235 | 0.0204 | 0.2018 | 0.1988 | 0.1888 | 0.1995 |
| PVL | 0.0146 | 0.0161 | 0.0171 | 0.0166 | 0.1462 | 0.1564 | 0.1643 | 0.1476 |
| Myeloid | 0.0129 | 0.0148 | 0.0166 | 0.0165 | 0.1367 | 0.1528 | 0.1637 | 0.1466 |
| Endothelial | 0.0107 | 0.0114 | 0.0130 | 0.0127 | 0.1242 | 0.1335 | 0.1525 | 0.1302 |
| CAFs | 0.0087 | 0.0091 | 0.0110 | 0.0104 | 0.1318 | 0.1361 | 0.1676 | 0.1279 |
| all | 0.0329 | 0.0314 | 0.0386 | 0.0330 | 0.1818 | 0.1924 | 0.1838 | 0.1819 |

### 5.2 Anchors

**The section, in situ.** Receptor genes in Cancer Epithelial cells, against a
per-gene technical null (weighted log N offset plus theta m for the gene,
robustly trimmed, as the labels' null). Descriptive only; no clinical subtype is assigned.

| gene | detected (y >= 1) | above null, p < 0.05 | above null, p < 0.001 | transcripts per 1,000 captured |
| --- | --- | --- | --- | --- |
| ESR1 | 0.4048 | 0.0641 | 0.0030 | 2.9009 |
| PGR | 0.0324 | 0.0271 | 0.0011 | 0.1426 |
| ERBB2 | 0.2335 | 0.0408 | 0.0015 | 1.2379 |

For reference, Normal Epithelial and all other types:

| gene | stratum | detected (y >= 1) | above null, p < 0.05 | transcripts per 1,000 captured |
| --- | --- | --- | --- | --- |
| ESR1 | Normal Epithelial | 0.1960 | 0.0236 | 2.3307 |
| ESR1 | all other types | 0.0274 | 0.0054 | 0.5853 |
| PGR | Normal Epithelial | 0.0149 | 0.0131 | 0.1178 |
| PGR | all other types | 0.0024 | 0.0022 | 0.0381 |
| ERBB2 | Normal Epithelial | 0.1190 | 0.0199 | 1.1226 |
| ERBB2 | all other types | 0.0148 | 0.0043 | 0.2631 |

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
| MIX mean pi | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.55 (unclear) |
| Rogan-Gladen BASE-T (borrows from MIX) | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.55 (unclear) |
| BASE-T rate | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.55 (unclear) |
| BASE-O rate | pass (rank 1) | pass | pass (4/4 in bottom 5) | +0.68 (pass) |
| SCORE rate | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.53 (unclear) |
| Wu S/G2M, all patients | unclear (rank 2) | unclear | pass (4/4 in bottom 5) |  |
| Wu S/G2M, ER+ patients | fail (rank 5) | unclear | pass (4/4 in bottom 5) |  |

### 6.1-6.2 Where the arms disagree

Mean out-of-fold difference by cell type (* = 95% interval excludes 0):

| type | MIX - NB-c, BASE-T | MIX - NB-c, BASE-O | MIX - LOG-T, BASE-T | MIX - LOG-O, BASE-O | NB-c - NB-d, BASE-T | NB-c - NB-d, BASE-O | LOG-O - LOG-S (percentile rank) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | +0.0017* | +0.0190* | +0.0003* | +0.0223* | -0.0003* | -0.0001* | +0.0124* |
| CAFs | +0.0019* | +0.0315* | +0.0006* | +0.0397* | +0.0012* | +0.0009* | -0.0196* |
| PVL | +0.0010* | +0.0079* | +0.0005* | +0.0167* | +0.0002* | +0.0004* | -0.0189* |
| B-cells | -0.0009* | -0.0100* | +0.0031* | -0.0107* | +0.0004* | -0.0005* | +0.1074* |
| T-cells | -0.0007* | -0.0162* | +0.0041* | -0.0120* | -0.0002* | +0.0002* | +0.1070* |
| Myeloid | +0.0018* | +0.0109* | +0.0001 | +0.0171* | -0.0008* | +0.0000 | +0.0134* |
| Normal Epithelial | +0.0084* | -0.0197* | +0.0042* | -0.0038* | +0.0023* | +0.0013* | -0.0934* |
| Plasmablasts | +0.0044* | -0.0203* | +0.0029* | -0.0017 | +0.0019* | +0.0002* | -0.0714* |
| Cancer Epithelial | +0.0304* | -0.0434* | +0.0209* | -0.0194* | -0.0016* | +0.0011* | -0.0687* |

Along the continuous strata, first level to last (decile 1 to 10; tumour
edge from 100 um or more inside to 300 um or more outside):

| difference | neighbourhood tumour fraction decile | density decile | vascular proximity decile (u) | distance to tumour edge (um; inside < 0) |
| --- | --- | --- | --- | --- |
| MIX - NB-c, BASE-T | +0.0005 to +0.0259 | +0.0016 to +0.0000 | +0.0090 to +0.0022 | +0.0263 to +0.0022 |
| MIX - NB-c, BASE-O | +0.0111 to -0.0438 | +0.0228 to -0.0064 | -0.0206 to +0.0184 | -0.0436 to +0.0049 |
| MIX - LOG-T, BASE-T | +0.0023 to +0.0165 | +0.0013 to +0.0029 | +0.0080 to +0.0012 | +0.0156 to +0.0029 |
| MIX - LOG-O, BASE-O | +0.0130 to -0.0115 | +0.0290 to -0.0081 | -0.0069 to +0.0230 | -0.0160 to +0.0105 |
| NB-c - NB-d, BASE-T | +0.0002 to -0.0005 | +0.0002 to +0.0002 | -0.0001 to +0.0001 | -0.0009 to +0.0002 |
| NB-c - NB-d, BASE-O | -0.0001 to +0.0012 | +0.0000 to -0.0001 | +0.0005 to +0.0000 | +0.0010 to +0.0001 |
| LOG-O - LOG-S (percentile rank) | +0.0581 to -0.1175 | +0.0225 to +0.0946 | -0.0227 to +0.0111 | -0.1232 to +0.0300 |

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
Cancer Epithelial, CAFs (BASE-O) --
**does not hold**.

### 6.3 Residual maps

Local Moran's I of the out-of-fold label residual L - p on 100 um bins, as
`run/spatial_analysis.py` computes it. HH bins are under-prediction (more
labelled cells than predicted), LL bins over-prediction. The reference is the
mixture's count-residual clustering from the spatial analysis: HH by quadrant
lower-right 46%, lower-left 30%, upper-left 19%, upper-right 5%; LL by quadrant lower-right 45%, upper-left 25%, upper-right 20%, lower-left 10% (quadrants in plotted
coordinates, y upward).

| label | arm | global I | HH bins (under-prediction) | LL bins (over-prediction) | HH overlap with count-residual HH (Jaccard) | LL overlap with count-residual LL (Jaccard) | HH by quadrant | LL by quadrant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE-T | NB-c | 0.110 | 67 | 55 | 0.206 | 0.182 | lower-left 60%, lower-right 40% | upper-right 44%, lower-left 35%, lower-right 16%, upper-left 5% |
| BASE-T | NB-d | 0.109 | 67 | 53 | 0.215 | 0.184 | lower-left 58%, lower-right 42% | upper-right 40%, lower-left 38%, lower-right 17%, upper-left 6% |
| BASE-T | MIX | 0.125 | 37 | 117 | 0.136 | 0.209 | lower-left 62%, lower-right 38% | lower-right 38%, lower-left 33%, upper-right 26%, upper-left 3% |
| BASE-T | LOG-T | 0.093 | 36 | 72 | 0.167 | 0.208 | lower-left 69%, lower-right 31% | lower-right 40%, upper-right 28%, lower-left 28%, upper-left 4% |
| BASE-O | NB-c | 0.165 | 59 | 142 | 0.300 | 0.299 | upper-left 37%, lower-right 31%, lower-left 19%, upper-right 14% | lower-right 44%, upper-right 31%, lower-left 22%, upper-left 4% |
| BASE-O | NB-d | 0.164 | 61 | 142 | 0.295 | 0.299 | upper-left 38%, lower-right 30%, lower-left 20%, upper-right 13% | lower-right 44%, upper-right 31%, lower-left 21%, upper-left 4% |
| BASE-O | MIX | 0.171 | 99 | 86 | 0.420 | 0.439 | lower-right 43%, lower-left 29%, upper-left 22%, upper-right 5% | upper-right 30%, lower-right 29%, upper-left 23%, lower-left 17% |
| BASE-O | LOG-O | 0.143 | 60 | 105 | 0.256 | 0.327 | upper-left 45%, lower-left 28%, lower-right 17%, upper-right 10% | lower-right 56%, upper-right 27%, lower-left 14%, upper-left 3% |
| SCORE | LOG-S | 0.182 | 112 | 158 | 0.244 | 0.320 | upper-left 32%, lower-left 32%, lower-right 31%, upper-right 4% | lower-right 53%, upper-right 25%, lower-left 14%, upper-left 9% |

### 6.4 Over-dispersed types

Stage 4's scores for Cancer Epithelial and Plasmablasts:

| label | type | arm | log loss | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- | --- |
| BASE-T | Plasmablasts | NB-c | 0.1399 [0.1330, 0.1478] | 0.6614 [0.6490, 0.6731] | 0.0571 [0.0528, 0.0633] | 0.9710 [0.9133, 1.0372] |
| BASE-T | Cancer Epithelial | NB-c | 0.2953 [0.2826, 0.3091] | 0.6520 [0.6440, 0.6599] | 0.1494 [0.1413, 0.1581] | 0.8115 [0.7815, 0.8434] |
| BASE-T | Plasmablasts | NB-d | 0.1400 [0.1330, 0.1480] | 0.6624 [0.6503, 0.6737] | 0.0574 [0.0531, 0.0636] | 0.9124 [0.8589, 0.9731] |
| BASE-T | Cancer Epithelial | NB-d | 0.2951 [0.2825, 0.3088] | 0.6511 [0.6429, 0.6590] | 0.1488 [0.1407, 0.1575] | 0.8290 [0.7973, 0.8624] |
| BASE-T | Plasmablasts | MIX | 0.1417 [0.1347, 0.1493] | 0.6496 [0.6356, 0.6626] | 0.0558 [0.0514, 0.0626] | 1.1037 [1.0376, 1.1816] |
| BASE-T | Cancer Epithelial | MIX | 0.2948 [0.2832, 0.3068] | 0.6531 [0.6448, 0.6608] | 0.1613 [0.1522, 0.1715] | 1.1445 [1.0977, 1.1906] |
| BASE-T | Plasmablasts | LOG-T | 0.1416 [0.1345, 0.1497] | 0.6355 [0.6217, 0.6517] | 0.0628 [0.0574, 0.0708] | 1.0162 [0.9496, 1.0910] |
| BASE-T | Cancer Epithelial | LOG-T | 0.2882 [0.2764, 0.3009] | 0.6853 [0.6782, 0.6919] | 0.1888 [0.1794, 0.1989] | 0.9152 [0.8812, 0.9491] |
| BASE-O | Plasmablasts | NB-c | 0.4753 [0.4687, 0.4823] | 0.6166 [0.6104, 0.6232] | 0.2640 [0.2556, 0.2747] | 1.0877 [1.0587, 1.1173] |
| BASE-O | Cancer Epithelial | NB-c | 0.5348 [0.5257, 0.5439] | 0.5969 [0.5900, 0.6028] | 0.2979 [0.2875, 0.3088] | 1.0872 [1.0595, 1.1153] |
| BASE-O | Plasmablasts | NB-d | 0.4752 [0.4686, 0.4822] | 0.6173 [0.6112, 0.6239] | 0.2645 [0.2561, 0.2752] | 1.0866 [1.0577, 1.1161] |
| BASE-O | Cancer Epithelial | NB-d | 0.5345 [0.5254, 0.5436] | 0.5970 [0.5901, 0.6030] | 0.2983 [0.2879, 0.3095] | 1.0824 [1.0544, 1.1105] |
| BASE-O | Plasmablasts | MIX | 0.4733 [0.4659, 0.4811] | 0.6198 [0.6137, 0.6270] | 0.2690 [0.2607, 0.2788] | 0.9814 [0.9570, 1.0051] |
| BASE-O | Cancer Epithelial | MIX | 0.5359 [0.5259, 0.5468] | 0.5968 [0.5902, 0.6033] | 0.2936 [0.2850, 0.3027] | 0.9014 [0.8785, 0.9261] |
| BASE-O | Plasmablasts | LOG-O | 0.4853 [0.4775, 0.4937] | 0.5492 [0.5419, 0.5573] | 0.2175 [0.2102, 0.2248] | 0.9901 [0.9585, 1.0205] |
| BASE-O | Cancer Epithelial | LOG-O | 0.5352 [0.5256, 0.5450] | 0.5932 [0.5853, 0.6005] | 0.2972 [0.2873, 0.3083] | 0.9845 [0.9582, 1.0104] |
| SCORE | Plasmablasts | LOG-S | 0.5752 [0.5663, 0.5836] | 0.5945 [0.5873, 0.6022] | 0.3318 [0.3194, 0.3466] | 0.9733 [0.9454, 1.0011] |
| SCORE | Cancer Epithelial | LOG-S | 0.6357 [0.6292, 0.6430] | 0.5861 [0.5795, 0.5923] | 0.4228 [0.4102, 0.4359] | 0.9955 [0.9727, 1.0195] |

Figures: `fraction_ladder.png`, `deviation_strata.png`, `residual_maps.png`.
Stage 5 complete in 1 min.


### Allocation model as adopted, and V9 (2026-10-13 02:23)

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
| Endothelial | 48729 | 0.03815 | 20.56 | 13.24 | 12 | 0.04059 | 0 |
| CAFs | 52079 | 0.02186 | 22.74 | 10.52 | 13 | 0.07259 | 0 |
| PVL | 42834 | 0.05858 | 20.03 | 9.721 | 14 | 0.07147 | 3 |
| B-cells | 59119 | 0.09017 | 26.62 | 9.162 | 12 | 0.03887 | 0 |
| T-cells | 87004 | 0.1104 | 34.43 | 12.2 | 13 | 0.03824 | 0 |
| Myeloid | 69669 | 0.04306 | 24.4 | 10.93 | 12 | 0.03813 | 0 |
| Normal Epithelial | 54508 | 0.0934 | 45.92 | 23.96 | 38 | 0.08351 | 0 |
| Plasmablasts | 42523 | 0.1159 | 41.05 | 18.38 | 32 | 0.08795 | 1 |
| Cancer Epithelial | 105432 | 0.2062 | 66.33 | 30.47 | 51 | 0.1004 | 2 |

**V9: FAIL**, every one of 197 tested type-gene pairs within
10% in expectation. The largest:

| type | gene | observed | expected | one draw | mean of 10 draws | rel. error (expected) |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | CENPF | 0.0610 | 0.0672 | 0.0675 | 0.0673 | 0.1004 |
| Cancer Epithelial | NEK2 | 0.0443 | 0.0485 | 0.0488 | 0.0486 | 0.0963 |
| Plasmablasts | BIRC5 | 0.0533 | 0.0580 | 0.0580 | 0.0576 | 0.0880 |
| Normal Epithelial | CENPF | 0.0215 | 0.0233 | 0.0236 | 0.0233 | 0.0835 |
| Plasmablasts | NEK2 | 0.0121 | 0.0131 | 0.0137 | 0.0132 | 0.0824 |
| Plasmablasts | CENPF | 0.0175 | 0.0189 | 0.0190 | 0.0188 | 0.0786 |

The SCORE decomposition (gene term minus a fixed control term) reproduces
scanpy's own score, recomputed with the Tirosh columns permuted among cells of
equal N, to 1.8e-08; the stored control term (float32) differs from the
float64 one by at most 9.1e-10. The observed variance-to-mean ratio of y
in Cancer Epithelial, which G2 matches, is 9.205 (the spec's 7.29 was the
SpaNorm-era table's). The inputs for the Spartan array are in `sim_inputs.npz`.


### STOPPED in Stage 6: V9 (allocation model) failed

Per spec section 8, nothing further was run.

_Superseded: the user accepted the V9 exception (Cancer Epithelial CENPF at 10.04%) on 2026-10-13; Stage 6 then ran ("Allocation model as adopted, and V9 (2026-10-13 09:49)" and "Stage 6 results" below)._


## Stage 7 -- sensitivities (2026-10-13 03:33)

Out of fold, scored as in Stage 4 (250-block bootstrap, 300 replicates;
paired differences within replicate). 65 new fits, all converged:
**yes**. V7 at the production ridge for every new LOG
arm: **pass** (largest sd(eta) 0.92).

### 1. The threshold grids, every arm

Each grid label now has its own LOG arm; the count arms are Stage 3's push-through.
The last column is the LOG arm's paired log-loss difference against MIX
(negative = LOG better).

**BASE-T**

| threshold | label rate | NB-c log loss | MIX log loss | LOG-T log loss | NB-c AUPRC | MIX AUPRC | LOG-T AUPRC | LOG-T - MIX log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_B 0.01 | 0.01138 | 0.0550 [0.0510, 0.0586] | 0.0552 [0.0512, 0.0587] | 0.0537 [0.0500, 0.0570] | 0.053 [0.049, 0.057] | 0.060 [0.054, 0.066] | 0.074 [0.068, 0.081] | -0.0015 [-0.0018, -0.0011] |
| alpha_B 0.001 | 0.00213 | 0.0136 [0.0122, 0.0150] | 0.0151 [0.0135, 0.0167] | 0.0132 [0.0118, 0.0144] | 0.014 [0.012, 0.016] | 0.011 [0.010, 0.014] | 0.020 [0.017, 0.025] | -0.0019 [-0.0024, -0.0015] |
| alpha_B 0.0001 | 0.0003951 | 0.0032 [0.0027, 0.0037] | 0.0041 [0.0034, 0.0048] | 0.0031 [0.0027, 0.0036] | 0.003 [0.003, 0.005] | 0.001 [0.001, 0.002] | 0.003 [0.002, 0.005] | -0.0010 [-0.0013, -0.0007] |

**BASE-O**

| threshold | label rate | NB-c log loss | MIX log loss | LOG-O log loss | NB-c AUPRC | MIX AUPRC | LOG-O AUPRC | LOG-O - MIX log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Td 0 | 0.2779 | 0.5461 [0.5387, 0.5526] | 0.5509 [0.5432, 0.5576] | 0.5571 [0.5495, 0.5639] | 0.438 [0.427, 0.448] | 0.421 [0.412, 0.430] | 0.413 [0.402, 0.423] | 0.0063 [0.0053, 0.0072] |
| Td 0.25 | 0.2269 | 0.5077 [0.5016, 0.5134] | 0.5110 [0.5048, 0.5172] | 0.5188 [0.5127, 0.5246] | 0.342 [0.334, 0.349] | 0.330 [0.324, 0.336] | 0.314 [0.306, 0.323] | 0.0078 [0.0067, 0.0087] |
| Td 0.75 | 0.1423 | 0.3911 [0.3867, 0.3961] | 0.3915 [0.3871, 0.3966] | 0.4016 [0.3971, 0.4068] | 0.223 [0.220, 0.228] | 0.223 [0.219, 0.227] | 0.186 [0.182, 0.191] | 0.0101 [0.0094, 0.0107] |
| Td 1 | 0.1107 | 0.3328 [0.3283, 0.3376] | 0.3330 [0.3286, 0.3378] | 0.3418 [0.3375, 0.3466] | 0.178 [0.175, 0.181] | 0.181 [0.177, 0.184] | 0.144 [0.141, 0.148] | 0.0088 [0.0083, 0.0093] |

The primary thresholds are in Stage 4; all of it is drawn in `stage7_grid.png`.

### 2. BASE-O variants (Td 0.5)

BASE-O-global uses one s+ for all cells; BASE-O-neg the negative-codeword floor
(decoding noise only, effectively 1{y >= 2}).

| label | arm | label rate | log loss | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| BASE-O-global | NB-c | 0.1777 | 0.4441 [0.4385, 0.4497] | 0.279 [0.272, 0.286] | 1.065 [1.052, 1.078] |
| BASE-O-global | NB-d | 0.1777 | 0.4439 [0.4383, 0.4495] | 0.279 [0.273, 0.286] | 1.062 [1.049, 1.074] |
| BASE-O-global | MIX | 0.1777 | 0.4456 [0.4398, 0.4515] | 0.273 [0.268, 0.279] | 0.998 [0.982, 1.011] |
| BASE-O-global | LOG-O | 0.1777 | 0.4552 [0.4497, 0.4611] | 0.248 [0.241, 0.256] | 1.000 [0.987, 1.012] |
| BASE-O-neg | NB-c | 0.2216 | 0.4115 [0.4004, 0.4217] | 0.584 [0.564, 0.604] | 1.042 [1.032, 1.052] |
| BASE-O-neg | NB-d | 0.2216 | 0.4115 [0.4004, 0.4217] | 0.584 [0.564, 0.604] | 1.041 [1.031, 1.050] |
| BASE-O-neg | MIX | 0.2216 | 0.4125 [0.4016, 0.4227] | 0.580 [0.560, 0.599] | 1.039 [1.029, 1.049] |
| BASE-O-neg | LOG-O | 0.2216 | 0.4381 [0.4263, 0.4482] | 0.540 [0.517, 0.561] | 1.000 [0.987, 1.015] |

### 3. Unpenalised arms

weight_ridge = 0 (balance term kept). Convergence and size of the fits:

| fit | converged | max_sd_eta | max_norm_a |
| --- | --- | --- | --- |
| LOG-O_unpenalised | 5 | 0.367 | 0.336 |
| LOG-S_unpenalised | 5 | 0.434 | 0.373 |
| LOG-T_unpenalised | 5 | 0.837 | 0.634 |
| NB-c unpenalised | 5 | 0.412 | 0.367 |

| label | arm | log loss | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- |
| BASE-T | LOG-T | 0.1293 [0.1231, 0.1351] | 0.125 [0.116, 0.134] | 1.002 [0.976, 1.025] |
| BASE-T | LOG-T unpenalised | 0.1293 [0.1231, 0.1351] | 0.124 [0.115, 0.133] | 1.002 [0.976, 1.026] |
| BASE-T | NB-c | 0.1294 [0.1231, 0.1354] | 0.105 [0.098, 0.112] | 0.953 [0.926, 0.981] |
| BASE-T | NB-c unpenalised | 0.1294 [0.1231, 0.1354] | 0.105 [0.098, 0.112] | 0.954 [0.927, 0.982] |
| BASE-T | MIX | 0.1298 [0.1236, 0.1357] | 0.114 [0.106, 0.123] | 1.173 [1.142, 1.206] |
| BASE-O | LOG-O | 0.4640 [0.4591, 0.4696] | 0.242 [0.237, 0.249] | 1.000 [0.986, 1.012] |
| BASE-O | LOG-O unpenalised | 0.4641 [0.4591, 0.4696] | 0.242 [0.237, 0.249] | 1.000 [0.986, 1.013] |
| BASE-O | NB-c | 0.4528 [0.4479, 0.4581] | 0.276 [0.271, 0.281] | 1.058 [1.045, 1.070] |
| BASE-O | NB-c unpenalised | 0.4529 [0.4479, 0.4581] | 0.276 [0.270, 0.281] | 1.058 [1.045, 1.070] |
| BASE-O | MIX | 0.4542 [0.4494, 0.4599] | 0.272 [0.267, 0.276] | 1.011 [0.996, 1.024] |
| SCORE | LOG-S | 0.5204 [0.5129, 0.5275] | 0.344 [0.333, 0.354] | 1.001 [0.988, 1.013] |
| SCORE | LOG-S unpenalised | 0.5205 [0.5129, 0.5275] | 0.344 [0.333, 0.354] | 1.001 [0.988, 1.013] |

Paired differences (first minus second):

| label | difference | metric | estimate |
| --- | --- | --- | --- |
| BASE-T | LOG-T unpenalised - LOG-T | log loss | -0.0000 [-0.0000, 0.0000] |
| BASE-T | LOG-T unpenalised - LOG-T | AUPRC | -0.0011 [-0.0015, -0.0008] |
| BASE-T | NB-c unpenalised - NB-c | log loss | 0.0000 [-0.0000, 0.0000] |
| BASE-T | NB-c unpenalised - NB-c | AUPRC | -0.0001 [-0.0002, -0.0000] |
| BASE-T | LOG-T unpenalised - MIX | log loss | -0.0006 [-0.0009, -0.0002] |
| BASE-T | LOG-T unpenalised - MIX | AUPRC | 0.0098 [0.0076, 0.0121] |
| BASE-T | LOG-T unpenalised - NB-c unpenalised | log loss | -0.0002 [-0.0005, 0.0002] |
| BASE-T | LOG-T unpenalised - NB-c unpenalised | AUPRC | 0.0191 [0.0165, 0.0218] |
| BASE-O | LOG-O unpenalised - LOG-O | log loss | 0.0000 [0.0000, 0.0000] |
| BASE-O | LOG-O unpenalised - LOG-O | AUPRC | -0.0001 [-0.0001, -0.0000] |
| BASE-O | NB-c unpenalised - NB-c | log loss | 0.0000 [0.0000, 0.0000] |
| BASE-O | NB-c unpenalised - NB-c | AUPRC | -0.0001 [-0.0002, -0.0000] |
| BASE-O | LOG-O unpenalised - MIX | log loss | 0.0098 [0.0090, 0.0106] |
| BASE-O | LOG-O unpenalised - MIX | AUPRC | -0.0294 [-0.0330, -0.0256] |
| BASE-O | LOG-O unpenalised - NB-c unpenalised | log loss | 0.0112 [0.0105, 0.0119] |
| BASE-O | LOG-O unpenalised - NB-c unpenalised | AUPRC | -0.0336 [-0.0361, -0.0310] |
| SCORE | LOG-S unpenalised - LOG-S | log loss | 0.0000 [-0.0000, 0.0000] |
| SCORE | LOG-S unpenalised - LOG-S | AUPRC | -0.0001 [-0.0001, 0.0000] |

X1 without the penalty: BASE-T: the unpenalised LOG arm still does not have the best log loss; BASE-O: the unpenalised LOG arm still does not have the best log loss.

### 4. MIX with theta = 1

The production mixture has fixed theta = 1 since 2026-10-06, so this row is
the production row in Stages 3-6; there is nothing separate to run.

### 5. Nucleus-only matrix

Not run: it needs the transcript-level file (nucleus overlap per
transcript), which is not on this machine (decided in Stage 1).

Every score and interval is in `stage7_scores.csv`; every fit in
`stage7_fits.csv`. Stage 7 complete in 70 min.


### Allocation model as adopted, and V9 (2026-10-13 09:49)

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
| Endothelial | 48729 | 0.03815 | 20.56 | 13.24 | 12 | 0.04059 | 0 |
| CAFs | 52079 | 0.02186 | 22.74 | 10.52 | 13 | 0.07259 | 0 |
| PVL | 42834 | 0.05858 | 20.03 | 9.721 | 14 | 0.07147 | 3 |
| B-cells | 59119 | 0.09017 | 26.62 | 9.162 | 12 | 0.03887 | 0 |
| T-cells | 87004 | 0.1104 | 34.43 | 12.2 | 13 | 0.03824 | 0 |
| Myeloid | 69669 | 0.04306 | 24.4 | 10.93 | 12 | 0.03813 | 0 |
| Normal Epithelial | 54508 | 0.0934 | 45.92 | 23.96 | 38 | 0.08351 | 0 |
| Plasmablasts | 42523 | 0.1159 | 41.05 | 18.38 | 32 | 0.08795 | 1 |
| Cancer Epithelial | 105432 | 0.2062 | 66.33 | 30.47 | 51 | 0.1004 | 2 |

**V9: pass**, every one of 197 tested type-gene pairs within
10% in expectation, except Cancer Epithelial CENPF at 10.04%, accepted by the user (2026-10-13). The largest:

| type | gene | observed | expected | one draw | mean of 10 draws | rel. error (expected) |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | CENPF | 0.0610 | 0.0672 | 0.0675 | 0.0673 | 0.1004 |
| Cancer Epithelial | NEK2 | 0.0443 | 0.0485 | 0.0488 | 0.0486 | 0.0963 |
| Plasmablasts | BIRC5 | 0.0533 | 0.0580 | 0.0580 | 0.0576 | 0.0880 |
| Normal Epithelial | CENPF | 0.0215 | 0.0233 | 0.0236 | 0.0233 | 0.0835 |
| Plasmablasts | NEK2 | 0.0121 | 0.0131 | 0.0137 | 0.0132 | 0.0824 |
| Plasmablasts | CENPF | 0.0175 | 0.0189 | 0.0190 | 0.0188 | 0.0786 |

The SCORE decomposition (gene term minus a fixed control term) reproduces
scanpy's own score, recomputed with the Tirosh columns permuted among cells of
equal N, to 1.8e-08; the stored control term (float32) differs from the
float64 one by at most 9.1e-10. The observed variance-to-mean ratio of y
in Cancer Epithelial, which G2 matches, is 9.205 (the spec's 7.29 was the
SpaNorm-era table's). The inputs for the Spartan array are in `sim_inputs.npz`.

### Stage 6 results (2026-10-13 13:57)

30 replicates (G1: 10, G2: 10, G3: 10), each
on 100,000 real cells (held-out fold about
19,860 cells). Converged fits: NB-c 30/30, MIX 30/30, LOG-T 30/30, LOG-O 30/30, LOG-S 30/30;
largest LOG sd(eta) 1.00. Values are the median
[min, max] over replicates.

**1. Rank agreement with pi_true** (Spearman, held-out fold):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.970 [0.957, 0.974] | 0.960 [0.954, 0.965] |
| MIX, BASE-T | 0.792 [0.757, 0.809] | 0.775 [0.735, 0.801] |
| MIX, BASE-O | 0.396 [0.375, 0.418] | 0.135 [0.116, 0.163] |
| NB-c, BASE-T | 0.718 [0.677, 0.755] | 0.671 [0.613, 0.703] |
| NB-c, BASE-O | 0.486 [0.463, 0.523] | 0.396 [0.350, 0.421] |
| LOG-T | 0.846 [0.812, 0.875] | 0.867 [0.845, 0.883] |
| LOG-O | 0.771 [0.725, 0.826] | 0.478 [0.397, 0.540] |
| LOG-S | 0.785 [0.779, 0.833] | 0.769 [0.752, 0.818] |

**2. Neighbourhood effect.** Pooled over the types with a true contrast of at
least 0.1 in logit, the slope of predicted on true contrast (1 =
recovered, below 1 = attenuated):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.974 [0.781, 1.153] | 0.958 [0.726, 0.984] |
| MIX, BASE-T | 0.557 [0.450, 0.669] | 0.470 [0.411, 0.610] |
| MIX, BASE-O | 0.144 [0.093, 0.175] | 0.086 [0.033, 0.093] |
| NB-c, BASE-T | 0.431 [0.322, 0.537] | 0.408 [0.313, 0.525] |
| NB-c, BASE-O | 0.207 [0.135, 0.252] | 0.183 [0.145, 0.243] |
| LOG-T | 0.730 [0.574, 0.874] | 0.588 [0.474, 0.676] |
| LOG-O | 0.130 [0.038, 0.165] | 0.043 [0.012, 0.108] |
| LOG-S | 0.229 [0.149, 0.271] | 0.188 [0.119, 0.236] |

Per type, the ratio of predicted to true contrast (median over replicates):

**G1**

| type | true contrast | median N | MIX pi | MIX, BASE-T | MIX, BASE-O | NB-c, BASE-T | NB-c, BASE-O | LOG-T | LOG-O | LOG-S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 1.526 | 48.000 | 0.963 | 0.542 | 0.140 | 0.390 | 0.197 | 0.632 | 0.125 | 0.201 |
| CAFs | 1.331 | 60.000 | 1.013 | 0.470 | 0.006 | 0.196 | 0.067 | 0.676 | -0.027 | 0.121 |
| PVL | 3.386 | 46.000 | 1.001 | 0.430 | 0.134 | 0.333 | 0.170 | 0.629 | 0.145 | 0.201 |
| B-cells | 0.922 | 36.000 | 1.043 | 0.556 | 0.210 | 0.541 | 0.251 | 0.648 | 0.262 | 0.294 |
| T-cells | 0.923 | 44.000 | 0.975 | 0.711 | 0.326 | 0.750 | 0.348 | 0.718 | 0.357 | 0.419 |
| Myeloid | 1.743 | 50.000 | 0.913 | 0.605 | 0.156 | 0.445 | 0.228 | 0.684 | 0.119 | 0.196 |
| Normal Epithelial | 0.920 | 81.000 | 0.967 | 1.445 | 0.170 | 1.007 | 0.454 | 1.539 | 0.022 | 0.455 |
| Plasmablasts | 0.920 | 61.000 | 1.116 | 1.343 | 0.232 | 0.925 | 0.385 | 1.359 | 0.135 | 0.466 |
| Cancer Epithelial | 0.158 | 216.000 | 1.180 | 6.313 | 0.823 | 5.025 | 2.339 | 6.263 | 0.722 | 2.018 |

**G2**

| type | true contrast | median N | MIX pi | MIX, BASE-T | MIX, BASE-O | NB-c, BASE-T | NB-c, BASE-O | LOG-T | LOG-O | LOG-S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 1.526 | 48.000 | 0.931 | 0.503 | 0.066 | 0.414 | 0.193 | 0.564 | 0.028 | 0.140 |
| CAFs | 1.331 | 60.000 | 1.012 | 0.329 | -0.012 | 0.263 | 0.088 | 0.384 | -0.018 | 0.120 |
| PVL | 3.386 | 46.000 | 0.890 | 0.331 | 0.074 | 0.280 | 0.135 | 0.498 | 0.043 | 0.146 |
| B-cells | 0.922 | 36.000 | 0.965 | 0.515 | 0.130 | 0.494 | 0.249 | 0.718 | 0.152 | 0.217 |
| T-cells | 0.923 | 44.000 | 0.927 | 0.679 | 0.242 | 0.661 | 0.330 | 0.680 | 0.271 | 0.361 |
| Myeloid | 1.743 | 50.000 | 0.923 | 0.525 | 0.099 | 0.444 | 0.223 | 0.594 | 0.125 | 0.195 |
| Normal Epithelial | 0.920 | 81.000 | 1.013 | 1.402 | 0.013 | 1.267 | 0.450 | 1.290 | -0.139 | 0.383 |
| Plasmablasts | 0.920 | 61.000 | 1.163 | 1.302 | 0.040 | 1.021 | 0.375 | 1.177 | -0.022 | 0.431 |
| Cancer Epithelial | 0.158 | 216.000 | 2.054 | 6.284 | 0.561 | 6.344 | 2.843 | 5.183 | -0.088 | 2.060 |

**3. Fraction error** (mean over types of |mean prediction - realised cycling
fraction|; only MIX's pi targets this quantity):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.007 [0.004, 0.009] | 0.032 [0.030, 0.035] |
| MIX, BASE-T | 0.055 [0.052, 0.059] | 0.054 [0.049, 0.057] |
| MIX, BASE-O | 0.091 [0.085, 0.093] | 0.089 [0.083, 0.094] |
| NB-c, BASE-T | 0.061 [0.059, 0.065] | 0.053 [0.047, 0.055] |
| NB-c, BASE-O | 0.104 [0.100, 0.108] | 0.096 [0.090, 0.102] |
| LOG-T | 0.055 [0.052, 0.059] | 0.055 [0.049, 0.057] |
| LOG-O | 0.089 [0.086, 0.093] | 0.081 [0.078, 0.085] |
| LOG-S | 0.132 [0.126, 0.140] | 0.124 [0.121, 0.131] |

**G3 (no latent classes): implied cycling fraction by type** (median over
replicates; the truth has no cycling class):

| prediction | Endothelial | CAFs | PVL | B-cells | T-cells | Myeloid | Normal Epithelial | Plasmablasts | Cancer Epithelial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIX pi | 0.100 | 0.072 | 0.147 | 0.224 | 0.260 | 0.124 | 0.217 | 0.251 | 0.374 |
| MIX, BASE-T | 0.012 | 0.011 | 0.016 | 0.024 | 0.030 | 0.016 | 0.027 | 0.029 | 0.073 |
| MIX, BASE-O | 0.142 | 0.150 | 0.163 | 0.192 | 0.222 | 0.159 | 0.195 | 0.213 | 0.293 |
| NB-c, BASE-T | 0.011 | 0.009 | 0.016 | 0.024 | 0.030 | 0.015 | 0.027 | 0.032 | 0.070 |
| NB-c, BASE-O | 0.138 | 0.140 | 0.162 | 0.195 | 0.224 | 0.155 | 0.196 | 0.215 | 0.281 |
| LOG-T | 0.011 | 0.010 | 0.016 | 0.024 | 0.029 | 0.015 | 0.028 | 0.032 | 0.068 |
| LOG-O | 0.137 | 0.134 | 0.158 | 0.196 | 0.223 | 0.155 | 0.194 | 0.212 | 0.285 |
| LOG-S | 0.153 | 0.158 | 0.191 | 0.222 | 0.260 | 0.178 | 0.257 | 0.278 | 0.388 |

**Label-scale scores, held-out fold:**

| generator | prediction | log loss | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- |
| G1 | MIX, BASE-T | 0.140 [0.133, 0.165] | 0.123 [0.107, 0.147] | 1.030 [0.874, 1.059] |
| G1 | MIX, BASE-O | 0.453 [0.448, 0.460] | 0.277 [0.269, 0.284] | 1.011 [0.972, 1.018] |
| G1 | NB-c, BASE-T | 0.142 [0.136, 0.171] | 0.118 [0.101, 0.138] | 0.798 [0.650, 0.823] |
| G1 | NB-c, BASE-O | 0.456 [0.452, 0.463] | 0.268 [0.262, 0.274] | 1.086 [1.059, 1.119] |
| G1 | LOG-T | 0.142 [0.135, 0.168] | 0.118 [0.106, 0.138] | 1.029 [0.868, 1.055] |
| G1 | LOG-O | 0.470 [0.463, 0.475] | 0.225 [0.214, 0.234] | 1.000 [0.982, 1.027] |
| G1 | LOG-S | 0.530 [0.513, 0.532] | 0.296 [0.281, 0.310] | 0.996 [0.968, 1.038] |
| G2 | MIX, BASE-T | 0.154 [0.138, 0.167] | 0.111 [0.102, 0.124] | 1.002 [0.966, 1.127] |
| G2 | MIX, BASE-O | 0.432 [0.427, 0.442] | 0.253 [0.247, 0.262] | 1.003 [0.974, 1.041] |
| G2 | NB-c, BASE-T | 0.154 [0.140, 0.168] | 0.109 [0.098, 0.123] | 1.031 [0.948, 1.130] |
| G2 | NB-c, BASE-O | 0.437 [0.432, 0.444] | 0.243 [0.236, 0.253] | 1.135 [1.100, 1.156] |
| G2 | LOG-T | 0.156 [0.140, 0.169] | 0.105 [0.095, 0.117] | 0.976 [0.917, 1.075] |
| G2 | LOG-O | 0.447 [0.444, 0.456] | 0.194 [0.189, 0.200] | 1.000 [0.974, 1.015] |
| G2 | LOG-S | 0.520 [0.509, 0.527] | 0.278 [0.268, 0.298] | 1.001 [0.984, 1.017] |
| G3 | MIX, BASE-T | 0.126 [0.116, 0.136] | 0.083 [0.069, 0.095] | 1.017 [0.915, 1.106] |
| G3 | MIX, BASE-O | 0.472 [0.469, 0.480] | 0.312 [0.297, 0.326] | 1.018 [0.984, 1.047] |
| G3 | NB-c, BASE-T | 0.124 [0.114, 0.132] | 0.092 [0.073, 0.110] | 0.979 [0.933, 1.082] |
| G3 | NB-c, BASE-O | 0.471 [0.467, 0.477] | 0.316 [0.301, 0.329] | 0.997 [0.974, 1.026] |
| G3 | LOG-T | 0.126 [0.116, 0.135] | 0.082 [0.068, 0.104] | 0.972 [0.920, 1.094] |
| G3 | LOG-O | 0.483 [0.480, 0.487] | 0.284 [0.267, 0.300] | 0.997 [0.969, 1.031] |
| G3 | LOG-S | 0.530 [0.523, 0.533] | 0.373 [0.354, 0.390] | 0.999 [0.976, 1.033] |

**X8** (under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from
1 than MIX, with the largest distortion in small, low-depth types), judged
clause by clause: median |log ratio| over the 9 types with a true contrast of at least 0.1: MIX pi 0.076, LOG-O 2.425, LOG-S 1.526 (both further from 1); largest distortion in CAFs (LOG-O) and CAFs (LOG-S); low-depth types (median N below the overall 57): B-cells, Endothelial, Myeloid, PVL, T-cells. **partly holds**.

Figure: `sim_recovery.png`, the neighbourhood-effect ratio by arm and
generator, and the rank agreement. Every replicate value is in
`sim_recovery.csv`. **Review point** (stages 1, 3 and 6 pause for review).


## Addendum 1 -- pre-registration (recorded 2026-10-13, before any addendum result)

Spec: `instructions/division_classifier_addendum_1.md`. Recorded here before A1 runs, so
before any A2 or simulation result exists. Not edited afterwards.

### Rule for the primary alpha_B and null (addendum section 2.3)

The primary alpha_B, and the primary null (naive, or truncated at some alpha_trim), are the
pair that maximises the median over replicates of LOG-T's pooled neighbourhood-effect slope
under G2. Ties within 0.02 go to the smaller alpha_B. The full surface of slope against
alpha_B is reported for every null under G1 and G2. The choice is fixed by this rule, not by
label rate or by comparability with OSDR. NB and MIX push-through slopes on every grid label
are reported but do not enter the rule.

### Pre-stated expectations X9-X15 (addendum section 8)

- **X9.** BASE-T-trunc gives a lower alpha0 than naive trimming, a higher label rate at each
  alpha_B, and fewer types with a negative Rogan-Gladen fraction. The gain grows with alpha_trim.
- **X10.** V2-trunc passes at every alpha_trim; truncation removes the bias that naive trimming
  would show at alpha_trim = 0.05.
- **X11.** Under G2, LOG-T's neighbourhood slope peaks at alpha_B between 0.05 and 0.1 for every
  null, and stays below 0.8 at its peak (MIX pi: 0.96).
- **X12.** BASE-M's overall rate is below BASE-T's rate at alpha_B = 0.05.
- **X13.** TWO-P's rank agreement with pi_true is within 0.05 of MIX's under G1 and G2. Its
  neighbourhood slope lies strictly between LOG-T's best and MIX's.
- **X14.** Poisson + NB fits clearly better than Poisson + Poisson (likelihood-ratio statistic
  in the hundreds or more), with most of the gain in the 15 to 30+ count range of Cancer
  Epithelial. TWO-PNB's multi-start optima agree. Under G3, TWO-PNB's implied cycling fraction
  is at least as large as MIX's in every type.
- **X15.** If BASE-O-state runs, LOG-O-state's G2 slope improves on LOG-O's by less than 0.1,
  and its per-type ratio spread is not smaller than LOG-T's at the primary alpha_B.

### Decisions on the addendum (user, 2026-10-13)

- V12 runs full (generalised) EM from the TWO-P solution, rates refitted each iteration, up to
  2,000 iterations (SQUAREM allowed); the addendum's section 3.4 records the pass rule and the
  three ways a failure is reported.
- BASE-O-state's s+ is the square root of a gamma GLM fitted to (r - m1(phi))^2 (section 4.2).
- In TWO-P, theta_m m stays outside the component ratio, as in MIX: mu_z = lambda_z + theta_m m.
- Step 2 of the two-step arm uses all 141 shared weight-block columns, density terms included.
- The alpha_B grid {0.001, 0.01, 0.05, 0.1, 0.2} adds 0.1 and 0.2; the 0.0001 labels already
  reported are kept.
- BASE-M comes from TWO-P's step 1; TWO-PNB's rates are reported beside it.

## Addendum stage A1 -- BASE-T-trunc (2026-10-13 22:28)

The MisTIC null (mu0 = Nt e^b0 + theta_m m, NB2) refitted by iterated right-truncation (addendum section 1.2) from the untrimmed fit, at alpha_trim 0.001, 0.01 and 0.05. Labels on the extended grid alpha_B in {0.001, 0.01, 0.05, 0.1, 0.2} for every null; the naive Stage 1 labels are unchanged and reproduced exactly at 0.05, 0.01 and 0.001.

### V11 -- known-null recovery

y simulated from the naive production null (b0 -4.9438, theta_m 2.8185, alpha0 0.7889) with the real Nt and m; each fit started from the untrimmed fit of the simulated data. The naive-trimmed fit at the same level is the contrast (not part of the criterion).

| alpha_trim | fit | beta0 | rel err beta0 | theta_m | rel err theta_m | alpha0 | rel err alpha0 | z beta0 | z log theta_m | z log alpha0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.001 | truncated | -4.95 | 0.001285 | 2.847 | 0.01028 | 0.787 | -0.002461 | -1.802 | 1.031 | -0.4189 |
| 0.001 | naive-trimmed (contrast) | -4.957 | 0.002735 | 2.831 | 0.004344 | 0.7672 | -0.02746 | nan | nan | nan |
| 0.01 | truncated | -4.952 | 0.00176 | 2.87 | 0.01819 | 0.7871 | -0.002254 | -2.259 | 1.686 | -0.3427 |
| 0.01 | naive-trimmed (contrast) | -5.019 | 0.01531 | 2.717 | -0.03592 | 0.6461 | -0.1811 | nan | nan | nan |
| 0.05 | truncated | -4.959 | 0.003089 | 2.889 | 0.02504 | 0.7817 | -0.009176 | -3.126 | 1.907 | -1.121 |
| 0.05 | naive-trimmed (contrast) | -5.409 | 0.09418 | 1.65 | -0.4146 | 0.1246 | -0.8421 | nan | nan | nan |

Largest relative error of the truncated fit: 0.0250 (criterion 0.02): **FAIL**. The z columns give the same errors in standard errors of the truncated fit (exact Hessian), so a miss can be told from sampling noise.

### V2-trunc -- null calibration, end to end

| alpha_trim | alpha_B | label rate (refit) | mean alpha_i | relative error | pass (+/-30%) |
| --- | --- | --- | --- | --- | --- |
| 0.001 | 0.001 | 0.0006407 | 0.0005897 | 0.08644 | pass |
| 0.001 | 0.01 | 0.006049 | 0.00596 | 0.01504 | pass |
| 0.001 | 0.05 | 0.02908 | 0.02896 | 0.00403 | pass |
| 0.001 | 0.1 | 0.05972 | 0.0594 | 0.005472 | pass |
| 0.001 | 0.2 | 0.1244 | 0.1242 | 0.001391 | pass |
| 0.01 | 0.001 | 0.0005054 | 0.0005327 | -0.05123 | pass |
| 0.01 | 0.01 | 0.005665 | 0.005474 | 0.03484 | pass |
| 0.01 | 0.05 | 0.02649 | 0.02646 | 0.001159 | pass |
| 0.01 | 0.1 | 0.05555 | 0.05561 | -0.0009339 | pass |
| 0.01 | 0.2 | 0.1177 | 0.1176 | 0.001059 | pass |
| 0.05 | 0.001 | 0.0005001 | 0.000467 | 0.07076 | pass |
| 0.05 | 0.01 | 0.005017 | 0.004941 | 0.01536 | pass |
| 0.05 | 0.05 | 0.02399 | 0.02382 | 0.007167 | pass |
| 0.05 | 0.1 | 0.05183 | 0.05154 | 0.005658 | pass |
| 0.05 | 0.2 | 0.111 | 0.1106 | 0.003302 | pass |

**pass**.

### STOPPED in addendum stage A1: V11 (truncated-likelihood recovery) failed

largest relative error 0.0250 > 0.02

Per the addendum's section 6, nothing further was run.

### A1 continued: V11 accepted (user, 2026-10-13) (2026-10-13 23:02)

V11 missed its 2% tolerance on theta_m at alpha_trim 0.05 (+2.5%, z = 1.9). The untrimmed maximum-likelihood fit of the same simulated draw is already +1.1% on theta_m (and -1.9 SE on b0), so the truncated fits inherit most of the miss; their departure from that fit is under 2 SE. The user accepted V11 on this evidence, to be tested on more simulated draws later. The section 1.4 diagnostics follow, written from the saved outputs (`run/division_addendum_a1_report.py`).

### Fitted nulls

| null | beta0 | theta_m | alpha0 | trimmed fraction | iterations | stopped on |
| --- | --- | --- | --- | --- | --- | --- |
| naive (Stage 1) | -4.944 | 2.818 | 0.7889 | 0.00213 | 2 | tolerance |
| trunc 0.001 | -4.946 | 2.847 | 0.7901 | 0.00212 | 10 | thresholds unchanged |
| trunc 0.01 | -5.063 | 2.64 | 0.5249 | 0.01879 | 16 | thresholds unchanged |
| trunc 0.05 | -5.137 | 1.963 | 0.2961 | 0.05487 | 20 | max_iter |

At alpha_trim 0.001 the truncated null matches the naive one, which also trims at 0.001. The thresholds at alpha_trim 0.05 still moved for a few dozen boundary cells at the 20-iteration cap; the parameters had settled to 1e-4.

Trimmed fraction (y >= c_trim at the final fit) by N decile and cell type:

| stratum | alpha_trim 0.001 | alpha_trim 0.01 | alpha_trim 0.05 |
| --- | --- | --- | --- |
| N decile 1 | 0.0007268 | 0.008047 | 0.0114 |
| N decile 2 | 0.0004275 | 0.005476 | 0.02019 |
| N decile 3 | 0.0003549 | 0.005081 | 0.03065 |
| N decile 4 | 0.0004361 | 0.006802 | 0.0404 |
| N decile 5 | 0.0004989 | 0.006236 | 0.03776 |
| N decile 6 | 0.0009691 | 0.007358 | 0.02891 |
| N decile 7 | 0.001401 | 0.01167 | 0.04563 |
| N decile 8 | 0.002292 | 0.01769 | 0.05949 |
| N decile 9 | 0.004343 | 0.03517 | 0.09407 |
| N decile 10 | 0.009756 | 0.08439 | 0.1821 |
| Endothelial | 0.0003694 | 0.004207 | 0.01849 |
| CAFs | 0.000192 | 0.002957 | 0.01667 |
| PVL | 0.0003735 | 0.006093 | 0.02542 |
| B-cells | 0.0007273 | 0.00724 | 0.0331 |
| T-cells | 0.00154 | 0.01169 | 0.0438 |
| Myeloid | 0.0003732 | 0.005397 | 0.02281 |
| Normal Epithelial | 0.001046 | 0.01394 | 0.05095 |
| Plasmablasts | 0.001458 | 0.01677 | 0.0579 |
| Cancer Epithelial | 0.007825 | 0.06301 | 0.1458 |

### Label rates

All cells:

| null | alpha_B 0.001 | alpha_B 0.01 | alpha_B 0.05 | alpha_B 0.1 | alpha_B 0.2 |
| --- | --- | --- | --- | --- | --- |
| naive (Stage 1) | 0.00213 | 0.01138 | 0.03293 | 0.0578 | 0.1121 |
| trunc 0.001 | 0.00212 | 0.01134 | 0.03286 | 0.05772 | 0.1121 |
| trunc 0.01 | 0.006183 | 0.01879 | 0.04266 | 0.06958 | 0.1249 |
| trunc 0.05 | 0.01352 | 0.02816 | 0.05487 | 0.08385 | 0.1407 |

By type at alpha_B 0.05:

| type | naive (Stage 1) | trunc 0.001 | trunc 0.01 | trunc 0.05 |
| --- | --- | --- | --- | --- |
| Endothelial | 0.01069 | 0.01069 | 0.01432 | 0.01849 |
| CAFs | 0.008679 | 0.008679 | 0.0124 | 0.01667 |
| PVL | 0.01464 | 0.01457 | 0.01856 | 0.02542 |
| B-cells | 0.01981 | 0.01977 | 0.02595 | 0.0331 |
| T-cells | 0.02727 | 0.02722 | 0.03485 | 0.0438 |
| Myeloid | 0.0129 | 0.01288 | 0.01734 | 0.02281 |
| Normal Epithelial | 0.02644 | 0.02634 | 0.03581 | 0.05093 |
| Plasmablasts | 0.03292 | 0.03288 | 0.0432 | 0.0579 |
| Cancer Epithelial | 0.09122 | 0.09105 | 0.1164 | 0.1458 |

### Rogan-Gladen, sensitivity borrowed from MIX

Fraction (rate - mean alpha_i) / (sens - mean alpha_i) per type, sens = mean P1(Y >= c^T_i) under MIX's out-of-fold cycling component; 95% block-bootstrap intervals (Stage 5's 300 replicates). At alpha_B 0.05:

| type | naive (Stage 1) | trunc 0.001 | trunc 0.01 | trunc 0.05 |
| --- | --- | --- | --- | --- |
| Endothelial | -0.062 [-0.066, -0.057] | -0.062 [-0.066, -0.057] | -0.031 [-0.034, -0.027] | -0.007 [-0.011, -0.004] |
| CAFs | -0.068 [-0.071, -0.065] | -0.068 [-0.071, -0.065] | -0.038 [-0.041, -0.035] | -0.016 [-0.019, -0.013] |
| PVL | -0.053 [-0.058, -0.047] | -0.053 [-0.059, -0.047] | -0.019 [-0.024, -0.013] | +0.013 [+0.008, +0.018] |
| B-cells | -0.025 [-0.032, -0.016] | -0.025 [-0.032, -0.016] | +0.017 [+0.010, +0.026] | +0.051 [+0.043, +0.059] |
| T-cells | +0.006 [-0.000, +0.012] | +0.006 [-0.001, +0.011] | +0.043 [+0.037, +0.049] | +0.074 [+0.068, +0.081] |
| Myeloid | -0.056 [-0.060, -0.052] | -0.056 [-0.060, -0.052] | -0.022 [-0.026, -0.019] | +0.004 [+0.001, +0.008] |
| Normal Epithelial | -0.016 [-0.022, -0.010] | -0.016 [-0.022, -0.010] | +0.020 [+0.015, +0.025] | +0.056 [+0.050, +0.063] |
| Plasmablasts | +0.011 [+0.002, +0.020] | +0.011 [+0.002, +0.020] | +0.049 [+0.041, +0.057] | +0.085 [+0.077, +0.092] |
| Cancer Epithelial | +0.110 [+0.101, +0.121] | +0.110 [+0.101, +0.120] | +0.137 [+0.128, +0.147] | +0.169 [+0.159, +0.180] |

Types (of 9) whose interval lies wholly below 0:

| null | alpha_B 0.001 | alpha_B 0.01 | alpha_B 0.05 | alpha_B 0.1 | alpha_B 0.2 |
| --- | --- | --- | --- | --- | --- |
| naive (Stage 1) | 3 | 5 | 6 | 6 | 7 |
| trunc 0.001 | 3 | 5 | 6 | 6 | 7 |
| trunc 0.01 | 0 | 2 | 4 | 4 | 5 |
| trunc 0.05 | 0 | 0 | 2 | 3 | 4 |

Every type, null and alpha_B is in `trunc_diagnostics.csv`.

### X9 and X10

- **X9** (lower alpha0, higher label rate at each alpha_B, fewer negative Rogan-Gladen types, gains growing with alpha_trim). alpha0 lower at every alpha_trim: **no** (0.7901, 0.5249, 0.2961 against 0.7889; at 0.001 it is marginally higher). Label rate higher at every alpha_B: **no**. Fewer types below 0: **no**. Gain grows with alpha_trim: **yes**. Overall: **partly holds**; every clause holds at alpha_trim 0.01 and 0.05, and at 0.001 the truncated null is the naive one.
- **X10** (V2-trunc passes at every alpha_trim; truncation removes the bias naive trimming shows at 0.05). V2-trunc **passes**. In V11 the naive-trimmed fit at 0.05 is off by up to 84% (alpha0), against 2.5% for the truncated fit. Overall: **holds**.

Stage A1 complete. **Review point** (A1, A3, A5 and A6 pause for review).

### Addendum 1 -- pre-registration of X16-X19 (recorded 2026-10-13, before any section 5 result)

Section 5 (E, depth diagnostics against the Wu reference) was added to the addendum after A1
ran. Recorded here before E1 runs; not edited afterwards.

- **X16.** Over shared technical genes, this section's transcripts per cell are at least 10 times
  lower than Wu's in every type, unlike Ren et al.'s comparable figures. The Tirosh genes' median
  sensitivity ratio is within 2 times of the technical genes', so there is no Tirosh-specific
  probe failure.
- **X17.** Coverage costs little and depth costs a lot. With panel genes at full depth (condition
  ii), SCORE's AUROC against the primary truth stays above 0.9. At this section's depth
  (condition iii, ladder level 1) it falls below 0.75 for SCORE and BASE-T, with w from the
  constant-pi mixture scoring highest of the three.
- **X18.** On the ladder, AUROC rises steeply between factors 1 and 10, and approaches condition
  (ii) by factor 25.
- **X19.** Wu's S/G2M versus G1 composite ratio is at least 4 under the stricter grouping. The
  constant-pi mixture on thinned Wu recovers a ratio closer to Wu's than to 1.9.

## Addendum stage A6 -- depth diagnostics against Wu et al. (2026-10-13 23:13)

### E1 -- per-gene sensitivity of this section

Shared genes (this panel and Wu's transcriptome): 4,917 -- the 53 Tirosh genes, 467 other blocklist genes and 4,397 technical genes. Means per cell of each type; s_gk = mean here / mean in Wu. Intervals pair a 1,000-replicate spatial block bootstrap here with a 1,000-replicate patient bootstrap in Wu.

| type | median transcripts/cell, technical, here | median transcripts/cell, technical, Wu | ratio here/Wu | ratio 95% CI | median s_gk, technical | median s_gk, technical, 95% CI | median s_gk, Tirosh | median s_gk, Tirosh, 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 46 | 648 | 0.07099 | 0.0602-0.134 | 0.1533 | 0.112-0.203 | 0.1746 | 0.122-0.322 |
| CAFs | 58 | 604 | 0.09603 | 0.0756-0.13 | 0.167 | 0.112-0.228 | 0.3485 | 0.233-0.484 |
| PVL | 44 | 438 | 0.1005 | 0.0749-0.159 | 0.2079 | 0.127-0.307 | 0.2177 | 0.121-0.412 |
| B-cells | 35 | 190 | 0.1842 | 0.148-0.232 | 0.3884 | 0.299-0.454 | 0.44 | 0.313-0.656 |
| T-cells | 43 | 221 | 0.1946 | 0.15-0.249 | 0.4607 | 0.337-0.595 | 0.1364 | 0.0952-0.242 |
| Myeloid | 48 | 772 | 0.06218 | 0.0507-0.105 | 0.1523 | 0.101-0.219 | 0.07286 | 0.0497-0.117 |
| Normal Epithelial | 78 | 727 | 0.1073 | 0.083-0.136 | 0.2269 | 0.148-0.283 | 0.5844 | 0.428-0.761 |
| Plasmablasts | 59 | 204 | 0.2892 | 0.187-0.46 | 0.8619 | 0.505-1.12 | 1.143 | 0.752-1.49 |
| Cancer Epithelial | 208 | 854 | 0.2436 | 0.167-0.318 | 0.3198 | 0.255-0.443 | 0.1525 | 0.105-0.285 |

ER+ patients only (the closest Wu subtype):

| type | ratio here/Wu | median s_gk, technical | median s_gk, Tirosh |
| --- | --- | --- | --- |
| Endothelial | 0.06345 | 0.1369 | 0.3017 |
| CAFs | 0.09667 | 0.1679 | 0.3592 |
| PVL | 0.09129 | 0.2017 | 0.3423 |
| B-cells | 0.2509 | 0.3661 | 0.6508 |
| T-cells | 0.3071 | 0.5668 | 0.3113 |
| Myeloid | 0.1098 | 0.2152 | 0.1323 |
| Normal Epithelial | 0.1041 | 0.1888 | 0.59 |
| Plasmablasts | 0.3806 | 0.6747 | 0.8863 |
| Cancer Epithelial | 0.2748 | 0.4958 | 0.5171 |

Every type, scope and gene group, with detection rates and genes per cell, is in `depth_sensitivity.csv`; every gene in `depth_sensitivity_genes.csv`.

Caveats. The Tirosh ratios mix sensitivity with biology: they also carry differences in cycling fraction between this tumour and Wu's patients, so read them against the technical genes. The ratio includes section geometry: a 5 um section captures part of each cell's volume, which is part of this section's sensitivity, not an artefact to remove.

**Against Ren et al. (2025)**, who found transcripts and genes per cell comparable to scRNA-seq on shared genes: this section captures at most 28.9% of Wu's transcripts per cell on shared technical genes in any type (Wu is at least 3.5 times higher), so it does not agree.

- **X16** (at least 10 times fewer transcripts per cell here in every type; Tirosh median sensitivity within 2 times of the technical genes'): first clause **does not hold**; second clause **does not hold** (Tirosh over technical median ratio 0.30-2.58 across types). Overall: **does not hold**.

### V15 -- thinning

Whole Wu transcriptome, 20,000 random cells, every entry thinned at p = 0.1. Of 10,190 genes with at least 1,000 counts: largest relative error 0.2919, literal 1% criterion **not met**. A correct binomial thinning cannot meet it at 1,000 counts, where the thinned total's SD is 9.5% of its mean. Judged instead: on the 3 genes with at least 1.44 million counts (where 1% is four SDs) the largest error is 0.0021 (**pass**), and the largest |z| over every gene with >= 1,000 counts is 3.70 against 4.5 (**pass**). With p = 1 for every cell the thinned panel matrix equals condition (ii) exactly: **pass**. V15: **pass**.

### V16 -- pipeline parity on this section's own counts

The code path run on thinned Wu, run instead on this section's raw counts (every panel gene): y and N equal Stage 1's for every cell: **True**; SCORE label equals Stage 1's LS: **True**; BASE-T at alpha_B 0.05 equals a plain-N rebuild (robust null with a log N offset, no MisTIC term): **True**. V16: **pass**.

### E2 -- thinning Wu to this section's depth

Same Wu cells in every condition: (i) whole transcriptome, (ii) panel genes only at full depth, (iii) panel genes binomially thinned so each cell's depth matches the same quantile of this section's depth distribution within its type, times the ladder factor (5 seeds per level; mean shown). Every condition keeps cells with N >= 10, as this section does. Methods as on this section: SCORE (scanpy score_genes, pool = the condition's genes minus the blocklist; robust half-normal threshold), BASE-T (NB null with a plain log N offset, trimmed as in Stage 1, alpha_B 0.05; statistic -log p) and the constant-pi Poisson mixture (posterior w; MAP call w > 0.5). Truth: Seurat's whole-transcriptome S/G2M calls (primary); stricter: S/G2M cells above their type's 75th percentile of max(S, G2M) against G1 cells with both scores below 0.

Primary truth, all cells:

| condition | thinning | factor | method | median depth | AUROC | kappa | sensitivity | specificity | call rate | calls with y <= 1 | Spearman (type rates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (i) whole transcriptome | none | nan | BASE-T | 3573 | 0.8848 | 0.5203 | 0.4238 | 0.9912 | 0.09263 | 0 | 0.9 |
| (i) whole transcriptome | none | nan | SCORE | 3573 | 0.9091 | 0.6157 | 0.606 | 0.9576 | 0.1563 | 0.05377 | 0.9667 |
| (i) whole transcriptome | none | nan | mixture w | 3573 | 0.8597 | 0.4353 | 0.3308 | 0.9974 | 0.06895 | 0 | 0.8833 |
| (ii) panel, full depth | none | nan | BASE-T | 475 | 0.8877 | 0.5248 | 0.4314 | 0.9898 | 0.09531 | 0 | 0.9667 |
| (ii) panel, full depth | none | nan | SCORE | 475 | 0.9123 | 0.6206 | 0.705 | 0.9202 | 0.2061 | 0.05615 | 0.9667 |
| (ii) panel, full depth | none | nan | mixture w | 475 | 0.8558 | 0.4357 | 0.3323 | 0.9968 | 0.06971 | 0 | 0.9333 |
| (iii) panel, thinned | gene-specific | 1 | BASE-T | 66 | 0.6983 | 0.2701 | 0.1906 | 0.9979 | 0.03928 | 0 | 0.93 |
| (iii) panel, thinned | gene-specific | 1 | SCORE | 66 | 0.7503 | 0.3817 | 0.5024 | 0.8787 | 0.1965 | 0.5339 | 0.9467 |
| (iii) panel, thinned | gene-specific | 1 | mixture w | 66 | 0.7753 | 0.3405 | 0.2545 | 0.9936 | 0.05534 | 0 | 0.9367 |
| (iii) panel, thinned | quantile-matched | 1 | BASE-T | 67.8 | 0.719 | 0.2903 | 0.2088 | 0.9966 | 0.04428 | 0 | 0.9167 |
| (iii) panel, thinned | quantile-matched | 1 | SCORE | 67.8 | 0.7623 | 0.3842 | 0.5637 | 0.8466 | 0.2351 | 0.5067 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 1 | mixture w | 67.8 | 0.7778 | 0.3423 | 0.2581 | 0.9926 | 0.05728 | 0 | 0.94 |
| (iii) panel, thinned | quantile-matched | 2 | BASE-T | 131 | 0.7717 | 0.3689 | 0.2766 | 0.9952 | 0.0597 | 0 | 0.9533 |
| (iii) panel, thinned | quantile-matched | 2 | SCORE | 131 | 0.8113 | 0.4459 | 0.643 | 0.8452 | 0.2534 | 0.3552 | 0.9667 |
| (iii) panel, thinned | quantile-matched | 2 | mixture w | 131 | 0.8063 | 0.3798 | 0.2882 | 0.994 | 0.06298 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 5 | BASE-T | 322.8 | 0.8636 | 0.4875 | 0.3905 | 0.9919 | 0.0854 | 0 | 0.9533 |
| (iii) panel, thinned | quantile-matched | 5 | SCORE | 322.8 | 0.8892 | 0.5828 | 0.6838 | 0.9087 | 0.211 | 0.1062 | 0.96 |
| (iii) panel, thinned | quantile-matched | 5 | mixture w | 322.8 | 0.8316 | 0.4119 | 0.3108 | 0.9969 | 0.06525 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 10 | BASE-T | 430 | 0.8856 | 0.5171 | 0.4212 | 0.991 | 0.09228 | 0 | 0.96 |
| (iii) panel, thinned | quantile-matched | 10 | SCORE | 430 | 0.9093 | 0.6147 | 0.7157 | 0.9128 | 0.2142 | 0.06016 | 0.9567 |
| (iii) panel, thinned | quantile-matched | 10 | mixture w | 430 | 0.8497 | 0.4254 | 0.3225 | 0.9971 | 0.06745 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 25 | BASE-T | 475 | 0.8877 | 0.5248 | 0.4314 | 0.9898 | 0.09531 | 0 | 0.9567 |
| (iii) panel, thinned | quantile-matched | 25 | SCORE | 475 | 0.9123 | 0.6205 | 0.7049 | 0.9202 | 0.2061 | 0.0561 | 0.9667 |
| (iii) panel, thinned | quantile-matched | 25 | mixture w | 475 | 0.8556 | 0.4352 | 0.3318 | 0.9968 | 0.06959 | 0 | 0.9333 |

Stricter truth, all cells:

| condition | thinning | factor | method | median depth | AUROC | kappa | sensitivity | specificity | call rate | calls with y <= 1 | Spearman (type rates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (i) whole transcriptome | none | nan | BASE-T | 3573 | 0.894 | 0.5528 | 0.4536 | 0.9912 | 0.09263 | 0 | 0.9 |
| (i) whole transcriptome | none | nan | SCORE | 3573 | 0.9148 | 0.6345 | 0.6318 | 0.9576 | 0.1563 | 0.05377 | 0.9667 |
| (i) whole transcriptome | none | nan | mixture w | 3573 | 0.8632 | 0.4686 | 0.3577 | 0.9974 | 0.06895 | 0 | 0.8833 |
| (ii) panel, full depth | none | nan | BASE-T | 475 | 0.8966 | 0.5576 | 0.462 | 0.9898 | 0.09531 | 0 | 0.9667 |
| (ii) panel, full depth | none | nan | SCORE | 475 | 0.9191 | 0.6302 | 0.7272 | 0.9202 | 0.2061 | 0.05615 | 0.9667 |
| (ii) panel, full depth | none | nan | mixture w | 475 | 0.86 | 0.4694 | 0.3596 | 0.9968 | 0.06971 | 0 | 0.9333 |
| (iii) panel, thinned | gene-specific | 1 | BASE-T | 66 | 0.7135 | 0.2946 | 0.2072 | 0.9979 | 0.03928 | 0 | 0.93 |
| (iii) panel, thinned | gene-specific | 1 | SCORE | 66 | 0.7613 | 0.3935 | 0.5245 | 0.8787 | 0.1965 | 0.5339 | 0.9467 |
| (iii) panel, thinned | gene-specific | 1 | mixture w | 66 | 0.7825 | 0.3686 | 0.2756 | 0.9936 | 0.05534 | 0 | 0.9367 |
| (iii) panel, thinned | quantile-matched | 1 | BASE-T | 67.8 | 0.7323 | 0.315 | 0.226 | 0.9966 | 0.04428 | 0 | 0.9167 |
| (iii) panel, thinned | quantile-matched | 1 | SCORE | 67.8 | 0.773 | 0.3902 | 0.5823 | 0.8466 | 0.2351 | 0.5067 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 1 | mixture w | 67.8 | 0.7837 | 0.3687 | 0.2779 | 0.9926 | 0.05728 | 0 | 0.94 |
| (iii) panel, thinned | quantile-matched | 2 | BASE-T | 131 | 0.7831 | 0.3982 | 0.2989 | 0.9952 | 0.0597 | 0 | 0.9533 |
| (iii) panel, thinned | quantile-matched | 2 | SCORE | 131 | 0.8213 | 0.449 | 0.6602 | 0.8452 | 0.2534 | 0.3552 | 0.9667 |
| (iii) panel, thinned | quantile-matched | 2 | mixture w | 131 | 0.8127 | 0.4089 | 0.3106 | 0.994 | 0.06298 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 5 | BASE-T | 322.8 | 0.8721 | 0.5203 | 0.4194 | 0.9919 | 0.0854 | 0 | 0.9533 |
| (iii) panel, thinned | quantile-matched | 5 | SCORE | 322.8 | 0.8968 | 0.5924 | 0.7065 | 0.9087 | 0.211 | 0.1062 | 0.96 |
| (iii) panel, thinned | quantile-matched | 5 | mixture w | 322.8 | 0.8383 | 0.4446 | 0.3367 | 0.9969 | 0.06525 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 10 | BASE-T | 430 | 0.8943 | 0.5501 | 0.4514 | 0.991 | 0.09228 | 0 | 0.96 |
| (iii) panel, thinned | quantile-matched | 10 | SCORE | 430 | 0.9162 | 0.6231 | 0.7375 | 0.9128 | 0.2142 | 0.06016 | 0.9567 |
| (iii) panel, thinned | quantile-matched | 10 | mixture w | 430 | 0.8547 | 0.4589 | 0.3493 | 0.9971 | 0.06745 | 0 | 0.9467 |
| (iii) panel, thinned | quantile-matched | 25 | BASE-T | 475 | 0.8966 | 0.5576 | 0.462 | 0.9898 | 0.09531 | 0 | 0.9567 |
| (iii) panel, thinned | quantile-matched | 25 | SCORE | 475 | 0.9191 | 0.6302 | 0.7272 | 0.9202 | 0.2061 | 0.0561 | 0.9667 |
| (iii) panel, thinned | quantile-matched | 25 | mixture w | 475 | 0.8598 | 0.4689 | 0.3591 | 0.9968 | 0.06959 | 0 | 0.9333 |

AUROC by type at this section's depth (ladder x1, primary truth):

| stratum | BASE-T | SCORE | mixture w |
| --- | --- | --- | --- |
| B-cells | 0.6178 | 0.6284 | 0.6296 |
| CAFs | 0.5325 | 0.6672 | 0.8213 |
| Cancer Epithelial | 0.8572 | 0.8812 | 0.8777 |
| Endothelial | 0.5849 | 0.7078 | 0.808 |
| Myeloid | 0.6919 | 0.746 | 0.7742 |
| Normal Epithelial | 0.4912 | 0.6266 | 0.7518 |
| PVL | 0.5926 | 0.7049 | 0.7755 |
| Plasmablasts | 0.6368 | 0.7135 | 0.816 |
| T-cells | 0.6647 | 0.6845 | 0.6824 |

Figure `thinning_ladder.png`; every condition, level, seed, method, truth and type in `thinning_ladder.csv`.

- **X17** (condition (ii) SCORE AUROC above 0.9; at x1 SCORE and BASE-T below 0.75 with w highest): (ii) SCORE 0.912 -> **yes**; x1 SCORE 0.762, BASE-T 0.719 -> **no**; w 0.778, highest -> **yes**. Overall: **partly holds**.
- **X18** (AUROC rises steeply from x1 to x10, approaches (ii) by x25): gain x1->x10 BASE-T +0.167, SCORE +0.147, mixture w +0.072, x10->x25 BASE-T +0.002, SCORE +0.003, mixture w +0.006 -> steep-then-flat **yes**; within 0.02 of (ii) at x25 **yes**. Overall: **holds**.

### E3 -- the cycling rate ratio in Wu

Tirosh composite per panel library (sum y / sum N) in S/G2M-called cells over G1-called cells, pooled and per type, with a 1,000-replicate patient bootstrap. Thinned: ladder x1, five seeds.

| data | truth | mean | min | max |
| --- | --- | --- | --- | --- |
| (ii) panel, full depth | primary | 7.963 | 7.963 | 7.963 |
| (ii) panel, full depth | stricter | 8.13 | 8.13 | 8.13 |
| (iii) thinned x1 | primary | 7.384 | 7.371 | 7.398 |
| (iii) thinned x1 | stricter | 7.564 | 7.549 | 7.582 |

Per type, every seed and both truths in `wu_rate_ratio.csv`. The constant-pi mixture fitted to the same data (`wu_rate_ratio_mixture.csv`):

| data | mixture pi | mixture e^delta | mixture rate ratio |
| --- | --- | --- | --- |
| (iii) thinned x1, seed 0 | 0.07898 | 2.311 | 10.09 |
| (iii) thinned x1, seed 1 | 0.07983 | 2.307 | 10.04 |
| (iii) thinned x1, seed 2 | 0.08174 | 2.297 | 9.947 |
| (iii) thinned x1, seed 3 | 0.08162 | 2.301 | 9.987 |
| (iii) thinned x1, seed 4 | 0.08024 | 2.302 | 9.989 |
| (ii) panel, full depth | 0.07339 | 2.297 | 9.943 |

**Scale.** This project's e^delta is the log of the component rate ratio: MIX on this section has e^delta = 1.90, a rate ratio of 6.70. The addendum's '1.9' is MIX's e^delta, so the like-for-like comparison with Wu's rate ratios is 6.70.

- **X19** (Wu's stricter-grouping ratio at least 4; the mixture on thinned Wu closer to Wu's ratio than to 1.9): Wu's ratio 8.13 (primary grouping 7.96) -> **yes**; the mixture on thinned Wu gives 10.01, closer to Wu's than to 1.9 -> **yes** (and than to MIX's rate ratio 6.70 -> yes). Overall, as written: **holds**.

Stage A6 complete in 17 min. **Review point** (A1, A3, A5 and A6 pause for review).


## Addendum stage A2 -- the two-step arm TWO (2026-10-14 00:02)

`run/division_addendum_a2_twostep.py`, `tissue_dynamics.division.twostep`.
Step 1 is a constant-pi two-component mixture on MIX's production rate block
(Nt offset with coefficient 1, theta_m m added to both component means and kept
outside the rate ratio, lambda_1 = exp(e^delta) lambda_0). TWO-P has Poisson
components, MIX's own; TWO-PNB swaps the cycling component for an NB with one
shared dispersion alpha1. Step 2 regresses step 1's posterior w on MIX's
weight block (a0 plus all 141 eta columns),
with MIX's penalty and weight_ridge scaled to the fitting n; nothing technical
enters it. **TWO-P is not independent of MIX:** its components are MIX's, so
it differs from MIX only in not estimating rates and weights jointly.

As a check on step 1, the constant-pi Poisson log-likelihood written here
equals MIX's own code at stage A (weight = intercept) at the TWO-P optimum:
-609,653.270416 against -609,653.270416.

### Step 1 fits

| arm | data | converged | iterations | grad_inf | pi | e^delta | rate ratio | theta_m | alpha1 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TWO-P | all cells | pass | 18 | 7.14e-11 | 0.1025 | 1.927 | 6.866 | 1.777 | nan | 0.768 |
| TWO-PNB | all cells | pass | 35 | 4.847e-07 | 0.4848 | 0.9426 | 2.567 | 2.328 | 0.9975 | 0.7886 |
| TWO-P | train fold 0 | pass | 17 | 9.04e-11 | 0.1027 | 1.925 | 6.857 | 1.772 | nan | 0.581 |
| TWO-P | train fold 1 | pass | 20 | 1.655e-09 | 0.1023 | 1.937 | 6.936 | 1.821 | nan | 0.5873 |
| TWO-P | train fold 2 | pass | 20 | 1.516e-10 | 0.1029 | 1.918 | 6.81 | 1.735 | nan | 0.5606 |
| TWO-P | train fold 3 | pass | 17 | 6.974e-09 | 0.1022 | 1.924 | 6.851 | 1.766 | nan | 0.4733 |
| TWO-P | train fold 4 | pass | 17 | 3.274e-11 | 0.1022 | 1.928 | 6.877 | 1.795 | nan | 0.4624 |
| TWO-PNB | train fold 0 | pass | 18 | 4.814e-07 | 0.4867 | 0.9469 | 2.578 | 2.331 | 0.9956 | 0.4659 |
| TWO-PNB | train fold 1 | pass | 16 | 4.051e-07 | 0.4892 | 0.9425 | 2.566 | 2.377 | 1.003 | 0.4925 |
| TWO-PNB | train fold 2 | pass | 17 | 1.067e-10 | 0.4795 | 0.9509 | 2.588 | 2.269 | 0.9886 | 0.4678 |
| TWO-PNB | train fold 3 | pass | 14 | 1.276e-10 | 0.479 | 0.9438 | 2.57 | 2.306 | 0.9982 | 0.4702 |
| TWO-PNB | train fold 4 | pass | 17 | 5.493e-11 | 0.4904 | 0.9287 | 2.531 | 2.36 | 1.002 | 0.4801 |

### V13 -- TWO-PNB multi-start (10 random starts, all cells)

Starts perturb b0, delta and eta_m about MIX's values and draw pi from
U(0.02, 0.30) and log alpha1 from N(log 0.3, 1). Criterion (reported, not
blocking): every start within 1e-4 (relative) of the best objective.
**agree**
(0 of 10 starts elsewhere).

| start | converged | objective | rel. gap to best | pi | e^delta | alpha1 | theta_m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | True | 601987 | 0 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 1 | True | 601987 | 3.8677e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 2 | True | 601987 | 0 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 3 | True | 601987 | 3.8677e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 4 | True | 601987 | 1.93385e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 5 | True | 601987 | 3.8677e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 6 | True | 601987 | 0 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 7 | True | 601987 | 1.93385e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 8 | True | 601987 | 3.8677e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |
| 9 | True | 601987 | 1.93385e-16 | 0.484837 | 0.942569 | 0.997537 | 2.3282 |

### Section 3.1b -- Poisson + Poisson against Poisson + NB (all cells, constant pi)

| fit | log-likelihood | pi | e^delta | rate ratio | alpha1 |
| --- | --- | --- | --- | --- | --- |
| Poisson + Poisson | -609,653.27 | 0.1025 | 1.9265 | 6.866 | -- |
| Poisson + NB | -601,987.02 | 0.4848 | 0.9426 | 2.567 | 0.9975 |

Likelihood-ratio statistic 2(ll_PNB - ll_PP) = **15,332.5** for one extra
parameter (alpha1). The boundary at alpha1 = 0 makes a chi-square(1) reference
conservative, so no p-value is given.

By type: observed and predicted variance-to-mean ratio of y, rootogram misfit
(sum of squared hanging-bar residuals on 0-29 and 30+) and the log-likelihood
gain:

| stratum | cells | observed mean | observed VMR | TWO-P VMR | TWO-PNB VMR | TWO-P rootogram misfit | TWO-PNB rootogram misfit | loglik gain (PNB - PP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 561897 | 1.041 | 7.724 | 4.533 | 4.858 | 1772 | 1167 | 7666 |
| Endothelial | 48729 | 0.3484 | 1.866 | 2.632 | 2.758 | 741.2 | 672.8 | 286.8 |
| CAFs | 52079 | 0.3883 | 1.998 | 2.789 | 2.967 | 1076 | 957 | 458.1 |
| PVL | 42834 | 0.4198 | 2.436 | 2.421 | 2.694 | 272.6 | 237.9 | 219.1 |
| B-cells | 59119 | 0.3918 | 2.694 | 1.812 | 1.956 | 103.7 | 102 | 60.64 |
| T-cells | 87004 | 0.4966 | 2.912 | 1.784 | 1.905 | 250.1 | 250 | 107.4 |
| Myeloid | 69669 | 0.4025 | 2.254 | 2.407 | 2.584 | 599.8 | 514.4 | 325.5 |
| Normal Epithelial | 54508 | 1.178 | 4.943 | 4.421 | 5.107 | 61.79 | 43.42 | 559.8 |
| Plasmablasts | 42523 | 1.03 | 5.239 | 3.799 | 4.509 | 50.58 | 24.25 | 384.8 |
| Cancer Epithelial | 105432 | 3.104 | 9.205 | 5.552 | 5.863 | 1404 | 1414 | 5264 |

Where the gain sits on the count axis (sum of per-cell log-likelihood
differences, PNB - PP, by observed y):

| stratum | y 0-4 | y 5-14 | y 15-29 | y >= 30 |
| --- | --- | --- | --- | --- |
| all | -249.4 | 5762.7 | 333.1 | 1819.8 |
| Cancer Epithelial | -1027.8 | 4575.0 | 57.7 | 1659.2 |
| Plasmablasts | -68.1 | 313.7 | 81.3 | 57.9 |

Rootograms: `rootogram_pp_vs_pnb.png` (same axes as the NB-against-mixture
write-up: 0-29 and a pooled 30+). Everything is in `component_family.csv`.
MIX is not changed. Poisson + NB fits clearly better on these numbers; whether MIX's cycling component should become NB is an open specification question for Lachy.

Known risks of TWO-PNB, as the addendum asks: any extra-Poisson variance among
non-cycling cells (segmentation, spillover variance the MisTIC mean term does
not remove) can only go into the NB cycling component, inflating pi; G3 will
expose this, since a single overdispersed NB with no latent classes will still
be given a cycling fraction; and G2 simulates exactly an NB cycling component
on a Poisson null, so TWO-PNB's G2 results are a home fixture, to be read as
MIX's G1 results are.

### BASE-M (hard MAP label from TWO-P's step 1, all cells; not a target)

| stratum | BASE-M rate | BASE-T (0.05) rate | mean w (TWO-P) | agreement |
| --- | --- | --- | --- | --- |
| all | 0.0543 | 0.0329 | 0.1025 | 0.9785 |
| Endothelial | 0.0178 | 0.0107 | 0.0695 | 0.9925 |
| CAFs | 0.0168 | 0.0087 | 0.0611 | 0.9919 |
| PVL | 0.0249 | 0.0146 | 0.0797 | 0.9893 |
| B-cells | 0.0324 | 0.0198 | 0.0986 | 0.9873 |
| T-cells | 0.0435 | 0.0273 | 0.1058 | 0.9837 |
| Myeloid | 0.0228 | 0.0129 | 0.0743 | 0.9898 |
| Normal Epithelial | 0.0517 | 0.0264 | 0.0956 | 0.9745 |
| Plasmablasts | 0.0588 | 0.0329 | 0.1075 | 0.9738 |
| Cancer Epithelial | 0.1431 | 0.0912 | 0.1669 | 0.9481 |

### Step 2 fits, V6, V7, V8

V6: the step 2 predictor a0 + eta_state has the same 141
eta columns as MIX in the same order, and equals MIX's `_logit_pi` at the
production estimate to 3.6e-15. **pass**.
V7: largest sd(eta) 0.420 (criterion 20). **pass**.
V8: refitting TWO-P's step 2 on training fold 0 reproduces it to 0.0e+00.
**pass**. Every step 2 fit started from MIX's weight
block for the same data (production estimate on all cells, Stage 3's fold fit
per fold) and ran to the mixture's convergence criteria.

| arm | data | converged | iterations | grad_inf | objective | sd(eta) | sigma1(B) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TWO-P | all cells | pass | 387 | 1.613e-09 | 1.813e+05 | 0.4155 | 0.1517 | 0.6437 |
| TWO-P | train fold 0 | pass | 407 | 9.767e-06 | 1.433e+05 | 0.418 | 0.1528 | 0.9133 |
| TWO-P | train fold 1 | pass | 275 | 5.702e-06 | 1.47e+05 | 0.413 | 0.1484 | 0.5115 |
| TWO-P | train fold 2 | pass | 319 | 4.283e-07 | 1.438e+05 | 0.4168 | 0.1539 | 0.3861 |
| TWO-P | train fold 3 | pass | 369 | 7.998e-06 | 1.448e+05 | 0.4199 | 0.1569 | 0.3336 |
| TWO-P | train fold 4 | pass | 407 | 4.241e-11 | 1.462e+05 | 0.412 | 0.1475 | 0.3748 |
| TWO-PNB | all cells | pass | 1122 | 2.53e-08 | 3.886e+05 | 0.09645 | 0.03585 | 2.08 |
| TWO-PNB | train fold 0 | pass | 849 | 3.256e-07 | 3.068e+05 | 0.09721 | 0.03585 | 1.834 |
| TWO-PNB | train fold 1 | pass | 1042 | 7.871e-07 | 3.155e+05 | 0.09418 | 0.03459 | 1.696 |
| TWO-PNB | train fold 2 | pass | 1923 | 1.269e-07 | 3.07e+05 | 0.09894 | 0.03795 | 1.938 |
| TWO-PNB | train fold 3 | pass | 1023 | 7.865e-07 | 3.108e+05 | 0.09867 | 0.0373 | 1.466 |
| TWO-PNB | train fold 4 | pass | 1322 | 7.986e-06 | 3.141e+05 | 0.09369 | 0.03433 | 1.672 |

### V4 -- push-through identity

50,000 random cells, y simulated from each arm's out-of-fold predictive
distribution (prior pi, step 1's components). Formal criterion on the primary
thresholds (LT_a05, LO_td050): within 3 Monte Carlo SE. **pass**.
Of the other 48 comparisons 0 are beyond 3 SE (about
0.13 expected by chance).

| arm | threshold | mean p | simulated rate | MC SE | z |
| --- | --- | --- | --- | --- | --- |
| TWO-P | LT_a05 | 0.039584 | 0.03972 | 0.00085899 | 0.15887 |
| TWO-P | LO_td050 | 0.18692 | 0.18998 | 0.0017141 | 1.7831 |
| TWO-PNB | LT_a05 | 0.033248 | 0.03282 | 0.00079872 | -0.5363 |
| TWO-PNB | LO_td050 | 0.18798 | 0.18752 | 0.0017207 | -0.26564 |

Out of fold w, pi_TWO and the pushed-through p for every threshold are in
`two_step.parquet`; parameters in `two_step_params.json`; fits in
`two_step_fits.csv`. X12 (BASE-M below BASE-T at 0.05): **does not hold**
(0.0543 against 0.0329).
X14's first clause (likelihood-ratio statistic in the hundreds or more, most of
the gain in Cancer Epithelial's 15-30+ range): statistic 15,332; Cancer
Epithelial gains 5,264.1 of 7,666.2, 1,716.9 of it
from cells with y >= 15. Its second clause (V13 agreement):
**holds**. Its third clause (G3) waits
for A5. Part 1 of A2 done in 30 min.


## Addendum stage A2 -- V12: iterated two-step converges to MIX (2026-10-14 00:07)

`run/division_addendum_a2_v12.py`. Generalised EM on the penalised MIX
objective over all 561,897 cells, from the TWO-P solution: E-step under the
current pi(x); weight M-step 15 L-BFGS steps on the penalised
fractional logistic objective (MIX's penalty); rate M-step b0, delta and eta_m
by Newton to convergence; SQUAREM (S3) with a monotone safeguard. Cap 2000
EM map evaluations.

The run starts from TWO-P itself (step 1 rates and step 2 weights) rather than
the constant-pi fixed point, because U = V = 0 is a saddle a partial weight
M-step cannot leave; TWO-P is where the section 3.4 path lands after its first
iteration with an exact weight M-step. The premise that the rates do not move
in that first iteration holds: a rate M-step on step 1's own posteriors moves
(b0, delta, eta_m) by 1.1e-16.

| | penalised objective (-loglik + penalty) |
| --- | --- |
| TWO-P (start) | 603,222.055937 |
| end of EM | 600,332.256348 |
| production MIX | 600,332.256348 |
| end - production | +3.801e-07 (+6.33e-13 relative) |

EM map evaluations: 315. The objective never increased: **yes**
(largest increase 0.00e+00). Both pass criteria first held after
12 map evaluations.

Largest coefficient difference from production MIX, by block (B = U V'):

| block | max |difference| |
| --- | --- |
| a0 | 5.15e-08 |
| a | 4.78e-08 |
| c | 6.94e-08 |
| h | 5.20e-07 |
| K | 1.04e-06 |
| b0 | 1.94e-09 |
| delta | 7.03e-11 |
| eta_m | 3.23e-09 |
| B | 4.41e-08 |

Trace (selected cycles; every cycle in `v12_trace.csv`):

| maps | objective | step | alpha | gap to production | max |coef diff| |
| --- | --- | --- | --- | --- | --- |
| 0 | 603222.06 | start | nan | nan | nan |
| 3 | 600365.5 | squarem | -2.4312485 | 33.248585 | 0.031803262 |
| 6 | 600333.34 | squarem | -2.7045365 | 1.0825609 | 0.011144256 |
| 15 | 600332.26 | squarem | -1.7147475 | 0.0034228868 | 0.00022768335 |
| 30 | 600332.26 | squarem | -1.1408225 | 0.0016257277 | 2.5164615e-05 |
| 75 | 600332.26 | squarem | -1.1275943 | 0.00024053233 | 2.1675111e-05 |
| 150 | 600332.26 | squarem | -1 | 3.3590011e-05 | 1.1449847e-05 |
| 300 | 600332.26 | squarem | -1 | 6.1735045e-07 | 6.5629851e-07 |
| 315 | 600332.26 | squarem | -1 | 3.8009603e-07 | 1.0409648e-06 |

**V12: pass.**


## Addendum stage A2 -- extended grid and TWO, scored out of fold (2026-10-14 01:02)

`run/division_addendum_a2_grid.py`. Every BASE-T label of the extended grid
(alpha_B 0.001-0.2) for the naive null and each truncated null: LOG-T fitted on
each (Stage 3's specification, five training folds), NB-c, NB-d and MIX pushed
through each (their out-of-fold predictive distributions, unchanged), and TWO-P
and TWO-PNB pushed through each (part 1). Scored as Stage 4 on the same 300
block-bootstrap replicates; * marks a 95% interval excluding 0 (differences) or
1 (calibration). Per the interpretive rule, LOG-T is trained on the label it is
scored against. The NB, MIX and TWO fits never see a label: across alpha_B
only their label-scale expression changes.

### LOG-T fits (V7, convergence)

All 95 new fold fits converged: **yes**. V7 (sd(eta) <= 20):
**pass** (largest 0.958).

| null | alpha_B | fits | converged | max sd(eta) | mean theta | minutes |
| --- | --- | --- | --- | --- | --- | --- |
| naive | 0.001 | 5 | 5/5 | 0.7113 | 1.084 | 1.404 |
| naive | 0.01 | 5 | 5/5 | 0.9207 | 0.835 | 2.097 |
| naive | 0.1 | 5 | 5/5 | 0.6486 | 0.3552 | 2.092 |
| naive | 0.2 | 5 | 5/5 | 0.5286 | 0.1506 | 2.863 |
| trunc 0.001 | 0.001 | 5 | 5/5 | 0.7101 | 1.082 | 1.335 |
| trunc 0.001 | 0.01 | 5 | 5/5 | 0.9161 | 0.8351 | 1.885 |
| trunc 0.001 | 0.05 | 5 | 5/5 | 0.7847 | 0.6 | 2.025 |
| trunc 0.001 | 0.1 | 5 | 5/5 | 0.6482 | 0.3552 | 1.939 |
| trunc 0.001 | 0.2 | 5 | 5/5 | 0.5279 | 0.1508 | 2.694 |
| trunc 0.01 | 0.001 | 5 | 5/5 | 0.9011 | 1.388 | 1.749 |
| trunc 0.01 | 0.01 | 5 | 5/5 | 0.9098 | 0.9146 | 2.16 |
| trunc 0.01 | 0.05 | 5 | 5/5 | 0.7551 | 0.6538 | 1.984 |
| trunc 0.01 | 0.1 | 5 | 5/5 | 0.6175 | 0.381 | 2.147 |
| trunc 0.01 | 0.2 | 5 | 5/5 | 0.5139 | 0.1897 | 3.141 |
| trunc 0.05 | 0.001 | 5 | 5/5 | 0.9584 | 1.488 | 1.939 |
| trunc 0.05 | 0.01 | 5 | 5/5 | 0.8993 | 0.955 | 2.024 |
| trunc 0.05 | 0.05 | 5 | 5/5 | 0.7162 | 0.686 | 2.148 |
| trunc 0.05 | 0.1 | 5 | 5/5 | 0.6063 | 0.4037 | 2.247 |
| trunc 0.05 | 0.2 | 5 | 5/5 | 0.5147 | 0.2268 | 2.809 |

### Log loss (all cells; lower is better)

| null | alpha_B | label rate | NB-c | NB-d | MIX | TWO-P | TWO-PNB | LOG-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | 0.001 | 0.0021 | 0.0136 | 0.0136 | 0.0151 | 0.0153 | 0.0139 | 0.0132 |
| naive | 0.01 | 0.0114 | 0.0550 | 0.0550 | 0.0552 | 0.0565 | 0.0577 | 0.0537 |
| naive | 0.05 | 0.0329 | 0.1294 | 0.1294 | 0.1298 | 0.1323 | 0.1368 | 0.1293 |
| naive | 0.1 | 0.0578 | 0.2017 | 0.2016 | 0.2018 | 0.2049 | 0.2109 | 0.2054 |
| naive | 0.2 | 0.1121 | 0.3281 | 0.3279 | 0.3286 | 0.3330 | 0.3387 | 0.3359 |
| trunc 0.001 | 0.001 | 0.0021 | 0.0136 | 0.0136 | 0.0150 | 0.0152 | 0.0139 | 0.0131 |
| trunc 0.001 | 0.01 | 0.0113 | 0.0548 | 0.0548 | 0.0550 | 0.0564 | 0.0576 | 0.0535 |
| trunc 0.001 | 0.05 | 0.0329 | 0.1293 | 0.1293 | 0.1297 | 0.1322 | 0.1366 | 0.1291 |
| trunc 0.001 | 0.1 | 0.0577 | 0.2015 | 0.2015 | 0.2016 | 0.2047 | 0.2107 | 0.2052 |
| trunc 0.001 | 0.2 | 0.1121 | 0.3280 | 0.3279 | 0.3285 | 0.3329 | 0.3386 | 0.3358 |
| trunc 0.01 | 0.001 | 0.0062 | 0.0316 | 0.0315 | 0.0320 | 0.0328 | 0.0329 | 0.0298 |
| trunc 0.01 | 0.01 | 0.0188 | 0.0795 | 0.0795 | 0.0797 | 0.0816 | 0.0845 | 0.0783 |
| trunc 0.01 | 0.05 | 0.0427 | 0.1557 | 0.1557 | 0.1561 | 0.1590 | 0.1643 | 0.1560 |
| trunc 0.01 | 0.1 | 0.0696 | 0.2294 | 0.2293 | 0.2296 | 0.2332 | 0.2391 | 0.2341 |
| trunc 0.01 | 0.2 | 0.1249 | 0.3511 | 0.3509 | 0.3521 | 0.3569 | 0.3619 | 0.3594 |
| trunc 0.05 | 0.001 | 0.0135 | 0.0567 | 0.0567 | 0.0562 | 0.0575 | 0.0602 | 0.0541 |
| trunc 0.05 | 0.01 | 0.0282 | 0.1060 | 0.1060 | 0.1067 | 0.1091 | 0.1133 | 0.1055 |
| trunc 0.05 | 0.05 | 0.0549 | 0.1853 | 0.1853 | 0.1860 | 0.1896 | 0.1948 | 0.1862 |
| trunc 0.05 | 0.1 | 0.0838 | 0.2598 | 0.2597 | 0.2605 | 0.2647 | 0.2698 | 0.2653 |
| trunc 0.05 | 0.2 | 0.1407 | 0.3760 | 0.3759 | 0.3777 | 0.3824 | 0.3869 | 0.3855 |

### AUPRC (all cells)

| null | alpha_B | label rate | NB-c | NB-d | MIX | TWO-P | TWO-PNB | LOG-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | 0.001 | 0.0021 | 0.0141 | 0.0135 | 0.0112 | 0.0100 | 0.0153 | 0.0202 |
| naive | 0.01 | 0.0114 | 0.0531 | 0.0521 | 0.0599 | 0.0528 | 0.0520 | 0.0743 |
| naive | 0.05 | 0.0329 | 0.1049 | 0.1048 | 0.1141 | 0.1063 | 0.0921 | 0.1250 |
| naive | 0.1 | 0.0578 | 0.1416 | 0.1418 | 0.1456 | 0.1400 | 0.1175 | 0.1456 |
| naive | 0.2 | 0.1121 | 0.2124 | 0.2127 | 0.2055 | 0.1950 | 0.1673 | 0.1938 |
| trunc 0.001 | 0.001 | 0.0021 | 0.0142 | 0.0135 | 0.0114 | 0.0102 | 0.0153 | 0.0202 |
| trunc 0.001 | 0.01 | 0.0113 | 0.0529 | 0.0519 | 0.0597 | 0.0527 | 0.0517 | 0.0739 |
| trunc 0.001 | 0.05 | 0.0329 | 0.1048 | 0.1047 | 0.1140 | 0.1062 | 0.0919 | 0.1249 |
| trunc 0.001 | 0.1 | 0.0577 | 0.1413 | 0.1415 | 0.1454 | 0.1397 | 0.1170 | 0.1454 |
| trunc 0.001 | 0.2 | 0.1121 | 0.2123 | 0.2125 | 0.2053 | 0.1948 | 0.1672 | 0.1937 |
| trunc 0.01 | 0.001 | 0.0062 | 0.0503 | 0.0483 | 0.0540 | 0.0468 | 0.0506 | 0.0762 |
| trunc 0.01 | 0.01 | 0.0188 | 0.0941 | 0.0931 | 0.1015 | 0.0916 | 0.0873 | 0.1203 |
| trunc 0.01 | 0.05 | 0.0427 | 0.1397 | 0.1398 | 0.1433 | 0.1359 | 0.1234 | 0.1579 |
| trunc 0.01 | 0.1 | 0.0696 | 0.1726 | 0.1728 | 0.1694 | 0.1649 | 0.1483 | 0.1730 |
| trunc 0.01 | 0.2 | 0.1249 | 0.2385 | 0.2388 | 0.2268 | 0.2151 | 0.1942 | 0.2200 |
| trunc 0.05 | 0.001 | 0.0135 | 0.1090 | 0.1069 | 0.1120 | 0.0996 | 0.1035 | 0.1452 |
| trunc 0.05 | 0.01 | 0.0282 | 0.1441 | 0.1434 | 0.1457 | 0.1349 | 0.1350 | 0.1687 |
| trunc 0.05 | 0.05 | 0.0549 | 0.1816 | 0.1817 | 0.1734 | 0.1684 | 0.1652 | 0.1925 |
| trunc 0.05 | 0.1 | 0.0838 | 0.2104 | 0.2107 | 0.1971 | 0.1937 | 0.1898 | 0.2057 |
| trunc 0.05 | 0.2 | 0.1407 | 0.2731 | 0.2734 | 0.2570 | 0.2485 | 0.2377 | 0.2522 |

### AUROC (all cells)

| null | alpha_B | label rate | NB-c | NB-d | MIX | TWO-P | TWO-PNB | LOG-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | 0.001 | 0.0021 | 0.8507 | 0.8486 | 0.7382 | 0.6812 | 0.8176 | 0.8418 |
| naive | 0.01 | 0.0114 | 0.8157 | 0.8144 | 0.7945 | 0.7613 | 0.7738 | 0.8076 |
| naive | 0.05 | 0.0329 | 0.7667 | 0.7664 | 0.7641 | 0.7435 | 0.7165 | 0.7563 |
| naive | 0.1 | 0.0578 | 0.7329 | 0.7330 | 0.7316 | 0.7181 | 0.6812 | 0.7013 |
| naive | 0.2 | 0.1121 | 0.6926 | 0.6928 | 0.6892 | 0.6751 | 0.6438 | 0.6515 |
| trunc 0.001 | 0.001 | 0.0021 | 0.8512 | 0.8491 | 0.7398 | 0.6834 | 0.8178 | 0.8416 |
| trunc 0.001 | 0.01 | 0.0113 | 0.8154 | 0.8141 | 0.7942 | 0.7610 | 0.7733 | 0.8073 |
| trunc 0.001 | 0.05 | 0.0329 | 0.7665 | 0.7662 | 0.7640 | 0.7434 | 0.7161 | 0.7561 |
| trunc 0.001 | 0.1 | 0.0577 | 0.7328 | 0.7329 | 0.7315 | 0.7180 | 0.6806 | 0.7012 |
| trunc 0.001 | 0.2 | 0.1121 | 0.6925 | 0.6927 | 0.6891 | 0.6750 | 0.6436 | 0.6514 |
| trunc 0.01 | 0.001 | 0.0062 | 0.8774 | 0.8759 | 0.8435 | 0.8170 | 0.8536 | 0.8787 |
| trunc 0.01 | 0.01 | 0.0188 | 0.8255 | 0.8247 | 0.8164 | 0.7925 | 0.7871 | 0.8190 |
| trunc 0.01 | 0.05 | 0.0427 | 0.7713 | 0.7711 | 0.7690 | 0.7539 | 0.7292 | 0.7603 |
| trunc 0.01 | 0.1 | 0.0696 | 0.7347 | 0.7349 | 0.7326 | 0.7218 | 0.6921 | 0.7016 |
| trunc 0.01 | 0.2 | 0.1249 | 0.6946 | 0.6948 | 0.6897 | 0.6758 | 0.6517 | 0.6542 |
| trunc 0.05 | 0.001 | 0.0135 | 0.8878 | 0.8868 | 0.8791 | 0.8657 | 0.8685 | 0.8889 |
| trunc 0.05 | 0.01 | 0.0282 | 0.8317 | 0.8314 | 0.8275 | 0.8113 | 0.8000 | 0.8244 |
| trunc 0.05 | 0.05 | 0.0549 | 0.7777 | 0.7776 | 0.7739 | 0.7636 | 0.7442 | 0.7661 |
| trunc 0.05 | 0.1 | 0.0838 | 0.7387 | 0.7388 | 0.7350 | 0.7266 | 0.7053 | 0.7060 |
| trunc 0.05 | 0.2 | 0.1407 | 0.7002 | 0.7004 | 0.6953 | 0.6848 | 0.6676 | 0.6604 |

### Calibration-in-the-large, mean p / label rate (all cells)

| null | alpha_B | label rate | NB-c | NB-d | MIX | TWO-P | TWO-PNB | LOG-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | 0.001 | 0.0021 | 0.517 | 0.565 | 0.474 | 0.560 | 0.971 | 1.001 |
| naive | 0.01 | 0.0114 | 0.668 | 0.681 | 1.017 | 1.080 | 0.909 | 1.002 |
| naive | 0.05 | 0.0329 | 0.953 | 0.950 | 1.173 | 1.200 | 1.009 | 1.002 |
| naive | 0.1 | 0.0578 | 1.060 | 1.054 | 1.081 | 1.100 | 1.030 | 1.001 |
| naive | 0.2 | 0.1121 | 1.107 | 1.101 | 1.002 | 1.020 | 1.036 | 1.001 |
| trunc 0.001 | 0.001 | 0.0021 | 0.517 | 0.565 | 0.476 | 0.562 | 0.973 | 1.002 |
| trunc 0.001 | 0.01 | 0.0113 | 0.669 | 0.683 | 1.019 | 1.083 | 0.911 | 1.002 |
| trunc 0.001 | 0.05 | 0.0329 | 0.953 | 0.951 | 1.174 | 1.201 | 1.010 | 1.002 |
| trunc 0.001 | 0.1 | 0.0577 | 1.061 | 1.055 | 1.081 | 1.100 | 1.030 | 1.001 |
| trunc 0.001 | 0.2 | 0.1121 | 1.107 | 1.101 | 1.002 | 1.020 | 1.036 | 1.001 |
| trunc 0.01 | 0.001 | 0.0062 | 0.526 | 0.550 | 0.787 | 0.827 | 0.789 | 1.003 |
| trunc 0.01 | 0.01 | 0.0188 | 0.766 | 0.771 | 1.160 | 1.180 | 0.899 | 1.003 |
| trunc 0.01 | 0.05 | 0.0427 | 1.029 | 1.024 | 1.121 | 1.135 | 1.011 | 1.002 |
| trunc 0.01 | 0.1 | 0.0696 | 1.100 | 1.093 | 1.035 | 1.051 | 1.029 | 1.001 |
| trunc 0.01 | 0.2 | 0.1249 | 1.110 | 1.105 | 0.989 | 1.006 | 1.033 | 1.001 |
| trunc 0.05 | 0.001 | 0.0135 | 0.617 | 0.627 | 1.048 | 1.034 | 0.744 | 1.004 |
| trunc 0.05 | 0.01 | 0.0282 | 0.898 | 0.896 | 1.164 | 1.158 | 0.919 | 1.003 |
| trunc 0.05 | 0.05 | 0.0549 | 1.093 | 1.085 | 1.044 | 1.052 | 1.006 | 1.001 |
| trunc 0.05 | 0.1 | 0.0838 | 1.128 | 1.121 | 0.993 | 1.008 | 1.026 | 1.001 |
| trunc 0.05 | 0.2 | 0.1407 | 1.109 | 1.104 | 0.982 | 1.000 | 1.031 | 1.001 |

### Paired differences in log loss (negative: the first arm is better)

| null | alpha_B | MIX - LOG-T | TWO-P - LOG-T | MIX - TWO-P | TWO-PNB - TWO-P |
| --- | --- | --- | --- | --- | --- |
| naive | 0.001 | +0.00194* | +0.00216* | -0.00023* | -0.00138* |
| naive | 0.01 | +0.00147* | +0.00285* | -0.00138* | +0.00120* |
| naive | 0.05 | +0.00056* | +0.00308* | -0.00251* | +0.00448* |
| naive | 0.1 | -0.00363* | -0.00050 | -0.00312* | +0.00597* |
| naive | 0.2 | -0.00736* | -0.00293* | -0.00444* | +0.00571* |
| trunc 0.001 | 0.001 | +0.00190* | +0.00213* | -0.00023* | -0.00136* |
| trunc 0.001 | 0.01 | +0.00147* | +0.00284* | -0.00137* | +0.00118* |
| trunc 0.001 | 0.05 | +0.00056* | +0.00306* | -0.00250* | +0.00446* |
| trunc 0.001 | 0.1 | -0.00362* | -0.00050 | -0.00312* | +0.00598* |
| trunc 0.001 | 0.2 | -0.00736* | -0.00293* | -0.00443* | +0.00571* |
| trunc 0.01 | 0.001 | +0.00219* | +0.00293* | -0.00073* | +0.00012 |
| trunc 0.01 | 0.01 | +0.00145* | +0.00335* | -0.00190* | +0.00285* |
| trunc 0.01 | 0.05 | +0.00015 | +0.00305* | -0.00290* | +0.00525* |
| trunc 0.01 | 0.1 | -0.00450* | -0.00086* | -0.00363* | +0.00593* |
| trunc 0.01 | 0.2 | -0.00723* | -0.00250* | -0.00472* | +0.00503* |
| trunc 0.05 | 0.001 | +0.00203* | +0.00339* | -0.00135* | +0.00267* |
| trunc 0.05 | 0.01 | +0.00116* | +0.00363* | -0.00248* | +0.00413* |
| trunc 0.05 | 0.05 | -0.00014 | +0.00337* | -0.00351* | +0.00527* |
| trunc 0.05 | 0.1 | -0.00480* | -0.00063 | -0.00418* | +0.00511* |
| trunc 0.05 | 0.2 | -0.00776* | -0.00303* | -0.00472* | +0.00446* |

### Paired differences in AUPRC (positive: the first arm is better)

| null | alpha_B | MIX - LOG-T | TWO-P - LOG-T | MIX - TWO-P | TWO-PNB - TWO-P |
| --- | --- | --- | --- | --- | --- |
| naive | 0.001 | -0.00902* | -0.01019* | +0.00118* | +0.00530* |
| naive | 0.01 | -0.01439* | -0.02152* | +0.00713* | -0.00080 |
| naive | 0.05 | -0.01097* | -0.01872* | +0.00775* | -0.01424* |
| naive | 0.1 | +0.00001 | -0.00559* | +0.00560* | -0.02245* |
| naive | 0.2 | +0.01169* | +0.00116 | +0.01053* | -0.02766* |
| trunc 0.001 | 0.001 | -0.00880* | -0.00998* | +0.00118* | +0.00515* |
| trunc 0.001 | 0.01 | -0.01419* | -0.02126* | +0.00707* | -0.00093 |
| trunc 0.001 | 0.05 | -0.01094* | -0.01866* | +0.00772* | -0.01435* |
| trunc 0.001 | 0.1 | -0.00004 | -0.00568* | +0.00565* | -0.02278* |
| trunc 0.001 | 0.2 | +0.01167* | +0.00114 | +0.01054* | -0.02757* |
| trunc 0.01 | 0.001 | -0.02218* | -0.02940* | +0.00721* | +0.00383* |
| trunc 0.01 | 0.01 | -0.01877* | -0.02872* | +0.00994* | -0.00425* |
| trunc 0.01 | 0.05 | -0.01460* | -0.02202* | +0.00742* | -0.01249* |
| trunc 0.01 | 0.1 | -0.00356* | -0.00812* | +0.00456* | -0.01651* |
| trunc 0.01 | 0.2 | +0.00680* | -0.00491* | +0.01171* | -0.02089* |
| trunc 0.05 | 0.001 | -0.03322* | -0.04555* | +0.01233* | +0.00389* |
| trunc 0.05 | 0.01 | -0.02297* | -0.03383* | +0.01086* | +0.00012 |
| trunc 0.05 | 0.05 | -0.01902* | -0.02402* | +0.00500* | -0.00324* |
| trunc 0.05 | 0.1 | -0.00857* | -0.01200* | +0.00343* | -0.00385* |
| trunc 0.05 | 0.2 | +0.00474* | -0.00376* | +0.00850* | -0.01077* |

### Calibration by type at the primary label (LT_a05; * = interval excludes 1)

| stratum | NB-c | NB-d | MIX | TWO-P | TWO-PNB | LOG-T |
| --- | --- | --- | --- | --- | --- | --- |
| all | 0.953* | 0.950* | 1.173* | 1.200* | 1.009 | 1.002 |
| Endothelial | 1.062 | 1.088 | 1.219* | 2.121* | 2.687* | 1.189* |
| CAFs | 1.052 | 0.919 | 1.269* | 2.753* | 3.626* | 1.197* |
| PVL | 1.103* | 1.092* | 1.171* | 1.660* | 1.929* | 1.137* |
| B-cells | 1.234* | 1.216* | 1.187* | 1.341* | 1.335* | 1.032 |
| T-cells | 1.159* | 1.166* | 1.134* | 1.131* | 1.060* | 0.982 |
| Myeloid | 1.149* | 1.208* | 1.290* | 2.003* | 2.262* | 1.282* |
| Normal Epithelial | 1.053 | 0.965 | 1.371* | 1.436* | 1.270* | 1.211* |
| Plasmablasts | 0.971 | 0.912* | 1.104* | 1.110* | 0.980 | 1.016 |
| Cancer Epithelial | 0.811* | 0.829* | 1.144* | 0.949* | 0.530* | 0.915* |

### TWO on the primary BASE-O label (LO_td050)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.4528 [0.4479, 0.4581] | 0.1427 [0.1406, 0.1449] | 0.6535 [0.6506, 0.6561] | 0.2759 [0.2706, 0.2815] | 1.0581 [1.0454, 1.0705] |
| NB-d | 0.4527 [0.4477, 0.4579] | 0.1426 [0.1406, 0.1449] | 0.6539 [0.6510, 0.6565] | 0.2763 [0.2710, 0.2818] | 1.0557 [1.0430, 1.0680] |
| MIX | 0.4542 [0.4494, 0.4599] | 0.1430 [0.1409, 0.1453] | 0.6476 [0.6449, 0.6502] | 0.2715 [0.2668, 0.2757] | 1.0106 [0.9959, 1.0239] |
| LOG-O | 0.4640 [0.4591, 0.4696] | 0.1458 [0.1437, 0.1481] | 0.6055 [0.6017, 0.6096] | 0.2423 [0.2368, 0.2489] | 1.0003 [0.9865, 1.0124] |
| TWO-P | 0.4588 [0.4537, 0.4647] | 0.1443 [0.1422, 0.1467] | 0.6295 [0.6262, 0.6327] | 0.2609 [0.2563, 0.2648] | 1.0258 [1.0059, 1.0448] |
| TWO-PNB | 0.4644 [0.4593, 0.4704] | 0.1459 [0.1438, 0.1483] | 0.6032 [0.5984, 0.6076] | 0.2441 [0.2391, 0.2479] | 1.0317 [1.0094, 1.0533] |

Per-type scores for every label and arm, and the Brier and AUROC differences,
are in `scores_grid.csv`; out-of-fold p for every arm and label in
`oof_grid.parquet`. The primary alpha_B and null are not chosen here: by the
section 2.3 rule they come from the A5 simulation. Part 3 of A2 done in
54 min.

## Addendum stage A3 -- Wu baseline gate for BASE-O-state (2026-10-14 01:02)

`run/division_addendum_a3_wu_gate.py`, addendum section 4.1. Wu et al. 2021,
100,064 cells and 26 patients, restricted to the 4,917
genes shared with this panel. y is the Tirosh composite (53 of
53 genes; the rest are not in Wu), N the panel transcripts
without the Tirosh and blocklist genes (4,397 genes), as Stage 1 builds
them; cells with N >= 10. In non-cycling cells, selected two ways, the
composite per 1,000 panel transcripts per type is 1000 sum(y) / sum(N), and its
ratio to the pooled value is shown with a 1,000-replicate patient bootstrap
(patients resampled; ER+ patients resampled among themselves).

Selections: **primary**, Seurat-called G1 (S and G2M scores both below 0);
**stricter**, both scores below their 25th percentile within the cell's type.

Ratio to pooled [95% patient-bootstrap interval]:

| type | all, primary | all, stricter | ER+, primary | ER+, stricter |
| --- | --- | --- | --- | --- |
| Endothelial | 0.84 [0.76, 0.91] | 1.00 [0.87, 1.07] | 0.89 [0.77, 1.00] | 0.98 [0.77, 1.09] |
| CAFs | 0.64 [0.57, 0.71] | 0.71 [0.60, 0.79] | 0.72 [0.60, 0.80] | 0.69 [0.53, 0.82] |
| PVL | 0.81 [0.73, 0.89] | 0.92 [0.73, 1.04] | 0.84 [0.75, 0.95] | 0.89 [0.55, 1.10] |
| B-cells | 1.49 [1.38, 1.63] | 1.02 [0.82, 1.16] | 1.59 [1.42, 1.74] | 1.02 [0.35, 1.31] |
| T-cells | 1.49 [1.34, 1.66] | 1.20 [1.07, 1.33] | 1.76 [1.35, 2.09] | 1.10 [0.79, 1.34] |
| Myeloid | 0.84 [0.75, 1.00] | 0.71 [0.65, 0.83] | 1.16 [0.95, 1.37] | 0.81 [0.68, 0.89] |
| Normal Epithelial | 1.01 [0.92, 1.12] | 1.22 [1.01, 1.39] | 1.06 [0.85, 1.20] | 1.24 [1.10, 1.53] |
| Plasmablasts | 0.93 [0.85, 1.07] | 0.94 [0.86, 1.24] | 1.12 [0.90, 1.30] | 0.97 [0.85, 1.23] |
| Cancer Epithelial | 1.09 [0.98, 1.20] | 1.10 [1.02, 1.23] | 1.01 [0.86, 1.16] | 1.03 [0.95, 1.19] |

Cells selected:

| type | ER+, primary | ER+, stricter | all, primary | all, stricter |
| --- | --- | --- | --- | --- |
| Endothelial | 5023 | 551 | 7188 | 800 |
| CAFs | 2506 | 272 | 6340 | 716 |
| PVL | 3140 | 383 | 5053 | 577 |
| B-cells | 438 | 37 | 2504 | 269 |
| T-cells | 6578 | 519 | 24744 | 3067 |
| Myeloid | 1491 | 83 | 8350 | 903 |
| Normal Epithelial | 2316 | 266 | 4117 | 401 |
| Plasmablasts | 575 | 63 | 3219 | 485 |
| Cancer Epithelial | 11129 | 2206 | 18331 | 2874 |

Summary (max/min across the nine types, with its bootstrap interval):

| scope | selection | pooled per 1,000 | max/min | max/min 95% CI | highest | lowest | types excluding 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all patients | primary (Seurat G1: S < 0 and G2M < 0) | 4.94 | 2.32 | 2.13-2.66 | B-cells | CAFs | 6 |
| all patients | stricter (S and G2M below the type's 25th percentile) | 2.93 | 1.74 | 1.63-2.16 | Normal Epithelial | CAFs | 5 |
| ER+ patients | primary (Seurat G1: S < 0 and G2M < 0) | 4.67 | 2.44 | 2.10-2.92 | T-cells | CAFs | 5 |
| ER+ patients | stricter (S and G2M below the type's 25th percentile) | 3.05 | 1.8 | 1.60-4.07 | Normal Epithelial | CAFs | 3 |

Against the suggested reading (max/min above 1.5 with an interval excluding 1,
in both selections): all patients **met**;
ER+ patients **met**.
These baseline differences support a state-dependent null. Note: MIX's rates also omit state, so this is a specification question for MIX as well, out of scope here.

Every type, selection and scope is in `wu_baseline.csv`. Stage A3 complete in
0 min. **Stopped for Lachy's decision** on whether
A4 (BASE-O-state) runs.

### A2 and A3: verdicts and notes (2026-10-14)

- **X12** (BASE-M's rate below BASE-T's at alpha_B 0.05): **does not hold**; BASE-M 0.0543 against 0.0329. MAP labelling under the constant-pi mixture calls more cells than the 5% tail does, in every type.
- **X14**, first two clauses. Likelihood-ratio statistic 15,333, far into the hundreds: **yes**. Most of the gain in Cancer Epithelial's 15-30+ range: **no**; of Cancer Epithelial's 5,264 log-likelihood units, 1,717 come from y >= 15 and 4,575 from y 5-14 (y 0-4 loses 1,028). V13's ten starts agree: **yes**. Clause by clause, X14 so far **partly holds**; the G3 clause waits for A5. The better fit comes with pi = 0.48 and a cycling rate ratio of 2.6 (against TWO-P's 0.10 and 6.9), so the NB component is absorbing extra-Poisson variance among non-cycling cells -- the first of the known risks above -- rather than describing a cycling class. That bears on reading Poisson + NB as a candidate for MIX: it fits the counts better, but no longer separates a cycling minority.
- **V12 passes**: generalised EM from TWO-P reaches production MIX's penalised optimum (gap 6e-13 relative, largest coefficient difference 1e-6) with the objective never rising. TWO-P is therefore an early point on the EM path whose end is MIX, as section 3.4 argues.
- **Real-data scoring.** On log loss LOG-T leads only where it is trained, at the stricter alpha_B (0.001-0.05); from alpha_B 0.1 MIX and the NBs are better on log loss, and from 0.2 MIX is also better on AUPRC. MIX beats TWO-P on every label and metric shown (log loss and AUPRC intervals excluding 0), the real-data face of the joint-estimation gain. TWO-PNB is the worst-calibrated arm by type (CAFs 3.6x, Cancer Epithelial 0.53x at the primary label).
- **A3 reading.** The max/min ratio is at least 1 by construction, so its interval excluding 1 is a weak test; the per-type intervals are the informative ones (3-6 of 9 types exclude 1 in every scope and selection). The highest and lowest types move between selections (B- and T-cells highest under the primary selection, Normal Epithelial under the stricter; CAFs lowest throughout). Per 1,000 panel transcripts, a type's value also depends on which non-Tirosh panel genes it expresses, so these are differences in the composite relative to this panel's library, which is the quantity the null scales by.
