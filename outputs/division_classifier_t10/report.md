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


## Stage 1 -- labels (built and frozen 2026-10-17 17:41)

561,897 analysis cells. Cell type is the argmax of the column-standardised PhiSpace scores, agreeing with the stored label for 97.99% of cells.

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

### BASE-T null (label variant `t10`)

BASE-T uses the truncated-likelihood null at alpha_trim 0.2 (addendum section 1.2, from the untrimmed fit, stopping when the parameters settle to 0.001): alpha0 **0.1832**, theta_m 1.5550, 4 iterations (parameters settled). Primary alpha_B **0.1085**, the level at which this null labels 10% of cells (user, 2026-10-17, after the G4 and Wu calibrations in outputs/division_classifier/report.md). BASE-O and SCORE use the robust null above, unchanged. V2 below simulates from the truncated null and refits it the same way.

### Label rates

| cell type | cells | BASE-T 0.1085 | BASE-T 0.05 | BASE-T 0.01 | BASE-T 0.001 | BASE-O 0.5 | BASE-O-global 0.5 | BASE-O-neg 0.5 | SCORE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 561897 | 0.09998 | 0.06369 | 0.03518 | 0.01918 | 0.1818 | 0.1777 | 0.2216 | 0.2342 |
| Endothelial | 48729 | 0.04981 | 0.02167 | 0.008434 | 0.002257 | 0.1242 | 0.1156 | 0.1013 | 0.1408 |
| CAFs | 52079 | 0.04134 | 0.02001 | 0.006701 | 0.001843 | 0.1318 | 0.1104 | 0.1077 | 0.153 |
| PVL | 42834 | 0.06124 | 0.03037 | 0.01216 | 0.004016 | 0.1462 | 0.1363 | 0.1291 | 0.1819 |
| B-cells | 59119 | 0.07519 | 0.03855 | 0.0137 | 0.004736 | 0.2018 | 0.1907 | 0.1643 | 0.2267 |
| T-cells | 87004 | 0.08626 | 0.05011 | 0.02102 | 0.008643 | 0.2286 | 0.2163 | 0.1926 | 0.2662 |
| Myeloid | 69669 | 0.0552 | 0.02713 | 0.01068 | 0.003675 | 0.1367 | 0.1256 | 0.1172 | 0.1598 |
| Normal Epithelial | 54508 | 0.1018 | 0.06183 | 0.032 | 0.01508 | 0.1632 | 0.1578 | 0.2637 | 0.2368 |
| Plasmablasts | 42523 | 0.1081 | 0.06907 | 0.03544 | 0.01658 | 0.1906 | 0.185 | 0.2359 | 0.2724 |
| Cancer Epithelial | 105432 | 0.2185 | 0.1665 | 0.1124 | 0.07195 | 0.2338 | 0.2592 | 0.4686 | 0.3491 |

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

| N decile | N median | mu0 median | c^T median @0.1085 | mean alpha_i @0.1085 | c^T median @0.05 | mean alpha_i @0.05 | c^T median @0.01 | mean alpha_i @0.01 | c^T median @0.001 | mean alpha_i @0.001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0.07496 | 1 | 0.06773 | 2 | 0.005763 | 2 | 0.003345 | 3 | 0.0001388 |
| 2 | 21 | 0.1215 | 2 | 0.04613 | 2 | 0.00894 | 2 | 0.005335 | 3 | 0.0004025 |
| 3 | 30 | 0.1742 | 2 | 0.01725 | 2 | 0.01672 | 3 | 0.00197 | 4 | 0.0003698 |
| 4 | 40 | 0.2318 | 2 | 0.02842 | 2 | 0.02605 | 3 | 0.002935 | 4 | 0.0002643 |
| 5 | 51 | 0.2988 | 2 | 0.04338 | 2 | 0.03154 | 3 | 0.004942 | 4 | 0.0004748 |
| 6 | 65 | 0.383 | 2 | 0.06201 | 3 | 0.01842 | 3 | 0.005135 | 5 | 0.000406 |
| 7 | 86 | 0.5125 | 2 | 0.06735 | 3 | 0.02178 | 4 | 0.004037 | 5 | 0.0004683 |
| 8 | 125 | 0.7544 | 3 | 0.05389 | 4 | 0.02878 | 5 | 0.005354 | 6 | 0.0004931 |
| 9 | 206 | 1.269 | 4 | 0.07056 | 5 | 0.03108 | 6 | 0.005988 | 8 | 0.0005669 |
| 10 | 385 | 2.404 | 6 | 0.07873 | 7 | 0.03568 | 9 | 0.00691 | 12 | 0.0006732 |

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
| all | 561897 | 0.532 | 0.8201 | 0.2342 | 0.09998 | 0.1818 |
| Endothelial | 48729 | 0.4839 | 0.8764 | 0.1408 | 0.04981 | 0.1242 |
| CAFs | 52079 | 0.3854 | 0.8585 | 0.153 | 0.04134 | 0.1318 |
| PVL | 42834 | 0.4535 | 0.8458 | 0.1819 | 0.06124 | 0.1462 |
| B-cells | 59119 | 0.4343 | 0.9019 | 0.2267 | 0.07519 | 0.2018 |
| T-cells | 87004 | 0.413 | 0.8749 | 0.2662 | 0.08626 | 0.2286 |
| Myeloid | 69669 | 0.4692 | 0.8632 | 0.1598 | 0.0552 | 0.1367 |
| Normal Epithelial | 54508 | 0.5345 | 0.7615 | 0.2368 | 0.1018 | 0.1632 |
| Plasmablasts | 42523 | 0.4887 | 0.7627 | 0.2724 | 0.1081 | 0.1906 |
| Cancer Epithelial | 105432 | 0.6831 | 0.7242 | 0.3491 | 0.2185 | 0.2338 |

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
| 0.1085 | 0.05299 | 0.05378 | -0.0147 | pass | 0.05369 | 0.05371 |
| 0.05 | 0.02214 | 0.02235 | -0.009264 | pass | 0.0225 | 0.02228 |
| 0.01 | 0.004634 | 0.004613 | 0.004515 | pass | 0.004738 | 0.004603 |
| 0.001 | 0.0004236 | 0.0004256 | -0.004675 | pass | 0.0004431 | 0.0004243 |

BASE-O call rate on null data (reported; no pass criterion):

| Td | null call rate (rebuilt) | null call rate (frozen c) | observed call rate |
| --- | --- | --- | --- |
| 0 | 0.2926 | 0.292 | 0.2779 |
| 0.25 | 0.2509 | 0.2443 | 0.2269 |
| 0.5 | 0.2073 | 0.1978 | 0.1818 |
| 0.75 | 0.1666 | 0.1557 | 0.1423 |
| 1 | 0.132 | 0.1213 | 0.1107 |

Stage 1 complete in 1.0 min. **Review point** (stages 1,
3 and 6 pause for review).


## Stage 2 -- NB refits (2026-10-17 18:03)

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
| NB-c | all cells | 0 | 561897 | -5.936e+05 | 5.937e+05 | 4.898e-07 | 361 | 11 | pass | 2 | 1 | 0.7139 | 0.3613 | 0.1725 | 0.1765 | 0.4067 | no | 1.28 |
| NB-c | all cells | 1 | 561897 | -5.936e+05 | 5.937e+05 | 3.261e-10 | 239 | 16 | pass | 1 | 1 | 0.7139 | 0.3613 | 0.1725 | 0.1765 | 0.4067 | no | 1.594 |
| NB-d | all cells | 0 | 561897 | -5.934e+05 | 5.935e+05 | 1.895e-07 | 443 | 13 | pass | 1 | 1 | 0.524-0.853 | 0.3617 | 0.173 | 0.1766 | 0.4073 | no | 1.801 |
| NB-d | all cells | 1 | 561897 | -5.934e+05 | 5.935e+05 | 6.997e-06 | 381 | 10 | pass | 1 | 1 | 0.524-0.853 | 0.3617 | 0.173 | 0.1766 | 0.4073 | no | 1.518 |
| NB-c | train fold 0 | 0 | 443599 | -4.685e+05 | 4.685e+05 | 2.599e-06 | 391 | 12 | pass | 1 | 1 | 0.7151 | 0.3599 | 0.1728 | 0.1751 | 0.4099 | no | 1.154 |
| NB-c | train fold 1 | 0 | 455993 | -4.769e+05 | 4.769e+05 | 6.336e-06 | 402 | 7 | pass | 1 | 1 | 0.7182 | 0.361 | 0.1701 | 0.1753 | 0.4063 | no | 0.9305 |
| NB-c | train fold 2 | 0 | 444258 | -4.741e+05 | 4.741e+05 | 1.156e-07 | 422 | 11 | pass | 1 | 1 | 0.7126 | 0.359 | 0.1758 | 0.1771 | 0.4051 | no | 1.149 |
| NB-c | train fold 3 | 0 | 449795 | -4.826e+05 | 4.826e+05 | 1.995e-08 | 370 | 10 | pass | 1 | 1 | 0.7127 | 0.3613 | 0.1756 | 0.1774 | 0.4073 | no | 1.037 |
| NB-c | train fold 4 | 0 | 453943 | -4.723e+05 | 4.723e+05 | 5.012e-06 | 406 | 13 | pass | 1 | 1 | 0.7102 | 0.3644 | 0.1704 | 0.1787 | 0.4075 | no | 1.235 |
| NB-d | train fold 0 | 0 | 443599 | -4.684e+05 | 4.684e+05 | 2.579e-06 | 408 | 10 | pass | 1 | 1 | 0.538-0.855 | 0.3601 | 0.1736 | 0.1754 | 0.4105 | no | 1.247 |
| NB-d | train fold 1 | 0 | 455993 | -4.768e+05 | 4.768e+05 | 4.095e-07 | 394 | 7 | pass | 1 | 1 | 0.531-0.851 | 0.3613 | 0.17 | 0.1753 | 0.4067 | no | 1.058 |
| NB-d | train fold 2 | 0 | 444258 | -4.739e+05 | 4.74e+05 | 6.859e-11 | 422 | 13 | pass | 1 | 1 | 0.525-0.848 | 0.3599 | 0.176 | 0.1775 | 0.4057 | no | 1.442 |
| NB-d | train fold 3 | 0 | 449795 | -4.824e+05 | 4.825e+05 | 8.458e-07 | 373 | 8 | pass | 1 | 1 | 0.503-0.877 | 0.3615 | 0.177 | 0.1779 | 0.4081 | no | 1.1 |
| NB-d | train fold 4 | 0 | 453943 | -4.722e+05 | 4.722e+05 | 4.864e-07 | 540 | 12 | pass | 1 | 1 | 0.518-0.840 | 0.365 | 0.1703 | 0.1783 | 0.4082 | no | 1.484 |

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
in `oof_nb.parquet` for Stage 3. Stage 2 complete in 21 min.


## Stage 3 -- out-of-fold fits (2026-10-17 18:25)

Every arm fitted on four folds and predicted on the fifth, on the stored
partition. NB-c and NB-d are Stage 2's fold fits. MIX is refitted per training
fold (penalty scaled to training size) from the full-data point estimate, to
the mixture's convergence criteria. LOG-T, LOG-O and LOG-S regress the primary
labels (LT_a1085, LO_td050, LS) on the shared weight
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
3, lambda 300) to 6.5e-12, so starting
from the full-data estimate did not change where the fits end.

| data | converged | iterations | grad_inf | held-out loglik / cell | Spartan CV, same fold | e^delta | mean pi (test) | sd(eta) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train fold 0 | pass | 861 | 2.104e-05 | -1.068 | -1.068 | 1.9 | 0.09681 | 1.323 | 1.034 |
| train fold 1 | pass | 1227 | 2.155e-05 | -1.116 | -1.116 | 1.911 | 0.09925 | 1.319 | 1.401 |
| train fold 2 | pass | 988 | 1.213e-05 | -1.027 | -1.027 | 1.895 | 0.09688 | 1.303 | 1.147 |
| train fold 3 | pass | 1166 | 1.549e-05 | -1.001 | -1.001 | 1.901 | 0.09696 | 1.299 | 1.318 |
| train fold 4 | pass | 1160 | 1.283e-05 | -1.136 | -1.136 | 1.902 | 0.1026 | 1.306 | 1.322 |

### LOG fits and V7 -- separation

Criterion: sd(eta) <= 20 at the production ridge for every LOG fit.
**pass** (largest sd(eta) 0.594);
all converged: **yes**.

| arm | data | converged | iterations | grad_inf | theta (log N) | norm a | norm c | sigma1(B) | sd(eta) | sd(eta) > 20 (V7) | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOG-T | all cells | pass | 346 | 3.215e-06 | 0.3878 | 0.4741 | 0.2203 | 0.2289 | 0.5879 | no | 0.5198 |
| LOG-T | train fold 0 | pass | 435 | 1.353e-06 | 0.3891 | 0.4732 | 0.226 | 0.2307 | 0.5912 | no | 0.4337 |
| LOG-T | train fold 1 | pass | 431 | 5.254e-06 | 0.3926 | 0.4738 | 0.2181 | 0.2298 | 0.5938 | no | 0.4344 |
| LOG-T | train fold 2 | pass | 454 | 3.116e-08 | 0.39 | 0.4737 | 0.2173 | 0.2341 | 0.5818 | no | 0.5626 |
| LOG-T | train fold 3 | pass | 430 | 2.042e-08 | 0.3929 | 0.4757 | 0.222 | 0.2331 | 0.5876 | no | 0.5999 |
| LOG-T | train fold 4 | pass | 498 | 6.191e-06 | 0.3751 | 0.4754 | 0.2217 | 0.2221 | 0.5917 | no | 0.5362 |
| LOG-O | all cells | pass | 499 | 2.839e-07 | 0.1726 | 0.3255 | 0.1073 | 0.1609 | 0.3612 | no | 0.8549 |
| LOG-O | train fold 0 | pass | 483 | 6.635e-07 | 0.1748 | 0.3275 | 0.108 | 0.1652 | 0.362 | no | 0.6603 |
| LOG-O | train fold 1 | pass | 512 | 1.01e-08 | 0.1783 | 0.3256 | 0.1089 | 0.1644 | 0.36 | no | 0.7008 |
| LOG-O | train fold 2 | pass | 462 | 4.961e-07 | 0.1737 | 0.3303 | 0.1077 | 0.1636 | 0.3628 | no | 0.7287 |
| LOG-O | train fold 3 | pass | 491 | 4.077e-06 | 0.1675 | 0.3245 | 0.1098 | 0.1635 | 0.3638 | no | 0.7906 |
| LOG-O | train fold 4 | pass | 444 | 9.391e-11 | 0.169 | 0.3211 | 0.107 | 0.1566 | 0.3605 | no | 0.7254 |
| LOG-S | all cells | pass | 565 | 7.332e-06 | 0.2872 | 0.3623 | 0.155 | 0.172 | 0.4277 | no | 1.275 |
| LOG-S | train fold 0 | pass | 789 | 5.482e-09 | 0.2897 | 0.3613 | 0.1564 | 0.1692 | 0.4314 | no | 0.9493 |
| LOG-S | train fold 1 | pass | 475 | 2.347e-09 | 0.2967 | 0.361 | 0.1561 | 0.1747 | 0.4268 | no | 1.816 |
| LOG-S | train fold 2 | pass | 422 | 5.766e-06 | 0.2854 | 0.3678 | 0.1503 | 0.1725 | 0.4264 | no | 0.5413 |
| LOG-S | train fold 3 | pass | 577 | 2.917e-06 | 0.2767 | 0.362 | 0.1549 | 0.1711 | 0.4313 | no | 0.6304 |
| LOG-S | train fold 4 | pass | 895 | 7.124e-11 | 0.2877 | 0.3597 | 0.1582 | 0.1724 | 0.4295 | no | 0.6607 |

### V4 -- push-through identity

50,000 cells drawn at random; y simulated from each arm's out-of-fold
predictive distribution; the simulated label rate compared with the mean
pushed-through p. Formal criterion on the primary thresholds (cT_a1085,
cO_td050): within 3 Monte Carlo SE. **pass**.
The other grid thresholds are shown for information; with
21 further comparisons, one beyond 3 SE would be
expected about 0.06 times by chance.

| model | threshold | mean p | simulated rate | MC SE | z | within 3 SE |
| --- | --- | --- | --- | --- | --- | --- |
| NB-c | LT_a1085 | 0.11366 | 0.11534 | 0.0013561 | 1.2357 | pass |
| NB-c | LT_a05 | 0.072205 | 0.07302 | 0.0011069 | 0.73674 | pass |
| NB-c | LT_a01 | 0.034713 | 0.03578 | 0.00079071 | 1.3495 | pass |
| NB-c | LT_a001 | 0.013912 | 0.01436 | 0.00051088 | 0.87714 | pass |
| NB-c | LO_td000 | 0.28598 | 0.28822 | 0.0019136 | 1.1698 | pass |
| NB-c | LO_td025 | 0.23811 | 0.23986 | 0.0018384 | 0.94961 | pass |
| NB-c | LO_td050 | 0.19215 | 0.19408 | 0.0017169 | 1.1245 | pass |
| NB-c | LO_td075 | 0.15087 | 0.15262 | 0.0015677 | 1.1153 | pass |
| NB-c | LO_td100 | 0.11657 | 0.11874 | 0.0014099 | 1.5421 | pass |
| NB-d | LT_a1085 | 0.11303 | 0.1114 | 0.0013529 | -1.2044 | pass |
| NB-d | LT_a05 | 0.071721 | 0.07068 | 0.0011033 | -0.9432 | pass |
| NB-d | LT_a01 | 0.034552 | 0.03308 | 0.00078847 | -1.8663 | pass |
| NB-d | LT_a001 | 0.013977 | 0.0136 | 0.00051159 | -0.7367 | pass |
| NB-d | LO_td000 | 0.28573 | 0.28278 | 0.0019144 | -1.5416 | pass |
| NB-d | LO_td025 | 0.23772 | 0.23534 | 0.001838 | -1.2942 | pass |
| NB-d | LO_td050 | 0.1917 | 0.19036 | 0.0017154 | -0.78317 | pass |
| NB-d | LO_td075 | 0.15045 | 0.14876 | 0.0015656 | -1.0792 | pass |
| NB-d | LO_td100 | 0.11619 | 0.11508 | 0.0014075 | -0.78907 | pass |
| MIX | LT_a1085 | 0.097583 | 0.09624 | 0.0012905 | -1.0406 | pass |
| MIX | LT_a05 | 0.064122 | 0.06504 | 0.0010584 | 0.86696 | pass |
| MIX | LT_a01 | 0.039164 | 0.03814 | 0.0008334 | -1.2282 | pass |
| MIX | LT_a001 | 0.021536 | 0.02114 | 0.00062319 | -0.63514 | pass |
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
| BASE-T (LT_a1085) | 0.09998 | 0.11368 | 0.11303 | 0.09763 | 0.10008 | nan | nan |
| BASE-O (LO_td050) | 0.18184 | 0.19241 | 0.19197 | 0.18377 | nan | 0.18189 | nan |
| SCORE (LS) | 0.23423 | nan | nan | nan | nan | nan | 0.23436 |

Out-of-fold predictions for every arm and threshold are in
`oof_predictions.parquet`. Stage 3 complete in 21 min.
**Review point** (stages 1, 3 and 6 pause for review).


## Stage 4 -- label-scale scoring (2026-10-17 18:27)

Every score is out of fold (Stage 3's predictions; nothing refitted). Intervals
are 95% spatial block bootstrap: the 250 stored blocks resampled with
replacement, 300 replicates, every arm scored on the same resampled cells,
so a paired difference is taken within each replicate. Per the interpretive
rule, each LOG arm was trained on the label it is scored against, and a
label-scale win for it is not evidence that it infers division better.
Calibration-in-the-large is reported as mean p / label rate (1 = calibrated).
Cell types are Stage 1's.

### BASE-T (LT_a1085, label rate 0.1000)

| arm | log loss | Brier | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- |
| NB-c | 0.2925 [0.2829, 0.3012] | 0.0835 [0.0799, 0.0868] | 0.7371 [0.7317, 0.7422] | 0.2461 [0.2342, 0.2601] | 1.137 [1.117, 1.156] |
| NB-d | 0.2924 [0.2828, 0.3011] | 0.0834 [0.0798, 0.0867] | 0.7372 [0.7319, 0.7424] | 0.2464 [0.2344, 0.2603] | 1.130 [1.110, 1.149] |
| MIX | 0.2937 [0.2837, 0.3031] | 0.0839 [0.0802, 0.0875] | 0.7330 [0.7279, 0.7379] | 0.2271 [0.2172, 0.2386] | 0.976 [0.957, 0.994] |
| LOG-T | 0.2994 [0.2900, 0.3084] | 0.0843 [0.0807, 0.0878] | 0.7027 [0.6944, 0.7101] | 0.2337 [0.2209, 0.2481] | 1.001 [0.981, 1.019] |

Paired differences (first minus second; for log loss and Brier, negative means
the first arm is better; for AUROC and AUPRC, positive):

| difference | log loss | Brier | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| MIX - NB-c | 0.0012 [0.0006, 0.0019] | 0.0005 [0.0002, 0.0007] | -0.0041 [-0.0049, -0.0034] | -0.0190 [-0.0216, -0.0164] |
| NB-d - NB-c | -0.0001 [-0.0001, -0.0001] | -0.0000 [-0.0000, -0.0000] | 0.0001 [0.0000, 0.0002] | 0.0003 [0.0002, 0.0004] |
| LOG-T - MIX | 0.0057 [0.0049, 0.0063] | 0.0004 [0.0002, 0.0005] | -0.0303 [-0.0350, -0.0257] | 0.0067 [0.0031, 0.0100] |
| LOG-T - NB-c | 0.0069 [0.0063, 0.0074] | 0.0008 [0.0007, 0.0010] | -0.0344 [-0.0386, -0.0305] | -0.0124 [-0.0138, -0.0112] |

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
| NB-c | 0.083468 | 0.000400 | 0.006767 | 0.089986 | -0.000152 |
| NB-d | 0.083438 | 0.000381 | 0.006774 | 0.089986 | -0.000155 |
| MIX | 0.083926 | 0.000138 | 0.006204 | 0.089986 | 0.000006 |
| LOG-T | 0.084298 | 0.000194 | 0.005744 | 0.089986 | -0.000138 |

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
| BASE-T | NB-c | 0.2925 | 0.3140 | -0.0215 | 0.2461 | 0.2714 | -0.0253 | 0.7371 | 0.7456 |
| BASE-T | NB-d | 0.2924 | 0.3127 | -0.0203 | 0.2464 | 0.2698 | -0.0234 | 0.7372 | 0.7462 |
| BASE-T | MIX | 0.2937 | 0.2941 | -0.0004 | 0.2271 | 0.2068 | 0.0202 | 0.7330 | 0.7121 |
| BASE-T | LOG-T | 0.2994 | 0.2995 | -0.0001 | 0.2337 | 0.2158 | 0.0179 | 0.7027 | 0.7084 |
| BASE-O | NB-c | 0.4528 | 0.4643 | -0.0115 | 0.2759 | 0.3032 | -0.0273 | 0.6535 | 0.6606 |
| BASE-O | NB-d | 0.4527 | 0.4636 | -0.0110 | 0.2763 | 0.3021 | -0.0259 | 0.6539 | 0.6608 |
| BASE-O | MIX | 0.4542 | 0.4567 | -0.0024 | 0.2715 | 0.2810 | -0.0094 | 0.6476 | 0.6474 |
| BASE-O | LOG-O | 0.4640 | 0.4639 | 0.0002 | 0.2423 | 0.2447 | -0.0024 | 0.6055 | 0.6052 |
| SCORE | LOG-S | 0.5204 | 0.5202 | 0.0003 | 0.3437 | 0.3488 | -0.0051 | 0.6472 | 0.6469 |

### Calibration-in-the-large by cell type (mean p / rate; * = 95% interval excludes 1)

**BASE-T**

| stratum | NB-c | NB-d | MIX | LOG-T |
| --- | --- | --- | --- | --- |
| all | 1.137* | 1.130* | 0.976* | 1.001 |
| Endothelial | 1.132* | 1.140* | 1.180* | 1.000 |
| CAFs | 1.162* | 1.099* | 1.273* | 1.063* |
| PVL | 1.153* | 1.145* | 1.102* | 1.044* |
| B-cells | 1.102* | 1.097* | 0.980 | 0.998 |
| T-cells | 1.122* | 1.122* | 0.986 | 0.994 |
| Myeloid | 1.213* | 1.228* | 1.176* | 1.113* |
| Normal Epithelial | 1.232* | 1.208* | 1.017 | 1.071* |
| Plasmablasts | 1.151* | 1.135* | 0.966 | 1.050* |
| Cancer Epithelial | 1.106* | 1.104* | 0.868* | 0.948* |

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
| alpha_B 0.1085 (primary) | 0.09998 | 1.137 [1.117, 1.156] | 1.130 [1.110, 1.149] | 0.976 [0.957, 0.994] |
| alpha_B 0.05 | 0.06369 | 1.135 [1.110, 1.160] | 1.128 [1.102, 1.152] | 1.009 [0.984, 1.031] |
| alpha_B 0.01 | 0.03518 | 0.985 [0.957, 1.012] | 0.980 [0.953, 1.007] | 1.113 [1.080, 1.145] |
| alpha_B 0.001 | 0.01918 | 0.724 [0.700, 0.749] | 0.728 [0.704, 0.753] | 1.119 [1.077, 1.162] |

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
  15-30, pooled over both grids): mean p - rate +0.00729 at c_i 5-10 and
  -0.13909 at 15-30: **holds**.
- **X3** (MIX better calibrated than NB-c across the grids, but under-predicting
  at the strictest thresholds). The two clauses are judged separately, since
  the second concerns the strictest thresholds. Mean |log(p / rate)| over the
  seven grid thresholds other than alpha_B 0.001 and 0.0001: 0.032 for
  MIX against 0.064 for NB-c (**better**).
  At alpha_B 0.001 and 0.0001 MIX's p / rate is
  1.12 (**does not clearly under-predict**),
  far below NB-c's, so over all nine thresholds MIX is the worse calibrated
  (0.041 against 0.093). Overall: **partly holds**.

Figures: `reliability_BASE-T.png`, `reliability_BASE-O.png`,
`reliability_SCORE.png`, `calibration_sweep_BASE-T.png`,
`calibration_sweep_BASE-O.png`. Every score and interval is in `scores.csv`
(including the full per-type metrics and the Murphy components for every
stratum); ceilings in `ceilings.csv`. Stage 4 complete in
2 min.


## Stage 5 -- biology and deviation (2026-10-17 18:28)

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
| Cancer Epithelial | 0.218 [0.210, 0.229] | 0.234 [0.226, 0.242] | 0.349 [0.339, 0.361] | 0.197 [0.193, 0.202] | 0.205 [0.194, 0.217] | 0.251 | 0.063 |
| Plasmablasts | 0.108 [0.103, 0.113] | 0.191 [0.185, 0.196] | 0.272 [0.263, 0.281] | 0.117 [0.114, 0.121] | 0.108 [0.099, 0.116] | 0.087 | 0.089 |
| T-cells | 0.086 [0.083, 0.090] | 0.229 [0.225, 0.233] | 0.266 [0.260, 0.273] | 0.109 [0.106, 0.113] | 0.098 [0.092, 0.105] | 0.297 | 0.330 |
| Normal Epithelial | 0.102 [0.097, 0.107] | 0.163 [0.157, 0.169] | 0.237 [0.228, 0.247] | 0.100 [0.096, 0.104] | 0.078 [0.069, 0.087] | 0.055 | 0.047 |
| B-cells | 0.075 [0.072, 0.079] | 0.202 [0.197, 0.206] | 0.227 [0.221, 0.233] | 0.090 [0.087, 0.094] | 0.078 [0.071, 0.086] | 0.219 | 0.277 |
| PVL | 0.061 [0.058, 0.064] | 0.146 [0.142, 0.150] | 0.182 [0.176, 0.188] | 0.062 [0.059, 0.065] | 0.024 [0.018, 0.031] | 0.068 | 0.038 |
| Myeloid | 0.055 [0.053, 0.057] | 0.137 [0.133, 0.139] | 0.160 [0.156, 0.164] | 0.048 [0.046, 0.050] | 0.009 [0.004, 0.013] | 0.137 | 0.186 |
| Endothelial | 0.050 [0.048, 0.052] | 0.124 [0.121, 0.128] | 0.141 [0.137, 0.145] | 0.041 [0.039, 0.043] | -0.004 [-0.009, 0.001] | 0.055 | 0.035 |
| CAFs | 0.041 [0.039, 0.044] | 0.132 [0.128, 0.136] | 0.153 [0.148, 0.158] | 0.024 [0.022, 0.025] | -0.020 [-0.025, -0.015] | 0.035 | 0.026 |
| all | 0.100 [0.095, 0.105] | 0.182 [0.178, 0.186] | 0.234 [0.227, 0.241] | 0.098 [0.094, 0.102] | 0.090 [0.083, 0.098] |  |  |

The Rogan-Gladen fraction is negative, with its interval below 0, in
1 types (CAFs). There the BASE-T label
fires less often than its own mean alpha_i: the technical null, fitted across
all types with one dispersion (widened by the cycling cells),
overstates those types' false-positive rate. The correction is not usable
there; it says the label's false positives are type-dependent, not that the
fraction is below zero.

Each arm's mean p against its label (label scale):

| type | BASE-T rate | NB-c p, BASE-T | MIX p, BASE-T | LOG-T p, BASE-T | BASE-O rate | NB-c p, BASE-O | MIX p, BASE-O | LOG-O p, BASE-O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial | 0.2185 | 0.2417 | 0.1897 | 0.2071 | 0.2338 | 0.2542 | 0.2107 | 0.2302 |
| Plasmablasts | 0.1081 | 0.1244 | 0.1045 | 0.1135 | 0.1906 | 0.2073 | 0.1870 | 0.1887 |
| T-cells | 0.0863 | 0.0968 | 0.0850 | 0.0858 | 0.2286 | 0.2265 | 0.2102 | 0.2222 |
| Normal Epithelial | 0.1018 | 0.1255 | 0.1035 | 0.1090 | 0.1632 | 0.1854 | 0.1658 | 0.1695 |
| B-cells | 0.0752 | 0.0828 | 0.0737 | 0.0750 | 0.2018 | 0.1988 | 0.1888 | 0.1995 |
| PVL | 0.0612 | 0.0706 | 0.0675 | 0.0640 | 0.1462 | 0.1564 | 0.1643 | 0.1476 |
| Myeloid | 0.0552 | 0.0670 | 0.0649 | 0.0615 | 0.1367 | 0.1528 | 0.1637 | 0.1466 |
| Endothelial | 0.0498 | 0.0564 | 0.0588 | 0.0498 | 0.1242 | 0.1335 | 0.1525 | 0.1302 |
| CAFs | 0.0413 | 0.0480 | 0.0526 | 0.0439 | 0.1318 | 0.1361 | 0.1676 | 0.1279 |
| all | 0.1000 | 0.1137 | 0.0976 | 0.1001 | 0.1818 | 0.1924 | 0.1838 | 0.1819 |

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
| BASE-T rate | pass (rank 1) | unclear | unclear (3/4 in bottom 5) | +0.47 (unclear) |
| BASE-O rate | pass (rank 1) | pass | pass (4/4 in bottom 5) | +0.68 (pass) |
| SCORE rate | pass (rank 1) | pass | unclear (3/4 in bottom 5) | +0.53 (unclear) |
| Wu S/G2M, all patients | unclear (rank 2) | unclear | pass (4/4 in bottom 5) |  |
| Wu S/G2M, ER+ patients | fail (rank 5) | unclear | pass (4/4 in bottom 5) |  |

### 6.1-6.2 Where the arms disagree

Mean out-of-fold difference by cell type (* = 95% interval excludes 0):

| type | MIX - NB-c, BASE-T | MIX - NB-c, BASE-O | MIX - LOG-T, BASE-T | MIX - LOG-O, BASE-O | NB-c - NB-d, BASE-T | NB-c - NB-d, BASE-O | LOG-O - LOG-S (percentile rank) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | +0.0024* | +0.0190* | +0.0089* | +0.0223* | -0.0004* | -0.0001* | +0.0124* |
| CAFs | +0.0046* | +0.0315* | +0.0087* | +0.0397* | +0.0026* | +0.0009* | -0.0196* |
| PVL | -0.0031* | +0.0079* | +0.0035* | +0.0167* | +0.0005* | +0.0004* | -0.0189* |
| B-cells | -0.0092* | -0.0100* | -0.0013* | -0.0107* | +0.0004* | -0.0005* | +0.1074* |
| T-cells | -0.0117* | -0.0162* | -0.0007* | -0.0120* | -0.0000* | +0.0002* | +0.1070* |
| Myeloid | -0.0020* | +0.0109* | +0.0035* | +0.0171* | -0.0008* | +0.0000 | +0.0134* |
| Normal Epithelial | -0.0219* | -0.0197* | -0.0055* | -0.0038* | +0.0024* | +0.0013* | -0.0934* |
| Plasmablasts | -0.0200* | -0.0203* | -0.0090* | -0.0017 | +0.0017* | +0.0002* | -0.0714* |
| Cancer Epithelial | -0.0521* | -0.0434* | -0.0174* | -0.0194* | +0.0006* | +0.0011* | -0.0687* |

Along the continuous strata, first level to last (decile 1 to 10; tumour
edge from 100 um or more inside to 300 um or more outside):

| difference | neighbourhood tumour fraction decile | density decile | vascular proximity decile (u) | distance to tumour edge (um; inside < 0) |
| --- | --- | --- | --- | --- |
| MIX - NB-c, BASE-T | -0.0015 to -0.0509 | +0.0011 to -0.0079 | -0.0225 to -0.0003 | -0.0513 to -0.0051 |
| MIX - NB-c, BASE-O | +0.0111 to -0.0438 | +0.0228 to -0.0064 | -0.0206 to +0.0184 | -0.0436 to +0.0049 |
| MIX - LOG-T, BASE-T | +0.0039 to -0.0159 | +0.0067 to -0.0005 | -0.0044 to +0.0055 | -0.0179 to +0.0022 |
| MIX - LOG-O, BASE-O | +0.0130 to -0.0115 | +0.0290 to -0.0081 | -0.0069 to +0.0230 | -0.0160 to +0.0105 |
| NB-c - NB-d, BASE-T | +0.0004 to +0.0012 | +0.0005 to +0.0003 | +0.0006 to +0.0002 | +0.0009 to +0.0005 |
| NB-c - NB-d, BASE-O | -0.0001 to +0.0012 | +0.0000 to -0.0001 | +0.0005 to +0.0000 | +0.0010 to +0.0001 |
| LOG-O - LOG-S (percentile rank) | +0.0581 to -0.1175 | +0.0225 to +0.0946 | -0.0227 to +0.0111 | -0.1232 to +0.0300 |

The tumour region (100 um bins with Cancer Epithelial share >= 0.5) holds
13.1% of cells. Every level, with intervals and mean
|difference|, is in `deviation.csv`; the trends are drawn in
`deviation_strata.png`.

**X4** (NB-d moves predictions mainly in Cancer Epithelial): largest mean
|NB-c - NB-d| in CAFs (BASE-T) and CAFs (BASE-O) --
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
| BASE-T | NB-c | 0.181 | 51 | 178 | 0.254 | 0.237 | lower-left 53%, lower-right 27%, upper-left 14%, upper-right 6% | lower-right 46%, lower-left 26%, upper-right 25%, upper-left 3% |
| BASE-T | NB-d | 0.179 | 51 | 176 | 0.244 | 0.239 | lower-left 53%, lower-right 27%, upper-left 14%, upper-right 6% | lower-right 45%, lower-left 26%, upper-right 26%, upper-left 3% |
| BASE-T | MIX | 0.167 | 151 | 128 | 0.246 | 0.267 | lower-left 50%, lower-right 48%, upper-left 1%, upper-right 1% | upper-right 37%, lower-right 35%, lower-left 23%, upper-left 5% |
| BASE-T | LOG-T | 0.155 | 104 | 148 | 0.314 | 0.284 | lower-right 49%, lower-left 46%, upper-left 3%, upper-right 2% | lower-right 42%, upper-right 32%, lower-left 21%, upper-left 5% |
| BASE-O | NB-c | 0.165 | 59 | 142 | 0.300 | 0.299 | upper-left 37%, lower-right 31%, lower-left 19%, upper-right 14% | lower-right 44%, upper-right 31%, lower-left 22%, upper-left 4% |
| BASE-O | NB-d | 0.164 | 61 | 142 | 0.295 | 0.299 | upper-left 38%, lower-right 30%, lower-left 20%, upper-right 13% | lower-right 44%, upper-right 31%, lower-left 21%, upper-left 4% |
| BASE-O | MIX | 0.171 | 99 | 86 | 0.420 | 0.439 | lower-right 43%, lower-left 29%, upper-left 22%, upper-right 5% | upper-right 30%, lower-right 29%, upper-left 23%, lower-left 17% |
| BASE-O | LOG-O | 0.143 | 60 | 105 | 0.256 | 0.327 | upper-left 45%, lower-left 28%, lower-right 17%, upper-right 10% | lower-right 56%, upper-right 27%, lower-left 14%, upper-left 3% |
| SCORE | LOG-S | 0.182 | 112 | 158 | 0.244 | 0.320 | upper-left 32%, lower-left 32%, lower-right 31%, upper-right 4% | lower-right 53%, upper-right 25%, lower-left 14%, upper-left 9% |

### 6.4 Over-dispersed types

Stage 4's scores for Cancer Epithelial and Plasmablasts:

| label | type | arm | log loss | AUROC | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- | --- | --- |
| BASE-T | Plasmablasts | NB-c | 0.3264 [0.3165, 0.3354] | 0.6721 [0.6637, 0.6803] | 0.1883 [0.1784, 0.2001] | 1.1512 [1.1124, 1.1920] |
| BASE-T | Cancer Epithelial | NB-c | 0.4988 [0.4879, 0.5100] | 0.6626 [0.6569, 0.6687] | 0.3428 [0.3291, 0.3569] | 1.1065 [1.0764, 1.1386] |
| BASE-T | Plasmablasts | NB-d | 0.3261 [0.3161, 0.3352] | 0.6727 [0.6643, 0.6809] | 0.1885 [0.1785, 0.2003] | 1.1351 [1.0967, 1.1751] |
| BASE-T | Cancer Epithelial | NB-d | 0.4987 [0.4879, 0.5099] | 0.6627 [0.6570, 0.6687] | 0.3436 [0.3300, 0.3580] | 1.1038 [1.0738, 1.1362] |
| BASE-T | Plasmablasts | MIX | 0.3250 [0.3146, 0.3343] | 0.6696 [0.6610, 0.6782] | 0.1805 [0.1709, 0.1915] | 0.9664 [0.9343, 1.0011] |
| BASE-T | Cancer Epithelial | MIX | 0.5063 [0.4936, 0.5195] | 0.6423 [0.6362, 0.6490] | 0.3069 [0.2955, 0.3194] | 0.8682 [0.8447, 0.8957] |
| BASE-T | Plasmablasts | LOG-T | 0.3321 [0.3224, 0.3410] | 0.6360 [0.6253, 0.6465] | 0.1749 [0.1651, 0.1862] | 1.0497 [1.0150, 1.0885] |
| BASE-T | Cancer Epithelial | LOG-T | 0.5032 [0.4921, 0.5152] | 0.6502 [0.6441, 0.6562] | 0.3358 [0.3223, 0.3502] | 0.9480 [0.9219, 0.9763] |
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


## Stage 7 -- sensitivities (2026-10-18 00:01)

Out of fold, scored as in Stage 4 (250-block bootstrap, 300 replicates;
paired differences within replicate). 65 new fits, all converged:
**yes**. V7 at the production ridge for every new LOG
arm: **pass** (largest sd(eta) 0.99).

### 1. The threshold grids, every arm

Each grid label now has its own LOG arm; the count arms are Stage 3's push-through.
The last column is the LOG arm's paired log-loss difference against MIX
(negative = LOG better).

**BASE-T**

| threshold | label rate | NB-c log loss | MIX log loss | LOG-T log loss | NB-c AUPRC | MIX AUPRC | LOG-T AUPRC | LOG-T - MIX log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_B 0.05 | 0.06369 | 0.2049 [0.1948, 0.2143] | 0.2060 [0.1957, 0.2158] | 0.2063 [0.1963, 0.2159] | 0.212 [0.200, 0.225] | 0.194 [0.184, 0.206] | 0.216 [0.204, 0.230] | 0.0003 [-0.0001, 0.0006] |
| alpha_B 0.01 | 0.03518 | 0.1241 [0.1157, 0.1321] | 0.1254 [0.1169, 0.1335] | 0.1247 [0.1164, 0.1325] | 0.180 [0.170, 0.191] | 0.171 [0.160, 0.181] | 0.194 [0.182, 0.206] | -0.0008 [-0.0010, -0.0005] |
| alpha_B 0.001 | 0.01918 | 0.0723 [0.0654, 0.0789] | 0.0723 [0.0656, 0.0786] | 0.0704 [0.0638, 0.0767] | 0.156 [0.147, 0.165] | 0.155 [0.145, 0.165] | 0.185 [0.174, 0.196] | -0.0019 [-0.0022, -0.0016] |

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
| LOG-T_unpenalised | 5 | 0.602 | 0.491 |
| NB-c unpenalised | 5 | 0.412 | 0.367 |

| label | arm | log loss | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- |
| BASE-T | LOG-T | 0.2994 [0.2900, 0.3084] | 0.234 [0.221, 0.248] | 1.001 [0.981, 1.019] |
| BASE-T | LOG-T unpenalised | 0.2994 [0.2900, 0.3085] | 0.233 [0.221, 0.248] | 1.001 [0.981, 1.019] |
| BASE-T | NB-c | 0.2925 [0.2829, 0.3012] | 0.246 [0.234, 0.260] | 1.137 [1.117, 1.156] |
| BASE-T | NB-c unpenalised | 0.2925 [0.2830, 0.3012] | 0.246 [0.234, 0.260] | 1.137 [1.117, 1.156] |
| BASE-T | MIX | 0.2937 [0.2837, 0.3031] | 0.227 [0.217, 0.239] | 0.976 [0.957, 0.994] |
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
| BASE-T | LOG-T unpenalised - LOG-T | log loss | 0.0000 [-0.0000, 0.0000] |
| BASE-T | LOG-T unpenalised - LOG-T | AUPRC | -0.0003 [-0.0004, -0.0002] |
| BASE-T | NB-c unpenalised - NB-c | log loss | 0.0000 [0.0000, 0.0000] |
| BASE-T | NB-c unpenalised - NB-c | AUPRC | -0.0002 [-0.0002, -0.0001] |
| BASE-T | LOG-T unpenalised - MIX | log loss | 0.0057 [0.0049, 0.0064] |
| BASE-T | LOG-T unpenalised - MIX | AUPRC | 0.0063 [0.0029, 0.0096] |
| BASE-T | LOG-T unpenalised - NB-c unpenalised | log loss | 0.0069 [0.0063, 0.0074] |
| BASE-T | LOG-T unpenalised - NB-c unpenalised | AUPRC | -0.0125 [-0.0139, -0.0114] |
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
`stage7_fits.csv`. Stage 7 complete in 154 min.


### Allocation model as adopted, and V9 (2026-10-18 00:01)

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

### Stage 6 results (2026-10-18 10:01)

30 replicates (G1: 10, G2: 10, G3: 10), each
on 100,000 real cells (held-out fold about
19,860 cells). Converged fits: NB-c 30/30, MIX 30/30, LOG-T 30/30, LOG-O 30/30, LOG-S 30/30;
largest LOG sd(eta) 0.68. Values are the median
[min, max] over replicates.

**1. Rank agreement with pi_true** (Spearman, held-out fold):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.970 [0.957, 0.974] | 0.960 [0.954, 0.965] |
| MIX, BASE-T | 0.639 [0.590, 0.680] | 0.607 [0.566, 0.648] |
| MIX, BASE-O | 0.396 [0.375, 0.418] | 0.135 [0.116, 0.163] |
| NB-c, BASE-T | 0.566 [0.513, 0.611] | 0.566 [0.513, 0.610] |
| NB-c, BASE-O | 0.486 [0.463, 0.523] | 0.396 [0.350, 0.421] |
| LOG-T | 0.826 [0.779, 0.846] | 0.827 [0.787, 0.845] |
| LOG-O | 0.771 [0.725, 0.826] | 0.478 [0.397, 0.540] |
| LOG-S | 0.785 [0.779, 0.833] | 0.769 [0.752, 0.818] |

**2. Neighbourhood effect.** Pooled over the types with a true contrast of at
least 0.1 in logit, the slope of predicted on true contrast (1 =
recovered, below 1 = attenuated):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.974 [0.781, 1.153] | 0.958 [0.726, 0.984] |
| MIX, BASE-T | 0.355 [0.306, 0.476] | 0.294 [0.252, 0.407] |
| MIX, BASE-O | 0.144 [0.093, 0.175] | 0.086 [0.033, 0.093] |
| NB-c, BASE-T | 0.393 [0.301, 0.484] | 0.376 [0.274, 0.488] |
| NB-c, BASE-O | 0.207 [0.135, 0.252] | 0.183 [0.145, 0.243] |
| LOG-T | 0.383 [0.324, 0.485] | 0.332 [0.234, 0.437] |
| LOG-O | 0.130 [0.038, 0.165] | 0.043 [0.012, 0.108] |
| LOG-S | 0.229 [0.149, 0.271] | 0.188 [0.119, 0.236] |

Per type, the ratio of predicted to true contrast (median over replicates):

**G1**

| type | true contrast | median N | MIX pi | MIX, BASE-T | MIX, BASE-O | NB-c, BASE-T | NB-c, BASE-O | LOG-T | LOG-O | LOG-S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 1.526 | 48.000 | 0.963 | 0.313 | 0.140 | 0.328 | 0.197 | 0.339 | 0.125 | 0.201 |
| CAFs | 1.331 | 60.000 | 1.013 | 0.252 | 0.006 | 0.248 | 0.067 | 0.213 | -0.027 | 0.121 |
| PVL | 3.386 | 46.000 | 1.001 | 0.231 | 0.134 | 0.255 | 0.170 | 0.305 | 0.145 | 0.201 |
| B-cells | 0.922 | 36.000 | 1.043 | 0.573 | 0.210 | 0.566 | 0.251 | 0.626 | 0.262 | 0.294 |
| T-cells | 0.923 | 44.000 | 0.975 | 0.695 | 0.326 | 0.667 | 0.348 | 0.638 | 0.357 | 0.419 |
| Myeloid | 1.743 | 50.000 | 0.913 | 0.406 | 0.156 | 0.431 | 0.228 | 0.397 | 0.119 | 0.196 |
| Normal Epithelial | 0.920 | 81.000 | 0.967 | 1.054 | 0.170 | 1.250 | 0.454 | 0.969 | 0.022 | 0.455 |
| Plasmablasts | 0.920 | 61.000 | 1.116 | 1.119 | 0.232 | 1.183 | 0.385 | 0.927 | 0.135 | 0.466 |
| Cancer Epithelial | 0.158 | 216.000 | 1.180 | 4.004 | 0.823 | 5.741 | 2.339 | 4.069 | 0.722 | 2.018 |

**G2**

| type | true contrast | median N | MIX pi | MIX, BASE-T | MIX, BASE-O | NB-c, BASE-T | NB-c, BASE-O | LOG-T | LOG-O | LOG-S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 1.526 | 48.000 | 0.931 | 0.254 | 0.066 | 0.339 | 0.193 | 0.302 | 0.028 | 0.140 |
| CAFs | 1.331 | 60.000 | 1.012 | 0.215 | -0.012 | 0.277 | 0.088 | 0.207 | -0.018 | 0.120 |
| PVL | 3.386 | 46.000 | 0.890 | 0.176 | 0.074 | 0.211 | 0.135 | 0.218 | 0.043 | 0.146 |
| B-cells | 0.922 | 36.000 | 0.965 | 0.509 | 0.130 | 0.551 | 0.249 | 0.516 | 0.152 | 0.217 |
| T-cells | 0.923 | 44.000 | 0.927 | 0.590 | 0.242 | 0.643 | 0.330 | 0.533 | 0.271 | 0.361 |
| Myeloid | 1.743 | 50.000 | 0.923 | 0.329 | 0.099 | 0.406 | 0.223 | 0.369 | 0.125 | 0.195 |
| Normal Epithelial | 0.920 | 81.000 | 1.013 | 0.915 | 0.013 | 1.270 | 0.450 | 0.858 | -0.139 | 0.383 |
| Plasmablasts | 0.920 | 61.000 | 1.163 | 0.933 | 0.040 | 1.133 | 0.375 | 0.854 | -0.022 | 0.431 |
| Cancer Epithelial | 0.158 | 216.000 | 2.054 | 3.695 | 0.561 | 5.965 | 2.843 | 3.851 | -0.088 | 2.060 |

**3. Fraction error** (mean over types of |mean prediction - realised cycling
fraction|; only MIX's pi targets this quantity):

| prediction | G1 | G2 |
| --- | --- | --- |
| MIX pi | 0.007 [0.004, 0.009] | 0.032 [0.030, 0.035] |
| MIX, BASE-T | 0.017 [0.013, 0.020] | 0.021 [0.018, 0.024] |
| MIX, BASE-O | 0.091 [0.085, 0.093] | 0.089 [0.083, 0.094] |
| NB-c, BASE-T | 0.030 [0.025, 0.036] | 0.028 [0.024, 0.037] |
| NB-c, BASE-O | 0.104 [0.100, 0.108] | 0.096 [0.090, 0.102] |
| LOG-T | 0.017 [0.014, 0.020] | 0.018 [0.015, 0.022] |
| LOG-O | 0.089 [0.086, 0.093] | 0.081 [0.078, 0.085] |
| LOG-S | 0.132 [0.126, 0.140] | 0.124 [0.121, 0.131] |

**G3 (no latent classes): implied cycling fraction by type** (median over
replicates; the truth has no cycling class):

| prediction | Endothelial | CAFs | PVL | B-cells | T-cells | Myeloid | Normal Epithelial | Plasmablasts | Cancer Epithelial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIX pi | 0.100 | 0.072 | 0.147 | 0.224 | 0.260 | 0.124 | 0.217 | 0.251 | 0.374 |
| MIX, BASE-T | 0.040 | 0.035 | 0.049 | 0.064 | 0.077 | 0.048 | 0.083 | 0.083 | 0.191 |
| MIX, BASE-O | 0.142 | 0.150 | 0.163 | 0.192 | 0.222 | 0.159 | 0.195 | 0.213 | 0.293 |
| NB-c, BASE-T | 0.040 | 0.034 | 0.049 | 0.064 | 0.077 | 0.047 | 0.079 | 0.083 | 0.167 |
| NB-c, BASE-O | 0.138 | 0.140 | 0.162 | 0.195 | 0.224 | 0.155 | 0.196 | 0.215 | 0.281 |
| LOG-T | 0.038 | 0.035 | 0.050 | 0.066 | 0.076 | 0.048 | 0.080 | 0.087 | 0.164 |
| LOG-O | 0.137 | 0.134 | 0.158 | 0.196 | 0.223 | 0.155 | 0.194 | 0.212 | 0.285 |
| LOG-S | 0.153 | 0.158 | 0.191 | 0.222 | 0.260 | 0.178 | 0.257 | 0.278 | 0.388 |

**Label-scale scores, held-out fold:**

| generator | prediction | log loss | AUPRC | mean p / rate |
| --- | --- | --- | --- | --- |
| G1 | MIX, BASE-T | 0.299 [0.290, 0.319] | 0.213 [0.197, 0.225] | 1.004 [0.953, 1.052] |
| G1 | MIX, BASE-O | 0.453 [0.448, 0.460] | 0.277 [0.269, 0.284] | 1.011 [0.972, 1.018] |
| G1 | NB-c, BASE-T | 0.304 [0.294, 0.324] | 0.204 [0.189, 0.215] | 1.202 [1.151, 1.284] |
| G1 | NB-c, BASE-O | 0.456 [0.452, 0.463] | 0.268 [0.262, 0.274] | 1.086 [1.059, 1.119] |
| G1 | LOG-T | 0.307 [0.299, 0.327] | 0.196 [0.180, 0.211] | 1.013 [0.954, 1.050] |
| G1 | LOG-O | 0.470 [0.463, 0.475] | 0.225 [0.214, 0.234] | 1.000 [0.982, 1.027] |
| G1 | LOG-S | 0.530 [0.513, 0.532] | 0.296 [0.281, 0.310] | 0.996 [0.968, 1.038] |
| G2 | MIX, BASE-T | 0.289 [0.273, 0.303] | 0.187 [0.180, 0.204] | 0.985 [0.941, 1.054] |
| G2 | MIX, BASE-O | 0.432 [0.427, 0.442] | 0.253 [0.247, 0.262] | 1.003 [0.974, 1.041] |
| G2 | NB-c, BASE-T | 0.295 [0.281, 0.312] | 0.183 [0.173, 0.196] | 1.322 [1.218, 1.365] |
| G2 | NB-c, BASE-O | 0.437 [0.432, 0.444] | 0.243 [0.236, 0.253] | 1.135 [1.100, 1.156] |
| G2 | LOG-T | 0.296 [0.281, 0.310] | 0.175 [0.166, 0.186] | 0.997 [0.961, 1.056] |
| G2 | LOG-O | 0.447 [0.444, 0.456] | 0.194 [0.189, 0.200] | 1.000 [0.974, 1.015] |
| G2 | LOG-S | 0.520 [0.509, 0.527] | 0.278 [0.268, 0.298] | 1.001 [0.984, 1.017] |
| G3 | MIX, BASE-T | 0.255 [0.240, 0.273] | 0.189 [0.173, 0.201] | 1.073 [0.987, 1.119] |
| G3 | MIX, BASE-O | 0.472 [0.469, 0.480] | 0.312 [0.297, 0.326] | 1.018 [0.984, 1.047] |
| G3 | NB-c, BASE-T | 0.253 [0.238, 0.269] | 0.194 [0.177, 0.205] | 0.999 [0.949, 1.043] |
| G3 | NB-c, BASE-O | 0.471 [0.467, 0.477] | 0.316 [0.301, 0.329] | 0.997 [0.974, 1.026] |
| G3 | LOG-T | 0.261 [0.245, 0.275] | 0.177 [0.161, 0.192] | 1.007 [0.978, 1.048] |
| G3 | LOG-O | 0.483 [0.480, 0.487] | 0.284 [0.267, 0.300] | 0.997 [0.969, 1.031] |
| G3 | LOG-S | 0.530 [0.523, 0.533] | 0.373 [0.354, 0.390] | 0.999 [0.976, 1.033] |

**X8** (under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from
1 than MIX, with the largest distortion in small, low-depth types), judged
clause by clause: median |log ratio| over the 9 types with a true contrast of at least 0.1: MIX pi 0.076, LOG-O 2.425, LOG-S 1.526 (both further from 1); largest distortion in CAFs (LOG-O) and CAFs (LOG-S); low-depth types (median N below the overall 57): B-cells, Endothelial, Myeloid, PVL, T-cells. **partly holds**.

Figure: `sim_recovery.png`, the neighbourhood-effect ratio by arm and
generator, and the rank agreement. Every replicate value is in
`sim_recovery.csv`. **Review point** (stages 1, 3 and 6 pause for review).

**Against the original labels' Stage 6 (written after the results).** Same draws, cells and
arms; only BASE-T differs (naive null at alpha_B 0.05, rate 3.3%, against the trunc 0.2 null at
0.1085, rate 10%).

| prediction | Spearman with pi_true G1 / G2, original | t10 | neighbourhood slope G1 / G2, original | t10 | fraction error G1 / G2, original | t10 |
| --- | --- | --- | --- | --- | --- | --- |
| LOG-T | 0.846 / 0.867 | 0.826 / 0.827 | 0.730 / 0.588 | 0.383 / 0.332 | 0.055 / 0.055 | 0.017 / 0.018 |
| MIX, BASE-T | 0.792 / 0.775 | 0.639 / 0.607 | 0.557 / 0.470 | 0.355 / 0.294 | 0.055 / 0.054 | 0.017 / 0.021 |
| NB-c, BASE-T | 0.718 / 0.671 | 0.566 / 0.566 | 0.431 / 0.408 | 0.393 / 0.376 | 0.061 / 0.053 | 0.030 / 0.028 |

The 10% label gets the level right: fraction error falls about threefold, because its rate
matches the true cycling fraction. It loses the spatial signal, though. Every arm trained on it, or
pushed through it, recovers about half the neighbourhood effect the original label recovered
(LOG-T slope 0.73 -> 0.38 under G1), and rank agreement with pi_true drops. At a 10% rate about
half the labelled cells are false positives, and that misclassification dilutes every contrast.
This is the attenuation the G4 calibration showed (MCC peaks at a 5-7% real rate).

Caveat: G1 and G2 have Poisson non-cycling cells, so on these simulated sections the trunc 0.2 null
collapses towards Poisson (alpha0 0.00-0.02, against 0.18 on the real counts). The simulated label
is therefore not built from the same null shape as the real one. G4 reproduces the real nulls,
but Stage 6 does not include it.
