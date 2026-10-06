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
