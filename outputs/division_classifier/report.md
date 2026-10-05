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
| LOG-O regression | "OSDR's pipeline" | matched covariates, pooled over types | OSDR fits a separate logistic regression per cell type on neighbour counts within 80 um (`tdm.model.Model`, "two models per cell_type"). LOG-O reproduces OSDR's **label** exactly in translation, but its regression is the shared weight block, for covariate parity. It is OSDR's labelling with our regression, not OSDR end to end. |


## Stage 1 -- labels (built and frozen 2026-10-05 13:53)

678,885 analysis cells. Cell type is the argmax of column-standardised PhiSpace
scores, agreeing with the stored label for 99.90% of cells.

**Outcome and library size.** y recomputed from the count matrix with the
mixture's 53-gene Tirosh list equals the table's `y_count` for every
cell; N (excluding the Tirosh genes and the 530-gene blocklist) equals the
table's `N` for every cell.

### Technical null (spec 2.1)

log mu0 = b0 + log N + s(log A), area spline as the NB arms (df 8), raw log N as
a fixed offset. Robust fit by iterative trimming at P(Y >= y) < 0.001; stopped on
**tolerance** after 2 fits. Final alpha0 = **0.8293**.

| iteration | n_fit | alpha0 | flagged | flagged_frac | changed | converged |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 678885 | 0.89 | 1027 | 0.001513 | 1027 | True |
| 2 | 677858 | 0.8293 | 1237 | 0.001822 | 210 | True |

### Label rates

| cell type | cells | BASE-T 0.05 | BASE-T 0.01 | BASE-T 0.001 | BASE-T 0.0001 | BASE-O 0.5 | BASE-O-global 0.5 | BASE-O-neg 0.5 | SCORE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 678885 | 0.03272 | 0.009866 | 0.001822 | 0.000327 | 0.1508 | 0.1438 | 0.1912 | 0.0607 |
| Endothelial | 60853 | 0.02432 | 0.006228 | 0.001397 | 0.0003615 | 0.06162 | 0.07549 | 0.0469 | 0.05226 |
| CAFs | 52491 | 0.01179 | 0.002 | 0.0001715 | 1.905e-05 | 0.1076 | 0.09737 | 0.06731 | 0.03686 |
| PVL | 36313 | 0.02313 | 0.005535 | 0.001157 | 0.0001377 | 0.1236 | 0.1268 | 0.0789 | 0.0686 |
| B-cells | 56234 | 0.02043 | 0.003894 | 0.0004446 | 1.778e-05 | 0.1868 | 0.1707 | 0.1385 | 0.06178 |
| T-cells | 97814 | 0.0223 | 0.004447 | 0.0004601 | 2.045e-05 | 0.218 | 0.1889 | 0.1615 | 0.06297 |
| Myeloid | 49317 | 0.01411 | 0.003082 | 0.0004258 | 6.083e-05 | 0.09788 | 0.1003 | 0.06781 | 0.04506 |
| Normal Epithelial | 95893 | 0.02202 | 0.005329 | 0.0009073 | 0.0001564 | 0.1113 | 0.1196 | 0.08267 | 0.05935 |
| Plasmablasts | 101074 | 0.0371 | 0.01369 | 0.002652 | 0.0003859 | 0.1444 | 0.1289 | 0.2502 | 0.0432 |
| Cancer Epithelial | 128896 | 0.07283 | 0.0257 | 0.005082 | 0.00104 | 0.2063 | 0.2 | 0.4688 | 0.09072 |

### Discreteness of BASE-T (spec 2.2)

c^T_i is the smallest count with P(Y >= c) < alpha_B; alpha_i is the realised
false-positive rate at that count, always below alpha_B.

| N decile | N median | mu0 median | c^T median @0.05 | mean alpha_i @0.05 | c^T median @0.01 | mean alpha_i @0.01 | c^T median @0.001 | mean alpha_i @0.001 | c^T median @0.0001 | mean alpha_i @0.0001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.02241 | 1 | 0.02165 | 2 | 0.002117 | 2 | 0.0003048 | 3 | 2.539e-05 |
| 2 | 9 | 0.06719 | 2 | 0.01351 | 2 | 0.003853 | 3 | 0.0002395 | 4 | 2.548e-05 |
| 3 | 17 | 0.1277 | 2 | 0.01259 | 3 | 0.003827 | 4 | 0.0003719 | 5 | 4.078e-05 |
| 4 | 27 | 0.2047 | 2 | 0.0277 | 3 | 0.004386 | 4 | 0.0005346 | 5 | 4.572e-05 |
| 5 | 38 | 0.2926 | 2 | 0.02897 | 4 | 0.005304 | 5 | 0.0004448 | 6 | 5.113e-05 |
| 6 | 52 | 0.4002 | 3 | 0.02169 | 4 | 0.005555 | 6 | 0.000533 | 7 | 5.446e-05 |
| 7 | 71 | 0.5496 | 3 | 0.03461 | 5 | 0.005813 | 7 | 0.000627 | 9 | 6.197e-05 |
| 8 | 102 | 0.8148 | 4 | 0.03214 | 6 | 0.00677 | 9 | 0.0006535 | 11 | 6.569e-05 |
| 9 | 174 | 1.48 | 6 | 0.0382 | 9 | 0.00763 | 13 | 0.0007606 | 17 | 7.576e-05 |
| 10 | 356 | 3.607 | 12 | 0.04372 | 18 | 0.008736 | 26 | 0.0008725 | 34 | 8.715e-05 |

### BASE-O (spec 2.3)

Primary BASE-O scales the residual within each cell type, as OSDR's own code
does (`is_dividing`: one cutoff per type, `Td * sigma_type + Tn`, ddof 0).
Per-type s+: Endothelial 2.676, CAFs 1.286, PVL 1.630, B-cells 1.212, T-cells 1.140, Myeloid 1.554, Normal Epithelial 1.734, Plasmablasts 1.207, Cancer Epithelial 1.416.
BASE-O-global uses one s+ = 1.4921 for all cells.
Negative-codeword floor (BASE-O-neg, decoding noise only, no spillover):
rho = 4.051e-06 per um^2 over 609 negative-control codewords,
|T| = 53, mean b_i = 2.32e-05 against mean mu0 = 0.8182;
s+ = 514.8577.

Caveat: per-type scaling raises the effective threshold in types with more
cycling cells, so it partly removes between-type proliferation differences.
That is a property of OSDR's construction, inherited here.

### SCORE (spec 2.4)

scanpy `score_genes`, 53 genes, n_bins 25, ctrl_size 50, control pool
4,571 genes (panel minus blocklist), log1p(X s / N) with s = median
N = 44. Robust null: KDE mode 0.0000, left-half
RMS sigma 0.0050, threshold 0.0155 (upper-tail p <
0.001).

| stratum | n | kappa vs BASE-T | kappa vs BASE-O | SCORE rate | BASE-T rate | BASE-O rate |
| --- | --- | --- | --- | --- | --- | --- |
| all | 678885 | 0.05494 | 0.5334 | 0.0607 | 0.001822 | 0.1508 |
| Endothelial | 60853 | 0.04948 | 0.9104 | 0.05226 | 0.001397 | 0.06162 |
| CAFs | 52491 | 0.008921 | 0.482 | 0.03686 | 0.0001715 | 0.1076 |
| PVL | 36313 | 0.03096 | 0.6861 | 0.0686 | 0.001157 | 0.1236 |
| B-cells | 56234 | 0.01342 | 0.4455 | 0.06178 | 0.0004446 | 0.1868 |
| T-cells | 97814 | 0.01361 | 0.3884 | 0.06297 | 0.0004601 | 0.218 |
| Myeloid | 49317 | 0.0179 | 0.6061 | 0.04506 | 0.0004258 | 0.09788 |
| Normal Epithelial | 95893 | 0.02838 | 0.67 | 0.05935 | 0.0009073 | 0.1113 |
| Plasmablasts | 101074 | 0.1112 | 0.4221 | 0.0432 | 0.002652 | 0.1444 |
| Cancer Epithelial | 128896 | 0.0974 | 0.5546 | 0.09072 | 0.005082 | 0.2063 |

**Zero-count diagnostic.**

| cells with y = 0 | score sd, y = 0 | score sd, all cells | score IQR, y = 0 | score 1st-99th pct, y = 0 | SCORE call rate among y = 0 | share of SCORE calls with y <= 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 461525 | 0.002762 | 0.009062 | 0.003987 | -0.010 to 0.000 | 0 | 0.3756 |

### Validation

- **V3** survivor functions vs scipy: max |diff| NB 1.63e-14, Poisson
  0.00e+00 (< 1e-12): **pass**.
- **V10** BASE-O discrete equivalence, every cell, every Td, all three
  variants: **pass**.
- **V2** null calibration, end to end (simulate from the null, refit the robust
  null, rebuild the labels): **pass**.

| alpha_B | label rate (refit) | mean alpha_i | relative error | pass (+/-30%) | label rate (frozen c) | mean alpha_i (frozen) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.02807 | 0.02729 | 0.02885 | pass | 0.02755 | 0.02739 |
| 0.01 | 0.005556 | 0.005348 | 0.039 | pass | 0.005325 | 0.005372 |
| 0.001 | 0.0005848 | 0.0005308 | 0.1017 | pass | 0.0005553 | 0.0005323 |
| 0.0001 | 6.039e-05 | 5.351e-05 | 0.1287 | pass | 5.303e-05 | 5.308e-05 |

BASE-O call rate on null data (reported; no pass criterion):

| Td | null call rate (rebuilt) | null call rate (frozen c) | observed call rate |
| --- | --- | --- | --- |
| 0 | 0.2421 | 0.2416 | 0.2337 |
| 0.25 | 0.2111 | 0.2005 | 0.1921 |
| 0.5 | 0.1726 | 0.1563 | 0.1508 |
| 0.75 | 0.1358 | 0.1191 | 0.1168 |
| 1 | 0.1066 | 0.08952 | 0.08972 |

Stage 1 complete in 1.9 min. **Review point** (stages 1,
3 and 6 pause for review).

### Stage 1 review notes

All Stage 1 tests pass. Labels were rebuilt once, on 2026-10-05, after OSDR's
code showed that its Ki67 SD is computed per cell type; BASE-O is now per type,
matching OSDR. Five findings bear on how the later stages read.

**1. The primary BASE-T label is nearly empty.** At alpha_B = 0.001 it calls
1,237 cells (0.18%), against a realised null rate of about 0.05%. Under the
mixture (this borrows from MIX), the labels' operating characteristics are:

| label | call rate | sensitivity | false-positive rate | PPV |
| --- | --- | --- | --- | --- |
| BASE-T 0.05 | 3.27% | 26.7% | 0.54% | 0.86 |
| BASE-T 0.01 | 0.99% | 6.0% | 0.04% | 0.95 |
| BASE-T 0.001 | 0.18% | 0.5% | ~0 | 0.98 |
| BASE-T 1e-4 | 0.03% | ~0 | ~0 | 0.99 |
| BASE-O 0 | 23.37% | 66.0% | 17.65% | 0.33 |
| BASE-O 0.5 | 15.08% | 60.9% | 8.93% | 0.47 |
| BASE-O 1 | 8.97% | 49.3% | 4.06% | 0.61 |
| BASE-O-global 0.5 | 14.38% | 60.1% | 8.36% | 0.48 |

Cycling cells mostly carry small counts (median mixture rate 1.30, against 0.23
for non-cycling), so a strict tail threshold catches very few of them. At 0.001
the per-type positive counts are tiny (CAFs: 9 cells), so per-type AUPRC and
reliability curves at that threshold will be dominated by noise.

**2. BASE-O is no more selective than the technical null, a strong form of X6.**
At Td = 0.5 it calls 15.1% of cells. Data simulated from the technical null give
17.3% with the label rebuilt end to end, and 15.6% with the thresholds frozen.
The observed rate does not exceed the null rate at any Td. Against the mixture's
tighter non-cycling component, its false-positive rate is 8.9%, and fewer than
half its calls are cycling cells.

For scale: OSDR's Ki67 labels give division fractions of 1-6% (Danenberg:
T cells 1.4%, B cells 0.7%). The translated label gives T cells 21.8% and B cells
18.7%. BASE-T at 0.05 (3.3% overall) is the only count label inside OSDR's range.

Per-type versus global scaling changes little (15.1% against 14.4% overall), and
neither changes the conclusion.

| cell type | BASE-O 0.5 | BASE-O-global 0.5 | mean pi (MIX) | BASE-O PPV under MIX |
| --- | --- | --- | --- | --- |
| Endothelial | 6.2% | 7.5% | 12.1% | 0.40 |
| CAFs | 10.8% | 9.7% | 5.4% | 0.18 |
| PVL | 12.4% | 12.7% | 10.6% | 0.33 |
| B-cells | 18.7% | 17.1% | 10.7% | 0.32 |
| T-cells | 21.8% | 18.9% | 12.8% | 0.37 |
| Myeloid | 9.8% | 10.0% | 7.2% | 0.24 |
| Normal Epithelial | 11.1% | 12.0% | 9.6% | 0.32 |
| Plasmablasts | 14.4% | 12.9% | 8.8% | 0.50 |
| Cancer Epithelial | 20.6% | 20.0% | 18.4% | 0.80 |

BASE-O labels T-cells above Cancer Epithelial; the mixture puts Cancer
Epithelial first. Only in Cancer Epithelial are most BASE-O calls cycling cells.

**3. SCORE's null is built on zero-count cells.** A cell with no Tirosh counts
scores minus its control-gene mean, so it can only score at or below 0. These
461,525 cells make up the left half of the distribution, and 70,451 of them sit
at exactly 0 (no control counts either), which is where the KDE mode lands. The
half-normal null (sigma 0.005) therefore measures control-gene noise in empty
cells (X7). SCORE calls no zero-count cell, but 37.6% of its calls have y <= 1.
Its agreement with BASE-T is near zero (kappa 0.055).

**4. BASE-O-neg is degenerate.** The decoding-noise floor averages 2.3e-5
counts, so at Td = 0.5 the label is effectively 1{y >= 2}.

**5. LOG-O is OSDR's labelling with our regression, not OSDR end to end.**
OSDR fits a separate logistic regression per cell type on neighbour counts within
80 um. LOG-O uses the shared weight block, pooled over types, for covariate
parity with MIX.

**Resolved.** OSDR's SD reading: per type, from the code (see Decisions).

**Open for the user.** Choice of primary alpha_B; whether SCORE keeps the
literal threshold.

## Decisions after the Stage 1 review (2026-10-05)

| item | spec default | adopted | reason |
| --- | --- | --- | --- |
| primary alpha_B (BASE-T) | 0.001 | **0.05** | at 0.001 the label calls 0.18% of cells, with 9 positives among CAFs; at 0.05 it calls 3.3% with PPV 0.86 under MIX, inside OSDR's range of label rates. The full grid is still reported. Decided by the user. |
| SCORE threshold | robust half-normal, p < 0.001 | **unchanged, literal** | it is how the pipeline is used in practice; its zero-count dependence is reported as a finding. Decided by the user. |
| OSDR's SD reading | check Supp. Fig. 1D | **per type** | confirmed by Supp. Fig. 1D, which shows the adjusted Ki67 distribution of each cell type, each scaled by its own SD, collapsing onto one shape; and by `is_dividing` in OSDR's code. |

Labels were not rebuilt: every alpha_B level was already frozen in
`labels.parquet`, so changing the primary only changes which column is the
headline.

**OSDR's label rates, for scale** (Supp. Fig. 1E, Danenberg, adjusted Ki67 >
0.5): B cells 0.8%, endothelial 1.1%, fibroblasts 1.5%, T cells 1.5%,
macrophages 1.6%, tumour 2.1%. The translated BASE-O at Td = 0.5 gives
6.2-21.8% by type, 7-27 times higher, and ranks T cells first rather than
tumour. The mechanism is the floor. In IMC, Ki67 rarely clears the 0.5 noise
level. In Xenium, r > 0 whenever y exceeds the technical expectation, which in a
low-depth cell is a single transcript; 23% of all cells clear the floor (BASE-O
at Td = 0).

Agreement (Cohen's kappa) at the adopted primary thresholds:

| stratum | SCORE vs BASE-T 0.05 | SCORE vs BASE-O 0.5 | BASE-T 0.05 vs BASE-O 0.5 |
| --- | --- | --- | --- |
| all | 0.675 | 0.533 | 0.320 |
| B-cells | 0.477 | 0.445 | 0.166 |
| CAFs | 0.472 | 0.482 | 0.180 |
| Cancer Epithelial | 0.861 | 0.555 | 0.464 |
| Endothelial | 0.623 | 0.910 | 0.550 |
| Myeloid | 0.463 | 0.606 | 0.233 |
| Normal Epithelial | 0.522 | 0.670 | 0.305 |
| PVL | 0.487 | 0.686 | 0.287 |
| Plasmablasts | 0.891 | 0.422 | 0.372 |
| T-cells | 0.502 | 0.388 | 0.151 |


### STOPPED in Stage 2: V5 (fold identity) failed

the regenerated folds do not reproduce the mixture CV's fold sizes:

| fold | n_train here | n_test here | n_test, mixture CV (Spartan) | match |
| --- | --- | --- | --- | --- |
| 0 | 538435 | 140450 | 138613 | NO |
| 1 | 550445 | 128440 | 126534 | NO |
| 2 | 538462 | 140423 | 143096 | NO |
| 3 | 542225 | 136660 | 138416 | NO |
| 4 | 545973 | 132912 | 132226 | NO |

Per spec section 8, nothing further was run.

**Diagnosis.** MiniBatchKMeans is deterministic within an environment but not
across them. This machine (scikit-learn 1.6.1, numpy 2.4.1) returns the same
partition on every rerun and under one thread. Spartan returned identical fold
sizes across all 40 mixture CV tasks. The two environments, Spartan's with an
unpinned scikit-learn, converge to slightly different partitions from the same
seed. The partition the mixture CV used exists only on Spartan.

**Fix (pending).** `run/division_make_folds.py` (`sbatch hpc/folds.slurm`)
writes the map in Spartan's environment and checks it against the CV outputs
there. Copied back, `folds.parquet` is loaded by Stage 2 and every later stage,
as spec 4.1 intends ("load the stored assignment if it exists").

Side effect for earlier work: the block measurements in the mixture write-up's
Spatial analysis tab (250 blocks, median 2,704 cells, median diameter 506 um)
were made on the local partition, not the one Spartan's bootstrap and CV used.
They describe the same construction, but not the identical blocks.
