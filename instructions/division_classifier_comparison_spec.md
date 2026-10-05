# Division classifier comparison: implementation spec

Status: draft for a Claude Code agent. Supersedes `threshold_baseline_spec.md`.

Read alongside:
- the mixture write-up "Division counts: from negative binomial to Poisson mixture" (main and Spatial analysis tabs);
- `run/compare_nb_mixture.py`, which already builds the matched NB;
- Somer et al. (OSDR), Methods "Ki67 thresholds" and Supplementary Fig. 1, which define the label construction that BASE-O translates.

## 0. Question, arms and estimands

**Question.** Starting from PhiSpace state and neighbourhood alone, how well does each model predict which cells divide? And where, and in which direction, do the models disagree with the data and with biological expectation?

### Labels (targets, built from observed Xenium data)

| Label | Construction | Can NB/MIX be pushed through it exactly? |
|---|---|---|
| **BASE-T** | Technical-null tail probability on the composite Tirosh S+G2M count (§2.2) | Yes |
| **BASE-O** | OSDR-faithful translation: background-subtracted, scaled count residual, thresholded at Td (§2.3) | Yes |
| **SCORE** | Seurat/Tirosh module score relative to bin-matched control genes, with a robust-null threshold (§2.4) | No. NB and MIX predict the composite count, not per-gene counts |

### Model arms

| Arm | Model | Output for cell i |
|---|---|---|
| **NB-c** | Matched NB, fixed log N offset, constant dispersion | p̂_i for each count-based label, by push-through (§3) |
| **NB-d** | As NB-c, with the state-dependent dispersion submodel | as above |
| **MIX** | Production Poisson mixture | Label scale: p̂_i by push-through. State scale: π_i |
| **LOG-T** | Logistic regression of BASE-T on the matched covariates | p̂_i directly |
| **LOG-O** | Logistic regression of BASE-O on the matched covariates. This is OSDR's pipeline with transcripts in place of Ki67 | p̂_i directly |
| **LOG-S** | Logistic regression of SCORE on the matched covariates | p̂_i directly |

### Two estimands; never compare across them

**Label scale.** P(L_i = 1 | x_i): the probability that a given label construction calls cell i cycling. Scoring against the observed L_i:
- For BASE-T and BASE-O: NB-c, NB-d, MIX and the matching LOG arm.
- For SCORE: LOG-S alone, plus the optional Monte Carlo push-through in §3.4.

**State scale.** P(Z_i = 1 | x_i): the probability that cell i is in the cycling state.
- Only MIX produces this directly (π_i).
- The LOG arms and NB arms produce label rates, which include each label's false positives and miss its false negatives.
- State-scale claims are tested only in §5 (biology) and §7 (simulation with known truth).

### Interpretive rule

Write this rule into `report.md` before any results.

- Each LOG arm is trained directly on its own label, so it is expected to win label-scale scores on that label.
- A label-scale win for a LOG arm is not evidence that it infers division better.
- NB and MIX can beat a LOG arm on the label scale only if their count likelihood carries information that binarisation discards.
- §7 decides which approach recovers the cycling state. In particular, §7 decides whether the OSDR pipeline (LOG-O) or a state-of-the-art score pipeline (LOG-S) matches MIX.

## 1. Covariate parity (blocking)

### Shared weight block

Every arm uses the same weight block:

η(x_i) = a0 + φ_iᵀa + φ̄_iᵀc + Σ_{r=1..2} (φ_iᵀU_r)(φ̄_iᵀV_r) + ψ·ox_i + H_iᵀh + φ_iᵀK·H_i

Conventions are those of the production mixture:
- φ is projected to the 8-dimensional sum-to-zero subspace;
- the bilinear term has rank 2;
- `ox` is the vascular proximity field;
- H is the density spline, with knots at the 25th, 50th, 75th and 90th percentiles, and K gives per-type density curves.

### Technical block, by arm

- **MIX:** b0 + θ log N_i + s(log A_i), as in production. s is a natural cubic spline with df 8. θ is free (fitted value 1.036).
- **NB-c, NB-d:** b0 + 1·log N_i (fixed offset) + s(log A_i), df 8.
- **LOG-T, LOG-O, LOG-S:** g(log N_i, log A_i) enters the logit, using the same s(log A) basis plus a linear log N term. Every label depends on depth, so omitting these terms would bias the state coefficients.

### Model equations

**NB:**
- log μ_i = technical_i + η(x_i), with y_i ~ NB(μ_i, α_i).
- NB-c: α_i = α.
- NB-d: log α_i = γ0 + γ_typeᵀ(type indicators), as in the earlier dispersion submodel.

**MIX:** unchanged from production.

**LOG arms:** logit P(L_i = 1) = η(x_i) + g(log N_i, log A_i).

### Penalty parity

MIX carries weight_ridge = 10 · n / 678,885 on all η coefficients except the intercept.
- Apply the identical ridge, with the same per-observation scaling, to the η coefficients of the NB and LOG arms.
- As a sensitivity, also fit the NB and LOG arms unpenalised.

For every fit, report ‖a‖, ‖c‖, σ1(UVᵀ) and sd(η). Flag any fit with sd(η) > 20. The LOG arms can separate in the same way the concomitant model did.

### Offset mismatch (record, do not fix silently)

The NB arms fix θ = 1, whereas MIX estimates θ̂ = 1.036 (95% CI 1.014 to 1.057). Run MIX with θ fixed at 1 as a sensitivity row.

## 2. Labels

### 2.1 Shared ingredients

**Outcome.** y_i is the summed Tirosh S+G2M count, on the same gene list as the mixture. Assert that the lists are equal.

**Technical null.** y_i ~ NB(μ0_i, α0), with log μ0_i = β0 + log N_i + s(log A_i), df 8.
- Use the same N definition, offset and area basis as the NB arms.
- Include no cell type, φ or neighbourhood terms. Under the DAG, those are the signal.

**Robust fit by iterative trimming.**
1. Fit the null on all cells.
2. Compute p_i = F̄0(y_i − 1) = P(Y ≥ y_i | μ0_i, α0).
3. Refit on cells with p_i ≥ 0.001.
4. Repeat until the set of cells with p_i < 0.001 changes by less than 0.1% of cells, or 20 iterations. Log the trace.

**Freezing.** All labels are target definitions. Fit them once on all cells and freeze them before any model is fitted. Record this in `report.md`.

### 2.2 BASE-T (tail probability)

- L^T_i = 1{p_i < α_B}. Primary α_B = 0.001; grid {0.05, 0.01, 0.001, 1e-4}.
- Equivalently, L^T_i = 1{y_i ≥ c^T_i}, where c^T_i = min{c : F̄0(c − 1) < α_B}. Store c^T_i.
- **Discreteness.** With P(y = 0) = 0.68, c^T_i is a small integer. The realised false-positive rate α_i = F̄0(c^T_i − 1) therefore varies with depth and sits below α_B. Store α_i and report its distribution by depth decile.

### 2.3 BASE-O (OSDR-faithful translation)

OSDR's construction (Somer et al., Methods "Ki67 thresholds"; Supp. Fig. 1C to 1E) has four steps:
1. Fix one global background level, Tn = 0.5 mean isotopic counts. Tn was read off the distributions of several IMC markers. No per-cell relative expression against other proteins is used.
2. Keep Ki67 values above Tn and subtract Tn.
3. Divide by the Ki67 SD.
4. Call division where the adjusted Ki67 exceeds Td (0.5 in Supp. Fig. 1E). The authors report robustness for Td in [0, 1].

The translation keeps each step. The background becomes per cell, because Xenium background scales with depth and area while IMC noise is roughly constant.

1. **Background.** Per-cell expected count μ0_i from the technical null in §2.1.
2. **Residual.** r_i = (y_i − μ0_i) / sqrt(μ0_i + α0 μ0_i²).
3. **Above-noise selection.** Keep cells with r_i > 0, which is OSDR's "Ki67 > Tn". Cells with r_i ≤ 0 are non-dividing.
4. **Scale.** Set s+ = SD of r_i over cells with r_i > 0, and define the adjusted value a_i = r_i / s+.
   - Primary: s+ is computed globally.
   - Variant BASE-O-type: s+ is computed within each coarse cell type. This is the literal reading if OSDR's SD was per type. Check Supp. Fig. 1D and record which reading was adopted.
   - Caveat to state in the report: per-type scaling raises the effective threshold in types with more cycling cells, so it partly removes between-type proliferation differences.
5. **Label.** L^O_i = 1{a_i > Td}. Primary Td = 0.5; grid {0, 0.25, 0.5, 0.75, 1}, OSDR's robustness range.
6. **Discrete equivalent.** Compute c^O_i = min{y integer : (y − μ0_i)/sd0_i > Td · s+}, and store it. For BASE-O-type, c^O_i also depends on the cell's type. That is acceptable, because the type is fixed and known at prediction time.
7. **Null call rate.** Report the BASE-O call rate under data simulated from the technical null, at each Td (V2). Unlike BASE-T, BASE-O does not control false positives by construction.

Sensitivity BASE-O-neg uses an alternative floor built from Xenium negative-control codewords. Background is the expected decoding false-positive count: b_i = ρ̂_neg · A_i · |T| / n_negcodewords, where ρ̂_neg is the global negative-codeword rate per µm² and n_negcodewords is the number of negative-control codewords. Replace μ0_i by b_i and sd0_i by sqrt(b_i).

This floor sees decoding noise only, not spillover from neighbouring cells. Label it as such.

### 2.4 SCORE (module score; state-scale and validation only)

1. Compute the score with scanpy `score_genes` on the S+G2M genes:
   - `gene_pool` excludes the proliferation blocklist and the Tirosh genes;
   - n_bins = 25 and ctrl_size = 50;
   - library size is computed without blocklist genes;
   - normalise with log1p(X · s / N), s = median N.
2. **Threshold:** robust half-normal null.
   - μ0^S = mode of the score's KDE.
   - σ0^S = RMS deviation from that mode over the left half of the distribution.
   - Call cells with upper-tail p < 0.001.
3. Report Cohen's κ against BASE-T and against BASE-O, overall and by cell type.
4. **Zero-count diagnostic.** Among cells with y_i = 0, report the spread of the score. These cells have identical cycle signal, so any spread comes entirely from the control-gene term. Also report the fraction of SCORE calls that have y_i ≤ 1.

## 3. Pushing each model through a label

For a held-out cell i and a count-based label with threshold c_i (either c^T_i or c^O_i):

- **NB:** p̂_i = F̄_NB(c_i − 1; μ̂_i, α̂_i).
  - Never use 1{μ̂_i ≥ c_i}.
- **MIX:** p̂_i = π̂_i · F̄_Pois(c_i − 1; μ̂1_i) + (1 − π̂_i) · F̄_Pois(c_i − 1; μ̂0_i).
  - Use π̂_i, never the posterior w_i.
- **LOG arms:** p̂_i is the fitted probability for that arm's own label.

### 3.4 Optional: Monte Carlo push-through for SCORE

Run this only if Stages 1 to 6 are complete.

**Allocation model.**
- For each coarse cell type, estimate per-gene proportions of the Tirosh count from the observed data. Pool cells with y ≥ 1 within the type.
- Fit a Dirichlet-multinomial so that its overdispersion matches the observed per-gene detection rates (V9).

**Procedure.**
1. For each held-out cell, draw y from the model's predictive distribution.
2. Allocate the draw across genes.
3. Hold the control genes at their observed values.
4. Recompute the score and apply the frozen SCORE threshold.
5. Repeat 200 times. p̂_i is the mean call.

**Labelling.** Report the results as "label scale under an allocation assumption". Never put them in the same table as §4.3.

## 4. Out-of-sample protocol and label-scale scoring

### 4.1 Folds

Reuse the existing partition:
- 250 k-means blocks (`MiniBatchKMeans`, seed 0; median diameter 506 µm);
- dealt into 5 folds of 50 blocks (shuffle seed 0).

Load the stored assignment if it exists. Otherwise regenerate it, and assert identity with the mixture CV by comparing hashes of the cell-to-fold map.

### 4.2 Fitting

1. For each fold, fit every arm on the other four folds, with the penalty scaled to training size.
2. Predict on the held-out fold.
3. Concatenate the out-of-fold predictions.

The NB and MIX fits do not depend on the label, so fit them once per fold and push them through every label.

### 4.3 Scores against each label (out of fold)

For BASE-T and BASE-O separately:
- log loss;
- Brier score with Murphy decomposition (reliability, resolution, uncertainty);
- AUROC and AUPRC (AUPRC is the more informative at low prevalence);
- calibration-in-the-large (mean p̂ − mean L), overall and per cell type;
- reliability curves (20 equal-count bins), overall and for Cancer Epithelial and Plasmablasts;
- a threshold sweep: calibration-in-the-large across the α_B grid (BASE-T) and the Td grid (BASE-O).

**Cell-type stratification.** Use the argmax of column-standardised PhiSpace scores, never the raw argmax. This rule reproduces the stored labels for 99.9% of cells.

**Intervals.** Spatial block bootstrap over the 250 blocks, 300 replicates, resampling the out-of-fold predictions without refitting. Compute paired differences between arms within each replicate.

For SCORE, report LOG-S's own scores alone. These are not comparable with the other tables.

### 4.4 Noise ceiling

For each fitted count model and each count-based label:
1. Simulate y from the model.
2. Apply the same c_i.
3. Compute the expected log loss and AUPRC the model would achieve if it were true.

Interpret each arm by the gap between its ceiling and its observed score.

## 5. Biological expectation (state scale and label rates)

### 5.1 Fractions per cell type

Report for each cell type:
- the BASE-T, BASE-O and SCORE label rates;
- each arm's mean p̂;
- MIX's mean π̂;
- a Rogan–Gladen corrected BASE-T fraction, (rate − ᾱ_i)/(sens − ᾱ_i), where the sensitivity is mean F̄_Pois(c^T_i − 1; μ̂1_i) taken from MIX. Label this row "borrows from MIX".

### 5.2 Anchors

**The section.** The 10x dataset page lists only Discovery Life Sciences, stage IIA (T1c N1 MX). There is no receptor status and no Ki67.
- Infer receptor context in situ: report the fraction of Cancer Epithelial cells whose ESR1, PGR and ERBB2 counts exceed a technical null for each gene.
- This is descriptive only. Do not assign a clinical subtype.

**Wu et al. 2021 (GSE176078).**
- Load the per-cell metadata (patient, subtype, `celltype_major`, and the authors' phase calls if present).
- Compute per-cell-type S/G2M fractions, overall and within the patients of the subtype closest to the in-situ receptor profile.
- If phase calls are absent, run Seurat `CellCycleScoring` on the reference.
- State the platform gap: droplet scRNA-seq, at higher depth.

**Literature Ki67 index by subtype.** Report as a range only.
- Ki67 protein marks G1 as well as S/G2/M, so a transcript S/G2M fraction is expected to sit below it.
- Treat it as an upper bound for Cancer Epithelial, not as a target.

### 5.3 Qualitative checks

Report each as pass, fail or unclear. Do not tune anything to pass them.
- Cancer Epithelial is highest.
- Plasmablasts and T-cells are intermediate.
- CAFs, PVL, Endothelial and Normal Epithelial are low.
- The cell-type ordering is broadly consistent with the Wu reference.

## 6. Where the arms deviate

### 6.1 Cell-level disagreement

Compute these out-of-fold differences:
- d_i = p̂_MIX − p̂_NB-c, per count-based label;
- MIX − LOG-T and MIX − LOG-O;
- NB-c − NB-d;
- LOG-O − LOG-S, as a rank comparison only, because their prevalences differ.

### 6.2 Stratification

Stratify each difference by:
- cell type;
- neighbourhood tumour fraction (deciles of φ̄ for Cancer Epithelial);
- density percentile;
- vascular proximity decile;
- signed distance to the tumour region edge. The tumour region is the set of 100 µm bins with Cancer Epithelial share ≥ 0.5; distances inside it are negative.

Report means with block-bootstrap intervals.

### 6.3 Maps

1. For each arm and label, compute the 100 µm binned mean of the held-out residual L_i − p̂_i.
2. Compute local Moran's I following the conventions of `run/spatial_analysis.py`: queen weights, 9,999 conditional permutations, BH at 5%.
3. Overlay the persistent regions from the mixture write-up (lower-left under-prediction, lower-right over-prediction). Report whether each arm shares them.

### 6.4 Over-dispersed types

Report every metric in §4.3 separately for Cancer Epithelial and Plasmablasts.

## 7. Known-truth simulation (decides the state-scale question)

### 7.1 Setup

Use the real x_i, N_i and A_i, and the real fold structure. Draw Z_i ~ Bernoulli(π_i^true), then y_i from one of three generators:
- **G1**, home advantage for MIX: the fitted mixture.
- **G2**, misspecified for MIX: NB components with dispersion within the cycling class, matched to the observed Cancer Epithelial variance-to-mean ratio of 7.29.
- **G3**, no latent classes: a single NB from the fitted NB-c. The true Z is undefined here. Report label-scale scores and each arm's implied cycling fraction only.

### 7.2 Procedure

For each replicate:
1. Build BASE-T and BASE-O from the simulated y. Build SCORE by allocating the simulated y across genes with the §3.4 allocation model, holding the control genes at their observed values.
2. Fit NB-c, MIX, LOG-T, LOG-O and LOG-S.

### 7.3 State-recovery metrics (G1 and G2)

1. **Rank agreement.** Spearman correlation of each arm's p̂ (or π̂) with π^true. This is scale-free, so arms with different prevalence are comparable.
2. **Neighbourhood effect,** the quantity the phase portrait is built on. For each own type, compute the predicted and the true log-odds contrast between tumour-rich and tumour-poor neighbourhoods (90th versus 10th percentile of φ̄ tumour fraction), and report the ratio of predicted to true.
   - Label noise attenuates this ratio toward 0 if it is non-differential.
   - Depth-dependent label noise can push the ratio either way.
3. **Fraction error.** Per-type absolute error of the predicted against the true cycling fraction. Only MIX targets this quantity directly; report it for the other arms with that caveat.

### 7.4 Scale and reading

Run 10 replicates per generator at n = 100,000 on Spartan, following the existing `sim_*` array pattern.

How to read the results:
- If LOG-O or LOG-S matches MIX on metrics 1 and 2 under G2, the mixture's extra machinery is not earning its keep for the phase portrait.
- If they fall well short, the replacement of OSDR's threshold-then-logistic step is justified on these data.

## 8. Validation and blocking tests (all must pass before §4)

| ID | Test | Pass criterion |
|---|---|---|
| V1 | Matched NB convergence. The existing fit stopped at 20,000 iterations with gradient 1.18 | rel_tol 1e-9 and grad_tol 1e-5 on the full data and every training fold |
| V2 | Null calibration: simulate y from the fitted technical null with no excess | BASE-T label rate within ±30% (relative) of mean α_i at every α_B. BASE-O null call rate reported at every Td (no pass criterion; it is a property to report) |
| V3 | Survivor functions against `scipy.stats` (nbinom with the correct parameterisation; poisson) on a 1,000-point grid | Maximum absolute difference < 1e-12 |
| V4 | Push-through identity. For each fitted model, simulate 50,000 cells, apply c^T_i and c^O_i, and compare with mean p̂ | Within 3 Monte Carlo SE |
| V5 | Fold identity with the mixture CV | Equal hash of the cell-to-fold map |
| V6 | Parity audit: print the η design columns for NB, MIX and every LOG arm | Identical names and order |
| V7 | LOG separation | sd(η) ≤ 20 at the production ridge for every LOG arm; otherwise stop and report |
| V8 | Determinism | Identical outputs on rerun with fixed seeds |
| V9 | Allocation model (needed for §3.4 and §7) | Simulated per-type, per-gene detection rates within 10% (relative) of observed for genes detected in ≥ 1% of cells |
| V10 | BASE-O discrete equivalence | 1{a_i > Td} equals 1{y_i ≥ c^O_i} for every cell at every Td |

On any failure:
1. Stop.
2. Write the failure to `report.md`.
3. Do not proceed.

## 9. Build stages

Stop after each stage and append a status note to `report.md`.

1. **Labels.** Technical null; BASE-T, BASE-O (all variants) and SCORE. Run V2, V3 and V10, the discreteness report and the SCORE zero-count diagnostic.
2. **NB refits.** Refit NB-c and NB-d to convergence (V1). Run V6.
3. **Out-of-fold fits.** Fit every arm out of fold. Run V5 and V7, then V4 and V8.
4. **Label-scale scoring** (§4), including the noise ceilings.
5. **Biology and deviation** (§5 and §6).
6. **Known-truth simulation** (§7). Run V9 first.
7. **Sensitivities.**
   - the α_B and Td grids;
   - BASE-O-type and BASE-O-neg;
   - unpenalised NB and LOG arms;
   - MIX with θ = 1;
   - all labels and arms on the nucleus-only matrix with nucleus area, as a spillover check.
8. **Optional:** the §3.4 Monte Carlo push-through for SCORE.

## 10. Outputs

All outputs go in `outputs/division_classifier/`.

**Data files.**
- `labels.parquet`: cell_id, y, N, A, μ0, α0, r, a (global and per type), c^T at each α_B, c^O at each Td, α_i, L^T, L^O, L^S, the SCORE value.
- `oof_predictions.parquet`: cell_id, fold, p̂ for every arm and label, π̂ (MIX), μ̂ and α̂ (NB arms), μ̂0 and μ̂1 (MIX).
- `scores.csv`: label, arm, threshold, stratum, metric, estimate, CI, plus paired differences.
- `ceilings.csv`, `biology.csv`, `deviation.csv`, `sim_recovery.csv`.

**Figures.**
- Reliability curves, per label.
- Calibration-in-the-large against threshold, by type, per label.
- The per-type fraction ladder: each label, each arm, π̂, and the Wu reference.
- Binned residual maps with local Moran clusters.
- Disagreement against neighbourhood tumour fraction and against density.
- The §7 neighbourhood-effect ratio by arm and generator. This is the headline figure for the OSDR-versus-mixture question.

**Report.** `report.md` contains the pre-stated expectations, the run log, deviations, test results, and headline tables on each scale.

## 11. Pre-stated expectations (record before Stage 4; do not edit)

- **X1.** Each LOG arm has the best label-scale log loss on its own label.
- **X2.** NB-c's calibration-in-the-large changes sign across the threshold grids:
  - over-prediction where c_i falls in 5 to 10;
  - under-prediction where c_i falls in 15 to 30.
  This follows the NB rootogram wave.
- **X3.** MIX is better calibrated than NB-c across the grids, but under-predicts at the strictest thresholds (its 30+ tail shortfall).
- **X4.** NB-d moves predictions mainly in Cancer Epithelial.
- **X5.** Disagreement between NB and MIX concentrates in Cancer Epithelial and Plasmablasts.
- **X6.** BASE-O's null call rate at Td = 0.5 is several percent, not near zero. Where μ0 is below 1, a count of 1 or 2 already clears the threshold, so OSDR's [0, 1] robustness range will not transfer.
- **X7.** Among cells with y = 0, the SCORE value has non-trivial spread, coming entirely from the control-gene term.
- **X8.** Under G2, LOG-O and LOG-S have neighbourhood-effect ratios further from 1 than MIX, with the largest distortion in small, low-depth cell types.

## 12. Open decisions

- **MIX headline row.** Production MIX or MIX with θ = 1. Default: production, with θ = 1 as a sensitivity.
- **Primary thresholds.** Defaults are α_B = 0.001 and Td = 0.5. The threshold sweeps are the more informative results, and the headline should not hinge on one threshold.
- **Reading of OSDR's SD.** Global or per type; record the reading adopted after checking Supp. Fig. 1D.
- **Test 07.** Unresolved in the mixture write-up. All results from this comparison inherit that caveat until it is resolved.
