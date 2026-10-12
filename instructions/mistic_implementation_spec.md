# MisTIC integration: implementation brief

## Goal

Replace cell area as a technical covariate everywhere with MisTIC-derived quantities.

1. Run MisTIC once and save per-transcript misassignment probabilities.
2. Build per-cell technical covariates and a MisTIC-weighted count matrix.
3. Swap these into: PhiSpace input, the BASE-T and BASE-O technical null, the Poisson mixture, and any other model that currently uses cell area.
4. Run the essential tests in Step 4.

**Scope is only the above.** Do not change any other model term (mixing-weight covariates, density spline, vascular proximity field, calibration, neighbourhood radius). Ask before touching anything not listed here.

**Execution rules.** Python throughout. Heavy jobs get a Slurm script, but do not submit anything: Lachy runs all jobs on Spartan himself.

## Notation

| Symbol | Meaning |
|---|---|
| p_k | MisTIC probability that transcript k is misassigned (its reassignment probability) |
| q_k | 1 - p_k |
| G_cyc | Tirosh S/G2M gene set (the proliferation outcome genes) |
| B | Existing PhiSpace blocklist (contains G_cyc) |
| N_i | Raw total transcript count of cell i |
| Y_i^G | Raw count of transcripts from gene set G in cell i |

---

## Step 1. Prepare and run MisTIC

### 1.1 Environment

- Create a separate conda env with Python 3.10, then `pip install txMisTIC`.
- Verify the install with `python -m MisTIC --version`.
- Do not install it into the main analysis env. Its pins (torch 1.13, numpy<2, anndata<0.11, scanpy 1.10) will clash with the spatialdata stack.

### 1.2 Build the inputs

Read from the Xenium output bundle and write each input as parquet. All coordinates are already in microns.

**detected_transcripts**, from `transcripts.parquet`:
- Keep `qv >= 20`.
- Keep gene features only. Use `codeword_category` if present; otherwise drop names starting with `NegControlProbe_`, `NegControlCodeword_`, `UnassignedCodeword_`, `DeprecatedCodeword_` or `BLANK_`.
- Drop rows with `cell_id == "UNASSIGNED"`.
- Keep the columns `x_location`, `y_location`, `feature_name`, `cell_id` and `transcript_id`. The `transcript_id` lets results be joined back later.
- Decode bytes columns to str if needed.

**cell_metadata**, from `cells.parquet`:
- Make `cell_id` the first column, followed by `x_centroid`, `y_centroid` and `cell_type`.
- Set `cell_type` to the argmax archetype of the **current** PhiSpace scores (the existing raw-input pipeline).
- These labels are cycle-blind. Do not leave `cell_type` empty: MisTIC would then Leiden-cluster on all genes, including the cycle genes.

**cell_boundary_polygons**, from `cell_boundaries.parquet`:
- This file stores vertex lists (`cell_id`, `vertex_x`, `vertex_y`).
- Build one shapely Polygon per cell into a GeoDataFrame indexed by `cell_id`, and save it as GeoParquet.

**Consistency check.** Confirm `cell_id`s match across all three tables. Drop cells without a valid polygon and log how many were dropped.

### 1.3 Run

Use the Python API, not the CLI. The CLI does not expose the per-transcript probabilities.

```python
from MisTIC.mistic_class import mistic
m = mistic(cell_centroid_x_col="x_centroid", cell_centroid_y_col="y_centroid",
           celltype_col="cell_type", tx_x_col="x_location", tx_y_col="y_location",
           gene_col="feature_name", cell_col="cell_id")
m.import_data(cell_metadata=META_PATH, cell_boundary_polygons=POLY_PATH,
              detected_transcripts=TX_PATH)   # pass paths, not dataframes (it deep-copies)
m.patchfy_data()
m.initialize_parameters()
m.training_loop(n_epochs=50)                  # early stopping applies
m.compute_reassign_probs()
m.tx_reassign_info.write_parquet(OUT_PROBS)   # MisTIC does not save this itself
m.save_model(dir_name=OUT_DIR, model_name="mistic", save_correction_result=False)
```

- Do **not** call `correct_tx`. We use the soft probabilities, not thresholded reassignment.
- Inspect the columns of `tx_reassign_info`, identify the probability column, and record its name in a README.

### 1.4 Slurm

- Write a GPU job script with generous memory.
- If the job runs out of memory, split the section into spatial tiles with a 50 µm overlap and run each tile separately. Keep results only for transcripts whose host-cell centroid lies in the tile core.

### 1.5 Join back

MisTIC's `molecule_id` is `tx_<row number>` of the input transcript table. Map it back to `transcript_id`.

Output: `transcripts_mistic.parquet` with columns `transcript_id`, `cell_id`, `gene`, `p`.

---

## Step 2. Per-cell covariates

Save the per-cell quantities to `cell_mistic.parquet`:

- **N_i**: raw transcript count.
- **Ñ_i** = Σ_{k in i} q_k, summed over all transcripts in the cell. This is the MisTIC-weighted library size.
- **f_i** = mean of p_k over transcripts in cell i whose gene is **not in B**. This is the contamination fraction, computed from non-cycle genes only so that the outcome genes never feed the correction.
- **s_i^G** = Σ_{j in nbr(i)} Y_j^G / Σ_{j in nbr(i)} N_j, where nbr(i) is the 6 nearest cells by centroid, excluding i. Use raw counts. This is the share of G transcripts in the cells that spillover could come from.
- **m_i^G** = f_i · N_i · s_i^G. This is the expected number of misassigned G transcripts in cell i.
  - Implement it as a function of the gene set G.
  - The main models use m_i = m_i^{G_cyc}.
  - The tests reuse the same function with other gene sets.

Save the weighted count matrix separately:

- **X̃_ig** = Σ_{k in i, gene g} q_k, as a sparse float matrix.
- Save it in whatever format the existing PhiSpace pipeline reads.

---

## Step 3. Swap into the models

### General rule (applies to every model)

- Remove every cell-area term: linear terms, splines and interactions.
- Replace N with Ñ wherever library size appears. The offset coefficient stays fixed at 1.
- Where the outcome is a raw count of a gene set G, add θ·m_i^G **additively** to the mean.
  - Parameterise θ = exp(η) and estimate it.
- Outcomes stay as raw integer counts. Never weight the outcome counts.

### 3.1 PhiSpace

- Use X̃ as the input instead of raw counts. Everything else stays the same: log-normalise, apply the blocklist on the full gene set after normalisation, subset, train, apply.
- Recompute φ̄ from the new φ.
- If any step requires integer counts, stop and report.
- Save both the raw-input φ and the corrected φ. The raw-input version is the sensitivity arm.

### 3.2 BASE-T and BASE-O technical null

- The new null model is NB with mean

  μ_i = Ñ_i · exp(β₀) + θ · m_i

- The area spline is removed.
- The additive term does not fit a log-link GLM, so write the NB negative log-likelihood directly.
  - Parameters: (β₀, η, log dispersion).
  - Optimise with `scipy.optimize.minimize` (L-BFGS-B), using analytic or autodiff gradients.
- Leave everything downstream of the null unchanged (how p-values and thresholds are computed).

### 3.3 Poisson mixture

- Component rates: remove the area term and replace log N with log Ñ.
- The observed count becomes

  Y_i ~ Σ_z π_zi · Poisson(λ_zi + θ · m_i)

  - θ is shared across components.
- If the code already has an additive background ρ_i, add θ·m_i to it. Do not replace it.
- The mixing weight π is unchanged.

### 3.4 Any other model

This includes the NB and logistic baselines in the division-classifier comparison. Apply the general rule.

---

## Step 4. Essential tests

**Blocking** tests must pass before any results are used. **Report** tests produce outputs for Lachy to review.

### T1. Likelihood recovery (blocking)

1. Simulate 200k cells from the new NB null, using the real Ñ and m values and known β₀, θ and dispersion.
2. Refit the model.
3. Pass if every parameter is recovered within 2 SE.

For the mixture, add the θ·m term to the existing known-parameter recovery test and confirm it still passes.

### T2. MisTIC sanity (blocking)

Check all of the following:

- Every retained transcript has a p in [0, 1].
- Ñ_i ≤ N_i for every cell.
- On a random subsample of 1M transcripts, bin each transcript by its distance to its own cell's polygon boundary, and plot mean p against that distance. Mean p should rise toward the boundary.

If the curve is flat, MisTIC is not capturing spatial misassignment: stop and report.

### T3. Null calibration negative control (blocking)

1. Build 20 random gene sets from genes **not in B**. Each set should be the same size as G_cyc and matched to it gene-by-gene on mean expression and detection rate.
2. Run BASE-T and BASE-O on each set, using m^G for that set.
3. Report the call rate at nominal α overall, by cell-area decile, and by φ̄[Cancer Epithelial] decile. Area is only a diagnostic axis here; it is not in the model.

Pass criteria:

- The overall call rate is within [0.5α, 2α].
- Within each binning, the max decile rate divided by the min decile rate is below 2.

### T4. Spillover control (report)

1. Choose G_epi: epithelial markers that are not in B and are present on the panel. Check candidates such as EPCAM, KRT8, KRT18, KRT19 and CDH1.
2. In non-epithelial cells (by argmax archetype), run BASE-T and BASE-O on G_epi, once with the θ·m^G term and once without.
3. Plot the call rate against distance to the nearest Cancer Epithelial cell, in 10 µm bins out to 100 µm.

Expected result: without the term, call rates rise near epithelium; with it, the curve flattens. Report the fitted θ.

### T5. PhiSpace feature check (report)

Compare the raw-input and corrected φ.

**(a) Shell correlation.** Compute the correlation between φ_i and φ̄_i using neighbours in the 0–15, 15–30 and 30–50 µm shells. Report it per archetype component and averaged. The inner shell should drop more than the outer shells after correction.

**(b) Sharpening check.**
- For each cell, compute the margin: the top PhiSpace score minus the second-highest score.
- Report the mean change in margin (corrected minus raw) by f_i decile.
- The increase should concentrate in the high-f_i deciles.
- Flag for Lachy if the lowest f_i decile changes by more than half as much as the highest decile. That pattern would mean MisTIC is pulling cells toward prototypes.

### T6. Old vs new comparison (report)

Produce a table comparing the area-based fits with the MisTIC-based fits:

- **Mixture:** δ, θ (with CI), overall cycling fraction, and IQR-scaled neighbourhood rate ratios for the priority archetype pairs.
- **BASE-T and BASE-O:** call rates per archetype.

There is no pass/fail criterion for this table.

---

## Deliverables

- **Scripts:** `mistic_prepare.py`, `mistic_run.py` with its `.slurm`, `mistic_covariates.py`, the model changes, and the test files.
- **Data:** `transcripts_mistic.parquet`, `cell_mistic.parquet`, and the weighted count matrix.
- **Report:** a short markdown file with the T2–T6 outputs and figures.

## Out of scope

- Robust or iterative refitting of the null.
- Any use of MisTIC probabilities for cycle-gene transcripts. They are computed but not used.
- Rerunning MisTIC without cycle genes.
- Changes to the density, vascular proximity or calibration terms.

**Known limitation (note it in the report).** MisTIC's internal model still sees cycle-gene transcripts when it estimates p for other genes.
