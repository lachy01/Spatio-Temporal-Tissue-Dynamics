# tissue-dynamics

Infer per-cell (or per-bin) probability of division, death, or remaining
unchanged from spatial transcriptomics data, using externally-trained
Phi-Space cell-state embeddings and each unit's spatial neighborhood. Reads
both single-cell (Xenium) and cell-free, spatially-binned (Visium / Visium
HD) data — see "Reading spatial data" below; everything past IO treats a
Visium spot exactly like a Xenium cell.

This is the **static** (single-timepoint) pipeline: read data → attach
Phi-Space embeddings → build the spatial neighborhood graph → score
division/death gene signatures → fit a multinomial fate model. A later phase
will add time-stepping simulation on top (`simulation.py` is a documented
stub for now, not implemented).

## Install

From the repo root:

```
pip install -e ".[xenium-io,parquet,download,test]"
```

## Pipeline overview

```python
from tissue_dynamics import read_xenium, run_static_pipeline

adata = read_xenium("data/<sample>/outs")
adata, model, history = run_static_pipeline(adata, "phi_space.csv", radius=80.0)

adata.obsm["fate_proba"]   # (n_cells, 3) P(divide, die, unchanged) per cell
```

`run_static_pipeline` wires together every step below; each is also usable
standalone.

### 0. Download a sample (optional)

```python
from tissue_dynamics import download_xenium_sample

outs_dir = download_xenium_sample("GSM7990532", dest_dir="data")
```

A full Xenium GEO deposit is typically 4-5GB, almost all of it morphology
images, per-molecule transcripts, segmentation boundaries, and 10x's own
analysis/report outputs that `read_xenium` never reads. `download_xenium_sample`
fetches only the handful of files it actually needs (the cell x gene matrix,
cell centroid table, and run/panel metadata — usually well under 100MB),
downloading each individually where GEO lists it that way, and otherwise
streaming whatever combined archive GEO does provide and extracting just the
matching members without ever writing the huge unwanted ones to disk. Pass
`include_mtx_fallback=True` if you don't have `spatialdata-io` installed, or
`include_transcripts=True` if you want `transcripts.csv.gz` for the manual
per-transcript QC path in `scripts/xenium_gse250346_loader.py`.

### 1. Read spatial data

`read_xenium(path)` reads a 10x Xenium `outs/` folder into an AnnData with
cell x gene counts, `obsm["spatial"]` centroids, `obs["cell_area"]`,
`obs["cell_id"]` (raw per-section barcode), and `obs["sample_id"]` (section
identity — pass `sample_id=` explicitly, or it defaults to the sample root
folder name). `load_xenium_samples(paths)` reads and concatenates several sections
into one AnnData, keeping `obs_names` unique across sections.

`read_visium(path)` reads a Visium or Visium HD sample into the same
contract — bins x genes counts, `obsm["spatial"]` in real micrometers,
`obs["cell_id"]` (the spot/bin barcode), `obs["sample_id"]`. This is the
"cell-free" path: each spatial bin gets an id and is treated exactly like a
Xenium cell by everything downstream (`neighbors`/`targets`/`models` never
know or care whether a row is a segmented cell or a spot). Accepts either a
raw Space Ranger output folder (classic Visium, or Visium HD's
`binned_outputs/square_XXXum/` — point at the specific resolution you want)
or an already-assembled `.h5ad` following the scanpy/squidpy spatial
convention (e.g. exported from CELLxGENE). `load_visium_samples(paths)`
mirrors `load_xenium_samples`.

```python
from tissue_dynamics import read_visium

adata = read_visium("GSM.../outs")            # raw Space Ranger folder, or
adata = read_visium("cellxgene_sample.h5ad")  # a pre-assembled spatial AnnData
```

Two Visium-specific details worth knowing:

- **Pixel-to-micron conversion.** Visium spot/bin coordinates are recorded
  in image-pixel space, not micrometers, so `read_visium` converts them
  using a `microns_per_pixel` factor. The obvious approach —
  `55um / scalefactors_json["spot_diameter_fullres"]`, since standard Visium
  spots are 55µm — turned out to be wrong by ~18% when checked against a
  real sample (GSE176078): `spot_diameter_fullres` doesn't sit in the
  expected ratio to the hardware-fixed 100µm center-to-center hex spacing.
  So `read_visium` self-calibrates instead, measuring the median
  nearest-neighbor spot/bin spacing in pixels and dividing the known fixed
  physical spacing (100µm for classic Visium's hex grid; auto-detected from
  a `square_XXXum` path, or pass `spot_spacing_um`, for Visium HD's square
  grid) by that. Pass `microns_per_pixel` directly to skip this entirely.
- **`radius=80.0` is tuned for Xenium, not Visium.** Classic Visium spots
  sit ~100µm apart center-to-center, so an 80µm neighborhood radius finds
  *zero* neighbors for every spot — `build_adjacency` will warn loudly
  ("all cells isolated") if this happens, but you need to pass a larger
  `radius` (e.g. 150–200µm to reach the 6 immediate hex neighbors) for
  Visium data; there's no single default that's correct for both
  resolutions. Visium HD's finer bins may need a radius closer to Xenium's.

Also worth knowing for CELLxGENE-sourced `.h5ad` files specifically:
`var_names` there are typically Ensembl gene ids (with the symbol in
`var["feature_name"]`) — `read_visium` remaps `var_names` to symbols
automatically (preserving the Ensembl id in `var["gene_ids"]`) so gene-set
matching (`score_division`, `score_death`, ...) works the same way it does
for Xenium.

### 2. Attach Phi-Space embeddings

Phi-Space embeddings are trained separately (e.g. in R, on a single-cell
reference) and handed off as a plain CSV/CSV.gz/Parquet table: a `cell_id`
column, phi_0..phi_{k-1} embedding columns, and (for multi-section data) a
sample column.

```python
from tissue_dynamics import read_phi_table, attach_phi

phi = read_phi_table("phi_space.csv", sample_col="sample")
attach_phi(adata, phi, sample_col="sample")   # -> adata.obsm["phi"]
```

Real PhiSpace output is often a wide table of *named* per-reference-class
scores (e.g. `"Cancer Cycling"`, `"CAFs myCAF-like"`, one column per cell
type in the reference atlas) rather than generic `phi_0/phi_1/...` columns —
pass `phi_cols="all"` (or an explicit column list) to use those directly;
the original names are preserved in `phi.attrs["phi_names"]` /
`adata.uns["phi_names"]` for interpretability.

`attach_phi` warns if any joined row is present but all-NaN — in real data
this typically means a near-empty segmentation artifact (~0 transcripts)
that the embedding method couldn't score; filter these out before
`average_neighbors`/`fit_fate_model`, or the NaNs will propagate.

Joins on `(cell_id, sample)` rather than `obs_names` directly, since Xenium
cell barcodes (and Visium spot/bin barcodes) are only unique within one
section.

### 3. Spatial neighborhood graph

```python
from tissue_dynamics import build_adjacency, average_neighbors

build_adjacency(adata, radius=80.0)   # micrometers; obsp["spatial_connectivity"]
average_neighbors(adata)              # obsm["phi_bar"], the neighbor-averaged embedding
```

Neighbors are computed per `sample_id` (never across sections) as a sparse
0/1 adjacency matrix. `phi_bar_i` is the row-normalized (mean, not sum)
neighbor embedding; isolated cells (no neighbors within radius) get
`phi_bar = 0`, flagged in `obs["is_isolated"]`.

### 4. Division/death signature scoring

Ground-truth proxies come from established gene signatures, reusing the
existing proliferation scorer and a new death/apoptosis one:

- **`score_division`** — auto-selects between `score_cell_cycle` (a full
  Tirosh et al. S/G2M/G1 split, used when the panel covers enough of those
  43+54 genes) and `score_proliferation` (`MKI67`, `TOP2A`, `PCNA`, `CCNB2`,
  `CDK1`, `CENPF`, used as the fallback for thinner panels), and unifies
  either behind one continuous `obs["proliferation_score"]` column (in
  `cell_cycle` mode this is `max(S_score, G2M_score)`) plus a boolean
  `obs["is_dividing"]`. This is what `run_static_pipeline` and
  `analyze_sample` both call — pass `division_kwargs={"method": "cell_cycle"}`
  to `run_static_pipeline` to force the phase split instead of auto-detecting.
- **`score_death`** — `CASP3`, `CASP8`, `CASP9`, `BAX`, `BAK1`, `BID`,
  `CYCS`, `APAF1` (intrinsic + extrinsic apoptosis pathways). Like the
  proliferation markers, this is a coarse mRNA-level proxy for a process
  that's substantially regulated post-translationally — check panel coverage
  (a warning fires if it's thin) or supply a custom `marker_genes` list.

Both set a continuous `_score` and a boolean call (`is_dividing`/`is_dying`)
via a 2-component Gaussian mixture (`method="gmm"`) or a fixed `threshold`.

### 5. Fate targets and model fit

```python
from tissue_dynamics import build_fate_targets, SpatialMultinomialFateModel, fit_fate_model, predict_fate_proba

build_fate_targets(adata, mode="hard")   # or mode="soft"
model = SpatialMultinomialFateModel(phi_dim=adata.obsm["phi"].shape[1])
history = fit_fate_model(model, adata.obsm["phi"], adata.obsm["phi_bar"], adata.obs["fate_code"])
proba = predict_fate_proba(model, adata.obsm["phi"], adata.obsm["phi_bar"])
```

`build_fate_targets` supports two target modes:

- `mode="hard"` — discrete divide/die/unchanged labels from the boolean
  signature calls (`obs["fate_code"]`, `obs["fate_label"]`).
- `mode="soft"` — per-sample z-scored, clipped-at-zero signature scores
  normalized into a 3-way target distribution (`obsm["fate_target"]`).

The model predicts each class's logit as
`eta^(k) = b0^(k) + a^(k)·phi_i + c^(k)·phi_bar_i + phi_i^T B^(k) phi_bar_i`
— an intercept, own-state and neighbor-state linear terms, and a bilinear
own-state × neighbor-state interaction `B^(k)`. "unchanged" is the reference
class (logit fixed at 0); `fate_loss` accepts either hard class indices or
soft target distributions, so the same model/training loop handles both
target modes.

The interaction term is controlled by `interaction_rank`:

- **`interaction_rank=2` (default)** — factorizes `B^(k) = U^(k) V^(k)^T`
  (each `D×r`), so the interaction is a sum of `r` "cross-talk modes"
  `Σ_j (u_j·phi)(v_j·phi_bar)`, using `2·D·r` parameters instead of `D²`.
  This is the default because a full `B`'s `D²` outer-product coefficients
  are extremely collinear (see §8): the low-rank form generalizes better on
  held-out data, is far easier to optimize, and its modes are interpretable
  (§9). It is *not* linear in its parameters, so `fate_parameter_stats`
  doesn't apply to it — use `likelihood_ratio_test` / `interaction_modes`.
- **`interaction_rank=None`** — the full dense `B^(k)`. Linear in
  parameters (so `fate_parameter_stats` works) but ill-conditioned.
- **`interaction_rank=0`** — no interaction (own + neighbor state only).
  The model to fit for clean Wald hypothesis testing (§8).

**Technical covariates.** Pass `n_covariates=C` (and a `covariates` matrix to
`fit_fate_model`/`predict_fate_proba`) to add a per-cell control block
`d^(k)·z_i` to each logit — for log library size and local cell density,
which any count-derived fate label is confounded by. `covariates.py` computes
them (`compute_library_size`, `compute_local_density`) and standardizes them
(`build_covariate_matrix`); `run_static_pipeline(..., covariate_keys=(...))`
wires the whole thing. These are confounder *controls*, so they are excluded
from weight decay (estimated unbiased, not shrunk), and `fate_parameter_stats`
reports their coefficients like any other. On the real breast-cancer sample
the log-library-size coefficient is enormous (z ≈ 140) — the fate labels are
strongly depth-driven — and controlling for depth + density shrinks the mean
neighbour-state coefficient by ~45%, i.e. much of the apparent neighbourhood
effect was confounded.

```python
from tissue_dynamics import compute_library_size, compute_local_density

compute_library_size(adata)                 # -> obs["log_total_counts"]
compute_local_density(adata, radius=50.0)   # -> obs["local_density"]
adata, model, history = run_static_pipeline(
    adata, phi_path, model_kwargs={"interaction_rank": 0},
    covariate_keys=("log_total_counts", "local_density"),
)
```

**Ambient background covariate.** `attach_control_counts` pulls the per-cell
negative-control counts from `cells.csv.gz` (Xenium's `cell_feature_matrix.h5`
may carry only Gene Expression), and `compute_ambient_background` forms the local
ambient fraction `pi_tilde_i = sum_N u_j / sum_N (m_j + u_j)` over a 500 µm
neighbourhood (control counts `u` over total counts) — a proxy for background
contamination — adding `obs["log_ambient_pi"]`. The 500 µm sum is done by binning
to a grid and disk-convolving (an exact radius graph there would have billions of
edges). On the breast data the negative-control *probes alone* are too sparse
(816 counts slide-wide) so the broader negative-control set is summed; the field
is strongly spatial (Moran's I ≈ 0.9) but, entered as a covariate, is **not
significant** for the division rate (z ≈ 0.8) and doesn't reduce the residual
spatial autocorrelation — the library-size offset already normalizes away the
proportional ambient inflation.

```python
from tissue_dynamics import attach_control_counts, compute_ambient_background
attach_control_counts(adata, sample_path,
                      control_cols=("control_probe_counts", "genomic_control_counts",
                                    "control_codeword_counts"))
compute_ambient_background(adata, radius=500.0)   # -> obs["log_ambient_pi"]
```

### 6. Cross-validated evaluation

```python
from tissue_dynamics import spatial_group_kfold, fate_metrics

for train_idx, test_idx in spatial_group_kfold(adata):
    ...  # fit on train_idx, predict on test_idx, score with fate_metrics
```

Always split by whole section (`sample_id`), never per-cell — a per-cell
random split would leak information through the neighbor-averaged `phi_bar`
term.

### 7. Spatial visualization

```python
from tissue_dynamics import plot_fate_spatial

fig, axes = plot_fate_spatial(adata, sample_id="section1")  # only needed if >1 section
```

Plots a 2 x 3 grid in physical (`obsm["spatial"]`) coordinates: top row is
the ground-truth signature call (`obs["fate_label"]`), bottom row the
model's predicted class (argmax of `obsm["fate_proba"]`), one column per
class. Each panel highlights only that column's class against all other
cells in gray — fate classes are typically dominated by "unchanged", so a
single blended categorical plot would bury the minority classes' spatial
pattern. Panel titles report each class's cell count; the suptitle reports
overall accuracy — but check the per-class panels too, since a single
accuracy number can look fine while one class (typically the rarest) is
almost never predicted; that failure is invisible in the aggregate number
but obvious once plotted spatially.

`plot_training_loss(history)` plots the per-epoch loss `fit_fate_model`
returns; pass a `{label: history}` dict to overlay runs (e.g. unweighted vs
balanced — though note their losses are on different scales when
`class_weight` differs, so read convergence shape, not absolute height).

### 8. Parameter inference / hypothesis testing

`SpatialMultinomialFateModel` is a multinomial logit, so its coefficients
come with the standard maximum-likelihood inference machinery —
`fate_parameter_stats` returns each coefficient's estimate, variance,
standard error, Wald z-statistic and p-value (validated to match
statsmodels' `Logit`/`MNLogit` to machine precision), and
`plot_coefficients` draws them as a forest plot with 95% confidence
intervals.

`fate_parameter_stats` only applies to models that are linear in their
parameters — `interaction_rank=0` (none) or `None` (full B) — and raises on
the low-rank default. Since the default is now low-rank, fit a dedicated
model for inference:

```python
from tissue_dynamics import fate_parameter_stats, plot_coefficients

# Inference is only valid on the UNPENALIZED, UNWEIGHTED MLE, and the full
# bilinear model's information matrix is numerically singular — so fit a
# no-interaction model with weight_decay=0, class_weight=None:
model = SpatialMultinomialFateModel(phi_dim=k, interaction_rank=0)
fit_fate_model(model, adata.obsm["phi"], adata.obsm["phi_bar"],
               adata.obs["fate_code"].to_numpy(), weight_decay=0.0)
stats = fate_parameter_stats(model, adata.obsm["phi"], adata.obsm["phi_bar"],
                              feature_names=adata.uns["phi_names"])
plot_coefficients(stats, top_n=20)
```

Two things the function warns about but can't fix for you: (1) weight decay
and class weighting both invalidate the Wald test, so fit a dedicated
`weight_decay=0, class_weight=None` model for inference even if you predict
with a penalized/weighted one; (2) the full `B` block's D*D collinear
outer-product terms make its information matrix near-singular (condition
number ~1e24) — its SEs are meaningless, so use `interaction_rank=0` for
inference. With n in the hundreds of thousands, essentially everything is
"significant", so rank by effect size / |z|, not by p-value.

To test whether an interaction term (low-rank or full) improves fit *as a
whole* — the right question when per-coefficient tests don't apply —
`likelihood_ratio_test` compares two nested fitted models:

```python
from tissue_dynamics import likelihood_ratio_test

null = SpatialMultinomialFateModel(phi_dim=k, interaction_rank=0)   # fit each
alt  = SpatialMultinomialFateModel(phi_dim=k, interaction_rank=2)   # (weight_decay=0)
res = likelihood_ratio_test(null, alt, phi, phi_bar, y)   # -> lr_statistic, df, p_value
```

(The chi-square p-value is exact for none-vs-full; for a low-rank
alternative the null sits on a manifold boundary, so treat it as indicative
and prefer a held-out log-loss comparison as the robust check.)

### 9. Interaction modes

`interaction_modes` SVDs each class's interaction matrix `B^(k)` (or `U V^T`
for a low-rank model) into ranked rank-1 "cross-talk modes"
`σ_j · p_j q_j^T`, and `plot_interaction_modes` draws each mode's own-state
loadings `p_j` and neighbor-state loadings `q_j` side by side. Unlike the
raw (rotation-ambiguous) `U`/`V` factors, the SVD modes are identified and
ordered by strength — so this is how to read *which own cell-state, next to
which neighbor cell-state, shifts fate*.

```python
from tissue_dynamics import interaction_modes, plot_interaction_modes

modes = interaction_modes(model, feature_names=adata.uns["phi_names"], n_modes=2)
plot_interaction_modes(modes, class_name="divide", n_modes=2, top_features=12)
```

### 10. MLP baseline

`fit_mlp_baseline` is an off-the-shelf `MLPClassifier` (standardized inputs)
to check whether the structured model's own-state × neighbor-state inductive
bias buys anything over throwing the same features at a generic net. Its
default input is `phi_bar` alone — the neighbors' averaged Phi-Space
phenotyping — so it asks "how well does a cell's spatial neighborhood
predict its fate?"; pass `input_features="both"` for own+neighbor, or
`"own"` for own-only.

```python
from tissue_dynamics import fit_mlp_baseline, predict_mlp_proba

mlp = fit_mlp_baseline(phi_train, phi_bar_train, y_train, input_features="both")
proba = predict_mlp_proba(mlp, phi_test, phi_bar_test, n_classes=3)
```

Note `MLPClassifier` has no class-weighting, so on imbalanced fate labels it
collapses the minority classes (the same failure the unweighted structured
model shows) — resample the training split if you need it to predict them.

### 11. Tile-level sanity checks and the MKI67 count model

Before trusting any per-cell fate model, two checks sidestep the noisy
hard-label thresholding (`tiles.py`, `count_glm.py`; the count model needs
the optional `statsmodels` dependency). A worked walkthrough on the real data
is in [`notebooks/exploration.ipynb`](../../notebooks/exploration.ipynb).

**Tile aggregation.** `assign_tiles` grids the slide (default 100 µm) and
`aggregate_tiles` collapses cells to one row per tile — per-tile mean `phi`,
fate fractions, summed gene counts + library size. The tile count is roughly
the *effective* sample size (adjacent cells share almost their whole
neighborhood, so it's slide-area / πr² — order 10⁴ tiles, not the cell
count). `binned_conditional_mean` + `plot_binned_conditional_mean` draw the
key OSDR figure: observed division fraction vs a binned neighborhood axis — a
monotone, wide-spanning curve is the neighborhood signal, visible even when
per-cell accuracy is at the majority baseline. `plot_tile_field` maps any
tile quantity in space.

```python
from tissue_dynamics import assign_tiles, aggregate_tiles, binned_conditional_mean

assign_tiles(adata, tile_size=100.0)
tiles = aggregate_tiles(adata, obs_means=["is_dividing"], gene_counts=["MKI67"])
```

**MKI67 count model.** Rather than hard-labelling a near-null cell-cycle
score, `fit_neighbourhood_count_glm` regresses a direct marker's counts on
the neighborhood as a negative-binomial GLM with `log(total_counts)` offset
(so coefficients describe detection *rate*), and `compare_count_glms` fits the
nested none/own/neighbor/own+neighbor ladder so you can read off whether the
neighborhood adds anything beyond own state. Judge by McFadden R² effect size,
not the p-value — at these n everything is "significant".

```python
from tissue_dynamics import compare_count_glms
compare_count_glms(adata, gene="MKI67", n_pcs=6)   # McFadden R² per predictor spec
```

**Division-count NB model (`fit_division_nb`).** A self-contained alternative
to the multinomial fate model that **drops death entirely** and models division
as a count. The response is `get_signature_counts` over a cell-cycle gene set
(default: the Tirosh S+G2M genes present in the panel — the same genes
`score_cell_cycle` uses), which is strongly over-dispersed, so it's fit as
negative binomial. Predictors are the named own- and neighbour-state embeddings;
covariates are the **same** `log_total_counts` + `local_density` as the fate
model but entered as *free* standardized coefficients rather than a fixed
`log(total_counts)` offset (so the depth term is estimated, ~1.1 on real data,
not forced to 1.0). It reports the NB dispersion, the coefficient table (log
rate-ratios), and a likelihood-ratio test of own+neighbour state against a
covariate-only null at the same dispersion — isolating what cell state adds for
division *after* depth and density are controlled for. On the breast-cancer
data this corroborates the fate model: own T-cell/Cancer-Epithelial/B-cell
score raises the division rate, CAFs lower it; being *surrounded by* Cancer
Epithelial raises it (z≈22) and by Normal Epithelial lowers it, and the
neighbourhood adds a small-but-highly-significant increment beyond own state.
Note these compositional celltype-major scores sum to a constant, so drop one
cell type as reference (see the fate-model note above) before fitting.
`interaction_rank` toggles the same bilinear `phi^T B phi_bar` term as the fate
model: `0` (default) none, `None` the full dense `B` (all own×neighbour outer
products as GLM columns — linear in the parameters, so valid LR inference, but a
collinear block best judged by its block LR test), and a positive int a low-rank
`B = U Vᵀ` (bilinear, so not fittable as a plain GLM — accepted by the API but
raises `NotImplementedError` for now; use the torch model for low rank). On the
breast data the full interaction is highly significant by block LR but adds only
a small increment (pseudo-R² 0.018 → 0.021).

The nuisance terms can each enter in the form that fits their role, so the
default free-covariate treatment isn't the only option: `offset_key` puts a
column in as a fixed **offset** (library size → a transcript-*rate* model),
`spline_keys` expands a column as a smooth **natural-cubic spline** (density,
whose effect needn't be linear), and `covariate_keys` keeps the rest as free
linear terms (e.g. cell area). The recommended rate model is
`offset_key="log_total_counts", spline_keys=("local_density",),
covariate_keys=("cell_area",)`. Each fit also attaches per-cell `fitted_mu`,
`prob_division` (= P(≥1 cell-cycle transcript) under the fitted NB), and
`pearson_resid` for downstream spatial diagnostics.

```python
from tissue_dynamics import fit_division_nb
fit = fit_division_nb(adata, predictors="own+neighbor",
                      covariate_keys=("log_total_counts", "local_density"))
fit["coef"]                    # log-rate-ratio table (own[...], nbr[...], covariates)
fit["lr_vs_covariate_only"]    # does own+neighbour state beat covariates alone?

full = fit_division_nb(adata, predictors="own+neighbor", interaction_rank=None)  # full dense B
rate = fit_division_nb(adata, predictors="own+neighbor",   # recommended division spec
                       offset_key=None, spline_keys=("local_density",),
                       covariate_keys=("log_total_counts", "cell_area"))
rate_B = fit_division_nb(adata, predictors="own+neighbor", interaction_rank=None,  # + interaction
                         offset_key=None, spline_keys=("local_density",),
                         covariate_keys=("log_total_counts", "cell_area"))
```

**Depth: free covariate for division, offset for death.** Library size enters
either as a fixed `log(total_counts)` **offset** (coefficient pinned at 1, the
"rate per transcript" reading) or as a **free standardized covariate**, whose
implied exponent is `beta / sd(log T)`. Notebook section 21 tests both on the
breast sample and the two responses come out opposite:

| model | implied `d log mu / d log T` | verdict |
|---|---:|---|
| division | 0.70 | free it — better log-likelihood (+880), PIT (0.049 -> 0.040) and rootogram (RMS 7.7 -> 6.3) |
| death | 0.93 | keep the offset — the likelihood gain is trivial at this n and the rootogram *degrades* (RMS 1.5 -> 1.8) |

For division the offset is not a neutral choice: it also inflates the fitted
dispersion (0.70 -> 0.77) by absorbing the depth mis-specification, so anything
that models the dispersion should be fit on the free-depth spec. Sections 9-20 of
the notebook predate this test and still use the offset, so their absolute fit
numbers are not directly comparable with sections 21 and 24.

**State-dependent dispersion (`dispersion_predictors`).** By default one NB
dispersion is shared by every cell, which is a strong assumption when the
populations differ as much as they do here. Setting `dispersion_predictors` frees
it, so each cell carries its own:

    log mu_i     = offset_i + X_i . beta          (mean, unchanged)
    log alpha_i  = gamma_0 + phi_i . gamma        (dispersion)

The **log link** keeps `alpha_i` positive — an unconstrained linear predictor on
the dispersion wanders negative and makes the likelihood undefined — so
`exp(gamma_j)` reads as the multiplicative effect of state dimension `j` on the
dispersion. This is no longer a GLM, so it is fit by block coordinate ascent on
the joint likelihood (a beta step by IRLS at the current `alpha_i`, a gamma step
by L-BFGS with an analytic score); both blocks climb the same objective, so the
ascent is monotone and it converges in a handful of outer iterations.

```python
fit = fit_division_nb(adata, predictors="own+neighbor", dispersion_predictors="own",
                      offset_key="log_total_counts", spline_keys=("local_density",),
                      covariate_keys=("cell_area",))
fit["alpha"]                                      # now a per-cell vector
d = fit["dispersion"]
d["coef"]                                         # gamma table (estimate, se, z, exp_estimate)
d["lr_vs_constant_alpha"]                         # against the constant-alpha *MLE*
d["fitted_mu_constant_alpha"], d["alpha_constant_alpha"]   # that baseline, for diagnostics
```

Two cautions. It **forces `fitter="irls"`** (statsmodels cannot take a
per-observation dispersion) and requires `family="nb"`. And the likelihood-ratio
test is deliberately against a constant-alpha **MLE** refit of the same mean
model, not against the default fit: every other model here estimates `alpha` by
the Cameron-Trivedi moment estimator, and on the breast sample that alone accounts
for about 60% of the naive chi-square (0.774 by moments versus 0.687 by MLE), so
comparing against it would badly overstate what the state dependence buys.
`plot_rootogram` and `plot_pit_histogram` both accept the per-cell `alpha` vector
directly. See notebook section 24 for the fitted result and its diagnostics.

Wide designs (the full interaction pushes the rate model to ~87 parameters) are
fit with a memory-efficient normal-equations IRLS instead of statsmodels'
SVD-over-the-tall-design — `fitter="auto"` (default) switches to it past ~30
predictors, so the full-B rate model fits on 680k cells without exhausting
memory (`fit["fitter"]` records which backend ran; the IRLS coefficients are
validated against statsmodels in the tests). On the breast data the interaction
block is highly significant (LR χ²≈3900 on 64 df) and nudges the residual
Moran's I down from 0.083 to 0.071 — it captures a little more of the spatial
clustering, but most of the residual structure is still not explained by cell
state alone.

**Neighbour reparameterisation (`neighbour_contrast=True`).** Represents the
neighbour state as `phi_bar - phi` (the neighbourhood's deviation from the cell)
rather than `phi_bar`, an attempt to decorrelate the own and neighbour blocks.
For the main-effects model it is an *exact* reparameterisation (identical fit;
`nbr_dev` coefficient = raw `nbr` coefficient, own absorbs the rest). On this
data it does **not** decorrelate them, though — `phi_bar` is a neighbourhood
*average* with ~40% the variance of `phi`, so `phi_bar - phi` is dominated by
`-phi` and ends up *more* correlated with own state (cond 136 → 275, |corr|
0.67 → 0.78); the textbook decorrelation assumes equal variances, which
neighbourhood averaging violates. It does moderate the collinear *interaction*
coefficients (mean |int| 3.2 → 1.9) — but there `phi ⊗ (phi_bar - phi)` implies
own×own terms, so it is a different model, not a pure reparam.

**Goodness of fit (`plot_rootogram`, `plot_pit_histogram`).** Two count-model
diagnostics against the fitted NB, taking `(y, fit["fitted_mu"], fit["alpha"])`.
The hanging rootogram compares observed vs model-expected count frequencies on a
√ scale (bars dipping below the zero line are under-predicted counts); the
non-randomized PIT histogram is a calibration check (uniform = calibrated,
U-shape = under-dispersed, hump = over-dispersed). On the breast data the PIT is
nearly uniform (max height deviation ~0.05 — well calibrated), while the
rootogram flags finer shape misfit (an excess of exactly-1 counts, mild
over-prediction across counts 5–9).

**Spatial residual check (`morans_i`).** After fitting, bin the slide into ~100 µm
tiles (`assign_tiles` / `aggregate_tiles`) and compare the observed division
fraction, the model's mean `prob_division`, and the mean `pearson_resid` over
space; `evaluate.morans_i(values, coords=...)` then measures the spatial
autocorrelation left in the residuals (kNN weights, permutation p-value). On the
breast data Moran's I falls from **0.68 on the observed signal to 0.08 on the
residuals** — the cell-state model captures ~88% of the spatial clustering in
proliferation, with a small but significant remainder it can't explain from
own+neighbour state alone.

**Annulus / spillover test (`annulus.py`).** The sharpest identifiability check in
the package: is the neighbourhood effect real, or **segmentation spillover**
(transcripts misassigned from a touching neighbour, so `phi_bar` merely echoes the
cell's own mis-measured state)? Spillover is a *contact* artefact — it can only
act between cells whose boundaries touch — so `annulus_scan` refits the model with
the neighbourhood restricted to successive rings (`DEFAULT_RINGS`: 0–15, 15–30,
30–50, 50–80, 80–120, 120–200 µm) and tracks the neighbour coefficients `c` and
the interaction `B`. In the same pass `annulus_neighbor_stats` accumulates the
exact Phi-Space correlogram `rho_phi(r)` over *every* pair in the ring (~7×10⁸
pairs at the outer ring — chunk-accumulated, never materialized).
`spillover_chi2` then tests the pure-spillover null `c_d(r) = k_d · rho_phi(r)`.
Each ring's `phi_bar` is z-scored before fitting (otherwise wider rings' smaller
`phi_bar` variance inflates coefficients mechanically), and `max_neighbors` runs
the count-matched control that equalizes regression dilution across rings. On the
breast data the null is decisively rejected *in the opposite direction to the
artefact*: `c` **grows** with distance (3.9× by the outer ring) while `rho_phi`
decays (0.44×), and the coefficients are largest at 120–200 µm where transcript
misassignment is physically impossible.

```python
from tissue_dynamics import annulus_scan, spillover_chi2, plot_annulus_profile
coef, summary = annulus_scan(adata, interaction=True,          # one refit per ring
                             offset_key="log_total_counts", spline_keys=("local_density",),
                             covariate_keys=("cell_area",), fitter="irls")
spillover_chi2(coef)                     # c(r) = k*rho_phi(r)?  chi2 / p / per-ring deviations
plot_annulus_profile(summary)            # effect vs autocorrelation, normalized
```

**Oxygen / perfusion field (`oxygen.py`).** Oxygen diffuses over hundreds of
micrometres, well past the 80 um model neighbourhood, so a cell's supply is set by
vasculature it never touches. `build_oxygen_field` solves a steady-state
screened-diffusion PDE on the rasterised section — `lap(u) = kappa u` with Robin
sources at soft-labelled endothelium and no-flux at the tissue edge — and
`attach_oxygen_field` writes `obs["oxygen_u"]` and `obs["oxygen_grad_u"]` ready for
`covariate_keys`. Use `source_xy`/`source_phi_endo` to solve over *all* cells while
attaching to the modelled subset, since vessels outside the subset still supply it.

Two things decide whether the field is any good, and both are testable. Source
calibration: the endothelial score must be soft-thresholded (`c`) hard enough that
the noise floor across hundreds of thousands of non-endothelial cells does not
outweigh the real vessels — on the Xenium breast panel, blood endothelium
(ACKR1+RGS5+CXCL12, excluding lymphatics, which carry no oxygenated blood) at
`c = 0.15` retains 2.6% of cells with a 100% signal share, and the field is
decoupled from cell density (corr 0.12). And the biological gate (spec test 7.9):
HIF-target expression must fall with `u`. **On this dataset 7.9 could not be
evaluated** — CA9 has zero counts in 86% of 100 um tiles and the composite is
depth-confounded (rho = -0.30 with tile depth) — so the field is *not* licensed as
a hypoxia covariate.

It is licensed as a **vessel-proximity / perfusion** covariate on separate
evidence: added to the division rate model it lowers the residual spatial
autocorrelation (Moran's I 0.088 -> 0.082) roughly 14x more than a 180-degree
rotated decoy with identical smoothness (-> 0.087), with beta = +0.046 (z = +15.2)
— better-perfused tissue divides more. Call it perfusion, not hypoxia, until 7.9
passes on a panel that can actually measure hypoxia.

```python
from tissue_dynamics import attach_oxygen_field, fit_division_nb
attach_oxygen_field(adata, phi_endo, source_xy=all_xy, source_phi_endo=all_phi,
                    h=5.0, L0=100.0, beta=1.0, c=0.15)
fit_division_nb(adata, predictors="own+neighbor", offset_key="log_total_counts",
                spline_keys=("local_density",), covariate_keys=("cell_area", "oxygen_u"))
```

**Death model (`fit_death_nb`).** The apoptosis analogue of `fit_division_nb` —
identical architecture (own + neighbour state, interaction, offset, spline,
covariates), just a different response: the summed counts over
`gene_sets.APOPTOSIS_SIGNATURE`, the pro-apoptotic machinery *in the dying cell*
(caspases, pro-apoptotic Bcl-2/BH3-only, apoptosome, death receptors, p53),
excluding death ligands/cytotoxic-effector genes that mark the killer cell. It's
a thin wrapper (all `fit_division_nb` parameters pass through; per-cell death
probability is returned as `prob_death`). On the breast data it fits even better
than the division model (rootogram near-perfect, PIT max-dev ≈ 0.01, residual
Moran's I ≈ 0.02) and the biology is coherent: Cancer-Epithelial cells have a
*lower* death rate (apoptosis evasion) while dividing more, immune cells (T/B)
turn over faster, and being surrounded by myeloid/T-cells raises death
(immune-mediated killing).

```python
from tissue_dynamics import fit_death_nb
death = fit_death_nb(adata, predictors="own+neighbor", interaction_rank=None,
                     offset_key="log_total_counts", spline_keys=("local_density",),
                     covariate_keys=("cell_area",))
death["coef"]        # own[...]/nbr[...] log-rate-ratios for apoptosis
death["prob_death"]  # per-cell P(>=1 apoptosis transcript)
```

**Stepwise selection (`stepwise_division_nb`).** Backward elimination *from* the
full-interaction model (own + neighbour + full dense `B` + covariates) to a
parsimonious one. At this n every term is "significant", so it selects by **BIC**
(penalty `ln(n)` per parameter, ≫ AIC's 2). Each step drops the eligible term
with the smallest Wald |z| and refits, stopping when no single removal lowers
BIC. The technical covariates are protected (never dropped) and **marginality**
is respected (a main effect isn't removed while an interaction on that cell type
survives). The path runs at a fixed NB dispersion via a fast p×p IRLS
(`_nb2_irls`, validated against statsmodels), so the ~40-step search over 680k
cells takes ~2 min rather than hours; the selected model is then refit once with
statsmodels. On the breast data it prunes the full 83-parameter model to ~45 —
keeping every main effect and covariate but discarding ~38 of the 64
interaction terms (the collinear, uninformative own×neighbour pairs).

```python
from tissue_dynamics import stepwise_division_nb
sel = stepwise_division_nb(adata, criterion="bic")
sel["selected_terms"], sel["dropped_terms"]   # what survived / was pruned
sel["path"]                                    # BIC/AIC/loglik at each elimination step
```

## Extending to other spatial technologies

Every stage past IO (`neighbors`, `targets`, `models`) only assumes an
AnnData with `obsm["spatial"]` coordinates (in real physical units) and an
`obs["sample_id"]` grouping — it never assumes a row is a single segmented
cell. `read_visium` is the existence proof: it's a completely separate
module from `read_xenium`, and nothing in `neighbors.py`/`targets.py`/
`models.py`/`pipeline.py` changed to support it. A loader for another
technology just needs to emit an AnnData with that same contract.

## Note on the example panel

The 541-gene lung panel under `data/` covers 6/6 core proliferation markers
but only 1/8 core apoptosis markers — check your own panel's coverage before
trusting either signature score, or supply a custom marker list.
