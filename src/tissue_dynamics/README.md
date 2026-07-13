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
per-transcript QC path in `xenium_gse250346_loader.py`.

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
— an intercept, own-state and neighbor-state linear terms, and a learned
bilinear own-state × neighbor-state interaction `B^(k)`. "unchanged" is the
reference class (logit fixed at 0); `fate_loss` accepts either hard class
indices or soft target distributions, so the same model/training loop
handles both target modes.

### 6. Cross-validated evaluation

```python
from tissue_dynamics import spatial_group_kfold, fate_metrics

for train_idx, test_idx in spatial_group_kfold(adata):
    ...  # fit on train_idx, predict on test_idx, score with fate_metrics
```

Always split by whole section (`sample_id`), never per-cell — a per-cell
random split would leak information through the neighbor-averaged `phi_bar`
term.

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
