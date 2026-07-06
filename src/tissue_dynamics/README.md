# tissue-dynamics

Infer per-cell probability of division, death, or remaining unchanged from
spatial transcriptomics data, using externally-trained Phi-Space cell-state
embeddings and each cell's spatial neighborhood. Built on Xenium today;
designed to extend to other spatial technologies (e.g. supercellular Visium
bins) later — see "Extending to other spatial technologies" below.

This is the **static** (single-timepoint) pipeline: read data → attach
Phi-Space embeddings → build the spatial neighborhood graph → score
division/death gene signatures → fit a multinomial fate model. A later phase
will add time-stepping simulation on top (`simulation.py` is a documented
stub for now, not implemented).

## Install

From the repo root:

```
pip install -e ".[xenium-io,parquet,test]"
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

### 1. Read spatial data

`read_xenium(path)` reads a 10x Xenium `outs/` folder into an AnnData with
cell x gene counts, `obsm["spatial"]` centroids, `obs["cell_area"]`,
`obs["cell_id"]` (raw per-section barcode), and `obs["sample_id"]` (section
identity — pass `sample_id=` explicitly, or it defaults to the sample root
folder name). `load_samples(paths)` reads and concatenates several sections
into one AnnData, keeping `obs_names` unique across sections.

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
cell barcodes are only unique within one section.

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

- **`score_proliferation`** — `MKI67`, `TOP2A`, `PCNA`, `CCNB2`, `CDK1`,
  `CENPF` (or `score_cell_cycle` for a full Tirosh et al. S/G2M/G1 split, if
  the panel covers enough of those 43+54 genes).
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
AnnData with `obsm["spatial"]` coordinates and an `obs["sample_id"]` grouping
— it never assumes a row is a single segmented cell. A future loader for
supercellular/binned data (e.g. Visium) just needs to emit an AnnData with
that same contract; nothing downstream needs to change.

## Note on the example panel

The 541-gene lung panel under `data/` covers 6/6 core proliferation markers
but only 1/8 core apoptosis markers — check your own panel's coverage before
trusting either signature score, or supply a custom marker list.
