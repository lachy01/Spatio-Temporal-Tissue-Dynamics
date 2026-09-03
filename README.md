# Spatio-Temporal-Tissue-Dynamics

Infer spatial cell-fate (division / death) dynamics from spatial
transcriptomics (Xenium, Visium / Visium HD) and externally-trained Phi-Space
cell-state embeddings.

The core is an installable Python package, **`tissue_dynamics`**, implementing
the static (single-timepoint) pipeline: read data → attach Phi-Space
embeddings → build the spatial neighbourhood graph → score division/death gene
signatures → fit a multinomial cell-fate model, plus inference, baselines, and
tile-level count diagnostics. Temporal simulation is a documented interface
stub (`simulation.py`) for a later phase.

## Install

```
pip install -e ".[xenium-io,parquet,download,stats,test]"
```

## Getting started

- **Package documentation:** [`src/tissue_dynamics/README.md`](src/tissue_dynamics/README.md)
  — the full API walkthrough, stage by stage.
- **Worked example:** [`notebooks/exploration.ipynb`](notebooks/exploration.ipynb)
  — tile-level sanity checks and the MKI67 count model on a real Xenium
  breast-cancer sample, executed with outputs.

```python
from tissue_dynamics import read_xenium, run_static_pipeline

adata = read_xenium("data/<sample>/outs")
adata, model, history = run_static_pipeline(adata, "phi_space.csv", radius=80.0)
adata.obsm["fate_proba"]   # (n_cells, 3) P(divide, die, unchanged) per cell
```

## Layout

| Path | Contents |
|------|----------|
| `src/tissue_dynamics/` | the package (IO, neighbours, scoring, model, inference, plotting, tiles) |
| `tests/` | pytest suite (synthetic fixtures, no real data needed) |
| `notebooks/` | exploratory analyses |
| `scripts/` | standalone helper scripts (GEO Xenium loader, Phi-Space reference builder) |
| `visuals/` | generated figures |
| `data/` | input samples (git-ignored) |
