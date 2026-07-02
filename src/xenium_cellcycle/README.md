# xenium-cellcycle

Read a 10x Xenium spatial transcriptomics sample and score cells for
cell-cycle / proliferation activity.

## Install

From the repo root:

```
pip install -e .
```

## Usage

```python
from xenium_cellcycle import analyze_sample

adata = analyze_sample("data/<sample>/outs")
adata.obs["is_dividing"]        # bool per cell
adata.obs["proliferation_score"]  # or S_score/G2M_score/phase, see below
```

`analyze_sample` reads the sample, then picks a scoring strategy automatically:

- **`score_cell_cycle`** — classic Tirosh et al. S/G2M/G1 phase assignment
  (`sc.tl.score_genes_cell_cycle`). Used when the gene panel covers enough
  of the canonical 43 S-phase / 54 G2M-phase genes to trust the phase split.
- **`score_proliferation`** — a single signature score over a small set of
  well-established proliferation markers (`MKI67`, `TOP2A`, `PCNA`,
  `CCNB2`, `CDK1`, `CENPF`), with dividing/non-dividing calls from a
  2-component Gaussian mixture fit to the score. Used as a fallback for
  targeted panels (most Xenium panels) that don't resolve enough
  canonical cell-cycle genes for a reliable S/G2M split.

Both are also callable directly on an `AnnData` you've already loaded
(e.g. via `read_xenium`), and both accept custom gene lists.

**Note on this repo's example panel:** the 541-gene lung panel under
`data/` only includes 1 of the 43 canonical S-phase genes and 5 of the 54
canonical G2M genes, so `analyze_sample` falls back to
`score_proliferation` for it (with a warning). This is typical for
targeted Xenium panels — check your own panel's coverage before trusting a
full S/G2M phase split.
