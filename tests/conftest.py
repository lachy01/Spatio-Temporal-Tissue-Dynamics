"""Shared synthetic fixtures for the tissue_dynamics test suite (no real data required)."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.gene_sets import (
    CORE_APOPTOSIS_MARKERS,
    CORE_PROLIFERATION_MARKERS,
    TIROSH_G2M_GENES,
    TIROSH_S_GENES,
)


@pytest.fixture
def small_adata():
    """8 cells across 2 sections with hand-computable adjacency at radius=80.

    Each section has cells at x = 0, 30, 60, 200 (y=0): the first three are
    mutually within 80um of each other, the fourth is isolated. `cell_id` is
    deliberately duplicated *across* sections (as real Xenium barcodes would
    be) to exercise the (cell_id, sample) composite join in `attach_phi`.
    """
    coords = np.array([[0, 0], [30, 0], [60, 0], [200, 0]] * 2, dtype=float)
    sample_id = ["A"] * 4 + ["B"] * 4
    cell_id = [f"c{i}" for i in range(4)] * 2
    obs_names = [f"{s}_{c}" for s, c in zip(sample_id, cell_id)]

    adata = AnnData(
        X=np.zeros((8, 3), dtype=float),
        obs=pd.DataFrame({"cell_id": cell_id, "sample_id": sample_id}, index=obs_names),
        var=pd.DataFrame(index=["GENE1", "GENE2", "GENE3"]),
    )
    adata.obsm["spatial"] = coords
    return adata


@pytest.fixture
def phi_frame(small_adata):
    """A raw Phi-Space table matching `small_adata`, row order shuffled.

    `phi_0` is set to each row's position in `small_adata` (0..n-1), so a
    correct (cell_id, sample) composite join recovers `phi_0 == arange(n)`
    regardless of table row order or the duplicated cell_id across sections.
    """
    n = small_adata.n_obs
    df = pd.DataFrame({
        "cell_id": small_adata.obs["cell_id"].to_numpy(),
        "sample": small_adata.obs["sample_id"].to_numpy(),
        "phi_0": np.arange(n, dtype=float),
        "phi_1": np.arange(n, dtype=float) * 10,
        "phi_2": np.arange(n, dtype=float) * 100,
    })
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def _make_counts_adata(n_cells, dividing_frac=0.2, dying_frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    marker_genes = list(CORE_PROLIFERATION_MARKERS) + list(CORE_APOPTOSIS_MARKERS)
    other_genes = [f"GENE{i}" for i in range(50)]
    genes = marker_genes + other_genes

    counts = rng.poisson(2, size=(n_cells, len(genes))).astype(float)

    n_dividing = int(n_cells * dividing_frac)
    n_dying = int(n_cells * dying_frac)
    dividing_idx = np.arange(0, n_dividing)
    dying_idx = np.arange(n_dividing, n_dividing + n_dying)

    prolif_cols = [genes.index(g) for g in CORE_PROLIFERATION_MARKERS]
    death_cols = [genes.index(g) for g in CORE_APOPTOSIS_MARKERS]
    counts[np.ix_(dividing_idx, prolif_cols)] += rng.poisson(30, size=(n_dividing, len(prolif_cols)))
    counts[np.ix_(dying_idx, death_cols)] += rng.poisson(30, size=(n_dying, len(death_cols)))

    obs_names = [f"cell{i}" for i in range(n_cells)]
    adata = AnnData(
        X=counts,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.uns["dividing_idx"] = dividing_idx
    adata.uns["dying_idx"] = dying_idx
    return adata


@pytest.fixture
def counts_adata():
    """60 cells x panel (proliferation + apoptosis markers + filler genes).

    Cells 0..11 have boosted proliferation-marker counts ("dividing"), cells
    12..23 have boosted apoptosis-marker counts ("dying"); the rest are
    baseline. `uns["dividing_idx"]`/`uns["dying_idx"]` record which.
    """
    return _make_counts_adata(n_cells=60)


@pytest.fixture
def cell_cycle_adata():
    """60 cells x panel with 5 Tirosh S-phase + 5 Tirosh G2M-phase genes present
    (well above MIN_GENES_PER_PHASE=3), so `score_division(method="auto")`
    resolves to the "cell_cycle" branch instead of falling back to
    "proliferation". Cells 0..11 have boosted S-phase-gene counts, cells
    12..23 have boosted G2M-phase-gene counts; the rest are baseline.
    """
    rng = np.random.default_rng(2)
    s_genes = list(TIROSH_S_GENES[:5])
    g2m_genes = list(TIROSH_G2M_GENES[:5])
    other_genes = [f"GENE{i}" for i in range(50)]
    genes = s_genes + g2m_genes + other_genes

    n_cells = 60
    counts = rng.poisson(2, size=(n_cells, len(genes))).astype(float)

    s_idx = np.arange(0, 12)
    g2m_idx = np.arange(12, 24)
    s_cols = [genes.index(g) for g in s_genes]
    g2m_cols = [genes.index(g) for g in g2m_genes]
    counts[np.ix_(s_idx, s_cols)] += rng.poisson(30, size=(len(s_idx), len(s_cols)))
    counts[np.ix_(g2m_idx, g2m_cols)] += rng.poisson(30, size=(len(g2m_idx), len(g2m_cols)))

    obs_names = [f"cell{i}" for i in range(n_cells)]
    adata = AnnData(
        X=counts,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.uns["s_idx"] = s_idx
    adata.uns["g2m_idx"] = g2m_idx
    return adata


@pytest.fixture
def pipeline_adata():
    """40 cells x panel, with spatial coordinates and two sections — enough
    structure to run the full static pipeline end to end."""
    adata = _make_counts_adata(n_cells=40, dividing_frac=0.3, dying_frac=0.3, seed=42)
    rng = np.random.default_rng(42)
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = ["S1"] * 20 + ["S2"] * 20
    adata.obsm["spatial"] = rng.uniform(0, 300, size=(adata.n_obs, 2))
    return adata
