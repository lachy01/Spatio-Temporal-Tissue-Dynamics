import numpy as np
import pytest

from tissue_dynamics.cellcycle import score_proliferation


def test_score_proliferation_scores_dividing_subset_higher(counts_adata):
    score_proliferation(counts_adata)

    dividing_idx = counts_adata.uns["dividing_idx"]
    other_idx = np.setdiff1d(np.arange(counts_adata.n_obs), dividing_idx)
    scores = counts_adata.obs["proliferation_score"].to_numpy()

    assert scores[dividing_idx].mean() > scores[other_idx].mean()
    assert counts_adata.obs["is_dividing"].to_numpy()[dividing_idx].mean() > 0.5


def test_score_proliferation_threshold_method(counts_adata):
    score_proliferation(counts_adata, method="threshold", threshold=0.0)
    assert counts_adata.obs["is_dividing"].dtype == bool


def test_score_proliferation_raises_on_too_few_marker_genes(counts_adata):
    with pytest.raises(ValueError):
        score_proliferation(counts_adata, marker_genes=["NOT_A_REAL_GENE"])
