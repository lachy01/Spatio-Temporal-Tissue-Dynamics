import numpy as np
import pytest

from tissue_dynamics.death import score_death


def test_score_death_scores_dying_subset_higher(counts_adata):
    score_death(counts_adata)

    dying_idx = counts_adata.uns["dying_idx"]
    other_idx = np.setdiff1d(np.arange(counts_adata.n_obs), dying_idx)
    scores = counts_adata.obs["death_score"].to_numpy()

    assert scores[dying_idx].mean() > scores[other_idx].mean()
    assert counts_adata.obs["is_dying"].to_numpy()[dying_idx].mean() > 0.5


def test_score_death_threshold_method(counts_adata):
    score_death(counts_adata, method="threshold", threshold=0.0)
    assert counts_adata.obs["is_dying"].dtype == bool


def test_score_death_raises_on_too_few_marker_genes(counts_adata):
    with pytest.raises(ValueError):
        score_death(counts_adata, marker_genes=["NOT_A_REAL_GENE"])


def test_score_death_and_score_proliferation_are_independent_columns(counts_adata):
    from tissue_dynamics.cellcycle import score_proliferation

    score_proliferation(counts_adata)
    score_death(counts_adata)
    assert "proliferation_score" in counts_adata.obs
    assert "death_score" in counts_adata.obs
    assert "is_dividing" in counts_adata.obs
    assert "is_dying" in counts_adata.obs
