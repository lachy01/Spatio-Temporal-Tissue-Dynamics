import numpy as np
import pytest

from tissue_dynamics.cellcycle import score_cell_cycle, score_division, score_proliferation


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


def test_score_division_auto_falls_back_to_proliferation_on_thin_coverage(counts_adata):
    # counts_adata's panel has no Tirosh S/G2M genes, so "auto" must fall back.
    with pytest.warns(UserWarning, match="falling back to a single proliferation score"):
        score_division(counts_adata)
    assert "proliferation_score" in counts_adata.obs
    assert "phase" not in counts_adata.obs


def test_score_division_auto_uses_cell_cycle_when_panel_covers_it(cell_cycle_adata):
    # cell_cycle_adata has 5 S-phase + 5 G2M-phase Tirosh genes -> no fallback warning.
    score_division(cell_cycle_adata)

    assert "phase" in cell_cycle_adata.obs
    assert "S_score" in cell_cycle_adata.obs
    assert "G2M_score" in cell_cycle_adata.obs
    # the unified score must equal max(S_score, G2M_score) per cell
    expected = cell_cycle_adata.obs[["S_score", "G2M_score"]].max(axis=1).to_numpy()
    np.testing.assert_allclose(cell_cycle_adata.obs["proliferation_score"].to_numpy(), expected)


def test_score_division_can_be_forced_to_proliferation_even_with_good_coverage(cell_cycle_adata):
    score_division(cell_cycle_adata, method="proliferation")
    assert "phase" not in cell_cycle_adata.obs
    assert "proliferation_score" in cell_cycle_adata.obs


def test_score_division_forced_cell_cycle_matches_score_cell_cycle(cell_cycle_adata):
    reference = cell_cycle_adata.copy()
    score_cell_cycle(reference)

    score_division(cell_cycle_adata, method="cell_cycle")

    np.testing.assert_allclose(
        cell_cycle_adata.obs["S_score"].to_numpy(), reference.obs["S_score"].to_numpy(),
    )
    np.testing.assert_allclose(
        cell_cycle_adata.obs["proliferation_score"].to_numpy(),
        reference.obs[["S_score", "G2M_score"]].max(axis=1).to_numpy(),
    )


def test_score_division_unknown_method_raises(counts_adata):
    with pytest.raises(ValueError):
        score_division(counts_adata, method="nonsense")
