import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.targets import build_fate_targets


@pytest.fixture
def toy_adata():
    obs = pd.DataFrame({
        "sample_id": ["A"] * 6,
        "proliferation_score": [5.0, 0.1, 0.0, 0.0, 0.2, 0.0],
        "death_score": [0.0, 0.0, 5.0, 0.1, 0.0, 0.0],
        "is_dividing": [True, False, False, False, False, False],
        "is_dying": [False, False, True, False, False, False],
    }, index=[f"cell{i}" for i in range(6)])
    return AnnData(X=np.zeros((6, 1)), obs=obs)


def test_hard_mode_labels_from_boolean_calls(toy_adata):
    build_fate_targets(toy_adata, mode="hard")
    assert toy_adata.obs["fate_label"].tolist() == [
        "divide", "unchanged", "die", "unchanged", "unchanged", "unchanged",
    ]
    assert toy_adata.obs["fate_code"].tolist() == [0, 2, 1, 2, 2, 2]


def test_hard_mode_conflict_resolved_by_stronger_zscored_signal(toy_adata):
    toy_adata.obs.loc[toy_adata.obs_names[0], "is_dying"] = True  # now flagged both
    build_fate_targets(toy_adata, mode="hard")
    # cell 0's proliferation_score (5.0) is far more extreme than its death_score (0.0)
    assert toy_adata.obs["fate_label"].iloc[0] == "divide"


def test_soft_mode_targets_are_a_valid_distribution(toy_adata):
    build_fate_targets(toy_adata, mode="soft")
    target = toy_adata.obsm["fate_target"]

    assert target.shape == (6, 3)
    np.testing.assert_allclose(target.sum(axis=1), 1.0)
    assert (target >= 0).all()
    assert toy_adata.uns["fate_target_class_names"] == ["divide", "die", "unchanged"]


def test_soft_mode_dividing_cell_has_higher_divide_weight(toy_adata):
    build_fate_targets(toy_adata, mode="soft")
    target = toy_adata.obsm["fate_target"]
    # cell 0 has the strongest proliferation signal -> highest "divide" (col 0) weight
    assert target[0, 0] == target[:, 0].max()


def test_unknown_mode_raises(toy_adata):
    with pytest.raises(ValueError):
        build_fate_targets(toy_adata, mode="nonsense")
