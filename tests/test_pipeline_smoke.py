"""Synthetic end-to-end smoke test for the static pipeline (no real Xenium data needed)."""

import numpy as np
import pandas as pd
import pytest

from tissue_dynamics.pipeline import run_static_pipeline


def _write_phi_csv(tmp_path, adata, phi_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    phi_df = pd.DataFrame(
        rng.normal(size=(adata.n_obs, phi_dim)), columns=[f"phi_{i}" for i in range(phi_dim)],
    )
    phi_df.insert(0, "sample_id", adata.obs["sample_id"].to_numpy())
    phi_df.insert(0, "cell_id", adata.obs["cell_id"].to_numpy())
    path = tmp_path / "phi.csv"
    phi_df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("target_mode", ["hard", "soft"])
def test_run_static_pipeline_end_to_end(tmp_path, pipeline_adata, target_mode):
    phi_path = _write_phi_csv(tmp_path, pipeline_adata)

    adata, model, history = run_static_pipeline(
        pipeline_adata, phi_path,
        radius=80.0, sample_key="sample_id", target_mode=target_mode,
        cell_id_col="cell_id", sample_col="sample_id",
        fit_kwargs={"n_epochs": 20, "lr": 0.05},
    )

    assert adata is pipeline_adata
    assert adata.obsm["phi"].shape == (adata.n_obs, 5)
    assert adata.obsm["phi_bar"].shape == (adata.n_obs, 5)
    assert "spatial_connectivity" in adata.obsp
    assert "proliferation_score" in adata.obs
    assert "death_score" in adata.obs

    proba = adata.obsm["fate_proba"]
    assert proba.shape == (adata.n_obs, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert not np.isnan(proba).any()
    assert len(history) == 20
    assert np.isfinite(history).all()
