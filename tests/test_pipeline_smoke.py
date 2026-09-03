"""Synthetic end-to-end smoke test for the static pipeline (no real Xenium data needed)."""

import numpy as np
import pytest

from tissue_dynamics.pipeline import run_static_pipeline


@pytest.mark.parametrize("target_mode", ["hard", "soft"])
def test_run_static_pipeline_end_to_end(pipeline_adata, target_mode, write_phi_csv):
    phi_path = write_phi_csv(pipeline_adata)

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


def test_run_static_pipeline_with_covariates(pipeline_adata, write_phi_csv):
    from tissue_dynamics.covariates import compute_library_size, compute_local_density

    compute_library_size(pipeline_adata)
    compute_local_density(pipeline_adata, radius=80.0)
    phi_path = write_phi_csv(pipeline_adata)

    adata, model, history = run_static_pipeline(
        pipeline_adata, phi_path,
        cell_id_col="cell_id", sample_col="sample_id",
        model_kwargs={"interaction_rank": 0},
        covariate_keys=("log_total_counts", "local_density"),
        fit_kwargs={"n_epochs": 15, "lr": 0.05},
    )

    assert model.n_covariates == 2
    assert adata.obsm["covariates"].shape == (adata.n_obs, 2)
    proba = adata.obsm["fate_proba"]
    assert proba.shape == (adata.n_obs, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert not np.isnan(proba).any()
