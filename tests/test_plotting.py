import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tissue_dynamics.inference import fate_parameter_stats, interaction_modes
from tissue_dynamics.pipeline import run_static_pipeline
from tissue_dynamics.plotting import (
    _pit_histogram,
    _rootogram_counts,
    plot_coefficients,
    plot_fate_spatial,
    plot_interaction_modes,
    plot_pit_histogram,
    plot_rootogram,
    plot_training_loss,
)


@pytest.fixture
def scored_adata(pipeline_adata, write_phi_csv):
    phi_path = write_phi_csv(pipeline_adata)
    adata, _model, _history = run_static_pipeline(
        pipeline_adata, phi_path,
        radius=80.0, sample_key="sample_id", target_mode="hard",
        cell_id_col="cell_id", sample_col="sample_id",
        fit_kwargs={"n_epochs": 5, "lr": 0.05},
    )
    return adata


def test_plot_fate_spatial_requires_sample_id_when_multiple_sections(scored_adata):
    with pytest.raises(ValueError, match="sample_id"):
        plot_fate_spatial(scored_adata)


def test_plot_fate_spatial_returns_grid_of_axes(scored_adata):
    fig, axes = plot_fate_spatial(scored_adata, sample_id="S1")
    assert axes.shape == (2, 3)
    plt.close(fig)


def test_plot_fate_spatial_panel_titles_include_class_and_count(scored_adata):
    fig, axes = plot_fate_spatial(scored_adata, sample_id="S1", class_names=("divide", "die", "unchanged"))
    assert "divide" in axes[0, 0].get_title()
    assert "n=" in axes[0, 0].get_title()
    plt.close(fig)


def test_plot_fate_spatial_raises_on_missing_ground_truth(scored_adata):
    del scored_adata.obs["fate_label"]
    with pytest.raises(ValueError):
        plot_fate_spatial(scored_adata, sample_id="S1")


def test_plot_fate_spatial_raises_on_missing_proba(scored_adata):
    del scored_adata.obsm["fate_proba"]
    with pytest.raises(ValueError):
        plot_fate_spatial(scored_adata, sample_id="S1")


def test_plot_training_loss_single_history():
    history = [1.0, 0.8, 0.7, 0.65]
    fig, ax = plot_training_loss(history)
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [1, 2, 3, 4]
    plt.close(fig)


def test_plot_training_loss_overlays_multiple_runs():
    fig, ax = plot_training_loss({"unweighted": [1.0, 0.8], "balanced": [1.2, 1.0]})
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_coefficients_shows_top_n_terms():
    rng = np.random.default_rng(0)
    n, d = 400, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    from tissue_dynamics.models import SpatialMultinomialFateModel, fit_fate_model
    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=None)
    fit_fate_model(model, phi, phi_bar, target, n_epochs=20, lr=0.1, weight_decay=0.0)
    stats = fate_parameter_stats(model, phi, phi_bar)

    fig, ax = plot_coefficients(stats, top_n=8)
    # one point per selected term + the vertical zero reference line
    assert len(ax.get_yticklabels()) == 8
    plt.close(fig)


def test_plot_interaction_modes_grid_shape():
    from tissue_dynamics.models import SpatialMultinomialFateModel, fit_fate_model
    rng = np.random.default_rng(0)
    n, d = 300, 4
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=2)
    fit_fate_model(model, phi, phi_bar, target, n_epochs=20, lr=0.1)
    modes = interaction_modes(model, feature_names=list("wxyz"))

    # one free class, 2 modes -> 2 rows x 2 cols (own | neighbor)
    fig, axes = plot_interaction_modes(modes, class_name="divide", n_modes=2)
    assert axes.shape == (2, 2)
    plt.close(fig)


def _nb_sample(n=20000, alpha=0.5, seed=0):
    """Draw NB2 counts with varying per-cell means; return (y, mu, alpha)."""
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0.0, 0.6, size=n))          # heterogeneous means ~ lognormal
    r = 1.0 / alpha                                     # NB2 size
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)
    return y.astype(float), mu, alpha


def test_rootogram_counts_expected_matches_observed_when_correct():
    y, mu, alpha = _nb_sample()
    ks, observed, expected = _rootogram_counts(y, mu, alpha, max_count=12)
    assert ks[0] == 0 and len(ks) == 13
    # totals agree (both count the mass up to max_count), and per-bin the fitted
    # expected frequencies track the observed within sampling error
    assert expected.sum() == pytest.approx(observed.sum(), rel=0.02)
    denom = np.sqrt(expected.sum())
    assert np.max(np.abs(np.sqrt(observed) - np.sqrt(expected))) < 0.05 * denom


def test_pit_histogram_uniform_when_model_correct():
    y, mu, alpha = _nb_sample()
    edges, heights = _pit_histogram(y, mu, alpha, n_bins=10)
    assert len(heights) == 10
    assert heights.mean() == pytest.approx(1.0, abs=1e-6)   # densities average to 1
    assert np.max(np.abs(heights - 1.0)) < 0.15             # ~flat for a calibrated model


def test_pit_histogram_u_shaped_when_predictive_too_narrow():
    # data are over-dispersed NB, but scored as if Poisson (predictive too narrow)
    y, mu, _alpha = _nb_sample(alpha=1.0, seed=3)
    _edges, heights = _pit_histogram(y, mu, alpha=None, n_bins=10)  # Poisson predictive
    assert heights[0] > 1.2 and heights[-1] > 1.2              # mass piles into both tails
    assert heights[len(heights) // 2] < 0.9                    # dip in the middle


def test_plot_rootogram_and_pit_return_axes():
    y, mu, alpha = _nb_sample(n=3000)
    fig1, ax1 = plot_rootogram(y, mu, alpha, max_count=10)
    fig2, ax2 = plot_pit_histogram(y, mu, alpha, n_bins=8)
    assert ax1.patches and ax2.patches                        # bars were drawn
    plt.close(fig1); plt.close(fig2)


def test_rootogram_and_pit_accept_per_cell_dispersion():
    import matplotlib
    matplotlib.use("Agg")
    from tissue_dynamics.plotting import plot_pit_histogram, plot_rootogram
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.uniform(0.5, 4.0, size=n)
    alpha = np.exp(rng.normal(-0.5, 0.4, size=n))      # a per-cell dispersion
    r = 1.0 / alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    fig, ax = plot_rootogram(y, mu, alpha, max_count=10)
    assert ax.patches                                   # bars were drawn
    fig2, ax2 = plot_pit_histogram(y, mu, alpha, n_bins=10)
    heights = np.array([p.get_height() for p in ax2.patches])
    assert np.max(np.abs(heights - 1.0)) < 0.15         # correctly specified -> ~uniform
