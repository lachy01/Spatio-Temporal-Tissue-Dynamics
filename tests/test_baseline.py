import numpy as np
import pytest

from tissue_dynamics.baseline import build_mlp_features, fit_mlp_baseline, predict_mlp_proba


def test_build_mlp_features_selects_inputs():
    phi = np.zeros((5, 3))
    phi_bar = np.ones((5, 4))
    assert build_mlp_features(phi, phi_bar, "neighbors").shape == (5, 4)
    assert build_mlp_features(phi, phi_bar, "own").shape == (5, 3)
    assert build_mlp_features(phi, phi_bar, "both").shape == (5, 7)
    with pytest.raises(ValueError):
        build_mlp_features(phi, phi_bar, "nonsense")


def test_fit_mlp_baseline_learns_separable_classes():
    rng = np.random.default_rng(0)
    n, d = 300, 3
    target = rng.integers(0, 3, size=n)
    means = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]], dtype=float)
    phi_bar = rng.normal(size=(n, d)) * 0.4 + means[target]
    phi = rng.normal(size=(n, d))

    pipeline = fit_mlp_baseline(phi, phi_bar, target, input_features="neighbors",
                                 hidden_layer_sizes=(16,), max_iter=500)
    proba = predict_mlp_proba(pipeline, phi, phi_bar)

    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    accuracy = (proba.argmax(axis=1) == target).mean()
    assert accuracy > 0.8  # separable neighbor signal -> easily learned


def test_predict_mlp_proba_reinserts_absent_class_column():
    rng = np.random.default_rng(1)
    n, d = 120, 2
    # only classes 0 and 2 appear in training; class 1 is absent
    target = rng.choice([0, 2], size=n)
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))

    pipeline = fit_mlp_baseline(phi, phi_bar, target, input_features="both",
                                 hidden_layer_sizes=(8,), max_iter=200)
    proba = predict_mlp_proba(pipeline, phi, phi_bar, n_classes=3)

    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba[:, 1], 0.0)          # absent class -> zero column
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_fit_mlp_baseline_rejects_soft_targets():
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(10, 2))
    phi_bar = rng.normal(size=(10, 2))
    soft = rng.random((10, 3))
    soft /= soft.sum(axis=1, keepdims=True)
    with pytest.raises(ValueError):
        fit_mlp_baseline(phi, phi_bar, soft)
