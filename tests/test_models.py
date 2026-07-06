import numpy as np
import torch

from tissue_dynamics.models import SpatialMultinomialFateModel, fate_loss, fit_fate_model, predict_fate_proba


def test_forward_shape_and_reference_class_fixed_at_zero_logit():
    model = SpatialMultinomialFateModel(phi_dim=4, class_names=("divide", "die", "unchanged"))
    phi = torch.randn(10, 4)
    phi_bar = torch.randn(10, 4)

    logits = model(phi, phi_bar)

    assert logits.shape == (10, 3)
    assert torch.allclose(logits[:, -1], torch.zeros(10))


def test_fate_loss_accepts_hard_and_soft_targets():
    model = SpatialMultinomialFateModel(phi_dim=3)
    logits = model(torch.randn(5, 3), torch.randn(5, 3))

    hard_target = torch.tensor([0, 1, 2, 0, 1])
    soft_target = torch.softmax(torch.randn(5, 3), dim=1)

    hard_loss = fate_loss(logits, hard_target)
    soft_loss = fate_loss(logits, soft_target)

    assert hard_loss.ndim == 0
    assert soft_loss.ndim == 0
    assert torch.isfinite(hard_loss)
    assert torch.isfinite(soft_loss)


def test_fit_fate_model_reduces_loss_with_hard_targets():
    rng = np.random.default_rng(0)
    n, d = 40, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d)
    history = fit_fate_model(model, phi, phi_bar, target, n_epochs=50, lr=0.1)

    assert history[-1] <= history[0]

    proba = predict_fate_proba(model, phi, phi_bar)
    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()
    assert not np.isnan(proba).any()


def test_fit_fate_model_reduces_loss_with_soft_targets():
    rng = np.random.default_rng(1)
    n, d = 30, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    raw = rng.random(size=(n, 3))
    target = raw / raw.sum(axis=1, keepdims=True)

    model = SpatialMultinomialFateModel(phi_dim=d)
    history = fit_fate_model(model, phi, phi_bar, target, n_epochs=30, lr=0.1)

    assert history[-1] <= history[0]


def test_fit_fate_model_with_minibatches():
    rng = np.random.default_rng(2)
    n, d = 37, 3  # not evenly divisible by batch_size, to exercise the tail batch
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d)
    history = fit_fate_model(model, phi, phi_bar, target, n_epochs=10, lr=0.1, batch_size=8)

    assert len(history) == 10
    assert all(np.isfinite(h) for h in history)


def test_weight_decay_only_applied_to_a_c_b_not_b0():
    model = SpatialMultinomialFateModel(phi_dim=3)
    optimizer = torch.optim.Adam([
        {"params": model.regularized_parameters(), "weight_decay": 1e-2},
        {"params": [model.b0], "weight_decay": 0.0},
    ])
    weight_decays = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
    assert weight_decays[id(model.b0)] == 0.0
    assert weight_decays[id(model.a)] == 1e-2
    assert weight_decays[id(model.B)] == 1e-2
