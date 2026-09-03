import numpy as np
import pytest
import torch

from tissue_dynamics.models import (
    SpatialMultinomialFateModel,
    _resolve_class_weight,
    fate_loss,
    fit_fate_model,
    predict_fate_proba,
)


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


def test_fate_loss_hard_weight_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    target = torch.tensor([0, 0, 1, 2, 1, 0])
    weight = torch.tensor([2.0, 1.0, 0.5])

    got = fate_loss(logits, target, weight=weight)
    expected = torch.nn.functional.cross_entropy(logits, target, weight=weight)
    assert torch.allclose(got, expected)


def test_fate_loss_soft_weight_reduces_to_hard_weight_for_onehot_targets():
    # A weighted soft-target loss on one-hot targets must equal the weighted
    # hard-label cross-entropy — the generalization has to be consistent.
    torch.manual_seed(1)
    logits = torch.randn(6, 3)
    hard_target = torch.tensor([0, 0, 1, 2, 1, 0])
    soft_target = torch.eye(3)[hard_target]
    weight = torch.tensor([2.0, 1.0, 0.5])

    soft_loss = fate_loss(logits, soft_target, weight=weight)
    hard_loss = fate_loss(logits, hard_target, weight=weight)
    assert torch.allclose(soft_loss, hard_loss, atol=1e-6)


def test_resolve_class_weight_balanced_upweights_rare_class_hard_targets():
    # counts [6, 3, 1]: class 0 most common, class 2 rarest -> weights ascend.
    target = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 2])
    weight = _resolve_class_weight("balanced", target, n_classes=3)

    assert weight.shape == (3,)
    assert weight[2] > weight[1] > weight[0]
    # sklearn convention: n_samples / (n_classes * count)
    np.testing.assert_allclose(weight.numpy(), 10.0 / (3 * np.array([6, 3, 1])), rtol=1e-6)


def test_resolve_class_weight_balanced_uses_mass_for_soft_targets():
    soft = torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.4, 0.0], [0.1, 0.1, 0.8]])
    weight = _resolve_class_weight("balanced", soft, n_classes=3)
    mass = soft.sum(dim=0).numpy()
    np.testing.assert_allclose(weight.numpy(), 3.0 / (3 * mass), rtol=1e-6)


def test_resolve_class_weight_explicit_and_errors():
    target = torch.tensor([0, 1, 2])
    explicit = _resolve_class_weight([1.0, 2.0, 3.0], target, n_classes=3)
    np.testing.assert_allclose(explicit.numpy(), [1.0, 2.0, 3.0])

    assert _resolve_class_weight(None, target, n_classes=3) is None

    with pytest.raises(ValueError, match="shape"):
        _resolve_class_weight([1.0, 2.0], target, n_classes=3)
    with pytest.raises(ValueError, match="balanced"):
        _resolve_class_weight("nonsense", target, n_classes=3)


def test_fit_fate_model_balanced_predicts_rare_class_more_than_unweighted():
    # Heavily imbalanced hard targets where the rare class (2) is genuinely
    # separable in phi; balanced weighting should recover it where an
    # unweighted fit collapses to never predicting it.
    rng = np.random.default_rng(0)
    n, d = 600, 3
    # 90% class 0, 7% class 1, 3% class 2
    target = rng.choice([0, 1, 2], size=n, p=[0.90, 0.07, 0.03])
    # give each class a separated mean in phi so the signal exists to learn
    means = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    phi = rng.normal(size=(n, d)) * 0.5 + means[target]
    phi_bar = rng.normal(size=(n, d)) * 0.5

    unweighted = SpatialMultinomialFateModel(phi_dim=d)
    fit_fate_model(unweighted, phi, phi_bar, target, n_epochs=150, lr=0.1)
    unweighted_pred = predict_fate_proba(unweighted, phi, phi_bar).argmax(axis=1)

    balanced = SpatialMultinomialFateModel(phi_dim=d)
    fit_fate_model(balanced, phi, phi_bar, target, n_epochs=150, lr=0.1, class_weight="balanced")
    balanced_pred = predict_fate_proba(balanced, phi, phi_bar).argmax(axis=1)

    is_rare = target == 2
    unweighted_recall = (unweighted_pred[is_rare] == 2).mean()
    balanced_recall = (balanced_pred[is_rare] == 2).mean()
    assert balanced_recall > unweighted_recall


def test_default_interaction_rank_is_2_lowrank():
    model = SpatialMultinomialFateModel(phi_dim=5)
    assert model.interaction_rank == 2
    assert model.interaction_mode == "lowrank"
    assert model.U.shape == (2, 5, 2) and model.V.shape == (2, 5, 2)


def test_interaction_modes_property_and_params_by_mode():
    none = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=0)
    full = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=None)
    low = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=3)

    assert none.interaction_mode == "none"
    assert full.interaction_mode == "full"
    assert low.interaction_mode == "lowrank"

    assert not hasattr(none, "B") and not hasattr(none, "U")
    assert any(p is full.B for p in full.regularized_parameters())
    reg_low = low.regularized_parameters()
    assert any(p is low.U for p in reg_low) and any(p is low.V for p in reg_low)


def test_interaction_rank_zero_matches_manual_own_neighbor_logits():
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=0)
    with torch.no_grad():
        model.a.copy_(torch.randn(2, 3))
        model.c.copy_(torch.randn(2, 3))
        model.b0.copy_(torch.randn(2))
    phi = torch.randn(6, 3)
    phi_bar = torch.randn(6, 3)
    expected_free = model.b0 + phi @ model.a.T + phi_bar @ model.c.T
    logits = model(phi, phi_bar)
    assert torch.allclose(logits[:, :2], expected_free, atol=1e-6)
    assert torch.allclose(logits[:, 2], torch.zeros(6))


def test_lowrank_interaction_matches_manual_bilinear():
    model = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=2)
    phi = torch.randn(5, 4)
    phi_bar = torch.randn(5, 4)
    logits = model(phi, phi_bar)
    # interaction for free class k = phi @ (U_k V_k^T) @ phi_bar
    B = model.U @ model.V.transpose(-1, -2)  # (2,4,4)
    manual = model.b0 + phi @ model.a.T + phi_bar @ model.c.T
    manual = manual + torch.einsum("nd,kde,ne->nk", phi, B, phi_bar)
    assert torch.allclose(logits[:, :2], manual, atol=1e-5)


def test_interaction_matrix_matches_UVt_for_lowrank():
    model = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=2)
    B = model.interaction_matrix()
    assert B.shape == (2, 4, 4)
    expected = model.U @ model.V.transpose(-1, -2)
    assert torch.allclose(B, expected)


def test_interaction_matrix_raises_when_no_interaction():
    model = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=0)
    with pytest.raises(ValueError):
        model.interaction_matrix()


def test_lowrank_fits_and_updates_UV():
    rng = np.random.default_rng(0)
    n, d = 200, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=2)
    U0 = model.U.detach().clone()
    history = fit_fate_model(model, phi, phi_bar, target, n_epochs=50, lr=0.1)
    assert history[-1] <= history[0]
    assert not torch.allclose(model.U, U0)  # low-rank factors actually train


def test_use_interaction_deprecated_alias_maps_to_rank():
    with pytest.warns(DeprecationWarning):
        full = SpatialMultinomialFateModel(phi_dim=3, use_interaction=True)
    assert full.interaction_mode == "full"
    with pytest.warns(DeprecationWarning):
        none = SpatialMultinomialFateModel(phi_dim=3, use_interaction=False)
    assert none.interaction_mode == "none"


def test_invalid_interaction_rank_raises():
    with pytest.raises(ValueError):
        SpatialMultinomialFateModel(phi_dim=3, interaction_rank=-1)
    with pytest.raises(ValueError):
        SpatialMultinomialFateModel(phi_dim=3, interaction_rank=2.5)


def test_weight_decay_only_applied_to_regularized_params_not_b0():
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=None)
    optimizer = torch.optim.Adam([
        {"params": model.regularized_parameters(), "weight_decay": 1e-2},
        {"params": [model.b0], "weight_decay": 0.0},
    ])
    weight_decays = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
    assert weight_decays[id(model.b0)] == 0.0
    assert weight_decays[id(model.a)] == 1e-2
    assert weight_decays[id(model.B)] == 1e-2


# --- covariate support ---

def test_covariates_add_linear_term_to_logits():
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=0, n_covariates=2)
    assert model.n_covariates == 2 and model.d.shape == (2, 2)
    with torch.no_grad():
        model.d.copy_(torch.randn(2, 2))
    phi = torch.randn(6, 3)
    phi_bar = torch.randn(6, 3)
    cov = torch.randn(6, 2)
    expected_free = (model.b0 + phi @ model.a.T + phi_bar @ model.c.T + cov @ model.d.T)
    logits = model(phi, phi_bar, covariates=cov)
    assert torch.allclose(logits[:, :2], expected_free, atol=1e-6)


def test_covariates_excluded_from_weight_decay():
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=0, n_covariates=2)
    reg = model.regularized_parameters()
    unreg = model.unregularized_parameters()
    assert all(p is not model.d for p in reg)      # not shrunk
    assert any(p is model.d for p in unreg)
    assert any(p is model.b0 for p in unreg)


def test_covariates_required_when_declared_and_rejected_when_not():
    with_cov = SpatialMultinomialFateModel(phi_dim=2, interaction_rank=0, n_covariates=1)
    with pytest.raises(ValueError, match="no covariates were passed"):
        with_cov(torch.randn(4, 2), torch.randn(4, 2))

    no_cov = SpatialMultinomialFateModel(phi_dim=2, interaction_rank=0)
    with pytest.raises(ValueError, match="n_covariates=0"):
        no_cov(torch.randn(4, 2), torch.randn(4, 2), covariates=torch.randn(4, 1))


def test_fit_with_covariates_recovers_covariate_effect():
    # fate logit driven purely by a covariate -> its coefficient must be learned
    rng = np.random.default_rng(0)
    n, d = 1500, 2
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    z = rng.normal(size=(n, 1))
    eta = np.stack([2.5 * z[:, 0], np.zeros(n), np.zeros(n)], axis=1)
    probs = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    target = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=0, n_covariates=1)
    fit_fate_model(model, phi, phi_bar, target, covariates=z, n_epochs=300, lr=0.1, weight_decay=0.0)
    proba = predict_fate_proba(model, phi, phi_bar, covariates=z)
    assert proba.shape == (n, 3)
    assert model.d[0, 0].item() > 1.0                # positive divide-class covariate effect


def test_invalid_n_covariates_raises():
    with pytest.raises(ValueError):
        SpatialMultinomialFateModel(phi_dim=3, n_covariates=-1)
