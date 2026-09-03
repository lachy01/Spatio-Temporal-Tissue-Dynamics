import numpy as np
import pytest

from tissue_dynamics.inference import (
    _build_design,
    _mnl_observed_information,
    fate_parameter_stats,
    interaction_modes,
    likelihood_ratio_test,
)
from tissue_dynamics.models import SpatialMultinomialFateModel, fit_fate_model


def test_observed_information_matches_statsmodels_logit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    n, p = 500, 4
    X = np.concatenate([np.ones((n, 1)), rng.normal(size=(n, p - 1))], axis=1)
    eta = X @ np.array([0.3, 1.0, -0.5, 0.8])
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)

    res = sm.Logit(y, X).fit(disp=0)
    proba_free = res.predict(X)[:, None]
    info = _mnl_observed_information(X, proba_free)
    my_se = np.sqrt(np.diag(np.linalg.inv(info)))

    np.testing.assert_allclose(my_se, res.bse, rtol=1e-4)


def test_observed_information_matches_statsmodels_mnlogit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n, p = 800, 3
    X = np.concatenate([np.ones((n, 1)), rng.normal(size=(n, p - 1))], axis=1)
    coef = np.array([[0.2, 0.8, -0.4], [-0.3, -0.6, 0.9]])
    eta = np.concatenate([np.zeros((n, 1)), X @ coef.T], axis=1)
    probs = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    res = sm.MNLogit(y, X).fit(disp=0)
    proba_free = res.predict(X)[:, 1:]
    info = _mnl_observed_information(X, proba_free)
    my_se = np.sqrt(np.diag(np.linalg.inv(info))).reshape(2, p)

    np.testing.assert_allclose(my_se, np.asarray(res.bse).T, rtol=1e-3)


def test_observed_information_chunking_is_exact():
    rng = np.random.default_rng(2)
    X = _build_design(rng.normal(size=(200, 3)), rng.normal(size=(200, 3)))
    proba = rng.random((200, 2))
    proba = proba / proba.sum(axis=1, keepdims=True) * 0.9  # leave mass for reference class

    one_shot = _mnl_observed_information(X, proba, chunk_size=10_000)
    chunked = _mnl_observed_information(X, proba, chunk_size=37)
    np.testing.assert_allclose(one_shot, chunked, rtol=1e-10)


def test_build_design_column_order_matches_B_flattening():
    phi = np.array([[2.0, 3.0]])
    phi_bar = np.array([[5.0, 7.0]])
    design = _build_design(phi, phi_bar)
    # [1, phi(2), phi_bar(2), outer(2x2)] ; outer[d,e] = phi[d]*phi_bar[e]
    expected = np.array([[1, 2, 3, 5, 7, 2 * 5, 2 * 7, 3 * 5, 3 * 7]], dtype=float)
    np.testing.assert_allclose(design, expected)


def test_build_design_inserts_covariates_before_outer_block():
    phi = np.array([[2.0, 3.0]])
    phi_bar = np.array([[5.0, 7.0]])
    cov = np.array([[9.0]])
    design = _build_design(phi, phi_bar, covariates=cov)
    # [1, phi(2), phi_bar(2), cov(1), outer(2x2)]
    expected = np.array([[1, 2, 3, 5, 7, 9, 2 * 5, 2 * 7, 3 * 5, 3 * 7]], dtype=float)
    np.testing.assert_allclose(design, expected)


def test_fate_parameter_stats_with_covariates_labels_and_requires_matrix():
    rng = np.random.default_rng(11)
    n, d = 500, 2
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    cov = rng.normal(size=(n, 2))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=0, n_covariates=2)
    fit_fate_model(model, phi, phi_bar, target, covariates=cov, n_epochs=30, lr=0.1, weight_decay=0.0)

    stats = fate_parameter_stats(model, phi, phi_bar, covariates=cov,
                                 feature_names=["x", "y"], covariate_names=["depth", "density"])
    # per class: b0 + a(2) + c(2) + d(2) = 7 terms, no B (interaction_rank=0)
    assert len(stats) == 2 * 7
    assert "d[depth]" in set(stats["term"]) and "d[density]" in set(stats["term"])

    with pytest.raises(ValueError, match="pass the covariates"):
        fate_parameter_stats(model, phi, phi_bar)   # model has covariates but none supplied


def test_fate_parameter_stats_covariate_effect_is_significant():
    # A covariate with a strong true effect on the divide class must come out
    # significant with the right sign in the Wald table.
    rng = np.random.default_rng(12)
    n, d = 2000, 2
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    cov = rng.normal(size=(n, 1))
    eta = np.stack([2.5 * cov[:, 0], np.zeros(n), np.zeros(n)], axis=1)
    probs = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    target = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=0, n_covariates=1)
    fit_fate_model(model, phi, phi_bar, target, covariates=cov, n_epochs=400, lr=0.1, weight_decay=0.0)
    stats = fate_parameter_stats(model, phi, phi_bar, covariates=cov, covariate_names=["depth"])

    row = stats[(stats["class"] == "divide") & (stats["term"] == "d[depth]")].iloc[0]
    assert row["estimate"] > 1.0
    assert row["p_value"] < 1e-6


def test_fate_parameter_stats_returns_labeled_table():
    rng = np.random.default_rng(3)
    n, d = 400, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=None)
    fit_fate_model(model, phi, phi_bar, target, n_epochs=30, lr=0.1, weight_decay=0.0)

    stats = fate_parameter_stats(model, phi, phi_bar, feature_names=["x", "y", "z"])

    n_terms_per_class = 1 + 2 * d + d * d
    assert len(stats) == 2 * n_terms_per_class          # 2 free classes
    assert set(stats["class"]) == {"divide", "die"}     # not the reference class
    assert (stats["variance"] >= 0).all()
    np.testing.assert_allclose(stats["std_error"] ** 2, stats["variance"], rtol=1e-6)
    assert ((stats["p_value"] >= 0) & (stats["p_value"] <= 1)).all()
    assert "b0" in set(stats["term"])
    assert "a[x]" in set(stats["term"])
    assert "condition_number" in stats.attrs


def test_fate_parameter_stats_no_interaction_is_smaller_and_better_conditioned():
    rng = np.random.default_rng(5)
    n, d = 500, 4
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    target = rng.integers(0, 3, size=n)

    full = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=None)
    fit_fate_model(full, phi, phi_bar, target, n_epochs=20, lr=0.1, weight_decay=0.0)
    reduced = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=0)
    fit_fate_model(reduced, phi, phi_bar, target, n_epochs=20, lr=0.1, weight_decay=0.0)

    full_stats = fate_parameter_stats(full, phi, phi_bar)
    reduced_stats = fate_parameter_stats(reduced, phi, phi_bar)

    # reduced model: 2 free classes x (1 + 2d) params, no B terms
    assert len(reduced_stats) == 2 * (1 + 2 * d)
    assert not reduced_stats["term"].str.startswith("B[").any()
    # dropping the collinear outer-product block improves conditioning
    assert reduced_stats.attrs["condition_number"] < full_stats.attrs["condition_number"]


def test_fate_parameter_stats_recovers_a_known_significant_coefficient():
    # Build data where free class 0's logit depends strongly on phi dim 0,
    # so that coefficient must come out large and highly significant.
    rng = np.random.default_rng(4)
    n, d = 2000, 2
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    eta = np.stack([3.0 * phi[:, 0], np.zeros(n), np.zeros(n)], axis=1)
    probs = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    target = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    model = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=0)
    fit_fate_model(model, phi, phi_bar, target, n_epochs=300, lr=0.1, weight_decay=0.0)

    stats = fate_parameter_stats(model, phi, phi_bar, feature_names=["s0", "s1"])
    row = stats[(stats["class"] == "divide") & (stats["term"] == "a[s0]")].iloc[0]
    assert row["estimate"] > 1.0
    assert row["p_value"] < 1e-6


def test_fate_parameter_stats_rejects_lowrank_model():
    rng = np.random.default_rng(6)
    phi = rng.normal(size=(50, 3))
    phi_bar = rng.normal(size=(50, 3))
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=2)
    with pytest.raises(ValueError, match="linear in its parameters"):
        fate_parameter_stats(model, phi, phi_bar)


def _fit(rank, phi, phi_bar, target, d):
    m = SpatialMultinomialFateModel(phi_dim=d, interaction_rank=rank)
    fit_fate_model(m, phi, phi_bar, target, n_epochs=150, lr=0.1, weight_decay=0.0)
    return m


def test_likelihood_ratio_test_detects_real_interaction():
    # Fate driven by a genuine own x neighbor product -> interaction model must
    # fit significantly better than the no-interaction null.
    rng = np.random.default_rng(7)
    n, d = 3000, 3
    phi = rng.normal(size=(n, d))
    phi_bar = rng.normal(size=(n, d))
    signal = 3.0 * phi[:, 0] * phi_bar[:, 1]
    eta = np.stack([signal, np.zeros(n), np.zeros(n)], axis=1)
    probs = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    target = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    null = _fit(0, phi, phi_bar, target, d)
    alt = _fit(1, phi, phi_bar, target, d)
    res = likelihood_ratio_test(null, alt, phi, phi_bar, target)

    assert res["loglik_alt"] > res["loglik_null"]
    assert res["lr_statistic"] > 0
    # df = n_free * (2*D*r - r^2) with n_free=2, D=3, r=1  ->  2*(6-1) = 10
    assert res["df"] == 2 * (2 * d * 1 - 1)
    assert res["p_value"] < 1e-3


def test_likelihood_ratio_test_requires_nesting():
    rng = np.random.default_rng(8)
    phi = rng.normal(size=(40, 2))
    phi_bar = rng.normal(size=(40, 2))
    target = rng.integers(0, 3, size=40)
    m0 = SpatialMultinomialFateModel(phi_dim=2, interaction_rank=0)
    with pytest.raises(ValueError, match="nested"):
        likelihood_ratio_test(m0, m0, phi, phi_bar, target)


def test_interaction_modes_svd_of_lowrank_matches_full():
    # A rank-1 model's single mode's own/neighbor loadings must recover the
    # unit vectors whose outer product defines B (up to sign).
    model = SpatialMultinomialFateModel(phi_dim=4, interaction_rank=1)
    modes = interaction_modes(model, feature_names=["w", "x", "y", "z"], n_modes=1)
    assert set(modes["class"]) == {"divide", "die"}
    assert set(modes["feature"]) == {"w", "x", "y", "z"}

    # reconstruct B from the reported top mode and compare to U V^T
    B = (model.U @ model.V.transpose(-1, -2)).detach().numpy()
    for ci, cls in enumerate(["divide", "die"]):
        sub = modes[modes["class"] == cls].sort_values("feature")
        own = sub["own_loading"].to_numpy()
        nbr = sub["neighbor_loading"].to_numpy()
        sigma = sub["singular_value"].to_numpy()[0]
        recon = sigma * np.outer(own, nbr)
        # feature order is alphabetical here; B is in original order w,x,y,z -> same
        np.testing.assert_allclose(recon, B[ci], atol=1e-5)


def test_interaction_modes_raises_without_interaction():
    model = SpatialMultinomialFateModel(phi_dim=3, interaction_rank=0)
    with pytest.raises(ValueError):
        interaction_modes(model)
