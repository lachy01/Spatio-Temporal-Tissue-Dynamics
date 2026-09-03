import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.count_glm import (
    _division_design,
    _estimate_nb_alpha,
    _nb2_irls,
    build_predictors,
    compare_count_glms,
    fit_count_glm,
    fit_death_nb,
    fit_division_nb,
    fit_neighbourhood_count_glm,
    get_gene_counts,
    get_signature_counts,
    stepwise_division_nb,
)


def _counts_adata(n=2000, beta_neighbor=1.0, seed=0):
    """AnnData whose gene 'MKI67' counts depend on neighbor axis phi_bar[:,0]."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, 2))
    phi_bar = rng.normal(size=(n, 2))
    total = np.full(n, 100.0)
    log_mu = np.log(total) - 5.0 + beta_neighbor * phi_bar[:, 0]
    mki = rng.poisson(np.exp(log_mu)).astype(float)
    # a well-populated library so observed total_counts (used as the offset)
    # is nonzero for every cell and close to the generating value of 100
    other = rng.poisson(100.0, size=n).astype(float)

    X = np.column_stack([mki, other])
    a = AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=["MKI67", "OTHER"]))
    a.obsm["phi"] = phi
    a.obsm["phi_bar"] = phi_bar
    return a


def test_get_gene_counts_prefers_counts_layer():
    a = AnnData(np.array([[1.0, 2.0], [3.0, 4.0]]),
                obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["G0", "G1"]))
    a.layers["counts"] = np.array([[10.0, 20.0], [30.0, 40.0]])
    g, total = get_gene_counts(a, "G1")
    np.testing.assert_allclose(g, [20.0, 40.0])          # from layers, not X
    np.testing.assert_allclose(total, [30.0, 70.0])


def test_build_predictors_shapes_by_spec():
    a = _counts_adata(n=50)
    assert build_predictors(a, "none")[0].shape == (50, 0)
    assert build_predictors(a, "own", n_pcs=None)[0].shape == (50, 2)
    assert build_predictors(a, "neighbor", n_pcs=None)[0].shape == (50, 2)
    assert build_predictors(a, "own+neighbor", n_pcs=None)[0].shape == (50, 4)
    # PCA reduction caps the column count
    assert build_predictors(a, "own+neighbor", n_pcs=1)[0].shape == (50, 2)


def test_estimate_nb_alpha_positive_on_overdispersed():
    rng = np.random.default_rng(0)
    mu = np.full(4000, 3.0)
    # NB-ish overdispersed counts
    y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    assert _estimate_nb_alpha(y, mu) > 0


def test_fit_neighbourhood_count_glm_detects_real_neighbor_effect():
    pytest.importorskip("statsmodels")
    a = _counts_adata(beta_neighbor=1.0)
    fit = fit_neighbourhood_count_glm(a, gene="MKI67", predictors="neighbor",
                                       n_pcs=None, family="nb")
    assert fit["n"] == a.n_obs
    assert fit["alpha"] is not None and fit["alpha"] > 0
    assert fit["mcfadden_r2"] > 0
    assert fit["lr_vs_null"]["p_value"] < 1e-3           # neighbor axis is real
    # coefficient on the true neighbor axis (nbr_0) is positive and significant
    row = fit["coef"].set_index("term").loc["nbr_0"]
    assert row["estimate"] > 0.3
    assert row["p_value"] < 1e-3


def test_fit_neighbourhood_count_glm_null_effect_not_significant():
    pytest.importorskip("statsmodels")
    a = _counts_adata(beta_neighbor=0.0, seed=3)          # no neighbor signal
    fit = fit_neighbourhood_count_glm(a, gene="MKI67", predictors="neighbor",
                                       n_pcs=None, family="nb")
    assert fit["lr_vs_null"]["p_value"] > 0.01            # nothing real to find


def test_compare_count_glms_orders_and_columns():
    pytest.importorskip("statsmodels")
    a = _counts_adata(beta_neighbor=1.0)
    table = compare_count_glms(a, gene="MKI67", specs=("none", "neighbor"), n_pcs=None)
    assert list(table["spec"]) == ["none", "neighbor"]
    assert {"n_params", "loglik", "mcfadden_r2", "lr_vs_null_p"} <= set(table.columns)
    # adding the neighbor block improves log-likelihood
    assert table.set_index("spec").loc["neighbor", "loglik"] > table.set_index("spec").loc["none", "loglik"]


def _division_adata(n=4000, beta_own=0.8, beta_int=0.0, seed=0, phi_dim=2, response_genes=None):
    """AnnData whose summed 'response gene' counts rise with own axis phi[:,0].

    The response genes (default DIV0..2) plus a well-populated filler block so
    every cell's library size is nonzero and the depth covariate has variance.
    `beta_int` adds a true own[:,0] x neighbour[:,0] interaction effect;
    `phi_dim` sets the embedding width; `response_genes` names the response
    block (e.g. apoptosis markers for the death model).
    """
    rng = np.random.default_rng(seed)
    resp = list(response_genes) if response_genes is not None else [f"DIV{i}" for i in range(3)]
    phi = rng.normal(size=(n, phi_dim))
    phi_bar = rng.normal(size=(n, phi_dim))
    log_mu = -1.0 + beta_own * phi[:, 0] + beta_int * phi[:, 0] * phi_bar[:, 0]
    div = rng.poisson(np.exp(log_mu)[:, None] * np.ones((1, len(resp)))).astype(float)
    filler = rng.poisson(20.0, size=(n, 4)).astype(float)

    genes = resp + [f"F{i}" for i in range(4)]
    X = np.column_stack([div, filler])
    a = AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=genes))
    a.obsm["phi"] = phi
    a.obsm["phi_bar"] = phi_bar
    a.uns["phi_names"] = [f"axis{i}" for i in range(phi_dim)]
    total = X.sum(1)
    a.obs["log_total_counts"] = np.log(total)
    a.obs["local_density"] = rng.normal(size=n)          # unrelated confounder
    a.obs["cell_area"] = rng.uniform(20.0, 200.0, size=n)
    return a, resp


def test_get_signature_counts_sums_matched_genes():
    a = AnnData(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["G0", "G1", "G2"]))
    a.layers["counts"] = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    y, total, matched = get_signature_counts(a, ["G0", "G2", "ABSENT"])
    assert matched == ["G0", "G2"]                        # only present genes, order preserved
    np.testing.assert_allclose(y, [40.0, 100.0])          # (10+30), (40+60) from layers
    np.testing.assert_allclose(total, [60.0, 150.0])


def test_get_signature_counts_raises_when_none_present():
    a = AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=["a", "b"]),
                var=pd.DataFrame(index=["G0", "G1"]))
    with pytest.raises(ValueError, match="none of"):
        get_signature_counts(a, ["NOPE1", "NOPE2"])


def test_fit_count_glm_respects_fixed_alpha():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata()
    y, total, _ = get_signature_counts(a, div_genes)
    keep = total > 0
    X = a.obsm["phi"][keep]
    auto = fit_count_glm(y[keep], X, ["a0", "a1"], family="nb")
    fixed = fit_count_glm(y[keep], X, ["a0", "a1"], family="nb", alpha=0.5)
    assert fixed["alpha"] == 0.5                          # used verbatim, not re-estimated
    assert auto["alpha"] != 0.5


def test_fit_division_nb_recovers_own_state_effect():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.8)
    fit = fit_division_nb(a, genes=div_genes, predictors="own",
                          covariate_keys=("log_total_counts", "local_density"))
    assert fit["genes"] == div_genes
    assert fit["n"] == a.n_obs
    assert fit["alpha"] is not None and fit["alpha"] > 0
    assert {"pseudo_r2_vs_covariates", "lr_vs_covariate_only"} <= set(fit)
    # own state adds real signal beyond the covariates
    assert fit["lr_vs_covariate_only"]["p_value"] < 1e-3
    row = fit["coef"].set_index("term").loc["own[axis0]"]
    assert row["estimate"] > 0.3 and row["p_value"] < 1e-3
    # covariates are present as free, named terms
    assert {"log_total_counts", "local_density"} <= set(fit["coef"]["term"])


def test_fit_division_nb_null_is_intercept_only_without_covariates():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.8)
    fit = fit_division_nb(a, genes=div_genes, predictors="own", covariate_keys=None)
    # with no covariates the state block (2 own axes) is tested against a bare intercept
    assert fit["lr_vs_covariate_only"]["df"] == 2
    assert "log_total_counts" not in set(fit["coef"]["term"])


def test_fit_division_nb_invalid_predictors_raises():
    a, div_genes = _division_adata()
    with pytest.raises(ValueError, match="Unknown predictors"):
        fit_division_nb(a, genes=div_genes, predictors="bogus")


def test_fit_division_nb_full_interaction_adds_outer_product_block():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata()                      # phi is 2-dim -> 2x2 = 4 int terms
    base = fit_division_nb(a, genes=div_genes, predictors="own+neighbor", interaction_rank=0)
    full = fit_division_nb(a, genes=div_genes, predictors="own+neighbor", interaction_rank=None)
    int_terms = [t for t in full["coef"]["term"] if t.startswith("int[")]
    assert len(int_terms) == 4                            # D_own * D_nbr
    assert "int[axis0 x axis0]" in int_terms
    assert full["n_params"] == base["n_params"] + 4       # exactly the outer-product columns


def test_fit_division_nb_recovers_true_interaction():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.5, beta_int=0.8, seed=5)
    fit = fit_division_nb(a, genes=div_genes, predictors="own+neighbor", interaction_rank=None)
    row = fit["coef"].set_index("term").loc["int[axis0 x axis0]"]
    assert row["estimate"] > 0.3 and row["p_value"] < 1e-3


def test_fit_division_nb_full_interaction_requires_both_blocks():
    a, div_genes = _division_adata()
    with pytest.raises(ValueError, match="needs predictors='own\\+neighbor'"):
        fit_division_nb(a, genes=div_genes, predictors="own", interaction_rank=None)


def test_fit_division_nb_low_rank_interaction_not_implemented():
    a, div_genes = _division_adata()
    with pytest.raises(NotImplementedError, match="low-rank"):
        fit_division_nb(a, genes=div_genes, predictors="own+neighbor", interaction_rank=2)


def test_fit_division_nb_rejects_bad_interaction_rank():
    a, div_genes = _division_adata()
    with pytest.raises(ValueError, match="interaction_rank must be"):
        fit_division_nb(a, genes=div_genes, predictors="own+neighbor", interaction_rank=-1)


def test_division_design_meta_and_shapes():
    a, _ = _division_adata()                              # phi is 2-dim
    X, names, meta, cov, cov_names = _division_design(
        a, "own+neighbor", None, ("log_total_counts", "local_density"), True)
    kinds = [m["kind"] for m in meta]
    assert kinds.count("own") == 2 and kinds.count("nbr") == 2
    assert kinds.count("int") == 4 and kinds.count("cov") == 2
    assert X.shape[1] == len(names) == 10                 # 2 + 2 + 4 + 2
    assert names[:2] == ["own[axis0]", "own[axis1]"]
    assert cov.shape[1] == 2 and cov_names == ["log_total_counts", "local_density"]
    # meta is aligned with names and marks the interaction structure
    m = dict(zip(names, meta))
    assert m["int[axis0 x axis1]"] == {"kind": "int", "own": "axis0", "nbr": "axis1"}


def test_nb2_irls_matches_statsmodels_at_fixed_alpha():
    pytest.importorskip("statsmodels")
    import statsmodels.api as sm

    a, div_genes = _division_adata(beta_own=0.6, seed=1)
    y, total, _ = get_signature_counts(a, div_genes)
    X = a.obsm["phi"]                                     # 2 predictors + intercept
    Xc = sm.add_constant(X, has_constant="add")
    alpha = 0.5

    ref = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    beta, z, se, ll = _nb2_irls(y, Xc, alpha)

    np.testing.assert_allclose(beta, ref.params, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ll, ref.llf, rtol=1e-6)
    np.testing.assert_allclose(se, ref.bse, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(z, ref.tvalues, rtol=2e-3, atol=1e-3)


def test_nb2_irls_offset_matches_statsmodels():
    pytest.importorskip("statsmodels")
    import statsmodels.api as sm

    a, div_genes = _division_adata(beta_own=0.5, seed=2)
    y, total, _ = get_signature_counts(a, div_genes)
    Xc = sm.add_constant(a.obsm["phi"], has_constant="add")
    offset = np.log(total)
    alpha = 0.6

    ref = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    beta, z, se, ll = _nb2_irls(y, Xc, alpha, offset=offset)
    np.testing.assert_allclose(beta, ref.params, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ll, ref.llf, rtol=1e-6)


def test_division_design_spline_and_covariate_nuisance_block():
    a, _ = _division_adata()
    X, names, meta, nuis, nuis_names = _division_design(
        a, "own", 0, ("cell_area",), True, spline_keys=("local_density",), spline_df=4)
    kinds = [m["kind"] for m in meta]
    assert kinds.count("spline") == 4 and kinds.count("cov") == 1
    # nuisance block = spline basis (4) + free covariate (1), in that order
    assert nuis.shape[1] == 5
    assert nuis_names == ["spline[local_density]_1", "spline[local_density]_2",
                          "spline[local_density]_3", "spline[local_density]_4", "cell_area"]
    assert names[-5:] == nuis_names                       # nuisance comes last in the design


def test_fit_division_nb_offset_spline_and_free_covariate():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.8)
    fit = fit_division_nb(a, genes=div_genes, predictors="own",
                          offset_key="log_total_counts", spline_keys=("local_density",),
                          spline_df=4, covariate_keys=("cell_area",))
    terms = set(fit["coef"]["term"])
    assert "log_total_counts" not in terms               # it's the offset, not a coefficient
    assert "cell_area" in terms                           # free covariate
    assert sum(t.startswith("spline[local_density]") for t in terms) == 4
    assert fit["offset_key"] == "log_total_counts"
    # per-cell predictions are attached and sane
    assert fit["fitted_mu"].shape == (a.n_obs,)
    assert fit["pearson_resid"].shape == (a.n_obs,)
    p = fit["prob_division"]
    assert np.all((p[np.isfinite(p)] >= 0) & (p[np.isfinite(p)] <= 1))
    # own state still recovered as a rate effect under the offset model
    assert fit["coef"].set_index("term").loc["own[axis0]", "z"] > 3


def test_fit_division_nb_irls_matches_statsmodels():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.8)
    kw = dict(genes=div_genes, predictors="own+neighbor", offset_key="log_total_counts",
              covariate_keys=("cell_area",))
    fsm = fit_division_nb(a, fitter="statsmodels", **kw)
    firls = fit_division_nb(a, fitter="irls", **kw)
    assert fsm["fitter"] == "statsmodels" and firls["fitter"] == "irls"
    assert "result" not in firls                          # no heavy statsmodels object on the irls path
    sm_c = fsm["coef"].set_index("term")["estimate"]
    ir_c = firls["coef"].set_index("term")["estimate"]
    np.testing.assert_allclose(ir_c, sm_c.loc[ir_c.index], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.nan_to_num(firls["fitted_mu"]),
                               np.nan_to_num(fsm["fitted_mu"]), rtol=1e-2, atol=1e-2)


def test_fit_division_nb_auto_uses_irls_for_wide_designs():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(phi_dim=6)             # own6+nbr6+int36 = 48 predictors >= 30
    wide = fit_division_nb(a, genes=div_genes, predictors="own+neighbor",
                           interaction_rank=None, fitter="auto")
    narrow = fit_division_nb(a, genes=div_genes, predictors="own+neighbor",
                             interaction_rank=0, fitter="auto")
    assert wide["fitter"] == "irls"                       # auto-switched (design too wide for statsmodels)
    assert narrow["fitter"] == "statsmodels"


def test_fit_death_nb_uses_apoptosis_signature_and_shares_architecture():
    pytest.importorskip("statsmodels")
    from tissue_dynamics.gene_sets import APOPTOSIS_SIGNATURE

    resp = list(APOPTOSIS_SIGNATURE[:3])                  # a few apoptosis markers as the response
    a, _ = _division_adata(beta_own=0.8, response_genes=resp)
    fit = fit_death_nb(a, predictors="own+neighbor", interaction_rank=None,
                       offset_key="log_total_counts", spline_keys=("local_density",),
                       covariate_keys=("cell_area",))
    # defaulted to the apoptosis signature (only panel-present genes)
    assert set(fit["genes"]) == set(resp) and set(fit["genes"]) <= set(APOPTOSIS_SIGNATURE)
    # death-specific probability key, and no division key
    assert "prob_death" in fit and "prob_division" not in fit
    assert fit["prob_death"].shape == (a.n_obs,)
    # same architecture: interaction block + offset present, own state recovered
    assert any(t.startswith("int[") for t in fit["coef"]["term"])
    assert "log_total_counts" not in set(fit["coef"]["term"])   # it's the offset
    assert fit["coef"].set_index("term").loc["own[axis0]", "z"] > 3


def test_fit_death_nb_defaults_to_full_apoptosis_signature():
    from tissue_dynamics.gene_sets import APOPTOSIS_SIGNATURE

    a, _ = _division_adata(response_genes=list(APOPTOSIS_SIGNATURE))   # all present
    fit = fit_death_nb(a, predictors="own", covariate_keys=None)
    assert set(fit["genes"]) == set(APOPTOSIS_SIGNATURE)


def test_division_design_neighbour_contrast_uses_phi_bar_minus_phi():
    a, _ = _division_adata()
    X, names, meta, _nuis, _cn = _division_design(
        a, "own+neighbor", 0, None, False, neighbour_contrast=True)
    # neighbour columns are phi_bar - phi and labelled nbr_dev[...]
    nbr_cols = [i for i, m in enumerate(meta) if m["kind"] == "nbr"]
    np.testing.assert_allclose(X[:, nbr_cols], a.obsm["phi_bar"] - a.obsm["phi"])
    assert all(names[i].startswith("nbr_dev[") for i in nbr_cols)


def test_fit_division_nb_neighbour_contrast_is_exact_reparam_of_main_effects():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.7, seed=4)
    kw = dict(genes=div_genes, predictors="own+neighbor", covariate_keys=("cell_area",))
    raw = fit_division_nb(a, **kw)
    con = fit_division_nb(a, neighbour_contrast=True, **kw)

    # same column span -> identical fit (log-likelihood, dispersion, params)
    assert con["loglik"] == pytest.approx(raw["loglik"], rel=1e-5)
    assert con["n_params"] == raw["n_params"]
    rc = raw["coef"].set_index("term")["estimate"]
    cc = con["coef"].set_index("term")["estimate"]
    # neighbour-deviation coefficient equals the raw neighbour coefficient...
    np.testing.assert_allclose(cc["nbr_dev[axis0]"], rc["nbr[axis0]"], rtol=1e-3, atol=1e-3)
    # ...and the own coefficient absorbs it: own_contrast = own_raw + nbr_raw
    np.testing.assert_allclose(cc["own[axis0]"], rc["own[axis0]"] + rc["nbr[axis0]"],
                               rtol=1e-3, atol=1e-3)


def test_fit_division_nb_neighbour_contrast_decorrelates_blocks():
    a, _ = _division_adata(seed=1)
    # spatial autocorrelation: neighbourhood = own state + a local deviation
    rng = np.random.default_rng(0)
    a.obsm["phi_bar"] = a.obsm["phi"] + 0.35 * rng.normal(size=a.obsm["phi"].shape)
    own = a.obsm["phi"][:, 0]
    raw_corr = abs(np.corrcoef(own, a.obsm["phi_bar"][:, 0])[0, 1])
    con_corr = abs(np.corrcoef(own, a.obsm["phi_bar"][:, 0] - own)[0, 1])
    assert raw_corr > 0.8                                 # own vs raw neighbour: collinear
    assert con_corr < 0.2                                 # own vs deviation: decorrelated


def test_stepwise_division_nb_prunes_nulls_and_keeps_signal():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.8, beta_int=0.0, seed=0)   # only own[axis0] real
    res = stepwise_division_nb(a, genes=div_genes, criterion="bic")

    full_params = 1 + 2 + 2 + 4 + 2                       # intercept + own + nbr + int + cov
    assert res["n_params"] < full_params                 # something was pruned
    assert res["criterion"] == "bic"
    assert "own[axis0]" in res["selected_terms"]          # the real effect survives
    # covariates are protected -> always retained (even the noise one)
    assert {"log_total_counts", "local_density"} <= set(res["selected_terms"])
    # path is a full trace from the full model down
    assert set(res["path"].columns) == {"step", "dropped", "n_terms", "n_params", "loglik", "aic", "bic"}
    assert res["path"].iloc[0]["dropped"] is None
    assert res["path"]["bic"].iloc[-1] <= res["path"]["bic"].iloc[0]   # BIC improved (or equal)


def test_stepwise_division_nb_respects_hierarchy():
    pytest.importorskip("statsmodels")
    a, div_genes = _division_adata(beta_own=0.5, beta_int=0.8, seed=5)  # real interaction present
    res = stepwise_division_nb(a, genes=div_genes, criterion="bic", respect_hierarchy=True)
    selected = set(res["selected_terms"])
    for t in selected:
        if t.startswith("int["):
            own_t, nbr_t = t[len("int["):-1].split(" x ")
            assert f"own[{own_t}]" in selected            # marginality: main effects kept
            assert f"nbr[{nbr_t}]" in selected


def test_stepwise_division_nb_rejects_bad_criterion():
    a, div_genes = _division_adata()
    with pytest.raises(ValueError, match="criterion must be"):
        stepwise_division_nb(a, genes=div_genes, criterion="rsquared")


def _vardisp_data(n=40000, g1=1.2, seed=0):
    """NB counts whose dispersion depends on state axis 0: log alpha = g0 + g1*phi0."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, 2))
    mu = np.exp(-0.3 + 0.7 * phi[:, 0] + 0.4 * phi[:, 1])
    alpha = np.exp(-0.5 + g1 * phi[:, 0])
    r = 1.0 / alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    X = np.column_stack([np.ones(n), phi])          # design incl. intercept
    Wd = np.column_stack([np.ones(n), phi])         # dispersion design incl. intercept
    return y, X, Wd, alpha


def test_nb_dispersion_fit_recovers_state_dependent_dispersion():
    from tissue_dynamics.count_glm import _nb_dispersion_fit
    y, X, Wd, alpha_true = _vardisp_data(g1=1.2)
    beta, gamma, alpha, ll, gse, nit = _nb_dispersion_fit(y, X, Wd)
    assert gamma[1] == pytest.approx(1.2, abs=0.12)      # the true dispersion slope
    assert abs(gamma[2]) < 0.12                          # axis 1 does not drive dispersion
    assert gamma[0] == pytest.approx(-0.5, abs=0.15)     # intercept
    assert np.corrcoef(alpha, alpha_true)[0, 1] > 0.98   # per-cell dispersion recovered
    assert beta[1] == pytest.approx(0.7, abs=0.08)       # mean model still right
    assert gse.shape == gamma.shape and (gse > 0).all()


def test_nb_dispersion_fit_beats_constant_alpha_when_dispersion_varies():
    from tissue_dynamics.count_glm import _nb2_irls, _nb_dispersion_fit
    y, X, Wd, _a = _vardisp_data(g1=1.2, seed=3)
    _b, _g, _al, ll_var, _se, _it = _nb_dispersion_fit(y, X, Wd)
    _b0, mu0 = __import__("tissue_dynamics.count_glm", fromlist=["_poisson_irls"])._poisson_irls(y, X, None)
    a_const = __import__("tissue_dynamics.count_glm", fromlist=["_estimate_nb_alpha"])._estimate_nb_alpha(y, mu0)
    _b2, _z2, _se2, ll_const = _nb2_irls(y, X, a_const)
    assert ll_var > ll_const + 50                        # a real gain, not noise


def test_nb_dispersion_fit_returns_flat_gamma_when_dispersion_is_constant():
    from tissue_dynamics.count_glm import _nb_dispersion_fit
    y, X, Wd, _a = _vardisp_data(g1=0.0, seed=7)         # dispersion does not vary
    _beta, gamma, alpha, _ll, _gse, _it = _nb_dispersion_fit(y, X, Wd)
    assert abs(gamma[1]) < 0.1 and abs(gamma[2]) < 0.1   # no spurious state dependence
    assert alpha.std() / alpha.mean() < 0.15             # essentially a constant alpha


def test_fit_division_nb_state_dependent_dispersion_api():
    a, div_genes = _division_adata(beta_own=0.8, n=6000)
    fit = fit_division_nb(a, genes=div_genes, predictors="own", fitter="irls",
                          dispersion_predictors="own",
                          covariate_keys=("log_total_counts",))
    d = fit["dispersion"]
    assert d["predictors"] == "own"
    assert np.ndim(fit["alpha"]) == 1 and len(fit["alpha"]) == fit["n"]   # per-cell now
    assert d["coef"].shape[0] == len(d["gamma"]) == a.obsm["phi"].shape[1] + 1
    assert set(d["lr_vs_constant_alpha"]) == {"statistic", "df", "p_value"}
    assert d["lr_vs_constant_alpha"]["df"] == a.obsm["phi"].shape[1]
    assert d["loglik"] >= d["loglik_constant_alpha"] - 1e-6      # joint MLE never loses
    # residuals use the per-cell dispersion, and stay finite
    assert np.isfinite(fit["pearson_resid"][fit["keep"]]).all()


def test_fit_division_nb_dispersion_predictors_validated():
    a, div_genes = _division_adata(n=500)
    with pytest.raises(ValueError, match="dispersion_predictors"):
        fit_division_nb(a, genes=div_genes, dispersion_predictors="bogus")
    with pytest.raises(ValueError, match="family='nb'"):
        fit_division_nb(a, genes=div_genes, dispersion_predictors="own", family="poisson")


def test_fit_division_nb_dispersion_exposes_constant_alpha_baseline():
    a, div_genes = _division_adata(beta_own=0.8, n=4000)
    fit = fit_division_nb(a, genes=div_genes, predictors="own", fitter="irls",
                          dispersion_predictors="own",
                          covariate_keys=("log_total_counts",))
    d = fit["dispersion"]
    mu_c = d["fitted_mu_constant_alpha"]
    assert mu_c.shape == (a.n_obs,)
    assert d["alpha_constant_alpha"] > 0
    assert np.isfinite(mu_c[fit["keep"]]).all()
    # the two mean models agree closely - only the dispersion differs
    assert np.corrcoef(mu_c[fit["keep"]], fit["fitted_mu"][fit["keep"]])[0, 1] > 0.99


def test_fit_division_nb_dispersion_terms_are_named_cell_types():
    a, div_genes = _division_adata(n=3000)
    a.uns["phi_names"] = ["T-cells", "CAFs"]
    fit = fit_division_nb(a, genes=div_genes, predictors="own", fitter="irls",
                          dispersion_predictors="own", covariate_keys=("log_total_counts",))
    assert list(fit["dispersion"]["terms"]) == ["const", "own[T-cells]", "own[CAFs]"]
