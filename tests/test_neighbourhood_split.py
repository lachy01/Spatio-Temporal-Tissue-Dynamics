"""The contact / regional neighbourhood split, and the portrait's view of it."""
import copy

import numpy as np
import pandas as pd
import pytest

from tissue_dynamics.mixture.data import load_config, split_neighbourhood


def _compositional(rng, n, d=9, total=-7.0):
    X = rng.normal(0.0, 1.0, (n, d))
    return X - X.mean(1, keepdims=True) + total / d


def test_split_recombines_to_the_disc():
    """Contact (0-15 um) + regional (15-80 um) means recombine to the 80 um
    disc mean that build_adjacency + average_neighbors produce, over the same
    typed pool; gaps are filled as documented."""
    from sklearn.neighbors import radius_neighbors_graph

    rng = np.random.default_rng(0)
    n = 3000
    xy = rng.uniform(0, 600, (n, 2))
    phi = _compositional(rng, n)
    pool = rng.uniform(size=n) < 0.8
    contact, n_c, regional, n_r = split_neighbourhood(xy, phi, pool, 15.0, 80.0, chunk=700)

    A = radius_neighbors_graph(xy, 80.0, mode="connectivity", include_self=False)
    A = A[:, np.flatnonzero(pool)]
    deg = np.asarray(A.sum(1)).ravel()
    disc = np.asarray(A @ phi[pool]) / np.where(deg > 0, deg, 1)[:, None]
    np.testing.assert_array_equal(n_c + n_r, deg)
    has = deg > 0
    recon = (n_c[has, None] * contact[has] + n_r[has, None] * regional[has]) / deg[has, None]
    np.testing.assert_allclose(recon, disc[has], atol=1e-12)
    # no contact neighbour -> the contact mean is the regional one
    gap = (n_c == 0) & (n_r > 0)
    assert gap.any()
    np.testing.assert_allclose(contact[gap], regional[gap])


@pytest.mark.parametrize("neighbourhood, whiten", [("split", True), ("disc", True),
                                                   ("disc", False)])
def test_portrait_fold_matches_the_weight_model(neighbourhood, whiten):
    """With contact and regional compositions equal (the portrait's mean-field
    closure), PortraitModel.from_fit reproduces the fit's own logit(pi) exactly,
    whatever the whitening and the neighbour block's layout."""
    import jax.numpy as jnp

    from tissue_dynamics.mixture.basis import build_basis, spec_from_basis
    from tissue_dynamics.mixture.model import _logit_pi, build_data, init_params
    from tissue_dynamics.portrait.core import PortraitModel

    rng = np.random.default_rng(1)
    n, d = 4000, 9

    def table(mean_field):
        phi = _compositional(rng, n)
        blocks = {pre: _compositional(rng, n) for pre in ("nbr_phi_", "nbrc_phi_", "nbrr_phi_")}
        if mean_field:                       # every neighbourhood has the same composition
            blocks = {pre: blocks["nbr_phi_"] for pre in blocks}
        df = pd.DataFrame({f"phi_{j:02d}": phi[:, j] for j in range(d)})
        for pre, X in blocks.items():
            for j in range(d):
                df[f"{pre}{j:02d}"] = X[:, j]
        df["density"] = np.exp(rng.normal(1.0, 0.4, n))
        df["area"] = np.exp(rng.normal(4.0, 0.3, n))
        df["N"] = rng.integers(10, 500, n)
        df["y_count"] = rng.poisson(1.0, n)
        df.attrs["phi_names"] = [f"type{j}" for j in range(d)]
        return df

    cfg = copy.deepcopy(load_config())
    cfg["state"].update(neighbourhood=neighbourhood, whiten=whiten)
    # the basis is fitted on data whose blocks differ (as on the real table);
    # the comparison is made on cells satisfying the mean-field closure
    basis = build_basis(table(mean_field=False), cfg, verbose=False)
    df = table(mean_field=True)
    spec = spec_from_basis(basis, cfg, weight_ridge=1.0)
    data = build_data(df, basis, cfg, spec)
    p = init_params(spec, cfg, data=data, rng=np.random.default_rng(2))
    p = {k: (v + jnp.asarray(np.random.default_rng(3).normal(0, 0.3, np.shape(v)))
             if k in ("a", "c", "U", "V", "h", "K") else v) for k, v in p.items()}

    want = np.asarray(_logit_pi(p, data.as_jax(), spec))
    model = PortraitModel.from_fit({k: np.asarray(v) for k, v in p.items()}, basis)
    got = model.eta(basis.own(df), basis.disc_in_own(df), df["density"].to_numpy(), 0.0)
    np.testing.assert_allclose(got, want, atol=1e-10)
