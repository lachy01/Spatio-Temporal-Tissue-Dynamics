"""Spatial cross-validation and evaluation metrics for the fate model."""

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def spatial_group_kfold(adata, sample_key="sample_id", n_splits=None):
    """Yield (train_idx, test_idx) splits by whole sample/section.

    Never splits cells within one section into both train and test — this
    avoids the leakage a per-cell random split would introduce through the
    neighbor-averaged `phi_bar` term (a held-out cell's neighbors would
    otherwise include training-set cells from the same section). `phi_bar`
    itself is still legitimately computed once over all available cells
    before splitting, since it's an observable covariate available at
    prediction time in real use; only the train/test *evaluation* split
    needs to respect section boundaries.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds. Defaults to leave-one-section-out (one fold per
        unique `sample_key` value).
    """
    groups = adata.obs[sample_key].to_numpy()
    X = np.zeros((adata.n_obs, 1))

    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    yield from splitter.split(X, groups=groups)


def fate_metrics(target, proba, class_names=("divide", "die", "unchanged")):
    """Compute cross-entropy and accuracy for predicted class probabilities.

    `target` may be 1D integer class codes (hard labels) or a 2D array of
    per-class probabilities (soft targets); `proba` is (n, n_classes) as
    returned by `predict_fate_proba`.
    """
    target = np.asarray(target)
    proba = np.asarray(proba)
    log_proba = np.log(np.clip(proba, 1e-12, 1.0))

    if target.ndim == 1:
        true_dist = np.eye(len(class_names))[target]
        true_labels = target
    else:
        true_dist = target
        true_labels = target.argmax(axis=1)

    cross_entropy = float(-(true_dist * log_proba).sum(axis=1).mean())
    accuracy = float((proba.argmax(axis=1) == true_labels).mean())
    return {"cross_entropy": cross_entropy, "accuracy": accuracy, "class_names": list(class_names)}


def morans_i(values, coords=None, W=None, n_neighbors=8, permutations=999, seed=0):
    """Moran's I spatial autocorrelation of `values`.

    The standard test for whether a spatial field clusters: applied to model
    *residuals* it asks whether structure the model failed to explain is
    spatially patterned (a positive I means neighbouring residuals are alike —
    the model is missing something spatial). Provide either `coords` (n x 2; a
    row-standardized k-nearest-neighbour weight matrix is built) or a
    precomputed weight matrix `W`. Non-finite `values` (and their coords/rows)
    are dropped first.

    `I = (n / S0) * (z' W z) / (z' z)` with `z = values - mean` and `S0` the sum
    of weights; under no autocorrelation `E[I] = -1/(n-1)`. With `permutations`
    a one-sided p-value (for positive autocorrelation) and a z-score are
    computed against a permutation null (labels shuffled over locations).

    Returns a dict: `I`, `expected_I`, `n`, and (if `permutations` > 0)
    `p_value`, `z_score`, `permutations`.
    """
    from scipy import sparse

    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 finite values for Moran's I")

    if W is None:
        if coords is None:
            raise ValueError("provide either coords (n x 2) or a weight matrix W")
        from sklearn.neighbors import kneighbors_graph
        c = np.asarray(coords, dtype=float)[ok]
        W = kneighbors_graph(c, n_neighbors=min(n_neighbors, n - 1),
                             mode="connectivity", include_self=False)
    else:
        W = sparse.csr_matrix(W)[ok][:, ok]

    W = W.astype(float)
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    Wn = sparse.diags(1.0 / rowsum) @ W                   # row-standardized weights
    S0 = float(Wn.sum())

    z = x - x.mean()
    denom = float(z @ z)

    def moran(zv):
        return (n / S0) * float(zv @ (Wn @ zv)) / denom

    observed = moran(z)
    out = {"I": observed, "expected_I": -1.0 / (n - 1), "n": int(n)}
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        perm = np.array([moran(rng.permutation(z)) for _ in range(permutations)])
        out["p_value"] = float((np.sum(perm >= observed) + 1) / (permutations + 1))
        out["z_score"] = float((observed - perm.mean()) / (perm.std() + 1e-12))
        out["permutations"] = int(permutations)
    return out
