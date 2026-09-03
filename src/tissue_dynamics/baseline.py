"""A small MLP baseline to compare against the SpatialMultinomialFateModel.

The structured model predicts fate from a cell's own Phi-Space state, its
neighbors' averaged state, and a learned bilinear interaction between them.
This baseline asks a simpler question: how far does a generic, off-the-shelf
neural net get from the same inputs, with no bilinear structure imposed? It's
the reference point that tells you whether the structured model's inductive
bias is buying anything over "throw the features at an MLP".

`fit_mlp_baseline` wraps sklearn's `MLPClassifier` (standardizing inputs
first, since raw Phi-Space scores aren't on a common scale). The default
input is `phi_bar` alone — each cell's neighbors' Phi-Space phenotyping,
averaged — so the baseline predicts a cell's fate purely from its spatial
neighborhood's cell-state composition. Pass `input_features="both"` to also
give it the cell's own state (a fairer head-to-head with the structured
model, which uses both), or `"own"` for own-state only.
"""

import numpy as np


def build_mlp_features(phi, phi_bar, input_features="neighbors"):
    """Assemble the MLP input matrix from own/neighbor Phi-Space embeddings.

    input_features : "neighbors" | "both" | "own"
        Which embeddings to feed the MLP — `phi_bar` only (default),
        both `phi` and `phi_bar` concatenated, or `phi` only.
    """
    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    if input_features == "neighbors":
        return phi_bar
    if input_features == "own":
        return phi
    if input_features == "both":
        return np.concatenate([phi, phi_bar], axis=1)
    raise ValueError(f"Unknown input_features {input_features!r}; expected 'neighbors', 'both', or 'own'")


def fit_mlp_baseline(phi, phi_bar, target, input_features="neighbors",
                      hidden_layer_sizes=(32,), max_iter=300, random_state=0,
                      **mlp_kwargs):
    """Fit a small standardized MLP classifier to predict fate.

    Parameters
    ----------
    phi, phi_bar : arrays (n, D)
        Own-state and neighbor-averaged Phi-Space embeddings.
    target : array (n,)
        Integer class codes (e.g. `adata.obs["fate_code"]`). Soft targets
        aren't supported here — the baseline trains on hard labels.
    input_features : "neighbors" | "both" | "own"
        See `build_mlp_features`. Defaults to neighbors-only (`phi_bar`).
    hidden_layer_sizes, max_iter, random_state, **mlp_kwargs
        Forwarded to `sklearn.neural_network.MLPClassifier`.

    Returns
    -------
    A fitted `sklearn.pipeline.Pipeline` (StandardScaler -> MLPClassifier).
    Use `predict_mlp_proba(pipeline, phi, phi_bar, input_features)` to get
    per-class probabilities aligned with the model's class order.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = build_mlp_features(phi, phi_bar, input_features)
    target = np.asarray(target)
    if target.ndim != 1:
        raise ValueError("target must be 1D integer class codes; soft targets aren't supported by the MLP baseline")

    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
                       random_state=random_state, **mlp_kwargs),
    )
    pipeline.fit(x, target)
    pipeline.input_features_ = input_features  # remember for predict helper
    return pipeline


def predict_mlp_proba(pipeline, phi, phi_bar, input_features=None, n_classes=None):
    """Per-class probabilities from a fitted baseline, in full class order.

    sklearn only emits columns for classes seen in training; if a class was
    absent from the training split this reinserts its (zero) column so the
    output is always (n, n_classes) and aligned with the model's class order.
    `input_features` defaults to whatever `fit_mlp_baseline` was called with.
    """
    if input_features is None:
        input_features = getattr(pipeline, "input_features_", "neighbors")
    x = build_mlp_features(phi, phi_bar, input_features)
    proba = pipeline.predict_proba(x)

    classes = pipeline.classes_
    if n_classes is None:
        n_classes = int(classes.max()) + 1
    if proba.shape[1] == n_classes and list(classes) == list(range(n_classes)):
        return proba

    full = np.zeros((proba.shape[0], n_classes))
    full[:, classes] = proba
    return full
