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
