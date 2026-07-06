"""Shared helper for turning a continuous signature score into a boolean call."""

import numpy as np


def classify_bimodal(scores, method, threshold):
    """Classify a continuous per-cell score into a boolean "high" call.

    Parameters
    ----------
    scores : array-like
        Continuous per-cell signature score.
    method : "gmm" or "threshold"
        "gmm" fits a 2-component Gaussian mixture to the score distribution
        and calls the higher-scoring component "high" — a data-driven cut
        that adapts to each sample's score distribution.
        "threshold" calls a cell "high" if its score exceeds `threshold`.
    threshold : float
        Cutoff used when `method="threshold"`; ignored for `"gmm"`.
    """
    if method == "threshold":
        return np.asarray(scores) > threshold
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        values = np.asarray(scores).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(values)
        high_component = np.argmax(gmm.means_.ravel())
        labels = gmm.predict(values)
        return labels == high_component
    raise ValueError(f"Unknown method {method!r}; expected 'threshold' or 'gmm'")
