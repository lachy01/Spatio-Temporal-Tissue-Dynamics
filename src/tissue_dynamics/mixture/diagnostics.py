"""Diagnostics that must accompany every reported fit (spec 8.7, 8.8, 10).

Spec 10 is explicit that a result reported without the entropy and the confound
R-squared "is not usable", so these are computed alongside the estimates rather
than left to the reader to request.
"""

from __future__ import annotations

import numpy as np


def _r2(X, y):
    """R^2 of an OLS fit of `y` on `[1, X]`."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    resid = np.asarray(y, dtype=float) - X @ beta
    return float(1.0 - np.var(resid) / max(np.var(y), 1e-300))


def confound_diagnostic(w, df, basis):
    """Spec 8.8: is `w` a cell-state readout, or a size-and-detection classifier?

    Regresses the responsibilities on the nuisance covariates and, separately,
    on the cell state, and returns both R-squared values. If the nuisance R^2
    substantially exceeds the state R^2, the model is classifying cells on how
    well they were captured rather than on what they are -- and the spec
    requires that number be reported prominently whatever it shows.
    """
    w = np.asarray(w, dtype=float)
    nuis = np.column_stack([
        np.log(df["area"].to_numpy(float)), np.log(df["N"].to_numpy(float)),
        np.log(df["density"].to_numpy(float)), df["oxygen"].to_numpy(float)])
    r_nuis, r_state = _r2(nuis, w), _r2(basis.own(df), w)
    per = {nm: float(np.corrcoef(v, w)[0, 1]) for nm, v in
           (("log_area", nuis[:, 0]), ("log_N", nuis[:, 1]),
            ("log_density", nuis[:, 2]), ("oxygen", nuis[:, 3]))}
    return {
        "confound_r2_nuisance": r_nuis,
        "confound_r2_state": r_state,
        "confound_ratio_state_over_nuisance": float(r_state / max(r_nuis, 1e-12)),
        "is_size_detection_classifier": bool(r_nuis > r_state),
        "confound_corr": per,
    }
