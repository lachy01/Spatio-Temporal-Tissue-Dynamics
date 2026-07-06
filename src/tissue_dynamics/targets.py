"""Turn division/death signature scores into multinomial fate-model training targets."""

import numpy as np
import pandas as pd


def _zscore_by_group(series, groups):
    def z(x):
        std = x.std()
        if not std or pd.isna(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / std

    return series.groupby(groups).transform(z)


def build_fate_targets(adata, mode="hard", class_names=("divide", "die", "unchanged"),
                         baseline_weight=1.0, sample_key="sample_id",
                         proliferation_col="proliferation_score", death_col="death_score",
                         dividing_col="is_dividing", dying_col="is_dying",
                         target_key="fate_target", label_key="fate_label", code_key="fate_code"):
    """Build multinomial fate-model targets from division/death signature scores.

    `class_names` is `(divide_class, die_class, reference_class)` — the
    reference class (last entry) is the one fixed at logit 0 by
    `SpatialMultinomialFateModel` and represents "unchanged".

    Parameters
    ----------
    mode : "hard" or "soft"
        "hard" derives a discrete label per cell from the boolean
        `dividing_col`/`dying_col` calls (e.g. from `score_proliferation` /
        `score_death`); a rare cell flagged both dividing and dying is
        assigned to whichever raw score is more standard deviations above
        its per-`sample_key` mean. Sets `obs[label_key]` (str) and
        `obs[code_key]` (int, indexing into `class_names`).
        "soft" z-scores `proliferation_col`/`death_col` per `sample_key`,
        clips at 0 (no negative evidence), and normalizes
        `[divide+, die+, baseline_weight]` into a probability distribution.
        Sets `obsm[target_key]` (n_obs x 3, columns ordered as `class_names`).

    Returns
    -------
    `adata`, modified in place.
    """
    if len(class_names) != 3:
        raise ValueError("class_names must have exactly 3 entries: (divide_class, die_class, reference_class)")
    divide_name, die_name, ref_name = class_names

    groups = adata.obs[sample_key] if sample_key in adata.obs.columns else pd.Series(
        np.zeros(adata.n_obs), index=adata.obs_names,
    )
    prolif_z = _zscore_by_group(adata.obs[proliferation_col], groups).to_numpy()
    death_z = _zscore_by_group(adata.obs[death_col], groups).to_numpy()

    if mode == "hard":
        for col in (dividing_col, dying_col):
            if col not in adata.obs.columns:
                raise ValueError(f"adata.obs[{col!r}] not found; run score_proliferation/score_death first")

        is_dividing = adata.obs[dividing_col].to_numpy(dtype=bool)
        is_dying = adata.obs[dying_col].to_numpy(dtype=bool)
        conflict = is_dividing & is_dying

        labels = np.full(adata.n_obs, ref_name, dtype=object)
        labels[is_dividing & ~conflict] = divide_name
        labels[is_dying & ~conflict] = die_name
        prefer_divide = prolif_z >= death_z
        labels[conflict & prefer_divide] = divide_name
        labels[conflict & ~prefer_divide] = die_name

        adata.obs[label_key] = labels
        adata.obs[code_key] = pd.Categorical(labels, categories=list(class_names)).codes
        return adata

    if mode == "soft":
        prolif_pos = np.clip(prolif_z, 0, None)
        death_pos = np.clip(death_z, 0, None)
        baseline = np.full(adata.n_obs, float(baseline_weight))
        raw = np.stack([prolif_pos, death_pos, baseline], axis=1)
        adata.obsm[target_key] = raw / raw.sum(axis=1, keepdims=True)
        adata.uns[f"{target_key}_class_names"] = list(class_names)
        return adata

    raise ValueError(f"Unknown mode {mode!r}; expected 'hard' or 'soft'")
