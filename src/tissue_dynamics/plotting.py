"""Matplotlib figures for the fate model: spatial fate maps, training loss,
coefficient forest plots, interaction modes, and tile-level diagnostics."""

import numpy as np


def plot_binned_conditional_mean(binned, ax=None, figsize=(6, 4), color=None,
                                  xlabel="predictor", ylabel="observed fraction",
                                  title="Binned conditional mean"):
    """Plot observed outcome vs binned predictor with error bars.

    Takes the DataFrame from `tiles.binned_conditional_mean` (columns
    `x_mean`, `y_mean`, `y_se`) and draws the conditional-mean curve — the
    OSDR sanity check. A monotone curve spanning a wide outcome range is the
    neighborhood signal, visible even when per-cell accuracy is near the
    majority baseline.

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.errorbar(binned["x_mean"], binned["y_mean"], yerr=binned["y_se"],
                fmt="o-", capsize=3, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_tile_field(tiles, value, ax=None, figsize=(6, 5), cmap="viridis",
                     size=12, invert_yaxis=True, title=None):
    """Spatial heat-scatter of a per-tile quantity across the slide.

    `tiles` is the DataFrame from `tiles.aggregate_tiles` (with `x`, `y` tile
    centroids); `value` is a column name to color by. A quick way to see
    whether e.g. tile division fraction or tile MKI67 rate is spatially
    structured (concentrated at a margin) rather than uniform.

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sc = ax.scatter(tiles["x"], tiles["y"], c=tiles[value], s=size, cmap=cmap, linewidths=0)
    fig.colorbar(sc, ax=ax, label=value)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or value)
    if invert_yaxis:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_training_loss(history, ax=None, logy=False, figsize=(6, 4), title="Training loss"):
    """Plot per-epoch training loss from `fit_fate_model`.

    Parameters
    ----------
    history : list of float, or dict of {label: list of float}
        A single loss history (as returned by `fit_fate_model`), or a dict
        mapping run labels to histories to overlay several runs on one axis
        (e.g. `{"unweighted": h1, "balanced": h2}`). NB: losses from runs
        with different `class_weight` are weighted-mean losses on different
        scales and aren't directly comparable in absolute terms — overlaying
        them shows convergence shape, not which run is "lower".
    logy : bool
        Log-scale the y-axis, useful when most of the drop is in the first
        few epochs.

    Returns
    -------
    (fig, ax) — the matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    histories = history if isinstance(history, dict) else {None: history}
    for label, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, marker=".", markersize=3, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if any(label is not None for label in histories):
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_coefficients(stats, top_n=20, by="abs_z", class_name=None,
                       figsize=None, title="Fate-model coefficients (95% CI)"):
    """Coefficient (forest) plot with 95% Wald confidence intervals.

    Visualizes the output of `inference.fate_parameter_stats`: each selected
    parameter's estimate as a point with a `estimate ± 1.96 * std_error`
    interval, so terms whose interval excludes zero (the dashed line) are the
    ones significant at the 5% Wald level. Shows the `top_n` terms with the
    largest `|z|` by default — with a big panel of coefficients (especially
    the bilinear B block) plotting all of them is unreadable.

    Parameters
    ----------
    stats : pandas.DataFrame
        As returned by `fate_parameter_stats`.
    by : "abs_z" or "estimate"
        Rank terms for selection by absolute z-statistic (default) or by
        absolute estimate.
    class_name : str, optional
        Restrict to one free class; otherwise all free classes are shown.

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    df = stats if class_name is None else stats[stats["class"] == class_name]
    key = df["z"].abs() if by == "abs_z" else df["estimate"].abs()
    df = df.assign(_rank=key).sort_values("_rank", ascending=False).head(top_n)
    df = df.iloc[::-1]  # largest at top when drawn bottom-up

    labels = [f"{c}: {t}" for c, t in zip(df["class"], df["term"])]
    estimate = df["estimate"].to_numpy()
    half_ci = 1.96 * df["std_error"].to_numpy()
    y = np.arange(len(df))

    if figsize is None:
        figsize = (7, max(3, 0.35 * len(df)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1, zorder=1)
    ax.errorbar(estimate, y, xerr=half_ci, fmt="o", markersize=4,
                capsize=3, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("coefficient estimate")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_interaction_modes(modes, class_name=None, n_modes=2, top_features=None, figsize=None):
    """Plot the own- and neighbor-state loadings of each interaction mode.

    Visualizes the output of `inference.interaction_modes`: for each selected
    (class, mode) it draws two horizontal bar charts side by side — the
    own-state loadings `p_j` and the neighbor-state loadings `q_j` over the
    embedding features — so you can read off which own cell-state, in the
    presence of which neighbor cell-state, shifts fate along that mode. Each
    row's title carries the mode's singular value (its strength).

    Parameters
    ----------
    modes : pandas.DataFrame
        As returned by `interaction_modes`.
    class_name : str, optional
        Restrict to one free class.
    n_modes : int
        Number of modes (per class) to show, strongest first.
    top_features : int, optional
        Show only the `top_features` features with the largest combined
        |own|+|neighbor| loading in each mode (useful when the embedding has
        many dimensions); default shows all.

    Returns
    -------
    (fig, axes).
    """
    import matplotlib.pyplot as plt

    df = modes if class_name is None else modes[modes["class"] == class_name]
    pairs = (df[["class", "mode", "singular_value"]]
             .drop_duplicates()
             .sort_values(["class", "mode"]))
    pairs = pairs[pairs["mode"] < n_modes]

    n_rows = len(pairs)
    if figsize is None:
        figsize = (9, max(2.2, 2.2 * n_rows))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, squeeze=False)

    for row, (_, pair) in enumerate(pairs.iterrows()):
        sub = df[(df["class"] == pair["class"]) & (df["mode"] == pair["mode"])].copy()
        if top_features is not None:
            sub["_mag"] = sub["own_loading"].abs() + sub["neighbor_loading"].abs()
            sub = sub.nlargest(top_features, "_mag")
        sub = sub.iloc[::-1]  # largest at top
        y = np.arange(len(sub))

        for col, (which, colname) in enumerate([("own", "own_loading"), ("neighbor", "neighbor_loading")]):
            ax = axes[row, col]
            ax.axvline(0.0, color="0.5", linewidth=0.8, zorder=1)
            ax.barh(y, sub[colname].to_numpy(), zorder=2)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["feature"], fontsize=7)
            ax.set_title(f"{pair['class']} mode {int(pair['mode'])} "
                         f"(σ={pair['singular_value']:.2f}): {which} loading", fontsize=9)

    fig.tight_layout()
    return fig, axes


def _resolve_sample(adata, sample_id, sample_key):
    if sample_key not in adata.obs.columns:
        return adata
    unique_samples = adata.obs[sample_key].unique()
    if sample_id is not None:
        return adata[adata.obs[sample_key] == sample_id]
    if len(unique_samples) > 1:
        raise ValueError(
            f"adata has {len(unique_samples)} distinct {sample_key!r} values "
            f"({list(unique_samples)}); pass sample_id to pick one to plot."
        )
    return adata


def plot_fate_spatial(adata, sample_id=None, class_names=("divide", "die", "unchanged"),
                        ground_truth_key="fate_label", proba_key="fate_proba",
                        sample_key="sample_id", spatial_key="spatial",
                        point_size=4, background_color="0.85", colors=None,
                        invert_yaxis=True, figsize=None):
    """Plot each fate class's spatial distribution, ground truth vs predicted.

    Produces a 2 x len(class_names) grid of scatter plots in physical
    (`obsm[spatial_key]`) coordinates: top row is the ground-truth signature
    call (`obs[ground_truth_key]`, from `targets.build_fate_targets`), bottom
    row is the model's predicted class (argmax of `obsm[proba_key]`, from
    `models.predict_fate_proba` / `pipeline.run_static_pipeline`). Each panel
    highlights only that column's class in color against every other cell in
    `background_color` — fate classes are typically imbalanced (e.g. mostly
    "unchanged"), so a single blended categorical plot would bury the
    minority classes' spatial pattern that this is meant to show.

    Parameters
    ----------
    sample_id : optional
        Which `obs[sample_key]` section to plot. Required if `adata` holds
        more than one section; ignored (whole `adata` used) if `sample_key`
        isn't present or there's only one section.
    invert_yaxis : bool
        Flip the y-axis so it matches the usual image-coordinate convention
        (y increasing downward) that Xenium/Visium centroids are recorded
        in. Doesn't affect the data, only the plot orientation.

    Returns
    -------
    (fig, axes) — the matplotlib Figure and its (2, len(class_names)) Axes array.
    Each panel's title reports how many cells fall in that class; the
    overall accuracy (argmax prediction vs ground truth) is in the suptitle.
    """
    import matplotlib.pyplot as plt

    if ground_truth_key not in adata.obs.columns:
        raise ValueError(f"adata.obs[{ground_truth_key!r}] not found; run build_fate_targets first")
    if proba_key not in adata.obsm:
        raise ValueError(f"adata.obsm[{proba_key!r}] not found; run predict_fate_proba first")
    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm[{spatial_key!r}] not found")

    sub = _resolve_sample(adata, sample_id, sample_key)

    coords = np.asarray(sub.obsm[spatial_key])
    truth = sub.obs[ground_truth_key].astype(str).to_numpy()
    proba = np.asarray(sub.obsm[proba_key])
    pred = np.array(class_names)[proba.argmax(axis=1)]

    if colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(class_names)}

    n_classes = len(class_names)
    if figsize is None:
        figsize = (3.2 * n_classes, 6.4)
    fig, axes = plt.subplots(2, n_classes, figsize=figsize, squeeze=False)

    accuracy = float((pred == truth).mean())
    sample_label = f" ({sample_id})" if sample_id is not None else ""
    fig.suptitle(f"Cell fate: ground truth vs predicted{sample_label} — overall accuracy {accuracy:.1%}")

    for row, (row_name, labels) in enumerate([("Ground truth", truth), ("Predicted", pred)]):
        for col, class_name in enumerate(class_names):
            ax = axes[row, col]
            is_class = labels == class_name

            ax.scatter(coords[~is_class, 0], coords[~is_class, 1],
                       s=point_size, c=background_color, linewidths=0, zorder=1)
            ax.scatter(coords[is_class, 0], coords[is_class, 1],
                       s=point_size, c=colors[class_name], linewidths=0, zorder=2)

            ax.set_title(f"{row_name}: {class_name} (n={int(is_class.sum())})", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if invert_yaxis:
                ax.invert_yaxis()

    fig.tight_layout()
    return fig, axes


def _count_pmf_cdf(mu, alpha):
    """Return `(pmf, cdf)` closures for the fitted count distribution.

    NB2 (`alpha` > 0) parameterized for scipy as `size = 1/alpha`,
    `prob = size/(size+mu)`; Poisson when `alpha` is None or 0. Each closure
    maps an integer count (scalar or array) to the per-cell probability/CDF
    evaluated at each cell's fitted mean `mu`.
    """
    from scipy import stats as sps

    mu = np.asarray(mu, dtype=float)
    if alpha is None or (np.isscalar(alpha) and alpha == 0):
        return (lambda k: sps.poisson.pmf(k, mu)), (lambda k: sps.poisson.cdf(k, mu))
    # alpha may be a scalar or a per-observation vector (state-dependent dispersion)
    r = 1.0 / np.asarray(alpha, dtype=float)
    p = r / (r + mu)
    return (lambda k: sps.nbinom.pmf(k, r, p)), (lambda k: sps.nbinom.cdf(k, r, p))


def _rootogram_counts(y, mu, alpha, max_count=None):
    """Observed and model-expected count frequencies for a rootogram.

    Returns `(ks, observed, expected)` over counts `0..max_count`, where
    `expected[k] = sum_i P(Y_i = k | mu_i, alpha)` (the fitted count
    distribution summed over cells) and `observed[k]` is the number of cells
    with `y == k`. Non-finite `y`/`mu` pairs are dropped.
    """
    y = np.asarray(y)
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    y = y[ok].astype(int)
    mu = mu[ok]
    if max_count is None:
        max_count = int(max(1, np.ceil(np.quantile(y, 0.999))))
    ks = np.arange(0, max_count + 1)
    pmf, _ = _count_pmf_cdf(mu, alpha)
    expected = np.array([pmf(int(k)).sum() for k in ks])
    observed = np.array([(y == k).sum() for k in ks], dtype=float)
    return ks, observed, expected


def _pit_histogram(y, mu, alpha, n_bins=10):
    """Non-randomized PIT histogram heights (Czado, Gneiting & Held 2009).

    Returns `(edges, heights)` for `n_bins` equal-width bins on [0, 1]. The
    non-randomized PIT averages the conditional CDF `\\bar F(u)` across cells;
    `heights = diff(\\bar F(edges)) * n_bins`, so a calibrated model gives
    heights ~ 1. Non-finite `y`/`mu` pairs are dropped.
    """
    y = np.asarray(y)
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    y = y[ok].astype(int)
    mu = mu[ok]
    _, cdf = _count_pmf_cdf(mu, alpha)
    F_y = cdf(y)
    F_y1 = np.where(y > 0, cdf(y - 1), 0.0)
    denom = np.clip(F_y - F_y1, 1e-12, None)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    f_bar = np.array([np.mean(np.clip((u - F_y1) / denom, 0.0, 1.0)) for u in edges])
    heights = np.diff(f_bar) * n_bins
    return edges, heights


def plot_rootogram(y, mu, alpha, ax=None, figsize=(7, 4), max_count=None,
                    color="C0", expected_color="C3", title="Hanging rootogram (NB fit)"):
    """Hanging rootogram of observed counts against the fitted NB distribution.

    Each bar (observed sqrt-frequency of a count) hangs from the curve of
    expected sqrt-frequencies `sqrt(sum_i P(Y_i = k))`: a bar whose foot dips
    *below* the zero line means the model under-predicts that count, one that
    stops *above* zero means it over-predicts. The sqrt scale keeps the rare
    high counts legible. `alpha` is the NB2 dispersion (None/0 -> Poisson); it may
    be a scalar or a per-cell vector, so a state-dependent-dispersion fit
    (`fit_division_nb(..., dispersion_predictors=...)`) can be passed straight in.
    Counts are shown up to `max_count` (default: the 99.9th observed percentile).

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    ks, observed, expected = _rootogram_counts(y, mu, alpha, max_count=max_count)
    se, so = np.sqrt(expected), np.sqrt(observed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.bar(ks, so, bottom=se - so, width=0.8, color=color, alpha=0.7,
           label="observed (hung from fit)")
    ax.plot(ks, se, "o-", color=expected_color, label="expected (NB fit)")
    ax.axhline(0.0, color="0.3", lw=1)
    ax.set_xlabel("count (cell-cycle transcripts per cell)")
    ax.set_ylabel(r"$\sqrt{\mathrm{frequency}}$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_pit_histogram(y, mu, alpha, ax=None, figsize=(7, 4), n_bins=10,
                        color="C0", title="Non-randomized PIT histogram (NB fit)"):
    """Non-randomized PIT histogram — a calibration check for the fitted count model.

    Bars near the uniform reference (dashed line at 1) mean the predictive
    distribution is well calibrated. A U-shape means it is too narrow
    (under-dispersed — real counts land in the tails more often than predicted),
    a central hump means it is too wide (over-dispersed), and a tilt means the
    mean is biased. `alpha` is the NB2 dispersion (None/0 -> Poisson).

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    edges, heights = _pit_histogram(y, mu, alpha, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.bar(edges[:-1], heights, width=1.0 / n_bins, align="edge", color=color,
           alpha=0.7, edgecolor="white")
    ax.axhline(1.0, color="C3", ls="--", label="uniform (calibrated)")
    ax.set_xlabel("probability integral transform")
    ax.set_ylabel("relative frequency")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_annulus_profile(summary, coef=None, ax=None, figsize=(7.5, 4.6),
                          c_col="c_mean_abs", rho_col="rho_phi",
                          title="Neighbour effect vs Phi-Space autocorrelation by ring"):
    """Compare the neighbour-effect profile across annuli with the spillover null.

    Plots the ring-by-ring neighbour coefficient size (`summary[c_col]`) against
    the Phi-Space correlogram (`summary[rho_col]`), each normalized to its value
    in the innermost ring, versus the ring midpoint. Under pure segmentation
    spillover the two curves coincide (`c(r) ∝ c(0)·rho_phi(r)`); a neighbour
    curve that stays above the autocorrelation curve is effect that spillover
    cannot explain. See `annulus.annulus_scan` / `annulus.spillover_chi2`.

    Returns
    -------
    (fig, ax).
    """
    import matplotlib.pyplot as plt

    mid = (summary["r_inner"].to_numpy() + summary["r_outer"].to_numpy()) / 2.0
    c = summary[c_col].to_numpy(dtype=float)
    rho = summary[rho_col].to_numpy(dtype=float)
    c_n = c / c[0] if c[0] else c
    rho_n = rho / rho[0] if rho[0] else rho

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(mid, c_n, "o-", color="C0", label=f"neighbour effect ({c_col}, normalized)")
    ax.plot(mid, rho_n, "s--", color="C3", label=r"spillover null: $\rho_\phi(r)$ (normalized)")
    ax.axhline(0.0, color="0.6", lw=1)
    ax.set_xlabel(r"ring midpoint distance ($\mu$m)")
    ax.set_ylabel("value relative to innermost ring")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
