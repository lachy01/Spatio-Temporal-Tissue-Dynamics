"""End-to-end convenience entrypoints: read/score a sample, and run the full static fate pipeline."""

import warnings

from .cellcycle import MIN_GENES_PER_PHASE, score_cell_cycle, score_proliferation
from .death import score_death
from .gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES, match_gene_set
from .io_phi import attach_phi, read_phi_table
from .io_xenium import read_xenium
from .models import SpatialMultinomialFateModel, fit_fate_model, predict_fate_proba
from .neighbors import average_neighbors, build_adjacency
from .targets import build_fate_targets


def analyze_sample(path, method="auto", **score_kwargs):
    """Read a Xenium sample and score its cells for cell-cycle activity.

    Parameters
    ----------
    path : str or Path
        Sample root directory or its `outs/` folder.
    method : "auto", "cell_cycle", or "proliferation"
        "cell_cycle" runs `score_cell_cycle` (S/G2M/G1 phase assignment).
        "proliferation" runs `score_proliferation` (single signature score).
        "auto" picks "cell_cycle" if the panel covers enough of the
        canonical S/G2M gene sets to trust the phase split, otherwise falls
        back to "proliferation".
    **score_kwargs
        Forwarded to whichever scoring function is used.

    Returns
    -------
    Scored AnnData (see `score_cell_cycle` / `score_proliferation` for the
    obs columns this adds).
    """
    adata = read_xenium(path)

    if method == "auto":
        s_present = match_gene_set(TIROSH_S_GENES, adata.var_names)
        g2m_present = match_gene_set(TIROSH_G2M_GENES, adata.var_names)
        if len(s_present) >= MIN_GENES_PER_PHASE and len(g2m_present) >= MIN_GENES_PER_PHASE:
            method = "cell_cycle"
        else:
            warnings.warn(
                f"Panel only covers {len(s_present)} S-phase / {len(g2m_present)} "
                "G2M-phase reference genes; falling back to a single "
                "proliferation score instead of S/G2M phase assignment.",
                stacklevel=2,
            )
            method = "proliferation"

    if method == "cell_cycle":
        return score_cell_cycle(adata, **score_kwargs)
    if method == "proliferation":
        return score_proliferation(adata, **score_kwargs)
    raise ValueError(f"Unknown method {method!r}; expected 'auto', 'cell_cycle', or 'proliferation'")


def run_static_pipeline(adata, phi_path, *, radius=80.0, sample_key="sample_id",
                          target_mode="hard", class_names=("divide", "die", "unchanged"),
                          cell_id_col="cell_id", sample_col=None, phi_prefix="phi_",
                          proliferation_kwargs=None, death_kwargs=None,
                          target_kwargs=None, fit_kwargs=None):
    """Run the full static fate-inference pipeline on an already-loaded AnnData.

    Wires together Phi-Space attachment, the spatial adjacency graph,
    neighbor averaging, division/death signature scoring, fate-target
    construction, and model fitting. Each step is also usable standalone
    (`io_phi`, `neighbors`, `cellcycle`/`death`, `targets`, `models`); this
    is the "fit on everything" convenience entrypoint — for cross-validated
    evaluation, use `evaluate.spatial_group_kfold` around the individual
    steps instead.

    Parameters
    ----------
    adata : AnnData
        Already read (e.g. via `read_xenium`), with `obs["cell_id"]` and
        `obs[sample_key]` set.
    phi_path : str or Path
        Phi-Space embedding table (see `io_phi.read_phi_table`).
    target_mode : "hard" or "soft"
        Forwarded to `targets.build_fate_targets`; determines whether the
        model is fit against discrete class codes or soft target
        distributions.
    proliferation_kwargs, death_kwargs, target_kwargs, fit_kwargs : dict, optional
        Forwarded to `score_proliferation`, `score_death`,
        `build_fate_targets`, and `fit_fate_model` respectively.

    Returns
    -------
    (adata, model, history) — `adata` modified in place with all
    intermediate results attached (`obsm["phi"]`, `obsm["phi_bar"]`,
    `obsp["spatial_connectivity"]`, signature scores, fate targets,
    `obsm["fate_proba"]`); `model` the fitted `SpatialMultinomialFateModel`;
    `history` the per-epoch training loss from `fit_fate_model`.
    """
    phi = read_phi_table(phi_path, cell_id_col=cell_id_col, sample_col=sample_col, phi_prefix=phi_prefix)
    attach_phi(adata, phi, cell_id_col=cell_id_col, sample_col=sample_col, sample_key=sample_key)

    build_adjacency(adata, radius=radius, sample_key=sample_key)
    average_neighbors(adata)

    score_proliferation(adata, **(proliferation_kwargs or {}))
    score_death(adata, **(death_kwargs or {}))

    build_fate_targets(adata, mode=target_mode, class_names=class_names, sample_key=sample_key,
                        **(target_kwargs or {}))

    model = SpatialMultinomialFateModel(phi_dim=adata.obsm["phi"].shape[1], class_names=class_names)
    target = adata.obs["fate_code"].to_numpy() if target_mode == "hard" else adata.obsm["fate_target"]
    history = fit_fate_model(model, adata.obsm["phi"], adata.obsm["phi_bar"], target, **(fit_kwargs or {}))

    adata.obsm["fate_proba"] = predict_fate_proba(model, adata.obsm["phi"], adata.obsm["phi_bar"])
    return adata, model, history
