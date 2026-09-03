"""Spatial cell-fate dynamics from spatial transcriptomics + Phi-Space embeddings.

Exports are resolved **lazily** (PEP 562). Importing this package used to pull in
torch, scanpy, anndata and matplotlib eagerly, which meant that anything reaching
the mixture code -- `tissue_dynamics.mixture.tasks`, the entry point for every
SLURM array task -- dragged the whole stack with it. A cluster environment then
had to carry ~2 GB of torch to run a model that never touches it.

`from tissue_dynamics import read_xenium` behaves exactly as before; the
submodule is imported on first attribute access instead of at package import.
"""

import importlib

_LAZY = {
    "DEFAULT_RINGS": "annulus",
    "OxygenFieldResult": "oxygen",
    "SpatialMultinomialFateModel": "models",
    "aggregate_tiles": "tiles",
    "analyze_sample": "pipeline",
    "annulus_neighbor_stats": "annulus",
    "annulus_scan": "annulus",
    "assign_tiles": "tiles",
    "attach_control_counts": "io_xenium",
    "attach_oxygen_field": "oxygen",
    "attach_phi": "io_phi",
    "average_neighbors": "neighbors",
    "binned_conditional_mean": "tiles",
    "build_adjacency": "neighbors",
    "build_covariate_matrix": "covariates",
    "build_fate_targets": "targets",
    "build_mlp_features": "baseline",
    "build_oxygen_field": "oxygen",
    "build_predictors": "count_glm",
    "calibrate_cutoff": "oxygen",
    "compare_count_glms": "count_glm",
    "compute_ambient_background": "covariates",
    "compute_library_size": "covariates",
    "compute_local_density": "covariates",
    "download_xenium_sample": "download",
    "ensure_lognormalized": "cellcycle",
    "fate_loss": "models",
    "fate_metrics": "evaluate",
    "fate_parameter_stats": "inference",
    "find_visium_path": "io_visium",
    "fit_count_glm": "count_glm",
    "fit_death_nb": "count_glm",
    "fit_division_nb": "count_glm",
    "fit_fate_model": "models",
    "fit_mlp_baseline": "baseline",
    "fit_neighbourhood_count_glm": "count_glm",
    "get_gene_counts": "count_glm",
    "get_signature_counts": "count_glm",
    "interaction_modes": "inference",
    "likelihood_ratio_test": "inference",
    "load_visium_samples": "io_visium",
    "load_xenium_samples": "io_xenium",
    "morans_i": "evaluate",
    "plot_annulus_profile": "plotting",
    "plot_binned_conditional_mean": "plotting",
    "plot_coefficients": "plotting",
    "plot_fate_spatial": "plotting",
    "plot_interaction_modes": "plotting",
    "plot_pit_histogram": "plotting",
    "plot_rootogram": "plotting",
    "plot_tile_field": "plotting",
    "plot_training_loss": "plotting",
    "predict_fate_proba": "models",
    "predict_mlp_proba": "baseline",
    "read_phi_table": "io_phi",
    "read_visium": "io_visium",
    "read_xenium": "io_xenium",
    "run_static_pipeline": "pipeline",
    "sample_at_cells": "oxygen",
    "score_cell_cycle": "cellcycle",
    "score_death": "death",
    "score_division": "cellcycle",
    "score_proliferation": "cellcycle",
    "solve_oxygen": "oxygen",
    "spatial_group_kfold": "evaluate",
    "spillover_chi2": "annulus",
    "stepwise_division_nb": "count_glm",
    "sweep_parameters": "oxygen",
}

__all__ = [
    "read_xenium",
    "load_xenium_samples",
    "attach_control_counts",
    "compute_ambient_background",
    "read_visium",
    "load_visium_samples",
    "find_visium_path",
    "download_xenium_sample",
    "ensure_lognormalized",
    "score_cell_cycle",
    "score_proliferation",
    "score_division",
    "score_death",
    "read_phi_table",
    "attach_phi",
    "build_adjacency",
    "average_neighbors",
    "build_fate_targets",
    "SpatialMultinomialFateModel",
    "fate_loss",
    "fit_fate_model",
    "predict_fate_proba",
    "spatial_group_kfold",
    "fate_metrics",
    "morans_i",
    "build_oxygen_field",
    "attach_oxygen_field",
    "OxygenFieldResult",
    "calibrate_cutoff",
    "sample_at_cells",
    "solve_oxygen",
    "sweep_parameters",
    "DEFAULT_RINGS",
    "annulus_neighbor_stats",
    "annulus_scan",
    "spillover_chi2",
    "plot_annulus_profile",
    "analyze_sample",
    "run_static_pipeline",
    "plot_fate_spatial",
    "plot_training_loss",
    "plot_coefficients",
    "plot_interaction_modes",
    "fate_parameter_stats",
    "likelihood_ratio_test",
    "interaction_modes",
    "fit_mlp_baseline",
    "predict_mlp_proba",
    "build_mlp_features",
    "assign_tiles",
    "aggregate_tiles",
    "binned_conditional_mean",
    "plot_binned_conditional_mean",
    "plot_tile_field",
    "plot_rootogram",
    "plot_pit_histogram",
    "get_gene_counts",
    "get_signature_counts",
    "build_predictors",
    "fit_count_glm",
    "fit_neighbourhood_count_glm",
    "compare_count_glms",
    "fit_division_nb",
    "fit_death_nb",
    "stepwise_division_nb",
    "compute_library_size",
    "compute_local_density",
    "build_covariate_matrix",
]


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value          # cache, so later access is a plain lookup
    return value


def __dir__():
    return sorted(set(__all__) | set(_LAZY) | set(globals()))
