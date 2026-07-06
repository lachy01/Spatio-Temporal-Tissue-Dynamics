"""Score cells for apoptosis / death signature activity."""

import warnings

import scanpy as sc

from ._classification import classify_bimodal
from .cellcycle import ensure_lognormalized
from .gene_sets import CORE_APOPTOSIS_MARKERS, match_gene_set


def score_death(adata, marker_genes=None, score_name="death_score",
                 preprocess=True, threshold=0.0, method="gmm", copy=False):
    """Score cells with a single apoptosis/death gene signature.

    Structural mirror of `score_proliferation`, using an apoptosis marker
    panel instead of a proliferation one.

    Adds `adata.obs[score_name]` and a boolean `adata.obs["is_dying"]`.

    Parameters
    ----------
    method : "gmm" or "threshold"
        "gmm" (default) fits a 2-component Gaussian mixture to the score
        distribution and calls the higher-scoring component dying — a
        data-driven cut that adapts to each sample's score distribution.
        "threshold" calls a cell dying if its score exceeds `threshold`
        (0 = no enrichment over the scanpy control-gene background); simpler
        but tends to over-call dying cells since roughly half the population
        sits above 0 by construction of the control-gene normalization.
    """
    if copy:
        adata = adata.copy()

    marker_genes = list(marker_genes) if marker_genes is not None else list(CORE_APOPTOSIS_MARKERS)
    present = match_gene_set(marker_genes, adata.var_names)

    if len(present) < 2:
        raise ValueError(
            f"Only {len(present)}/{len(marker_genes)} apoptosis marker genes "
            "found in this panel; need at least 2 to compute a signature score. "
            "Supply a custom marker_genes list appropriate for this panel."
        )
    if len(present) < len(marker_genes):
        warnings.warn(
            f"Using {len(present)}/{len(marker_genes)} apoptosis marker genes "
            f"present in this panel: {present}.",
            stacklevel=2,
        )

    if preprocess:
        ensure_lognormalized(adata)

    sc.tl.score_genes(adata, gene_list=present, score_name=score_name)
    adata.obs["is_dying"] = classify_bimodal(adata.obs[score_name], method, threshold)
    adata.uns["apoptosis_genes"] = present
    return adata
