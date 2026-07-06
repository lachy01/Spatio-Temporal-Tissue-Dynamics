"""Score cells for cell-cycle phase / proliferation activity."""

import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from ._classification import classify_bimodal
from .gene_sets import CORE_PROLIFERATION_MARKERS, TIROSH_G2M_GENES, TIROSH_S_GENES, match_gene_set

MIN_GENES_PER_PHASE = 3


def _looks_like_raw_counts(X):
    # NB: dense numpy arrays also have a `.data` attribute (a memoryview of
    # the buffer), so `hasattr(X, "data")` is not a valid sparse/dense check
    # — use scipy's own predicate instead.
    data = X.data if sp.issparse(X) else np.asarray(X)
    return data.size > 0 and np.all(data >= 0) and np.allclose(data, np.round(data))


def ensure_lognormalized(adata):
    """Normalize + log1p in place, preserving raw counts in `layers["counts"]`.

    No-ops if `adata.X` doesn't look like raw integer counts (e.g. it's
    already been normalized), so this is safe to call unconditionally.
    """
    if not _looks_like_raw_counts(adata.X):
        return adata
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def score_cell_cycle(adata, s_genes=None, g2m_genes=None, preprocess=True, copy=False):
    """Assign S/G2M/G1 phase per cell using the Tirosh et al. gene sets.

    Xenium panels are curated for cell typing and often cover only a
    handful of the 43 canonical S-phase / 54 canonical G2M-phase genes —
    warns loudly if coverage is too thin for the phase split to be
    trustworthy; consider `score_proliferation` instead in that case.

    Adds `adata.obs["S_score"]`, `["G2M_score"]`, `["phase"]`, and a boolean
    `["is_dividing"]` (True for S or G2M phase).
    """
    if copy:
        adata = adata.copy()

    s_genes = list(s_genes) if s_genes is not None else list(TIROSH_S_GENES)
    g2m_genes = list(g2m_genes) if g2m_genes is not None else list(TIROSH_G2M_GENES)

    s_present = match_gene_set(s_genes, adata.var_names)
    g2m_present = match_gene_set(g2m_genes, adata.var_names)

    if len(s_present) < MIN_GENES_PER_PHASE or len(g2m_present) < MIN_GENES_PER_PHASE:
        warnings.warn(
            f"Only {len(s_present)}/{len(s_genes)} S-phase and {len(g2m_present)}/"
            f"{len(g2m_genes)} G2M-phase reference genes are present in this panel. "
            "S/G2M phase scores are unlikely to be reliable with this little "
            "coverage — consider score_proliferation() instead, or pass a custom "
            "gene set curated for this panel.",
            stacklevel=2,
        )
    if not s_present or not g2m_present:
        raise ValueError(
            "No matching genes found for one of the S/G2M gene sets in this "
            "panel; cell-cycle phase scoring is not possible. Use "
            "score_proliferation() or supply custom s_genes/g2m_genes."
        )

    if preprocess:
        ensure_lognormalized(adata)

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
    adata.obs["is_dividing"] = adata.obs["phase"].isin(["S", "G2M"])
    adata.uns["cell_cycle_genes"] = {"s_genes": s_present, "g2m_genes": g2m_present}
    return adata


def score_proliferation(adata, marker_genes=None, score_name="proliferation_score",
                         preprocess=True, threshold=0.0, method="gmm", copy=False):
    """Score cells with a single proliferation gene signature.

    A more robust fallback than `score_cell_cycle` when a targeted panel
    doesn't resolve enough S- vs G2M-phase genes to separate the two.

    Adds `adata.obs[score_name]` and a boolean `adata.obs["is_dividing"]`.

    Parameters
    ----------
    method : "gmm" or "threshold"
        "gmm" (default) fits a 2-component Gaussian mixture to the score
        distribution and calls the higher-scoring component dividing — a
        data-driven cut that adapts to each sample's score distribution.
        "threshold" calls a cell dividing if its score exceeds `threshold`
        (0 = no enrichment over the scanpy control-gene background); simpler
        but tends to over-call dividing cells since roughly half the
        population sits above 0 by construction of the control-gene
        normalization.
    """
    if copy:
        adata = adata.copy()

    marker_genes = list(marker_genes) if marker_genes is not None else list(CORE_PROLIFERATION_MARKERS)
    present = match_gene_set(marker_genes, adata.var_names)

    if len(present) < 2:
        raise ValueError(
            f"Only {len(present)}/{len(marker_genes)} proliferation marker genes "
            "found in this panel; need at least 2 to compute a signature score. "
            "Supply a custom marker_genes list appropriate for this panel."
        )
    if len(present) < len(marker_genes):
        warnings.warn(
            f"Using {len(present)}/{len(marker_genes)} proliferation marker genes "
            f"present in this panel: {present}.",
            stacklevel=2,
        )

    if preprocess:
        ensure_lognormalized(adata)

    sc.tl.score_genes(adata, gene_list=present, score_name=score_name)
    adata.obs["is_dividing"] = classify_bimodal(adata.obs[score_name], method, threshold)
    adata.uns["proliferation_genes"] = present
    return adata
