"""End-to-end convenience entrypoint: read a Xenium sample and score it."""

import warnings

from .cellcycle import MIN_GENES_PER_PHASE, score_cell_cycle, score_proliferation
from .gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES, match_gene_set
from .io import read_xenium


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
