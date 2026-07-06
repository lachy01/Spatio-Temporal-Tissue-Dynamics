from tissue_dynamics.gene_sets import (
    CORE_APOPTOSIS_MARKERS,
    CORE_PROLIFERATION_MARKERS,
    TIROSH_G2M_GENES,
    TIROSH_S_GENES,
    match_gene_set,
)


def test_core_marker_lists_nonempty_and_unique():
    for genes in (CORE_PROLIFERATION_MARKERS, CORE_APOPTOSIS_MARKERS, TIROSH_S_GENES, TIROSH_G2M_GENES):
        assert len(genes) > 0
        assert len(genes) == len(set(genes))


def test_proliferation_and_apoptosis_markers_are_disjoint():
    assert not set(CORE_PROLIFERATION_MARKERS) & set(CORE_APOPTOSIS_MARKERS)


def test_match_gene_set_is_case_insensitive_and_order_preserving():
    var_names = ["casp3", "BAX", "unrelated"]
    matched = match_gene_set(["CASP3", "BAX", "CASP8"], var_names)
    assert matched == ["casp3", "BAX"]


def test_match_gene_set_returns_empty_for_no_overlap():
    assert match_gene_set(["CASP3"], ["UNRELATED1", "UNRELATED2"]) == []
