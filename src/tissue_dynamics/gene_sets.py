"""Reference gene sets used for cell-cycle / proliferation scoring."""

# Canonical Tirosh et al. (2016) / Regev lab cell-cycle gene lists, as used by
# the scanpy cell-cycle scoring tutorial. These were derived from full
# transcriptome scRNA-seq and are frequently a poor match to a targeted
# Xenium gene panel (which typically covers 300-500 genes chosen for cell
# typing, not cell-cycle phase resolution) — see `match_gene_set` for the
# panel-coverage check used before scoring with these.
TIROSH_S_GENES = (
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
)

TIROSH_G2M_GENES = (
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
)

# Small, well-established proliferation markers that tend to survive gene
# panel curation (used as a fallback single-score when a panel doesn't
# resolve enough S/G2M genes to separate the two phases reliably).
CORE_PROLIFERATION_MARKERS = ("MKI67", "TOP2A", "PCNA", "CCNB2", "CDK1", "CENPF")

# Core apoptosis pathway genes: intrinsic/mitochondrial (BAX, BAK1, BID, CYCS,
# APAF1, CASP9) and extrinsic (CASP8) pathways converging on the common
# executioner caspase (CASP3). As with the proliferation markers above, these
# are a coarse mRNA-level proxy for a process that's substantially regulated
# post-translationally (caspase cleavage/activation) rather than by
# transcription, and targeted panels frequently cover only a few of them —
# check panel coverage before trusting the resulting score, or supply a
# custom marker_genes list.
CORE_APOPTOSIS_MARKERS = ("CASP3", "CASP8", "CASP9", "BAX", "BAK1", "BID", "CYCS", "APAF1")

# A fuller pro-apoptotic signature for a death *count* response (the analogue of
# the S+G2M set for division): the intrinsic and extrinsic apoptosis machinery
# expressed *in the dying/targeted cell* — initiator and executioner caspases,
# pro-apoptotic Bcl-2 effectors and BH3-only sensitizers, the mitochondrial
# apoptosome, death receptors/adaptors, and the p53 axis. It deliberately
# EXCLUDES death ligands (FASLG, TNF, TNFSF10) and cytotoxic-effector genes
# (GZMB, PRF1): those mark the aggressor/killer cell, not the dying one, and
# would confound a death-count response with immune-killer identity. Same
# post-translational-regulation caveat as CORE_APOPTOSIS_MARKERS applies — check
# panel coverage (match_gene_set) before trusting the resulting count.
APOPTOSIS_SIGNATURE = (
    "CASP2", "CASP3", "CASP6", "CASP7", "CASP8", "CASP9", "CASP10",
    "BAX", "BAK1", "BID", "BAD", "BBC3", "PMAIP1", "BIK", "BOK", "HRK", "BMF", "BCL2L11",
    "APAF1", "CYCS", "DIABLO", "AIFM1", "ENDOG",
    "FAS", "TNFRSF10A", "TNFRSF10B", "TNFRSF1A", "FADD", "TRADD",
    "TP53",
)


def match_gene_set(genes, var_names):
    """Intersect a candidate gene list with what's actually in the panel.

    Matching is case-insensitive (Xenium panels are consistently upper-case
    gene symbols, but this guards against stray casing) and preserves the
    order of `genes`.
    """
    lookup = {name.upper(): name for name in var_names}
    return [lookup[g.upper()] for g in genes if g.upper() in lookup]
