"""Load a 10x Visium (or Visium HD) sample into an AnnData object.

Supports two input shapes:

1. A raw Space Ranger output folder (`spatial/tissue_positions*`,
   `filtered_feature_bc_matrix.h5`, `scalefactors_json.json`, including
   Visium HD's `binned_outputs/square_XXXum/` layout) -- read via scanpy's
   own `sc.read_visium`, which already produces the same `obsm["spatial"]`
   + `uns["spatial"]` convention this module builds on.
2. An already-assembled `.h5ad` file following that same scanpy/squidpy
   spatial convention (e.g. exported from CELLxGENE) -- read directly.

Either way, spots/bins are treated exactly like Xenium cells downstream:
`neighbors.py`/`targets.py`/`models.py` only need `obsm["spatial"]` (in real
physical units) + `obs["sample_id"]` -- they never assume a row is a
segmented cell. This is the "cell-free" path: each spatial bin gets an id
(`obs["cell_id"]`) and Phi-Space embeddings attach to it exactly as they
would to a Xenium cell.

Pixel -> micron conversion: Visium's `scalefactors_json` ships a
`spot_diameter_fullres` field that looks like it should give this directly
(`microns_per_pixel = 55.0 / spot_diameter_fullres`, since standard Visium
spots are 55um in diameter), but checked empirically against a real
GSE176078 Visium sample, that assumption is off by ~18% -- the recorded
diameter doesn't sit in the ratio you'd expect against the known,
hardware-fixed 100um center-to-center hex spacing (cross-validated two
independent ways: nearest-neighbor spot spacing, and the known 6.5mm x
6.5mm capture area bounding box -- both agree with each other and disagree
with the diameter-based formula). So by default this module self-calibrates
instead: it measures the median nearest-neighbor spot/bin spacing in pixels
and divides the known, fixed physical spacing (100um for classic Visium's
hex grid; auto-detected, or pass `spot_spacing_um`, for Visium HD's square
grid, e.g. 16.0 for 16um bins) by that -- robust to whatever
`spot_diameter_fullres` actually encodes. Pass `microns_per_pixel` directly
to skip auto-calibration entirely.

Verified against a real classic-Visium sample (GSE176078 / CELLxGENE
collection dea97145-f712-431c-a223-6b5f565f362a). The Visium HD
(`binned_outputs/square_XXXum/`) path is implemented per 10x's documented
output format but not yet verified against a real Visium HD sample --
treat it as provisional until tested against one.
"""

import re
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

CONTROL_PREFIXES = (
    "NegControlProbe_", "NegControlCodeword_", "BLANK_",
    "antisense_", "UnassignedCodeword_", "DEPRECATED_",
)

# Classic Visium's fixed hex-grid center-to-center spacing, in micrometers --
# a hardware constant of the printed slide, true for every standard Visium
# v1/v2 sample regardless of what a given scalefactors_json's
# spot_diameter_fullres happens to encode (see module docstring).
DEFAULT_SPOT_SPACING_UM = 100.0

_SQUARE_BIN_RE = re.compile(r"square_(\d+)um")


def find_visium_path(path):
    """Resolve a Visium sample root to whatever `sc.read_visium`/`ad.read_h5ad`
    expects: an `.h5ad` file as-is, or a Space Ranger sample directory
    (containing `spatial/tissue_positions*` directly, or found by resolving
    Visium HD's `binned_outputs/square_XXXum/` layout).

    For Visium HD with multiple binned resolutions present, point `path`
    directly at the desired `square_XXXum` folder (or its `outs/` subfolder)
    rather than the `binned_outputs/` parent -- this function deliberately
    does not guess which resolution you want.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".h5ad":
        return path
    if (path / "spatial" / "tissue_positions.csv").exists() or \
       (path / "spatial" / "tissue_positions.parquet").exists():
        return path
    if (path / "outs" / "spatial" / "tissue_positions.csv").exists() or \
       (path / "outs" / "spatial" / "tissue_positions.parquet").exists():
        return path / "outs"

    binned = path / "binned_outputs"
    if binned.is_dir():
        resolutions = sorted(p.name for p in binned.glob("square_*um") if p.is_dir())
        if len(resolutions) == 1:
            return find_visium_path(binned / resolutions[0])
        if len(resolutions) > 1:
            raise ValueError(
                f"Multiple Visium HD resolutions found under {binned}: {resolutions}. "
                "Point path directly at the one you want, e.g. "
                f"{binned / resolutions[0]}."
            )

    matches = list(path.rglob("tissue_positions.csv")) + list(path.rglob("tissue_positions.parquet"))
    if matches:
        return matches[0].parent.parent  # .../spatial/tissue_positions.* -> sample dir

    raise FileNotFoundError(f"No Visium (or Visium HD) output found under {path}")


def _bin_size_um_from_path(path):
    match = _SQUARE_BIN_RE.search(str(path))
    return float(match.group(1)) if match else None


def _spatial_library_entries(adata):
    """`uns["spatial"]` holds one dict per library id, but sometimes also a
    stray boolean `is_single` flag alongside them -- filter to real entries."""
    return {k: v for k, v in adata.uns.get("spatial", {}).items() if isinstance(v, dict)}


def _calibrate_microns_per_pixel(coords_px, spot_spacing_um):
    tree = cKDTree(coords_px)
    dists, _ = tree.query(coords_px, k=2)
    nn_px = np.median(dists[:, 1])
    return spot_spacing_um / nn_px


def read_visium(path, sample_id=None, in_tissue_only=True,
                  microns_per_pixel=None, spot_spacing_um=None, drop_controls=True):
    """Read a Visium (or Visium HD) sample into an AnnData with the same
    contract as `io_xenium.read_xenium`: bins x genes counts,
    `obsm["spatial"]` in real physical units (micrometers), `obs["cell_id"]`
    (the spot/bin barcode), and `obs["sample_id"]`.

    Parameters
    ----------
    path : str or Path
        A Space Ranger output folder (classic Visium, or Visium HD's
        `binned_outputs/square_XXXum/` -- see `find_visium_path`), or an
        already-assembled `.h5ad` following the scanpy/squidpy spatial
        convention (`obsm["spatial"]` + `uns["spatial"][library_id]
        ["scalefactors"]`).
    in_tissue_only : bool
        Drop spots/bins Space Ranger flagged as not overlapping tissue
        (`obs["in_tissue"] == 0`) -- typically most of the fixed
        capture-area grid for classic Visium. No-op if that column isn't
        present.
    microns_per_pixel : float, optional
        Skip auto-calibration and use this conversion factor directly.
    spot_spacing_um : float, optional
        Known fixed center-to-center spacing used for auto-calibration (see
        module docstring for why this is preferred over trusting
        `scalefactors_json`'s `spot_diameter_fullres` directly). Defaults to
        100.0 (classic Visium's hex pitch) unless the resolved path encodes
        a Visium HD bin size (e.g. `.../square_016um/` -> 16.0).
    drop_controls : bool
        Drop negative-control / blank features by name, as a safety net for
        probe-based (FFPE) Visium panels; a no-op for whole-transcriptome
        Visium, which has no such features.

    Returns
    -------
    AnnData, ready to feed into the same neighbors/targets/models pipeline
    as a Xenium sample.
    """
    resolved = find_visium_path(path)

    if resolved.is_file():
        adata = ad.read_h5ad(resolved)
        # CELLxGENE-schema .h5ad exports key var_names by Ensembl gene id,
        # with the human-readable symbol in var["feature_name"] instead --
        # gene_sets.match_gene_set (and everything built on it: score_division,
        # score_death, build_fate_targets) matches marker lists like "MKI67"
        # against var_names directly, so without this remap every signature
        # score would silently find zero matching genes. Raw Space Ranger
        # output (the other branch below) already uses gene symbols as
        # var_names via sc.read_visium's default, so this is a no-op there.
        if "feature_name" in adata.var.columns:
            adata.var["gene_ids"] = adata.var_names.astype(str)
            adata.var_names = adata.var["feature_name"].astype(str)
            adata.var_names_make_unique()
    else:
        import scanpy as sc

        adata = sc.read_visium(str(resolved))
        adata.var_names_make_unique()

    if "spatial" not in adata.obsm or not _spatial_library_entries(adata):
        raise ValueError(
            f"{path} doesn't look like Visium data: expected obsm['spatial'] "
            "(pixel coordinates) and uns['spatial'][library_id]['scalefactors']."
        )

    if in_tissue_only and "in_tissue" in adata.obs.columns:
        adata = adata[adata.obs["in_tissue"] == 1].copy()

    if spot_spacing_um is None:
        spot_spacing_um = _bin_size_um_from_path(resolved) or DEFAULT_SPOT_SPACING_UM

    if microns_per_pixel is None:
        microns_per_pixel = _calibrate_microns_per_pixel(adata.obsm["spatial"], spot_spacing_um)

    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64) * microns_per_pixel
    adata.uns["visium_microns_per_pixel"] = microns_per_pixel

    if drop_controls:
        is_control = adata.var_names.str.startswith(CONTROL_PREFIXES)
        if is_control.any():
            adata = adata[:, ~is_control].copy()

    library_entries = _spatial_library_entries(adata)
    resolved_sample_id = sample_id or (
        next(iter(library_entries)) if len(library_entries) == 1 else resolved.name
    )
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = resolved_sample_id
    return adata


def load_visium_samples(paths, sample_ids=None, **read_kwargs):
    """Read several Visium sections and concatenate them into one AnnData.
    Mirrors `io_xenium.load_xenium_samples`.

    Parameters
    ----------
    paths : sequence of str or Path
        Sample roots (or `.h5ad` files), one per section.
    sample_ids : sequence of str, optional
        Section identifiers, one per entry in `paths` (defaults to each
        section's own `read_visium`-derived `sample_id`). Must be unique.
    **read_kwargs
        Forwarded to `read_visium` for every sample.

    Returns
    -------
    AnnData with all sections stacked (`anndata.concat`, `join="outer"`),
    `obs["sample_id"]` distinguishing sections, `obsm["spatial"]` preserved.
    """
    if sample_ids is None:
        sample_ids = [None] * len(paths)
    if len(sample_ids) != len(paths):
        raise ValueError("sample_ids must be the same length as paths")

    adatas = [read_visium(p, sample_id=sid, **read_kwargs) for p, sid in zip(paths, sample_ids)]
    resolved_ids = [a.obs["sample_id"].iloc[0] for a in adatas]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError(f"sample_id values must be unique; got {resolved_ids}")

    return ad.concat(adatas, join="outer", index_unique="-", keys=resolved_ids)
