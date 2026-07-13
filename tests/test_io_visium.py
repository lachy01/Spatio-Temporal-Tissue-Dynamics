"""Tests for io_visium.py. Synthetic fixtures follow the scanpy/squidpy
spatial AnnData convention (obsm["spatial"] pixel coords + uns["spatial"]),
the same shape `sc.read_visium()` produces and what a real CELLxGENE Visium
.h5ad export looks like -- this module (including the pixel-to-micron
calibration approach) was additionally verified against a real GSE176078
Visium sample during development.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tissue_dynamics.io_visium import (
    _bin_size_um_from_path,
    _calibrate_microns_per_pixel,
    _spatial_library_entries,
    find_visium_path,
    load_visium_samples,
    read_visium,
)


def _make_visium_adata(n_side=4, pixel_spacing=20.0, library_id="sampleA",
                          spot_diameter_fullres=50.0, n_in_tissue=None, ensembl_var_names=False):
    n = n_side * n_side
    rng = np.random.default_rng(0)
    cols, rows = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords_px = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(float) * pixel_spacing

    n_genes = 6
    counts = rng.integers(0, 10, size=(n, n_genes)).astype("float32")
    obs_names = [f"spot{i}-1" for i in range(n)]

    gene_symbols = [f"GENE{i}" for i in range(n_genes)]
    if ensembl_var_names:
        var = pd.DataFrame({"feature_name": gene_symbols}, index=[f"ENSG{i:08d}" for i in range(n_genes)])
    else:
        var = pd.DataFrame(index=gene_symbols)

    if n_in_tissue is None:
        n_in_tissue = n
    in_tissue = np.zeros(n, dtype=int)
    in_tissue[:n_in_tissue] = 1
    obs = pd.DataFrame({"in_tissue": in_tissue}, index=obs_names)

    adata = AnnData(X=counts, obs=obs, var=var)
    adata.obsm["spatial"] = coords_px
    adata.uns["spatial"] = {
        library_id: {
            "scalefactors": {"spot_diameter_fullres": spot_diameter_fullres, "tissue_hires_scalef": 0.2},
            "images": {"hires": np.zeros((10, 10, 3), dtype="uint8")},
        },
        # real CELLxGENE exports carry this non-dict sibling key alongside
        # the library entries -- must not be mistaken for one.
        "is_single": True,
    }
    return adata


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def test_spatial_library_entries_filters_non_dict_keys():
    entries = _spatial_library_entries(_make_visium_adata())
    assert list(entries.keys()) == ["sampleA"]


def test_bin_size_um_from_path_parses_visium_hd_folder_name():
    assert _bin_size_um_from_path(Path("data/binned_outputs/square_016um")) == 16.0
    assert _bin_size_um_from_path(Path("data/classic_visium/outs")) is None


def test_calibrate_microns_per_pixel_recovers_known_spacing():
    coords_px = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
    mpp = _calibrate_microns_per_pixel(coords_px, spot_spacing_um=100.0)
    assert mpp == pytest.approx(5.0)


# --------------------------------------------------------------------------
# find_visium_path
# --------------------------------------------------------------------------

def test_find_visium_path_resolves_h5ad_file(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata().write_h5ad(path)
    assert find_visium_path(path) == path


def test_find_visium_path_resolves_classic_space_ranger_folder(tmp_path):
    d = tmp_path / "sample_outs"
    (d / "spatial").mkdir(parents=True)
    (d / "spatial" / "tissue_positions.csv").write_text("barcode,in_tissue\n")
    assert find_visium_path(d) == d


def test_find_visium_path_resolves_outs_subfolder(tmp_path):
    d = tmp_path / "sample_root"
    (d / "outs" / "spatial").mkdir(parents=True)
    (d / "outs" / "spatial" / "tissue_positions.csv").write_text("barcode,in_tissue\n")
    assert find_visium_path(d) == d / "outs"


def test_find_visium_path_resolves_single_visium_hd_resolution(tmp_path):
    d = tmp_path / "sample_root"
    resolution_dir = d / "binned_outputs" / "square_016um" / "spatial"
    resolution_dir.mkdir(parents=True)
    (resolution_dir / "tissue_positions.parquet").touch()
    assert find_visium_path(d) == d / "binned_outputs" / "square_016um"


def test_find_visium_path_raises_on_ambiguous_visium_hd_resolutions(tmp_path):
    d = tmp_path / "sample_root"
    for res in ("square_008um", "square_016um"):
        p = d / "binned_outputs" / res / "spatial"
        p.mkdir(parents=True)
        (p / "tissue_positions.parquet").touch()
    with pytest.raises(ValueError, match="Multiple Visium HD resolutions"):
        find_visium_path(d)


def test_find_visium_path_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_visium_path(tmp_path)


# --------------------------------------------------------------------------
# read_visium
# --------------------------------------------------------------------------

def test_read_visium_converts_pixels_to_known_microns(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(pixel_spacing=20.0).write_h5ad(path)

    adata = read_visium(path, in_tissue_only=False)  # default spot_spacing_um=100.0

    assert adata.uns["visium_microns_per_pixel"] == pytest.approx(5.0)
    x_um = np.sort(np.unique(adata.obsm["spatial"][:, 0]))
    np.testing.assert_allclose(np.diff(x_um), 100.0)


def test_read_visium_respects_explicit_microns_per_pixel(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(pixel_spacing=20.0).write_h5ad(path)

    adata = read_visium(path, in_tissue_only=False, microns_per_pixel=2.0)

    assert adata.uns["visium_microns_per_pixel"] == 2.0
    x_um = np.sort(np.unique(adata.obsm["spatial"][:, 0]))
    np.testing.assert_allclose(np.diff(x_um), 40.0)


def test_read_visium_filters_to_in_tissue_by_default(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(n_side=4, n_in_tissue=10).write_h5ad(path)

    assert read_visium(path).n_obs == 10
    assert read_visium(path, in_tissue_only=False).n_obs == 16


def test_read_visium_remaps_ensembl_var_names_to_symbols(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(ensembl_var_names=True).write_h5ad(path)

    adata = read_visium(path)

    assert "GENE0" in adata.var_names
    assert "gene_ids" in adata.var.columns
    assert adata.var.loc["GENE0", "gene_ids"] == "ENSG00000000"


def test_read_visium_leaves_symbol_var_names_untouched(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(ensembl_var_names=False).write_h5ad(path)

    adata = read_visium(path)
    assert list(adata.var_names) == [f"GENE{i}" for i in range(6)]


def test_read_visium_sets_sample_id_from_single_library_entry(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(library_id="mySample").write_h5ad(path)

    adata = read_visium(path)
    assert (adata.obs["sample_id"] == "mySample").all()


def test_read_visium_sample_id_override(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata(library_id="mySample").write_h5ad(path)

    adata = read_visium(path, sample_id="override")
    assert (adata.obs["sample_id"] == "override").all()


def test_read_visium_sets_cell_id_from_obs_names(tmp_path):
    path = tmp_path / "sample.h5ad"
    _make_visium_adata().write_h5ad(path)

    adata = read_visium(path)
    assert list(adata.obs["cell_id"]) == list(adata.obs_names)


def test_read_visium_raises_on_non_visium_h5ad(tmp_path):
    path = tmp_path / "not_visium.h5ad"
    AnnData(X=np.zeros((3, 2))).write_h5ad(path)
    with pytest.raises(ValueError, match="doesn't look like Visium data"):
        read_visium(path)


# --------------------------------------------------------------------------
# load_visium_samples
# --------------------------------------------------------------------------

def test_load_visium_samples_combines_sections(tmp_path):
    path_a, path_b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _make_visium_adata(library_id="A").write_h5ad(path_a)
    _make_visium_adata(library_id="B").write_h5ad(path_b)

    combined = load_visium_samples([path_a, path_b])

    assert set(combined.obs["sample_id"].unique()) == {"A", "B"}
    assert combined.n_obs == 32


def test_load_visium_samples_raises_on_duplicate_sample_ids(tmp_path):
    path_a, path_b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _make_visium_adata(library_id="same").write_h5ad(path_a)
    _make_visium_adata(library_id="same").write_h5ad(path_b)

    with pytest.raises(ValueError, match="must be unique"):
        load_visium_samples([path_a, path_b])
