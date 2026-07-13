"""Tests for io_xenium.py, focused on the manual-reader fallback behavior
(read_xenium falls back to it on any spatialdata-io failure, not just
ImportError -- see download.py's REQUIRED_FILES for why that matters)."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from tissue_dynamics.io_xenium import _read_10x_mtx_manual, find_outs_dir, read_xenium


def _write_cell_feature_matrix_h5(path, n_genes=5, n_cells=4):
    """A minimal but valid CellRanger/Xenium-format cell_feature_matrix.h5,
    readable by scanpy's sc.read_10x_h5 -- genes x cells, CSC-sparse."""
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 5, size=(n_genes, n_cells)).astype("int32")
    X = csc_matrix(dense)

    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = np.bytes_("matrix")
        grp = f.create_group("matrix")
        grp.create_dataset("shape", data=np.array([n_genes, n_cells], dtype="int32"))
        grp.create_dataset("data", data=X.data.astype("int32"))
        grp.create_dataset("indices", data=X.indices.astype("int32"))
        grp.create_dataset("indptr", data=X.indptr.astype("int32"))
        grp.create_dataset("barcodes", data=np.array([f"cell{i}".encode() for i in range(n_cells)]))
        feat = grp.create_group("features")
        feat.create_dataset("id", data=np.array([f"ENSG{i}".encode() for i in range(n_genes)]))
        feat.create_dataset("name", data=np.array([f"GENE{i}".encode() for i in range(n_genes)]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_genes))
        feat.create_dataset("genome", data=np.array([b"test_genome"] * n_genes))
    return dense.T  # cells x genes, for assertions


def _write_cells_csv(outs_dir, cell_ids):
    df = pd.DataFrame({
        "cell_id": cell_ids,
        "x_centroid": np.arange(len(cell_ids), dtype=float),
        "y_centroid": np.arange(len(cell_ids), dtype=float) * 10,
        "cell_area": np.full(len(cell_ids), 100.0),
    })
    df.to_csv(outs_dir / "cells.csv.gz", index=False, compression="gzip")


def _write_mtx_triplet(outs_dir, n_genes=3, n_cells=2):
    mtx_dir = outs_dir / "cell_feature_matrix"
    mtx_dir.mkdir()
    dense = np.arange(n_genes * n_cells, dtype="int32").reshape(n_genes, n_cells)
    with gzip.open(mtx_dir / "matrix.mtx.gz", "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        nnz = int((dense != 0).sum())
        f.write(f"{n_genes} {n_cells} {nnz}\n")
        for i in range(n_genes):
            for j in range(n_cells):
                if dense[i, j] != 0:
                    f.write(f"{i + 1} {j + 1} {dense[i, j]}\n")
    with gzip.open(mtx_dir / "features.tsv.gz", "wt") as f:
        for i in range(n_genes):
            f.write(f"ENSG{i}\tTRIPLET_GENE{i}\tGene Expression\n")
    with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
        for j in range(n_cells):
            f.write(f"cell{j}\n")


@pytest.fixture
def outs_dir_h5(tmp_path):
    """cells.csv.gz + cell_feature_matrix.h5 -- what download_xenium_sample fetches by default."""
    d = tmp_path / "sample" / "outs"
    d.mkdir(parents=True)
    _write_cell_feature_matrix_h5(d / "cell_feature_matrix.h5")
    _write_cells_csv(d, [f"cell{i}" for i in range(4)])
    return d


def test_find_outs_dir_locates_via_cells_csv(outs_dir_h5):
    assert find_outs_dir(outs_dir_h5.parent) == outs_dir_h5
    assert find_outs_dir(outs_dir_h5) == outs_dir_h5


def test_find_outs_dir_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_outs_dir(tmp_path)


def test_read_10x_mtx_manual_prefers_h5_over_triplet(tmp_path):
    d = tmp_path / "outs"
    d.mkdir()
    expected = _write_cell_feature_matrix_h5(d / "cell_feature_matrix.h5", n_genes=5, n_cells=4)
    _write_mtx_triplet(d, n_genes=3, n_cells=2)  # present too, should be ignored

    adata = _read_10x_mtx_manual(d)

    assert adata.shape == (4, 5)
    assert list(adata.var_names) == [f"GENE{i}" for i in range(5)]
    np.testing.assert_array_equal(adata.X.toarray(), expected)


def test_read_10x_mtx_manual_falls_back_to_triplet_without_h5(tmp_path):
    d = tmp_path / "outs"
    d.mkdir()
    _write_mtx_triplet(d, n_genes=3, n_cells=2)

    adata = _read_10x_mtx_manual(d)

    assert adata.shape == (2, 3)
    assert list(adata.var_names) == ["TRIPLET_GENE0", "TRIPLET_GENE1", "TRIPLET_GENE2"]


def test_read_xenium_uses_manual_reader_when_spatialdata_io_not_installed(outs_dir_h5, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spatialdata_io":
            raise ImportError("simulated: spatialdata_io not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    adata = read_xenium(outs_dir_h5)

    assert adata.n_obs == 4
    assert "sample_id" in adata.obs
    assert "cell_id" in adata.obs
    assert adata.obsm["spatial"].shape == (4, 2)


def test_read_xenium_falls_back_and_warns_when_spatialdata_io_errors(outs_dir_h5, monkeypatch):
    import builtins
    real_import = builtins.__import__

    class _FakeSpatialdataIoModule:
        @staticmethod
        def xenium(*args, **kwargs):
            raise FileNotFoundError("simulated: this spatialdata-io version wants a file we don't have")

    def fake_import(name, *args, **kwargs):
        if name == "spatialdata_io":
            return _FakeSpatialdataIoModule()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(UserWarning, match="falling back to the manual"):
        adata = read_xenium(outs_dir_h5)

    assert adata.n_obs == 4
    assert adata.obsm["spatial"].shape == (4, 2)


def test_read_xenium_sets_sample_id_and_cell_id(outs_dir_h5, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spatialdata_io":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    adata = read_xenium(outs_dir_h5, sample_id="my_section")
    assert (adata.obs["sample_id"] == "my_section").all()
    assert set(adata.obs["cell_id"]) == {f"cell{i}" for i in range(4)}
