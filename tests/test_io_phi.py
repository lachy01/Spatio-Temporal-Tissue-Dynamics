import numpy as np
import pandas as pd
import pytest

from tissue_dynamics.io_phi import attach_phi, read_phi_table


def test_read_phi_table_csv(tmp_path, phi_frame):
    path = tmp_path / "phi.csv"
    phi_frame.to_csv(path, index=False)

    table = read_phi_table(path, sample_col="sample")

    assert table.index.name == "cell_id"
    assert list(table.columns) == ["sample", "phi_0", "phi_1", "phi_2"]
    assert len(table) == len(phi_frame)


def test_read_phi_table_parquet(tmp_path, phi_frame):
    pytest.importorskip("pyarrow")
    path = tmp_path / "phi.parquet"
    phi_frame.to_parquet(path, index=False)

    table = read_phi_table(path, sample_col="sample")
    assert list(table.columns) == ["sample", "phi_0", "phi_1", "phi_2"]


def test_read_phi_table_missing_sample_col_raises(tmp_path, phi_frame):
    path = tmp_path / "phi.csv"
    phi_frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_phi_table(path, sample_col="does_not_exist")


def test_read_phi_table_renames_arbitrary_phi_prefix(tmp_path, phi_frame):
    renamed = phi_frame.rename(columns={"phi_0": "embed_0", "phi_1": "embed_1", "phi_2": "embed_2"})
    path = tmp_path / "phi.csv"
    renamed.to_csv(path, index=False)

    table = read_phi_table(path, sample_col="sample", phi_prefix="embed_")
    assert list(table.columns) == ["sample", "phi_0", "phi_1", "phi_2"]


def test_attach_phi_aligns_by_composite_key_regardless_of_table_order(tmp_path, small_adata, phi_frame):
    path = tmp_path / "phi.csv"
    phi_frame.to_csv(path, index=False)
    table = read_phi_table(path, sample_col="sample")

    attach_phi(small_adata, table, sample_col="sample")

    assert small_adata.obsm["phi"].shape == (small_adata.n_obs, 3)
    expected = np.arange(small_adata.n_obs, dtype=float)
    np.testing.assert_array_equal(small_adata.obsm["phi"][:, 0], expected)
    assert small_adata.uns["phi_dim"] == 3


def test_attach_phi_raises_on_missing_rows(tmp_path, small_adata, phi_frame):
    path = tmp_path / "phi.csv"
    phi_frame.iloc[1:].to_csv(path, index=False)  # drop one row
    table = read_phi_table(path, sample_col="sample")

    with pytest.raises(ValueError):
        attach_phi(small_adata, table, sample_col="sample")


def test_read_phi_table_phi_cols_all_uses_named_columns(tmp_path):
    df = pd.DataFrame({
        "cell_id": ["a", "b", "c"],
        "Cancer Cycling": [0.1, 0.2, 0.3],
        "T cells CD8+": [0.4, 0.5, 0.6],
    })
    path = tmp_path / "phispace.csv"
    df.to_csv(path, index=False)

    table = read_phi_table(path, phi_cols="all")

    assert list(table.columns) == ["phi_0", "phi_1"]
    assert table.attrs["phi_names"] == ["Cancer Cycling", "T cells CD8+"]


def test_read_phi_table_phi_cols_explicit_list(tmp_path):
    df = pd.DataFrame({
        "cell_id": ["a", "b"],
        "keep_me": [1.0, 2.0],
        "and_me": [3.0, 4.0],
        "not_this": [5.0, 6.0],
    })
    path = tmp_path / "phispace.csv"
    df.to_csv(path, index=False)

    table = read_phi_table(path, phi_cols=["keep_me", "and_me"])

    assert list(table.columns) == ["phi_0", "phi_1"]
    assert table.attrs["phi_names"] == ["keep_me", "and_me"]


def test_read_phi_table_no_prefix_match_raises_with_hint(tmp_path):
    df = pd.DataFrame({"cell_id": ["a", "b"], "Cancer Cycling": [0.1, 0.2]})
    path = tmp_path / "phispace.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="phi_cols"):
        read_phi_table(path)


def test_attach_phi_records_phi_names(small_adata, tmp_path):
    df = pd.DataFrame({
        "cell_id": small_adata.obs["cell_id"].unique(),
    })
    df["Cancer Cycling"] = np.arange(len(df), dtype=float)
    path = tmp_path / "phispace.csv"
    df.to_csv(path, index=False)

    table = read_phi_table(path, phi_cols="all")
    attach_phi(small_adata, table)

    assert small_adata.uns["phi_names"] == ["Cancer Cycling"]


def test_attach_phi_warns_on_nan_rows(small_adata, tmp_path):
    n = small_adata.n_obs
    unique_ids = small_adata.obs["cell_id"].unique()
    df = pd.DataFrame({
        "cell_id": list(unique_ids) * 2,
        "sample": ["A"] * len(unique_ids) + ["B"] * len(unique_ids),
        "phi_0": np.arange(n, dtype=float),
    })
    df.loc[0, "phi_0"] = np.nan  # one row (sample A's first cell_id) has a missing embedding
    path = tmp_path / "phi.csv"
    df.to_csv(path, index=False)
    table = read_phi_table(path, sample_col="sample")

    with pytest.warns(UserWarning, match="NaN"):
        attach_phi(small_adata, table, sample_col="sample")

    assert np.isnan(small_adata.obsm["phi"]).any()


def test_attach_phi_without_sample_col_uses_cell_id_only(small_adata):
    # cell_id is duplicated across sections in small_adata; without sample_col
    # the join key collapses to cell_id alone, so distinct sections sharing a
    # cell_id necessarily can't both align correctly -- attach_phi should
    # still run (no error) since only unique cell_ids need representation.
    unique_cell_ids = small_adata.obs["cell_id"].unique()
    table = pd.DataFrame(
        {"phi_0": np.arange(len(unique_cell_ids), dtype=float)},
        index=pd.Index(unique_cell_ids, name="cell_id"),
    )
    attach_phi(small_adata, table)
    assert small_adata.obsm["phi"].shape == (small_adata.n_obs, 1)
