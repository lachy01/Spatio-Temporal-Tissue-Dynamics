import numpy as np
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


def test_attach_phi_without_sample_col_uses_cell_id_only(small_adata):
    # cell_id is duplicated across sections in small_adata; without sample_col
    # the join key collapses to cell_id alone, so distinct sections sharing a
    # cell_id necessarily can't both align correctly -- attach_phi should
    # still run (no error) since only unique cell_ids need representation.
    unique_cell_ids = small_adata.obs["cell_id"].unique()
    import pandas as pd
    table = pd.DataFrame(
        {"phi_0": np.arange(len(unique_cell_ids), dtype=float)},
        index=pd.Index(unique_cell_ids, name="cell_id"),
    )
    attach_phi(small_adata, table)
    assert small_adata.obsm["phi"].shape == (small_adata.n_obs, 1)
