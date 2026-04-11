import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils import create_demo_blockmodel


def test_from_empty_parquet_raises_error(tmp_path):
    import pandas as pd
    parquet_path = tmp_path / "bar.parquet"
    parquet_path.touch()
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="Cannot infer block size/shape from degenerate centroid coordinates."):
        ParquetBlockModel.from_parquet(parquet_path)


def test_create_demo_block_model_creates_file(tmp_path):
    filename = tmp_path / "demo_block_model.parquet"
    model = ParquetBlockModel.create_demo_block_model(filename)
    assert isinstance(model, ParquetBlockModel)
    # Canonical ParquetBlockModel files now use a single ".pbm" suffix.
    assert model.blockmodel_path == filename.with_suffix(".pbm")
    assert model.name == filename.stem
    assert filename.exists()
    # Check if the file is a Parquet file
    assert filename.suffix == ".parquet"
    # Clean up the created file
    os.remove(filename) if filename.exists() else None
    os.remove(model.blockmodel_path) if filename.exists() else None


def test_block_model_with_categorical_data(tmp_path):
    # Create a Parquet file with categorical data
    parquet_path = tmp_path / "categorical_block_model.parquet"
    blocks = create_demo_blockmodel()
    blocks['category_col'] = pd.Categorical(['A', 'B', 'C'] * (len(blocks) // 3))

    blocks.to_parquet(parquet_path)

    # Load the block model
    block_model = ParquetBlockModel.from_parquet(parquet_path)

    # Check if the block model is created correctly
    assert isinstance(block_model, ParquetBlockModel)
    assert block_model.blockmodel_path == parquet_path.with_suffix(".pbm")
    assert block_model.name == parquet_path.stem
    assert block_model.data.shape[0] == 27
    assert "category_col" in block_model.columns


def test_sparse_block_model(tmp_path):
    # Create a Parquet file with sparse data
    parquet_path = tmp_path / "sparse_block_model.parquet"
    blocks = create_demo_blockmodel(shape=(4, 4, 4), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0))
    blocks = blocks.query('z!=1.5')  # Drop a single z value to create a sparse dataset
    blocks.to_parquet(parquet_path)
    # Load the block model
    block_model = ParquetBlockModel.from_parquet(parquet_path)
    # Check if the block model is created correctly
    assert isinstance(block_model, ParquetBlockModel)
    assert block_model.blockmodel_path == parquet_path.with_suffix(".pbm")
    assert block_model.name == parquet_path.stem

    block_model.to_dense_parquet(parquet_path.with_suffix('.dense.parquet'), show_progress=True)
    df = pd.read_parquet(parquet_path.with_suffix('.dense.parquet'))
    assert df.shape[0] == 4 * 4 * 4

def test_block_model_with_empty_parquet(tmp_path):
    # Create an empty Parquet file
    parquet_path = tmp_path / "empty_block_model.parquet"
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    # Attempt to load the block model
    with pytest.raises(ValueError, match="Cannot infer block size/shape from degenerate centroid coordinates."):
        ParquetBlockModel.from_parquet(parquet_path)


def test_read_without_xyz_uses_block_id(tmp_path):
    parquet_path = tmp_path / "source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=3)
    assert "x" not in pbm.columns
    assert "y" not in pbm.columns
    assert "z" not in pbm.columns
    assert "block_id" in pbm.columns

    df_ijk = pbm.read(columns=["depth"], index="ijk")
    assert list(df_ijk.index.names) == ["i", "j", "k"]
    assert len(df_ijk) == 8

    # xyz index is derived from geometry + block_id when xyz columns are absent.
    df_xyz = pbm.read(columns=["depth"], index="xyz")
    assert list(df_xyz.index.names) == ["x", "y", "z"]
    assert len(df_xyz) == 8


def test_sparse_validation_without_xyz(tmp_path):
    parquet_path = tmp_path / "source_sparse.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    sparse = blocks.iloc[:-2].copy()
    sparse.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    assert pbm.is_sparse
    assert pbm.validate_sparse()


def test_block_id_persisted_in_dense_pbm(tmp_path):
    parquet_path = tmp_path / "dense_source.parquet"
    blocks = create_demo_blockmodel(shape=(3, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    expected_block_ids = pbm.geometry.row_index_from_xyz(
        blocks["x"].to_numpy(),
        blocks["y"].to_numpy(),
        blocks["z"].to_numpy(),
    ).astype(np.int64)

    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].dtype == np.int64
    assert np.array_equal(persisted["block_id"].to_numpy(), expected_block_ids)


def test_block_id_persisted_in_sparse_pbm(tmp_path):
    parquet_path = tmp_path / "sparse_source.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    sparse = blocks.drop(index=[1, 4, 11]).reset_index(drop=True)
    sparse.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=3)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    expected_block_ids = pbm.geometry.row_index_from_xyz(
        sparse["x"].to_numpy(),
        sparse["y"].to_numpy(),
        sparse["z"].to_numpy(),
    ).astype(np.int64)

    assert pbm.is_sparse
    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].is_unique
    assert np.array_equal(persisted["block_id"].to_numpy(), expected_block_ids)


def test_validate_xyz_parquet_chunked(tmp_path):
    parquet_path = tmp_path / "xyz_defined.parquet"
    blocks = create_demo_blockmodel(shape=(4, 3, 2), block_size=(2.0, 1.0, 0.5), corner=(10.0, 20.0, -1.0)).reset_index()
    # Simulate xyz-defined input with no geometry metadata.
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    geometry = ParquetBlockModel.validate_xyz_parquet(parquet_path, chunk_size=5)
    assert geometry.shape == (4, 3, 2)
    assert geometry.block_size == (2.0, 1.0, 0.5)


def test_index_properties_match_demo_columns(tmp_path):
    shape = (3, 2, 4)
    parquet_path = tmp_path / "index_compare.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=shape)

    df = pbm.read(columns=["index_c", "index_f"], index=None)

    assert np.array_equal(pbm.index_c, df["index_c"].to_numpy())
    assert np.array_equal(pbm.index_f, df["index_f"].to_numpy())
    assert len(pbm.index_c) == int(np.prod(shape))


