import os
from pathlib import Path

import pandas as pd
import pytest

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils import create_demo_blockmodel


def test_from_empty_parquet_raises_error(tmp_path):
    import pandas as pd
    parquet_path = tmp_path / "bar.parquet"
    parquet_path.touch()
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="Parquet file is empty or does not contain valid centroid data."):
        ParquetBlockModel.from_parquet(parquet_path)


def test_create_demo_block_model_creates_file(tmp_path):
    filename = tmp_path / "demo_block_model.parquet"
    model = ParquetBlockModel.create_demo_block_model(filename)
    assert isinstance(model, ParquetBlockModel)
    assert model.path == filename.with_suffix(".pbm.parquet")
    assert model.name == filename.stem
    assert filename.exists()
    # Check if the file is a Parquet file
    assert filename.suffix == ".parquet"
    # Clean up the created file
    os.remove(filename) if filename.exists() else None
    os.remove(model.path) if filename.exists() else None


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
    assert block_model.path == parquet_path.with_suffix(".pbm.parquet")
    assert block_model.name == parquet_path.stem
    assert block_model.data.shape == (27, 4)  # 3x3x3 rows, 4 columns including index


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
    assert block_model.path == parquet_path.with_suffix(".pbm.parquet")
    assert block_model.name == parquet_path.stem

    block_model.to_dense_parquet(parquet_path.with_suffix('.dense.parquet'), show_progress=True)
    df = pd.read_parquet(parquet_path.with_suffix('.dense.parquet'))
    assert df.shape[0] == 4 * 4 * 4
