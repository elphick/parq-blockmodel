"""Tests for pandera schema support in ParquetBlockModel."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

pandera = pytest.importorskip("pandera", reason="pandera not installed")

from pandera import DataFrameSchema, Column, dtypes

from parq_blockmodel import ParquetBlockModel, RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_demo_parquet(tmp_path: Path) -> Path:
    """Write a small demo block model to a .parquet file and return its path."""
    df = create_demo_blockmodel(
        shape=(3, 3, 3),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
    )
    parquet_path = tmp_path / "demo.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


def _simple_schema(attr_col: str = "depth") -> DataFrameSchema:
    """Return a minimal pandera schema that coerces *attr_col* to float."""
    return DataFrameSchema(
        columns={attr_col: Column(float, coerce=True, nullable=True)},
        strict=False,
    )


# ---------------------------------------------------------------------------
# _load_schema
# ---------------------------------------------------------------------------

def test_load_schema_from_dataframeschema_object():
    schema = _simple_schema()
    loaded = ParquetBlockModel._load_schema(schema)
    assert loaded is schema


def test_load_schema_from_yaml_file(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    from df_eval.utils.pandera_io_compat import to_yaml

    schema = _simple_schema()
    yaml_path = tmp_path / "schema.yaml"
    to_yaml(schema, str(yaml_path))

    loaded = ParquetBlockModel._load_schema(yaml_path)
    assert isinstance(loaded, DataFrameSchema)


def test_load_schema_invalid_type_raises():
    with pytest.raises(TypeError, match="pandera DataFrameSchema"):
        ParquetBlockModel._load_schema(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# __init__ stores schema
# ---------------------------------------------------------------------------

def test_init_stores_schema(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    pbm = ParquetBlockModel.from_parquet(parquet_path)

    schema = _simple_schema()
    pbm2 = ParquetBlockModel(blockmodel_path=pbm.blockmodel_path,
                             geometry=pbm.geometry,
                             schema=schema)
    assert pbm2.schema is schema


def test_init_without_schema_is_none(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert pbm.schema is None


# ---------------------------------------------------------------------------
# from_parquet with schema
# ---------------------------------------------------------------------------

def test_from_parquet_with_schema_stores_schema(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    schema = _simple_schema()
    pbm = ParquetBlockModel.from_parquet(parquet_path, schema=schema)
    assert pbm.schema is schema


def test_from_parquet_without_schema_stores_none(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert pbm.schema is None


def test_from_parquet_schema_coerces_int_to_float(tmp_path):
    """Schema with coerce=True should resolve int→float conflicts across chunks."""
    parquet_path = _make_demo_parquet(tmp_path)

    # Overwrite with a version where 'depth' is integer in one row group
    df = pd.read_parquet(parquet_path)
    df["depth"] = df["depth"].astype(int)
    df.to_parquet(parquet_path)

    schema = DataFrameSchema(
        columns={"depth": Column(float, coerce=True, nullable=True)},
        strict=False,
    )
    pbm = ParquetBlockModel.from_parquet(parquet_path, schema=schema)
    assert pbm.schema is schema


# ---------------------------------------------------------------------------
# from_dataframe with schema
# ---------------------------------------------------------------------------

def test_from_dataframe_with_schema_stores_schema(tmp_path):
    df = create_demo_blockmodel(
        shape=(3, 3, 3), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)
    )
    # from_dataframe requires a MultiIndex with names ['x', 'y', 'z']
    df = df.set_index(["x", "y", "z"])
    schema = _simple_schema()
    filename = tmp_path / "demo.parquet"
    pbm = ParquetBlockModel.from_dataframe(df, filename=filename, schema=schema)
    assert pbm.schema is schema


def test_from_dataframe_without_schema_stores_none(tmp_path):
    df = create_demo_blockmodel(
        shape=(3, 3, 3), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)
    )
    df = df.set_index(["x", "y", "z"])
    filename = tmp_path / "demo.parquet"
    pbm = ParquetBlockModel.from_dataframe(df, filename=filename)
    assert pbm.schema is None


# ---------------------------------------------------------------------------
# from_geometry with schema
# ---------------------------------------------------------------------------

def test_from_geometry_with_schema_stores_schema(tmp_path):
    geometry = RegularGeometry(
        local=LocalGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(3, 3, 3),
        ),
        world=WorldFrame(),
    )
    schema = _simple_schema()
    pbm = ParquetBlockModel.from_geometry(
        geometry, path=tmp_path / "geo.pbm", schema=schema
    )
    assert pbm.schema is schema


# ---------------------------------------------------------------------------
# _validate_chunk
# ---------------------------------------------------------------------------

def test_validate_chunk_passes_for_valid_data():
    df = pd.DataFrame({"grade": [1.0, 2.0, 3.0]})
    schema = DataFrameSchema(
        columns={"grade": Column(float, nullable=False)},
        strict=False,
    )
    result = ParquetBlockModel._validate_chunk(df, schema)
    assert list(result["grade"]) == [1.0, 2.0, 3.0]


def test_validate_chunk_coerces_int_to_float():
    df = pd.DataFrame({"grade": [1, 2, 3]})
    schema = DataFrameSchema(
        columns={"grade": Column(float, coerce=True)},
        strict=False,
    )
    result = ParquetBlockModel._validate_chunk(df, schema)
    assert result["grade"].dtype == float


def test_validate_chunk_raises_on_invalid_data():
    df = pd.DataFrame({"grade": ["a", "b", "c"]})
    schema = DataFrameSchema(
        columns={"grade": Column(float, coerce=False)},
        strict=False,
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        ParquetBlockModel._validate_chunk(df, schema)


# ---------------------------------------------------------------------------
# validate() method
# ---------------------------------------------------------------------------

def test_validate_returns_true_with_valid_schema(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    schema = DataFrameSchema(
        columns={"depth": Column(float, nullable=True)},
        strict=False,
    )
    assert pbm.validate(schema=schema) is True


def test_validate_uses_stored_schema_when_none_passed(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    schema = DataFrameSchema(
        columns={"depth": Column(float, nullable=True)},
        strict=False,
    )
    pbm = ParquetBlockModel.from_parquet(parquet_path, schema=schema)
    assert pbm.validate() is True


def test_validate_raises_when_no_schema_available(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    with pytest.raises(ValueError, match="No schema provided"):
        pbm.validate()


def test_validate_sample_chunks(tmp_path):
    """Passing sample_chunks=1 should only validate the first chunk."""
    parquet_path = _make_demo_parquet(tmp_path)
    schema = DataFrameSchema(
        columns={"depth": Column(float, nullable=True)},
        strict=False,
    )
    pbm = ParquetBlockModel.from_parquet(parquet_path, schema=schema)
    assert pbm.validate(sample_chunks=1) is True


# ---------------------------------------------------------------------------
# Cross-chunk schema conflict (no pandera schema) — cast fallback
# ---------------------------------------------------------------------------

def test_cross_chunk_type_mismatch_resolved_without_schema(tmp_path):
    """Without a schema the cast fallback in _write_canonical_pbm should handle
    cross-chunk int/float differences on attribute columns."""
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner)

    # Force the 'depth' column to int
    df["depth"] = df["depth"].astype(int)
    parquet_path = tmp_path / "mixed.parquet"
    df.to_parquet(parquet_path, row_group_size=4)

    # Manually write a parquet with two row groups having different dtypes
    # (int64 first half, float64 second half) to simulate the real-world issue.
    half = len(df) // 2
    table1 = pa.Table.from_pandas(df.iloc[:half].assign(depth=df.iloc[:half]["depth"].astype("int64")),
                                   preserve_index=False)
    table2 = pa.Table.from_pandas(df.iloc[half:].assign(depth=df.iloc[half:]["depth"].astype("float64")),
                                   preserve_index=False)
    # Align schemas so the file itself is valid Arrow multi-chunk
    table2 = table2.cast(table1.schema)
    with pq.ParquetWriter(parquet_path, table1.schema) as writer:
        writer.write_table(table1)
        writer.write_table(table2)

    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert pbm is not None
    assert pbm.data.shape[0] == len(df)
