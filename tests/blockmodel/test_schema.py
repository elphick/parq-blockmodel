"""Tests for pandera schema support in ParquetBlockModel."""
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

pandera = pytest.importorskip("pandera", reason="pandera not installed")

from pandera import DataFrameSchema, Column

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
    from df_eval.pandera import dump_pandera_schema_yaml

    schema = _simple_schema()
    yaml_path = tmp_path / "schema.yaml"
    yaml_path.write_text(dump_pandera_schema_yaml(schema), encoding="utf-8")

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


def test_validate_accepts_yaml_schema_path(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    from df_eval.pandera import dump_pandera_schema_yaml

    parquet_path = _make_demo_parquet(tmp_path)
    schema = _simple_schema()
    yaml_path = tmp_path / "schema.yaml"
    yaml_path.write_text(dump_pandera_schema_yaml(schema), encoding="utf-8")

    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert pbm.validate(schema=yaml_path) is True

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


def test_schema_roundtrip_is_recovered_from_pbm_metadata(tmp_path):
    parquet_path = _make_demo_parquet(tmp_path)
    schema = _simple_schema()
    pbm = ParquetBlockModel.from_parquet(parquet_path, schema=schema)

    metadata = pq.read_metadata(pbm.blockmodel_path).metadata or {}
    assert b"parq-blockmodel-schema-yaml" in metadata

    reloaded = ParquetBlockModel(blockmodel_path=pbm.blockmodel_path, geometry=pbm.geometry)
    assert reloaded.schema is not None
    assert set(reloaded.schema.columns) == set(schema.columns)


def _calculated_schema() -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "density": Column(float, coerce=True, nullable=True),
            "grade": Column(float, coerce=True, nullable=True, required=False),
            "tonnes": Column(
                float,
                coerce=True,
                nullable=True,
                required=False,
                metadata={"df-eval": {"expr": "density * volume"}},
            ),
            "contained_metal": Column(
                float,
                coerce=True,
                nullable=True,
                required=False,
                metadata={"df-eval": {"expr": "tonnes * grade"}},
            ),
        },
        strict=False,
    )


def test_read_computes_simple_calculated_column(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    schema = _calculated_schema()

    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "simple_calc.parquet",
        schema=schema,
    )
    result = pbm.read(columns=["tonnes"], index="ijk", dense=True)
    assert list(result.columns) == ["tonnes"]
    np.testing.assert_allclose(result["tonnes"].to_numpy(), np.full(len(result), 2.5))


def test_read_computes_chained_calculated_columns(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = np.linspace(2.0, 3.4, len(df))
    df["grade"] = np.linspace(0.1, 0.8, len(df))
    schema = _calculated_schema()

    pbm = ParquetBlockModel.from_dataframe(
        df[["density", "grade"]],
        filename=tmp_path / "chained_calc.parquet",
        schema=schema,
    )
    result = pbm.read(columns=["contained_metal"], index="ijk", dense=True)
    expected = df["density"].to_numpy() * df["grade"].to_numpy()
    np.testing.assert_allclose(result["contained_metal"].to_numpy(), expected)


def test_read_include_calculated_materializes_schema_columns(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    df["grade"] = 0.5
    schema = _calculated_schema()

    pbm = ParquetBlockModel.from_dataframe(
        df[["density", "grade"]],
        filename=tmp_path / "materialized_calc.parquet",
        schema=schema,
    )
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" not in raw.columns
    assert "contained_metal" not in raw.columns

    result = pbm.read(index="ijk", dense=True, include_calculated=True)
    assert "volume" in result.columns
    assert "tonnes" in result.columns
    assert "contained_metal" in result.columns
    np.testing.assert_allclose(result["volume"].to_numpy(), np.full(len(result), 1.0))
    np.testing.assert_allclose(result["tonnes"].to_numpy(), np.full(len(result), 2.5))
    np.testing.assert_allclose(result["contained_metal"].to_numpy(), np.full(len(result), 1.25))


def test_read_calculated_column_raises_without_schema(tmp_path):
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "no_schema.parquet",
    )
    with pytest.raises(ValueError, match="defined by available df-eval operations|calculated"):
        pbm.read(columns=["tonnes"], index="ijk", dense=True)


def test_read_undefined_requested_column_raises(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    schema = _calculated_schema()

    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "undefined_calc.parquet",
        schema=schema,
    )
    with pytest.raises(ValueError, match="neither on-disk nor defined"):
        pbm.read(columns=["undefined_output"], index="ijk", dense=True)


def test_column_properties_distinguish_persisted_and_calculated(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    df["grade"] = 0.5
    schema = _calculated_schema()

    pbm = ParquetBlockModel.from_dataframe(
        df[["density", "grade"]],
        filename=tmp_path / "column_properties.parquet",
        schema=schema,
    )

    assert pbm.persisted_columns == pbm.columns
    assert set(pbm.position_columns) == {"block_id", "world_id", "x", "y", "z", "i", "j", "k"}
    assert set(pbm.persisted_attributes) == {"density", "grade"}
    assert set(pbm.calculated_columns) == {"volume", "tonnes", "contained_metal"}
    assert set(pbm.calculated_attributes) == {"tonnes", "contained_metal"}


def test_read_intrinsic_volume_without_schema(tmp_path):
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    pbm = ParquetBlockModel.from_dataframe(
        df[["depth"]],
        filename=tmp_path / "intrinsic_volume_only.parquet",
    )
    volume = pbm.read(columns=["volume"], index="ijk", dense=True)["volume"].to_numpy()
    np.testing.assert_allclose(volume, np.full(volume.shape, 1.0))


def test_include_calculated_adds_intrinsic_volume_without_schema(tmp_path):
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    pbm = ParquetBlockModel.from_dataframe(
        df[["depth"]],
        filename=tmp_path / "intrinsic_volume_include_calculated.parquet",
    )
    out = pbm.read(index="ijk", dense=True, include_calculated=True)
    assert "volume" in out.columns
    np.testing.assert_allclose(out["volume"].to_numpy(), np.full(len(out), 1.0))


def test_column_properties_calculated_columns_empty_without_schema(tmp_path):
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "column_properties_no_schema.parquet",
    )
    assert pbm.calculated_columns == ["volume"]
    assert pbm.calculated_attributes == []


# ---------------------------------------------------------------------------
# Engine Registration (Options 1 & 2)
# ---------------------------------------------------------------------------

def test_engine_initializer_at_construction(tmp_path):
    """engine_initializer provided at construction time works."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    from df_eval import DictResolver
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    # Create a DictResolver for density classification
    density_class_resolver = DictResolver({
        2.0: "Low",
        2.5: "Medium",
        3.0: "High",
    }, default="Unknown")
    
    def setup_engine(engine):
        engine.register_resolver("density_class", density_class_resolver)
        return engine
    
    # Create schema with lookup operation that uses resolver
    schema = DataFrameSchema(
        columns={
            "density": Column(float, coerce=True, nullable=True),
            "density_name": Column(
                str,
                coerce=True,
                nullable=True,
                required=False,
                metadata={"df-eval": {
                    "lookup": {
                        "resolver": "density_class",
                        "key": "density",
                        "on_missing": "default"
                    }
                }},
            ),
        },
        strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "with_initializer.parquet",
        schema=schema,
        engine_initializer=setup_engine,
    )
    
    # Verify schema was stored
    assert pbm.schema is not None
    assert pbm._engine_initializer is setup_engine


def test_configure_engine_post_load(tmp_path):
    """configure_engine() method stores initializer for post-load configuration."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    from df_eval import DictResolver
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    rock_type_resolver = DictResolver({
        1: "Granite",
        2: "Diorite",
        3: "Gabbro",
    })
    
    def setup_engine(engine):
        engine.register_resolver("rock_type", rock_type_resolver)
        return engine
    
    schema = DataFrameSchema(
        columns={
            "density": Column(float, coerce=True, nullable=True),
        },
        strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "post_config.parquet",
        schema=schema,
    )
    
    assert pbm._engine_initializer is None
    
    pbm.configure_engine(setup_engine)
    assert pbm._engine_initializer is setup_engine


def test_engine_initializer_none_by_default(tmp_path):
    """engine_initializer defaults to None for backward compatibility."""
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density"]],
       filename=tmp_path / "no_initializer.parquet",
    )
    
    assert pbm._engine_initializer is None


def test_configure_engine_can_be_called_multiple_times(tmp_path):
    """configure_engine() can be called multiple times."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    def setup1(engine):
       return engine
    
    def setup2(engine):
       return engine
    
    schema = DataFrameSchema(
       columns={"density": Column(float, nullable=True)},
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density"]],
       filename=tmp_path / "reconfig.parquet",
       schema=schema,
    )
    
    pbm.configure_engine(setup1)
    assert pbm._engine_initializer is setup1
    
    pbm.configure_engine(setup2)
    assert pbm._engine_initializer is setup2


def test_read_without_engine_initializer_works(tmp_path):
    """Reading without engine_initializer still works (backward compat)."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    schema = _calculated_schema()
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density"]],
       filename=tmp_path / "simple_read.parquet",
       schema=schema,
    )
    
    # Should work without engine_initializer
    result = pbm.read(columns=["tonnes"], index="ijk", dense=True)
    assert "tonnes" in result.columns
    np.testing.assert_allclose(result["tonnes"].to_numpy(), np.full(len(result), 2.5))


# ---------------------------------------------------------------------------
# Phase 1 & 2: Extract Required Columns & Persist Calculated Columns
# ---------------------------------------------------------------------------

def test_extract_required_columns_from_schema_includes_required_true():
    """_extract_required_columns_from_schema should include columns with required=True."""
    schema = DataFrameSchema(
       columns={
           "persisted_col": Column(float, required=True),
           "calc_col": Column(float, required=False),
       },
       strict=False,
    )
    required = ParquetBlockModel._extract_required_columns_from_schema(schema)
    assert "persisted_col" in required
    assert "calc_col" not in required


def test_extract_required_columns_defaults_to_required_true():
    """_extract_required_columns_from_schema should default to required=True if not specified."""
    schema = DataFrameSchema(
       columns={
           "default_col": Column(float),  # No required= specified, defaults to True
           "explicit_false": Column(float, required=False),
       },
       strict=False,
    )
    required = ParquetBlockModel._extract_required_columns_from_schema(schema)
    assert "default_col" in required
    assert "explicit_false" not in required


def test_extract_required_columns_empty_schema():
    """_extract_required_columns_from_schema should return empty set for None schema."""
    required = ParquetBlockModel._extract_required_columns_from_schema(None)
    assert required == set()


def test_from_dataframe_excludes_non_required_columns_from_persisted(tmp_path):
    """Columns with required=False should NOT be persisted to disk by from_dataframe."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    df["grade"] = 0.5
    
    schema = DataFrameSchema(
       columns={
           "density": Column(float, required=True),
           "grade": Column(float, required=True),
           "tonnes": Column(
               float,
               required=False,
               metadata={"df-eval": {"expr": "density * volume"}}
           ),
       },
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density", "grade"]],
       filename=tmp_path / "exclude_non_required.parquet",
       schema=schema,
    )
    
    # tonnes should NOT be in persisted columns
    assert "tonnes" not in pbm.persisted_columns
    assert "tonnes" in pbm.calculated_columns
    
    # Reading raw data should NOT include tonnes
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" not in raw.columns


def test_from_parquet_excludes_non_required_columns_from_persisted(tmp_path):
    """Columns with required=False should NOT be persisted by from_parquet."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["density"] = 2.5
    source_parquet = tmp_path / "source.parquet"
    df.to_parquet(source_parquet)
    
    schema = DataFrameSchema(
       columns={
           "density": Column(float, required=True),
           "tonnes": Column(
               float,
               required=False,
               metadata={"df-eval": {"expr": "density * volume"}}
           ),
       },
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_parquet(source_parquet, schema=schema)
    
    # tonnes should NOT be in persisted columns
    assert "tonnes" not in pbm.persisted_columns
    assert "tonnes" in pbm.calculated_columns
    
    # Reading raw data should NOT include tonnes
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" not in raw.columns


def test_from_dataframe_persists_calculated_columns_with_required_true(tmp_path):
    """Calculated columns with required=True should be persisted to disk."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = 2.5
    
    schema = DataFrameSchema(
       columns={
           "density": Column(float, required=True),
           "tonnes": Column(
               float,
               required=True,  # This should be persisted
               metadata={"df-eval": {"expr": "density * volume"}}
           ),
       },
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density"]],
       filename=tmp_path / "persist_required.parquet",
       schema=schema,
    )
    
    # tonnes should be in persisted columns
    assert "tonnes" in pbm.persisted_columns
    
    # Reading raw data should include tonnes
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" in raw.columns
    np.testing.assert_allclose(raw["tonnes"].to_numpy(), np.full(len(raw), 2.5))


def test_from_parquet_persists_calculated_columns_with_required_true(tmp_path):
    """Calculated columns with required=True should be persisted by from_parquet."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["density"] = 2.5
    source_parquet = tmp_path / "source.parquet"
    df.to_parquet(source_parquet)
    
    schema = DataFrameSchema(
       columns={
           "density": Column(float, required=True),
           "tonnes": Column(
               float,
               required=True,  # This should be persisted
               metadata={"df-eval": {"expr": "density * volume"}}
           ),
       },
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_parquet(source_parquet, schema=schema)
    
    # tonnes should be in persisted columns
    assert "tonnes" in pbm.persisted_columns
    
    # Reading raw data should include tonnes
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" in raw.columns
    np.testing.assert_allclose(raw["tonnes"].to_numpy(), np.full(len(raw), 2.5))


def test_from_dataframe_persists_chained_calculated_columns(tmp_path):
    """Chained calculated columns should be persisted correctly."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2)).set_index(["x", "y", "z"])
    df["density"] = np.linspace(2.0, 3.0, len(df))
    df["grade"] = np.linspace(0.1, 0.5, len(df))
    
    schema = DataFrameSchema(
       columns={
           "density": Column(float, required=True),
           "grade": Column(float, required=True),
           "tonnes": Column(
               float,
               required=True,
               metadata={"df-eval": {"expr": "density * volume"}}
           ),
           "contained_metal": Column(
               float,
               required=True,
               metadata={"df-eval": {"expr": "tonnes * grade"}}
           ),
       },
       strict=False,
    )
    
    pbm = ParquetBlockModel.from_dataframe(
       df[["density", "grade"]],
       filename=tmp_path / "chained_calc_persist.parquet",
       schema=schema,
    )
    
    # Both calculated columns should be persisted
    assert "tonnes" in pbm.persisted_columns
    assert "contained_metal" in pbm.persisted_columns
    
    # Reading raw data should include both
    raw = pbm.read(index="ijk", dense=True)
    assert "tonnes" in raw.columns
    assert "contained_metal" in raw.columns
    
    # Verify calculations are correct
    expected_tonnes = df["density"].to_numpy() * 1.0  # volume = 1.0
    expected_metal = expected_tonnes * df["grade"].to_numpy()
    np.testing.assert_allclose(raw["tonnes"].to_numpy(), expected_tonnes, rtol=1e-5)
    np.testing.assert_allclose(raw["contained_metal"].to_numpy(), expected_metal, rtol=1e-5)


def test_validation_remains_read_only_after_from_parquet(tmp_path):
    """Validation with external schema should not modify .pbm file."""
    pytest.importorskip("df_eval", reason="df-eval not installed")
    
    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["density"] = 2.5
    source_parquet = tmp_path / "source.parquet"
    df.to_parquet(source_parquet)
    
    # Create without schema
    pbm = ParquetBlockModel.from_parquet(source_parquet)
    pbm_size_before = pbm.blockmodel_path.stat().st_size
    
    # Validate with external schema (should not modify file)
    schema = DataFrameSchema(
       columns={"density": Column(float, coerce=True)},
       strict=False,
    )
    assert pbm.validate(schema=schema) is True
    
    pbm_size_after = pbm.blockmodel_path.stat().st_size
    assert pbm_size_before == pbm_size_after  # File size should not change
    
    # Persisted columns should not change
    assert "density" in pbm.persisted_columns
    
    # Re-read and verify data unchanged
    raw = pbm.read(index="ijk", dense=True)
    assert "density" in raw.columns


def test_from_parquet_alias_is_applied_before_dependent_calculations(tmp_path):
    """Alias + decimals metadata should run before dependent calculations."""
    pytest.importorskip("df_eval", reason="df-eval not installed")

    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["deposit"] = np.linspace(0.12345, 0.82345, len(df))
    source_parquet = tmp_path / "alias_source.parquet"
    df.to_parquet(source_parquet)

    schema = DataFrameSchema(
        columns={
            "deposit_code": Column(
                float,
                required=True,
                metadata={"df-eval": {"alias": "deposit", "decimals": 3}},
            ),
            "deposit_plus_one": Column(
                float,
                required=True,
                metadata={"df-eval": {"expr": "deposit_code + 1"}},
            ),
        },
        strict=False,
    )

    pbm = ParquetBlockModel.from_parquet(source_parquet, schema=schema)
    out = pbm.read(columns=["deposit_code", "deposit_plus_one"], index="ijk", dense=True)
    expected = np.round(df["deposit"].to_numpy(), 3)
    np.testing.assert_allclose(out["deposit_code"].to_numpy(), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["deposit_plus_one"].to_numpy(), expected + 1.0, rtol=0, atol=1e-12)


def test_from_parquet_alias_only_is_applied_before_validation_and_persisted(tmp_path):
    """Alias-only metadata should run before validation and persist rounded output."""
    pytest.importorskip("df_eval", reason="df-eval not installed")

    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["deposit"] = np.linspace(0.12345, 0.82345, len(df))
    source_parquet = tmp_path / "alias_only_source.parquet"
    df.to_parquet(source_parquet)

    schema = DataFrameSchema(
        columns={
            "deposit_code": Column(
                float,
                required=True,
                metadata={"df-eval": {"alias": "deposit", "decimals": 3}},
            ),
        },
        strict=False,
    )

    pbm = ParquetBlockModel.from_parquet(source_parquet, schema=schema)
    out = pbm.read(columns=["deposit_code"], index="ijk", dense=True)
    expected = np.round(df["deposit"].to_numpy(), 3)
    np.testing.assert_allclose(out["deposit_code"].to_numpy(), expected, rtol=0, atol=1e-12)


def test_from_parquet_persists_schema_columns_in_schema_order(tmp_path):
    """Schema-defined persisted columns should follow schema definition order."""
    pytest.importorskip("df_eval", reason="df-eval not installed")

    df = create_demo_blockmodel(shape=(2, 2, 2))
    df["density"] = 2.5
    df["grade"] = np.linspace(0.1, 0.8, len(df))
    df["deposit"] = np.linspace(0.12345, 0.82345, len(df))
    source_parquet = tmp_path / "schema_order_source.parquet"
    df.to_parquet(source_parquet)

    schema = DataFrameSchema(
        columns={
            "grade": Column(float, required=True),
            "deposit_code": Column(
                float,
                required=True,
                metadata={"df-eval": {"alias": "deposit", "decimals": 3}},
            ),
            "tonnes": Column(
                float,
                required=True,
                metadata={"df-eval": {"expr": "density * volume"}},
            ),
        },
        strict=False,
    )

    pbm = ParquetBlockModel.from_parquet(source_parquet, schema=schema)
    persisted_columns = pq.read_table(pbm.blockmodel_path).column_names

    # Check relative order of schema-defined columns.
    grade_idx = persisted_columns.index("grade")
    deposit_code_idx = persisted_columns.index("deposit_code")
    tonnes_idx = persisted_columns.index("tonnes")
    assert grade_idx < deposit_code_idx < tonnes_idx
