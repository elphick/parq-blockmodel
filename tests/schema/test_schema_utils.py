"""Tests for schema utility functions."""

import json
import pytest
from pathlib import Path
import pandas as pd

pytest_importorskip = pytest.importorskip


class TestLoadSchema:
    """Test schema loading utilities."""

    def test_load_schema_from_dataframeschema_object(self):
        """load_schema accepts DataFrameSchema object directly."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import load_schema

        schema = pandera.DataFrameSchema()
        result = load_schema(schema)
        assert result is schema

    def test_load_schema_from_yaml_path(self, tmp_path):
        """load_schema can load from YAML file."""
        pandera = pytest_importorskip("pandera")
        df_eval = pytest_importorskip("df_eval")
        from parq_blockmodel.schema.utils import load_schema, dump_schema_yaml

        # Create a simple schema and dump to YAML
        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
            "y": pandera.Column(float),
        })
        yaml_path = tmp_path / "test_schema.yaml"
        yaml_text = dump_schema_yaml(schema)
        yaml_path.write_text(yaml_text)

        # Load from YAML
        loaded = load_schema(yaml_path)
        assert loaded is not None
        assert hasattr(loaded, "columns")


class TestExtractRequiredColumns:
    """Test required columns extraction."""

    def test_extract_required_columns_includes_required_true(self):
        """extract_required_columns_from_schema includes columns with required=True."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import extract_required_columns_from_schema

        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float, required=True),
            "y": pandera.Column(float, required=False),
        })
        required = extract_required_columns_from_schema(schema)
        assert required == {"x"}

    def test_extract_required_columns_defaults_to_true(self):
        """extract_required_columns_from_schema defaults to required=True."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import extract_required_columns_from_schema

        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),  # No required specified
            "y": pandera.Column(float, required=False),
        })
        required = extract_required_columns_from_schema(schema)
        assert required == {"x"}


class TestExtractColumnAliases:
    """Test alias extraction from schema metadata."""

    def test_extract_column_aliases_from_df_eval_metadata(self):
        """extract_column_aliases_from_schema finds aliases in df-eval metadata."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import extract_column_aliases_from_schema

        schema = pandera.DataFrameSchema({
            "deposit_code": pandera.Column(
                str,
                metadata={"df-eval": {"alias": "deposit"}},
            ),
            "value": pandera.Column(float),
        })
        aliases = extract_column_aliases_from_schema(schema)
        assert aliases == {"deposit_code": "deposit"}

    def test_extract_column_aliases_empty_when_no_aliases(self):
        """extract_column_aliases_from_schema returns empty dict when no aliases."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import extract_column_aliases_from_schema

        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
            "y": pandera.Column(float),
        })
        aliases = extract_column_aliases_from_schema(schema)
        assert aliases == {}


class TestValidateChunk:
    """Test chunk validation."""

    def test_validate_chunk_passes_for_valid_data(self):
        """validate_chunk passes for valid DataFrame."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import validate_chunk

        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
            "y": pandera.Column(int),
        })
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1, 2]})
        result = validate_chunk(df, schema)
        assert len(result) == 2
        assert list(result.columns) == ["x", "y"]

    def test_validate_chunk_raises_for_invalid_data(self):
        """validate_chunk raises for invalid DataFrame."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.schema.utils import validate_chunk

        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
        })
        df = pd.DataFrame({"x": ["not_a_number"]})
        with pytest.raises(Exception):  # pandera.errors.SchemaError
            validate_chunk(df, schema)


class TestCompressionMetadata:
    """Test compression metadata helpers."""

    def test_resolve_active_compression_policy_default_is_snappy(self):
        from parq_blockmodel.schema.utils import resolve_active_compression_policy

        policy = resolve_active_compression_policy()
        assert policy["mode"] == "active"
        assert policy["default"] == {"codec": "snappy", "level": None}
        assert policy["columns"] == {}

    def test_resolve_active_compression_policy_integer_uses_zstd_level(self):
        from parq_blockmodel.schema.utils import resolve_active_compression_policy

        policy = resolve_active_compression_policy(5)
        assert policy["default"] == {"codec": "zstd", "level": 5}

    def test_build_schema_metadata_can_persist_compression(self):
        from parq_blockmodel.geometry import RegularGeometry
        from parq_blockmodel.schema.utils import build_schema_metadata, PBM_METADATA_KEY

        geometry = RegularGeometry.create()
        metadata = build_schema_metadata(
            geometry=geometry,
            schema=None,
            compression={
                "mode": "active",
                "default": {"codec": "snappy", "level": None},
                "columns": {},
            },
        )
        assert PBM_METADATA_KEY in metadata
        payload = json.loads(metadata[PBM_METADATA_KEY].decode("utf-8"))
        assert payload["compression"]["default"]["codec"] == "snappy"
