"""Tests for SchemaService class."""

import pytest
import pandas as pd
import numpy as np

pytest_importorskip = pytest.importorskip


class TestSchemaService:
    """Test SchemaService stateful operations."""

    def test_schema_service_initialization(self):
        """SchemaService initializes with geometry and schema."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry)
        assert service.geometry is geometry
        assert service.schema is None

    def test_intrinsic_operations(self):
        """SchemaService provides intrinsic volume operations."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(2, 3, 4), shape=(5, 5, 5)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry)
        ops = service.intrinsic_df_eval_operations()

        # Should have volume operation with correct block volume
        assert "volume" in ops
        assert ops["volume"]["kind"] == "expr"
        # Block volume = 2 * 3 * 4 = 24
        assert float(ops["volume"]["expr"]) == 24.0

    def test_available_operations_with_no_schema(self):
        """SchemaService returns only intrinsic operations without schema."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry, schema=None)
        ops = service.available_df_eval_operations()

        # Should have volume
        assert "volume" in ops

    def test_apply_intrinsic_operations(self):
        """SchemaService can apply intrinsic operations to DataFrame."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(2, 2, 2), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry)

        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        ops = {"volume": {"kind": "expr", "expr": repr(8.0)}}
        result = service.apply_intrinsic_operations(df, ops)

        assert "volume" in result.columns
        assert np.allclose(result["volume"].values, 8.0)

    def test_calculated_attribute_flags(self):
        """SchemaService classifies calculated columns as attributes."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry)

        calculated_cols = ["volume", "grade", "rock_type"]
        flags = service.calculated_attribute_flags(calculated_cols)

        # volume should be False (not an attribute), others True (attributes)
        assert flags.get("volume") is False
        assert flags.get("grade") is True
        assert flags.get("rock_type") is True

    def test_schema_column_names(self):
        """SchemaService extracts schema column names in order."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
            "grade": pandera.Column(float),
            "rock_type": pandera.Column(str),
        })
        service = SchemaService(geometry=geometry, schema=schema)

        names = service.get_schema_column_names()
        assert names == ["x", "grade", "rock_type"]

    def test_validate_chunk_without_schema_raises(self):
        """SchemaService.validate_chunk raises if no schema available."""
        pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        service = SchemaService(geometry=geometry, schema=None)
        df = pd.DataFrame({"x": [1.0]})

        with pytest.raises(ValueError, match="no schema available"):
            service.validate_chunk(df)

    def test_validate_chunk_with_schema(self):
        """SchemaService.validate_chunk validates DataFrame."""
        pandera = pytest_importorskip("pandera")
        from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
        from parq_blockmodel.schema.service import SchemaService

        geometry = RegularGeometry(
            local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
            world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
        )
        schema = pandera.DataFrameSchema({
            "x": pandera.Column(float),
        })
        service = SchemaService(geometry=geometry, schema=schema)
        df = pd.DataFrame({"x": [1.0, 2.0]})

        result = service.validate_chunk(df)
        assert len(result) == 2
