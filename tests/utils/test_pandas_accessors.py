"""Additional tests for parq_blockmodel.utils.pandas_accessors"""
import numpy as np
import pandas as pd
import pytest

from parq_blockmodel.utils.pandas_accessors import _geometry_from_df


def test_geometry_from_df_raises_without_attrs():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Cannot determine geometry"):
        _geometry_from_df(df)


def test_geometry_from_df_from_geometry_attrs():
    """Test path 2: lightweight 'geometry' attrs dict."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.attrs["geometry"] = {
        "corner": (0.0, 0.0, 0.0),
        "block_size": (5.0, 5.0, 5.0),
        "shape": (10, 10, 10),
        "rotation": {"azimuth": 0.0, "dip": 0.0, "plunge": 0.0},
    }
    geom = _geometry_from_df(df)
    assert geom is not None
    assert geom.local.shape == (10, 10, 10)


def test_geometry_from_df_from_geometry_attrs_no_rotation_key():
    """Test that missing 'rotation' key defaults to identity rotation."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.attrs["geometry"] = {
        "corner": (0.0, 0.0, 0.0),
        "block_size": (5.0, 5.0, 5.0),
        "shape": (4, 4, 4),
    }
    geom = _geometry_from_df(df)
    assert geom.local.block_size == (5.0, 5.0, 5.0)


def test_geometry_from_df_from_parq_blockmodel_attrs(tmp_path):
    """Test path 1: full 'parq-blockmodel' metadata attrs dict."""
    from parq_blockmodel import ParquetBlockModel, RegularGeometry
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "test_accessor.parquet", shape=(4, 4, 4)
    )
    df = pbm.read()
    # Manually inject 'parq-blockmodel' attrs (as from_dataframe would add)
    df.attrs["parq-blockmodel"] = pbm.geometry.to_metadata_dict()
    geom = _geometry_from_df(df)
    assert geom is not None
    assert geom.local.shape == (4, 4, 4)
    """Invalid grid_type should raise ValueError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Invalid grid_type"):
        df.to_pyvista(grid_type="unknown")


def test_to_pyvista_accessor_unstructured_without_block_size():
    """Unstructured grid without block_size should raise ValueError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="block_size must be provided"):
        df.to_pyvista(grid_type="unstructured")



