"""Tests for parq_blockmodel.utils.geometry_utils"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parq_blockmodel.utils.geometry_utils import (
    validate_geometry,
    validate_axes_orthonormal,
    angles_to_axes,
    axes_to_angles,
    rotate_points,
    dense_ijk_multiindex,
)


# ---------------------------------------------------------------------------
# dense_ijk_multiindex
# ---------------------------------------------------------------------------

def test_dense_ijk_multiindex_shape():
    idx = dense_ijk_multiindex((2, 3, 4))
    assert len(idx) == 2 * 3 * 4
    assert idx.names == ["i", "j", "k"]


# ---------------------------------------------------------------------------
# validate_geometry
# ---------------------------------------------------------------------------

def _write_parquet(tmp_path: Path, df: pd.DataFrame, fname="test.parquet") -> Path:
    table = pa.Table.from_pandas(df, preserve_index=False)
    p = tmp_path / fname
    pq.write_table(table, p)
    return p


def test_validate_geometry_missing_column(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})  # missing z
    p = _write_parquet(tmp_path, df)
    with pytest.raises(ValueError, match="Missing index columns"):
        validate_geometry(p)


def test_validate_geometry_nan_in_column(tmp_path):
    df = pd.DataFrame({
        "x": [1.0, None, 3.0],
        "y": [1.0, 2.0, 3.0],
        "z": [1.0, 2.0, 3.0],
    })
    p = _write_parquet(tmp_path, df)
    with pytest.raises(ValueError, match="contains NaN values"):
        validate_geometry(p)


def test_validate_geometry_too_few_unique(tmp_path):
    df = pd.DataFrame({
        "x": [1.0, 1.0],
        "y": [1.0, 1.0],
        "z": [1.0, 1.0],
    })
    p = _write_parquet(tmp_path, df)
    with pytest.raises(ValueError, match="not regular"):
        validate_geometry(p)


def test_validate_geometry_irregular_spacing(tmp_path):
    # x values are not evenly spaced
    df = pd.DataFrame({
        "x": [1.0, 2.0, 4.0],
        "y": [1.0, 2.0, 3.0],
        "z": [1.0, 2.0, 3.0],
    })
    p = _write_parquet(tmp_path, df)
    with pytest.raises(ValueError, match="evenly spaced"):
        validate_geometry(p)


def test_validate_geometry_valid(tmp_path):
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
        "z": [1.0, 2.0, 3.0],
    })
    p = _write_parquet(tmp_path, df)
    # Should not raise
    validate_geometry(p)


# ---------------------------------------------------------------------------
# validate_axes_orthonormal
# ---------------------------------------------------------------------------

def test_validate_axes_orthonormal_identity():
    assert validate_axes_orthonormal([1, 0, 0], [0, 1, 0], [0, 0, 1])


def test_validate_axes_orthonormal_non_unit():
    assert not validate_axes_orthonormal([2, 0, 0], [0, 1, 0], [0, 0, 1])


def test_validate_axes_orthonormal_non_orthogonal():
    assert not validate_axes_orthonormal([1, 0, 0], [1, 0, 0], [0, 0, 1])


# ---------------------------------------------------------------------------
# angles_to_axes
# ---------------------------------------------------------------------------

def test_angles_to_axes_identity():
    u, v, w = angles_to_axes(0, 0, 0)
    assert u == (1.0, 0.0, 0.0)
    assert v == (0.0, 1.0, 0.0)
    assert w == (0.0, 0.0, 1.0)


def test_angles_to_axes_azimuth_90():
    u, v, w = angles_to_axes(axis_azimuth=90, axis_dip=0, axis_plunge=0)
    # u should point roughly along Y
    assert np.isclose(u[0], 0.0, atol=1e-6)
    assert np.isclose(u[1], 1.0, atol=1e-6)


def test_angles_to_axes_returns_orthonormal():
    u, v, w = angles_to_axes(axis_azimuth=45, axis_dip=30, axis_plunge=15)
    assert validate_axes_orthonormal(u, v, w)


# ---------------------------------------------------------------------------
# axes_to_angles
# ---------------------------------------------------------------------------

def test_axes_to_angles_identity():
    az, dip, plunge = axes_to_angles([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert az == 0.0
    assert dip == 0.0
    assert plunge == 0.0


def test_axes_to_angles_non_identity():
    u, v, w = angles_to_axes(axis_azimuth=30, axis_dip=0, axis_plunge=0)
    az, dip, plunge = axes_to_angles(u, v, w)
    assert np.isclose(az, 30.0, atol=1e-6)


# ---------------------------------------------------------------------------
# rotate_points
# ---------------------------------------------------------------------------

def test_rotate_points_identity():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = rotate_points(pts, azimuth=0, dip=0, plunge=0)
    np.testing.assert_allclose(rotated, pts, atol=1e-10)


def test_rotate_points_azimuth_90():
    pts = np.array([[1.0, 0.0, 0.0]])
    rotated = rotate_points(pts, azimuth=90, dip=0, plunge=0)
    # After 90° azimuth, (1,0,0) maps to roughly (0,1,0)
    assert rotated.shape == (1, 3)

