import numpy as np
import pandas as pd
import pytest

from parq_blockmodel import RegularGeometry


def test_regular_geometry_init():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 2.0, 3.0),
        shape=(4, 5, 6),
    )
    assert geom.corner == (0.0, 0.0, 0.0)
    assert geom.block_size == (1.0, 2.0, 3.0)
    assert geom.shape == (4, 5, 6)


def test_centroid_properties():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs="my_srs",
    )
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    assert geom.srs == "my_srs"


def test_ijk_row_index_roundtrip():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(3, 4, 5),
    )
    rows = np.arange(np.prod(geom.shape))
    i, j, k = geom.ijk_from_row_index(rows)
    rows_back = geom.row_index_from_ijk(i, j, k)
    np.testing.assert_array_equal(rows_back, rows)


def test_to_multi_index_ijk():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    mi = geom.to_multi_index_ijk()
    expected = pd.MultiIndex.from_arrays(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],  # i
            [0, 0, 1, 1, 0, 0, 1, 1],  # j
            [0, 1, 0, 1, 0, 1, 0, 1],  # k
        ],
        names=["i", "j", "k"],
    )
    assert mi.equals(expected)


def test_to_multi_index_xyz():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    mi = geom.to_multi_index_xyz()
    expected = pd.MultiIndex.from_arrays(
        [
            [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5],
            [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5],
            [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5],
        ],
        names=["x", "y", "z"],
    )
    assert mi.equals(expected)


def test_to_dataframe_xyz():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    df = geom.to_dataframe_xyz()
    expected = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5],
            "y": [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5],
            "z": [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_axis_rotation_default():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        axis_w=(0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])


def test_axis_rotation_custom():
    from math import cos, sin, radians
    angle = radians(30)
    axis_u = (cos(angle), sin(angle), 0.0)
    axis_v = (-sin(angle), cos(angle), 0.0)
    axis_w = (0.0, 0.0, 1.0)
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        axis_u=axis_u,
        axis_v=axis_v,
        axis_w=axis_w,
    )

    expected_x = []
    expected_y = []
    expected_z = []
    for i in [0.5, 1.5]:
        for j in [0.5, 1.5]:
            for k in [0.5, 1.5]:
                x = i * cos(angle) + j * -sin(angle)
                y = i * sin(angle) + j * cos(angle)
                z = k
                expected_x.append(x)
                expected_y.append(y)
                expected_z.append(z)

    np.testing.assert_allclose(geom.centroid_x, expected_x)
    np.testing.assert_allclose(geom.centroid_y, expected_y)
    np.testing.assert_allclose(geom.centroid_z, expected_z)


def test_axis_rotation_invalid():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(1.0, 1.0, 0.0),
            axis_v=(0.0, 1.0, 0.0),
            axis_w=(0.0, 0.0, 1.0),
        )


def test_axis_rotation_non_orthogonal():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(1.0, 0.0, 0.0),
            axis_v=(1.0, 1.0, 0.0),
            axis_w=(0.0, 0.0, 1.0),
        )


def test_axis_rotation_non_normalized():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(2.0, 0.0, 0.0),
            axis_v=(0.0, 1.0, 0.0),
            axis_w=(0.0, 0.0, 1.0),
        )


def test_metadata_roundtrip():
    geom = RegularGeometry(
        corner=(1.0, 2.0, 3.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs="EPSG:4326",
    )
    meta = geom.to_metadata_dict()
    geom2 = RegularGeometry.from_metadata(meta)
    assert geom2.corner == geom.corner
    assert geom2.block_size == geom.block_size
    assert geom2.shape == geom.shape
    assert geom2.axis_u == geom.axis_u
    assert geom2.axis_v == geom.axis_v
    assert geom2.axis_w == geom.axis_w
    assert geom2.srs == geom.srs

