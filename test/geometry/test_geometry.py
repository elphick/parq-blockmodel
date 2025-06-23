import numpy as np
import pandas as pd
import pytest

from parq_blockmodel import RegularGeometry


def test_regular_geometry_init():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 2.0, 3.0),
        shape=(4, 5, 6)
    )
    assert geom.corner == (0.0, 0.0, 0.0)
    assert geom.block_size == (1.0, 2.0, 3.0)
    assert geom.shape == (4, 5, 6)
    assert geom.is_regular

def test_centroid_properties():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs='my_srs'
    )
    np.testing.assert_allclose(geom.centroid_u, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_v, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_w, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    assert geom.srs == 'my_srs'

def test_extents_and_bounding_box():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    extents = geom.extents
    assert extents[0] == (0.0, 2.0)
    assert extents[1] == (0.0, 2.0)
    assert extents[2] == (0.0, 2.0)
    assert geom.bounding_box == ((0.0, 2.0), (0.0, 2.0))

def test_to_json_and_from_json():
    geom = RegularGeometry(
        corner=(1.0, 2.0, 3.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    json_str = geom.to_json()
    geom2 = RegularGeometry.from_json(json_str)
    assert geom2.corner == [1.0, 2.0, 3.0]
    assert geom2.block_size == [1.0, 1.0, 1.0]
    assert geom2.shape == [2, 2, 2]

def test_nearest_centroid_lookup():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert geom.nearest_centroid_lookup(0.6, 0.6, 0.6) == (0.5, 0.5, 0.5)
    assert geom.nearest_centroid_lookup(1.4, 1.4, 1.4) == (1.5, 1.5, 1.5)

def test_is_compatible_true():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert geom1.is_compatible(geom2)

def test_is_compatible_false():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(0.5, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert not geom1.is_compatible(geom2)

def test_is_compatible_different_shapes():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(3, 3, 3)
    )
    assert not geom1.is_compatible(geom2)

def test_is_compatible_different_block_sizes():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(2.0, 2.0, 2.0),
        shape=(2, 2, 2)
    )
    assert not geom1.is_compatible(geom2)

def test_is_compatible_different_corners():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(1.0, 1.0, 1.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert geom1.is_compatible(geom2)

def test_is_compatible_different_srs():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs='EPSG:4326'
    )
    geom2 = RegularGeometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs='EPSG:3857'
    )
    assert not geom1.is_compatible(geom2)

def test_to_dataframe():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    df = geom.to_dataframe()
    expected_data = {
            'x': [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5],
            'y': [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5],
            'z': [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
        }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(df, expected_df)

    # check the df attrs contain the geometry information
    assert df.attrs['geometry']['corner'] == (0.0, 0.0, 0.0)
    assert df.attrs['geometry']['block_size'] == (1.0, 1.0, 1.0)
    assert df.attrs['geometry']['shape'] == (2, 2, 2)

def test_axis_rotation():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        axis_w=(0.0, 0.0, 1.0)
    )
    assert geom.axis_u == (1.0, 0.0, 0.0)
    assert geom.axis_v == (0.0, 1.0, 0.0)
    assert geom.axis_w == (0.0, 0.0, 1.0)

    # check the expected centroids
    np.testing.assert_allclose(geom.centroid_u, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_v, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_w, [0.5, 1.5])

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
        axis_w=axis_w
    )

    assert geom.axis_u == axis_u
    assert geom.axis_v == axis_v
    assert geom.axis_w == axis_w

    # Compute expected rotated centroids in x, y, z
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

    np.testing.assert_allclose(geom.centroid_u, np.unique(expected_x))
    np.testing.assert_allclose(geom.centroid_v, np.unique(expected_y))
    np.testing.assert_allclose(geom.centroid_w, np.unique(expected_z))

def test_axis_rotation_invalid():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(1.0, 1.0, 0.0),  # Not normalized
            axis_v=(0.0, 1.0, 0.0),
            axis_w=(0.0, 0.0, 1.0)
        )

def test_axis_rotation_non_orthogonal():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(1.0, 0.0, 0.0),
            axis_v=(1.0, 1.0, 0.0),  # Not orthogonal to axis_u
            axis_w=(0.0, 0.0, 1.0)
        )

def test_axis_rotation_non_normalized():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            corner=(0.0, 0.0, 0.0),
            block_size=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
            axis_u=(2.0, 0.0, 0.0),  # Not normalized
            axis_v=(0.0, 1.0, 0.0),
            axis_w=(0.0, 0.0, 1.0)
        )


def test_from_extents_block_alignment():
    # Define extents and block size that should align perfectly
    extents = ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))
    block_size = (1.0, 1.0, 1.0)
    # Should produce a 2x2x2 grid with centroids at 0.5, 1.5 in each axis
    geom = RegularGeometry.from_extents(extents, block_size)
    np.testing.assert_allclose(geom.corner, (0.0, 0.0, 0.0))
    assert geom.shape == (2, 2, 2)
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    # The extents should match the input
    assert geom.extents == extents

def test_from_extents_rounding_error():
    # Extents that are not perfectly divisible by block size
    extents = ((0.0, 2.1), (0.0, 2.1), (0.0, 2.1))
    block_size = (1.0, 1.0, 1.0)
    # Should produce a 3x3x3 grid, but check for off-by-one errors
    geom = RegularGeometry.from_extents(extents, block_size)
    assert geom.shape == (3, 3, 3)
    # The last block should cover the max extent
    assert geom.extents[0][1] >= extents[0][1]
    assert geom.extents[1][1] >= extents[1][1]
    assert geom.extents[2][1] >= extents[2][1]