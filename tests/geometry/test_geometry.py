import numpy as np
import pandas as pd
import pytest

from parq_blockmodel import RegularGeometry, LocalGeometry, WorldFrame


def make_geometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(2, 2, 2),
    axis_u=(1.0, 0.0, 0.0),
    axis_v=(0.0, 1.0, 0.0),
    axis_w=(0.0, 0.0, 1.0),
    srs=None,
):
    return RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w, srs=srs),
    )


def test_local_geometry_c_order_centroids():
    local = LocalGeometry(corner=(0.0, 0.0, 0.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2))
    centroids = local.centroids_uvw
    np.testing.assert_allclose(centroids[0], [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(centroids[1], [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(centroids[2], [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])


def test_world_frame_translation_and_rotation_roundtrip():
    frame = WorldFrame(
        world_origin=(10.0, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(-1.0, 0.0, 0.0),
        axis_w=(0.0, 0.0, 1.0),
    )
    local = np.array([[1.0], [2.0], [3.0]], dtype=float)
    world = frame.local_to_world(local)
    np.testing.assert_allclose(world.ravel(), [8.0, 1.0, 3.0])
    back = frame.world_to_local(world)
    np.testing.assert_allclose(back, local)


def test_regular_geometry_init():
    geom = make_geometry(block_size=(1.0, 2.0, 3.0), shape=(4, 5, 6))
    assert geom.local.corner == (0.0, 0.0, 0.0)
    assert geom.local.block_size == (1.0, 2.0, 3.0)
    assert geom.local.shape == (4, 5, 6)


def test_regular_geometry_composition_init():
    local = LocalGeometry(corner=(1.0, 2.0, 3.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2))
    world = WorldFrame(axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0), srs="EPSG:4326")
    geom = RegularGeometry(local=local, world=world)

    assert geom.local is local
    assert geom.world is world
    assert geom.local.corner == (1.0, 2.0, 3.0)
    assert geom.local.block_size == (1.0, 1.0, 1.0)
    assert geom.local.shape == (2, 2, 2)
    assert geom.world.srs == "EPSG:4326"


def test_regular_geometry_has_no_flat_legacy_properties():
    geom = make_geometry()

    with pytest.raises(AttributeError):
        _ = geom.corner

    with pytest.raises(AttributeError):
        _ = geom.axis_u


def test_centroid_properties():
    geom = make_geometry(srs="my_srs")
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    assert geom.world.srs == "my_srs"


def test_ijk_row_index_roundtrip():
    geom = make_geometry(shape=(3, 4, 5))
    rows = np.arange(np.prod(geom.local.shape))
    i, j, k = geom.ijk_from_row_index(rows)
    rows_back = geom.row_index_from_ijk(i, j, k)
    np.testing.assert_array_equal(rows_back, rows)


def test_to_multi_index_ijk():
    geom = make_geometry()
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
    geom = make_geometry()
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
    geom = make_geometry()
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
    geom = make_geometry()
    np.testing.assert_allclose(geom.centroid_x, [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_y, [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    np.testing.assert_allclose(geom.centroid_z, [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])


def test_axis_rotation_custom():
    from math import cos, sin, radians
    angle = radians(30)
    axis_u = (cos(angle), sin(angle), 0.0)
    axis_v = (-sin(angle), cos(angle), 0.0)
    axis_w = (0.0, 0.0, 1.0)
    geom = make_geometry(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

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


def test_to_pyvista_world_frame_is_default_and_rotated():
    pv = pytest.importorskip("pyvista")
    from parq_blockmodel.utils import angles_to_axes

    axis_u, axis_v, axis_w = angles_to_axes(axis_azimuth=30, axis_dip=0, axis_plunge=0)
    geom = make_geometry(shape=(2, 2, 1), axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

    grid = geom.to_pyvista()
    assert isinstance(grid, pv.ImageData)

    actual = np.asarray(grid.cell_centers().points)
    expected = geom.to_dataframe_xyz()[["x", "y", "z"]].to_numpy()

    actual = actual[np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))]
    expected = expected[np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))]
    np.testing.assert_allclose(actual, expected)


def test_to_pyvista_local_frame_is_axis_aligned():
    pv = pytest.importorskip("pyvista")
    from parq_blockmodel.utils import angles_to_axes

    axis_u, axis_v, axis_w = angles_to_axes(axis_azimuth=30, axis_dip=0, axis_plunge=0)
    geom = make_geometry(shape=(2, 2, 1), axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

    local_grid = geom.to_pyvista(frame="local")
    assert isinstance(local_grid, pv.ImageData)

    actual = np.asarray(local_grid.cell_centers().points)
    expected_local = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [1.5, 0.5, 0.5],
            [1.5, 1.5, 0.5],
        ],
        dtype=float,
    )

    actual = actual[np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))]
    expected_local = expected_local[np.lexsort((expected_local[:, 2], expected_local[:, 1], expected_local[:, 0]))]
    np.testing.assert_allclose(actual, expected_local)


def test_to_pyvista_invalid_frame_raises():
    geom = make_geometry()
    with pytest.raises(ValueError, match="frame must be one of"):
        geom.to_pyvista(frame="bad")


def test_axis_rotation_invalid():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            local=LocalGeometry(corner=(0.0, 0.0, 0.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2)),
            world=WorldFrame(axis_u=(1.0, 1.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0)),
        )


def test_axis_rotation_non_orthogonal():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            local=LocalGeometry(corner=(0.0, 0.0, 0.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2)),
            world=WorldFrame(axis_u=(1.0, 0.0, 0.0), axis_v=(1.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0)),
        )


def test_axis_rotation_non_normalized():
    with pytest.raises(ValueError, match="Axis vectors must be orthogonal and normalized."):
        RegularGeometry(
            local=LocalGeometry(corner=(0.0, 0.0, 0.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2)),
            world=WorldFrame(axis_u=(2.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0)),
        )


def test_metadata_roundtrip():
    geom = make_geometry(corner=(1.0, 2.0, 3.0), srs="EPSG:4326")
    meta = geom.to_metadata_dict()
    geom2 = RegularGeometry.from_metadata(meta)
    assert geom2.local.corner == geom.local.corner
    assert geom2.local.block_size == geom.local.block_size
    assert geom2.local.shape == geom.local.shape
    assert geom2.world.axis_u == geom.world.axis_u
    assert geom2.world.axis_v == geom.world.axis_v
    assert geom2.world.axis_w == geom.world.axis_w
    assert geom2.world.srs == geom.world.srs

