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
        origin=(10.0, 0.0, 0.0),
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


def test_regular_geometry_exposes_flat_legacy_properties():
    geom = make_geometry()
    assert geom.corner == geom.local.corner
    assert geom.origin == geom.world.origin
    assert geom.axis_u == geom.world.axis_u


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
    assert geom2.world.origin == geom.world.origin
    assert geom2.world.srs == geom.world.srs


def test_old_style_corner_not_double_translated():
    geom = RegularGeometry(corner=(10.0, 20.0, 30.0), block_size=(2.0, 4.0, 6.0), shape=(1, 1, 1))
    np.testing.assert_allclose([geom.centroid_x[0], geom.centroid_y[0], geom.centroid_z[0]], [11.0, 22.0, 33.0])


def test_create_corner_not_double_translated():
    geom = RegularGeometry.create(corner=(10.0, 20.0, 30.0), block_size=(2.0, 4.0, 6.0), shape=(1, 1, 1))
    np.testing.assert_allclose([geom.centroid_x[0], geom.centroid_y[0], geom.centroid_z[0]], [11.0, 22.0, 33.0])


# ============================================================================
# Extents tests
# ============================================================================

def test_extents_orthogonal_axis_aligned_bounds():
    """Test axis-aligned bounds for orthogonal geometry."""
    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    min_xyz, max_xyz = geom.extents.axis_aligned_bounds
    # Bounds should be exactly the grid bounds for orthogonal geometry
    np.testing.assert_allclose(min_xyz, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(max_xyz, (2.0, 2.0, 2.0))


def test_extents_orthogonal_convenience_properties():
    """Test convenience properties for orthogonal geometry."""
    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    extents = geom.extents
    assert extents.xmin == 0.0
    assert extents.xmax == 2.0
    assert extents.ymin == 0.0
    assert extents.ymax == 2.0
    assert extents.zmin == 0.0
    assert extents.zmax == 2.0
    assert extents.xy_bbox == (0.0, 0.0, 2.0, 2.0)
    assert extents.extent_xy == [0.0, 2.0, 0.0, 2.0]


def test_extents_orthogonal_corners():
    """Test corners for orthogonal geometry."""
    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    corners = geom.extents.corners
    # Should have 8 corners
    assert corners.shape == (8, 3)
    # Expected corners for a 2x2x2 grid of 1x1x1 blocks
    expected_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(corners, expected_corners)


def test_extents_orthogonal_corners_with_offset():
    """Test corners for orthogonal geometry with non-zero origin."""
    geom = make_geometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(2.0, 3.0, 4.0),
        shape=(2, 2, 2),
    )
    corners = geom.extents.corners
    # Expected corners
    expected_corners = np.array(
        [
            [10.0, 20.0, 30.0],
            [14.0, 20.0, 30.0],  # 10 + 2*2
            [10.0, 26.0, 30.0],  # 20 + 2*3
            [14.0, 26.0, 30.0],
            [10.0, 20.0, 38.0],  # 30 + 2*4
            [14.0, 20.0, 38.0],
            [10.0, 26.0, 38.0],
            [14.0, 26.0, 38.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(corners, expected_corners)


def test_extents_rotated_axis_aligned_bounds():
    """Test axis-aligned bounds for rotated geometry."""
    from math import cos, sin, radians

    angle = radians(45)
    axis_u = (cos(angle), sin(angle), 0.0)
    axis_v = (-sin(angle), cos(angle), 0.0)
    axis_w = (0.0, 0.0, 1.0)

    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(1, 1, 1),
        axis_u=axis_u,
        axis_v=axis_v,
        axis_w=axis_w,
    )

    min_xyz, max_xyz = geom.extents.axis_aligned_bounds
    
    # For 45-degree rotation with 1x1x1 blocks:
    # Block extends from local (0,0,0) to (1,1,1)
    # The rotation matrix R has columns (axis_u, axis_v, axis_w)
    # Corners in local: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
    # After transformation R @ local_point:
    # (0,0,0) -> (0, 0, 0)
    # (1,0,0) -> (cos(45), sin(45), 0) ≈ (0.707, 0.707, 0)
    # (0,1,0) -> (-sin(45), cos(45), 0) ≈ (-0.707, 0.707, 0)
    # (1,1,0) -> (0, √2, 0) ≈ (0, 1.414, 0)
    # (0,0,1) -> (0, 0, 1)
    # (1,0,1) -> (0.707, 0.707, 1)
    # (0,1,1) -> (-0.707, 0.707, 1)
    # (1,1,1) -> (0, 1.414, 1)
    # X range: [-0.707, 0.707]
    # Y range: [0, 1.414]
    # Z range: [0, 1]
    sqrt2 = np.sqrt(2)
    expected_xmin, expected_xmax = -0.5 / np.sqrt(2) - 0.5 / np.sqrt(2), 0.5 / np.sqrt(2) + 0.5 / np.sqrt(2)
    expected_ymin, expected_ymax = 0.0, sqrt2
    expected_zmin, expected_zmax = 0.0, 1.0

    np.testing.assert_allclose(
        min_xyz,
        (expected_xmin, expected_ymin, expected_zmin),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        max_xyz,
        (expected_xmax, expected_ymax, expected_zmax),
        atol=1e-10,
    )


def test_extents_rotated_corners_enclose_blocks():
    """Test that corners correctly enclose rotated blocks."""
    from math import cos, sin, radians

    angle = radians(30)
    axis_u = (cos(angle), sin(angle), 0.0)
    axis_v = (-sin(angle), cos(angle), 0.0)
    axis_w = (0.0, 0.0, 1.0)

    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(2.0, 2.0, 2.0),
        shape=(2, 2, 2),
        axis_u=axis_u,
        axis_v=axis_v,
        axis_w=axis_w,
    )

    corners = geom.extents.corners
    assert corners.shape == (8, 3)

    # All centroids should be inside the oriented bounding box
    centroids = np.column_stack([geom.centroid_x, geom.centroid_y, geom.centroid_z])
    min_xyz, max_xyz = geom.extents.axis_aligned_bounds

    # Every centroid should be within the axis-aligned bounds
    for i in range(centroids.shape[0]):
        x, y, z = centroids[i]
        assert min_xyz[0] <= x <= max_xyz[0], f"Centroid {i}: x={x} outside bounds [{min_xyz[0]}, {max_xyz[0]}]"
        assert min_xyz[1] <= y <= max_xyz[1], f"Centroid {i}: y={y} outside bounds [{min_xyz[1]}, {max_xyz[1]}]"
        assert min_xyz[2] <= z <= max_xyz[2], f"Centroid {i}: z={z} outside bounds [{min_xyz[2]}, {max_xyz[2]}]"


def test_extents_no_state_duplication():
    """Test that Extents is a lightweight view with no duplicated state."""
    geom = make_geometry(
        corner=(1.0, 2.0, 3.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    extents1 = geom.extents
    extents2 = geom.extents

    # Both should reference the same geometry
    assert extents1._geometry is extents2._geometry
    # Properties should compute consistently
    np.testing.assert_allclose(
        extents1.axis_aligned_bounds[0],
        extents2.axis_aligned_bounds[0],
    )


def test_extents_corners_match_bounds():
    """Test that corners are consistent with axis-aligned bounds."""
    geom = make_geometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    corners = geom.extents.corners
    min_xyz, max_xyz = geom.extents.axis_aligned_bounds

    # All corners should lie on the boundary of or inside the bounds
    for corner in corners:
        x, y, z = corner
        assert min_xyz[0] <= x <= max_xyz[0]
        assert min_xyz[1] <= y <= max_xyz[1]
        assert min_xyz[2] <= z <= max_xyz[2]

    # At least one corner should touch each boundary
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    z_coords = corners[:, 2]

    np.testing.assert_allclose(x_coords.min(), min_xyz[0], atol=1e-10)
    np.testing.assert_allclose(x_coords.max(), max_xyz[0], atol=1e-10)
    np.testing.assert_allclose(y_coords.min(), min_xyz[1], atol=1e-10)
    np.testing.assert_allclose(y_coords.max(), max_xyz[1], atol=1e-10)
    np.testing.assert_allclose(z_coords.min(), min_xyz[2], atol=1e-10)
    np.testing.assert_allclose(z_coords.max(), max_xyz[2], atol=1e-10)


def test_extents_with_world_origin():
    """Test that Extents accounts for world origin correctly."""
    from parq_blockmodel import WorldFrame

    geom = RegularGeometry(
        local=LocalGeometry(corner=(0.0, 0.0, 0.0), block_size=(1.0, 1.0, 1.0), shape=(2, 2, 2)),
        world=WorldFrame(origin=(100.0, 200.0, 300.0)),
    )
    min_xyz, max_xyz = geom.extents.axis_aligned_bounds
    # Bounds should be offset by world origin
    np.testing.assert_allclose(min_xyz, (100.0, 200.0, 300.0))
    np.testing.assert_allclose(max_xyz, (102.0, 202.0, 302.0))

