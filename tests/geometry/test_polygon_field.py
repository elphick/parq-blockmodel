import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon, box

from parq_blockmodel import LocalGeometry, PolygonField, RegularGeometry, WorldFrame


def make_geometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(3, 3, 1),
    axis_u=(1.0, 0.0, 0.0),
    axis_v=(0.0, 1.0, 0.0),
    axis_w=(0.0, 0.0, 1.0),
):
    return RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
    )


def test_polygon_field_evaluate_single_polygon():
    geometry = make_geometry(shape=(3, 3, 1))
    polygon = PolygonField.from_shapely(box(0.0, 0.0, 2.0, 2.0))

    mask = polygon.evaluate(geometry)

    expected = np.array([True, True, False, True, True, False, False, False, False], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_polygon_field_name_defaults_to_none():
    field = PolygonField.from_shapely(box(0.0, 0.0, 1.0, 1.0))
    assert field.name is None


def test_polygon_field_name_is_preserved():
    field = PolygonField.from_shapely(box(0.0, 0.0, 1.0, 1.0), name="lease_a")
    assert field.name == "lease_a"


def test_polygon_field_evaluate_multipolygon_collection():
    geometry = make_geometry(shape=(3, 3, 1))
    polygons = MultiPolygon([box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)])
    field = PolygonField.from_shapely(polygons)

    mask = field.evaluate(geometry)

    expected = np.array([True, False, False, False, True, False, False, False, False], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_polygon_field_evaluate_rotated_geometry():
    angle = np.deg2rad(45.0)
    axis_u = (float(np.cos(angle)), float(np.sin(angle)), 0.0)
    axis_v = (float(-np.sin(angle)), float(np.cos(angle)), 0.0)
    geometry = make_geometry(shape=(2, 2, 1), axis_u=axis_u, axis_v=axis_v)

    x, y = geometry.centroid_x[0], geometry.centroid_y[0]
    field = PolygonField.from_shapely(box(x - 0.1, y - 0.1, x + 0.1, y + 0.1))

    mask = field.evaluate(geometry)

    expected = np.array([True, False, False, False], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_polygon_field_validation_invalid_geometry():
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    with pytest.raises(ValueError, match="Invalid polygon geometry"):
        PolygonField.from_shapely(invalid)


def test_polygon_field_boundary_points_are_inside():
    geometry = make_geometry(shape=(1, 1, 1))
    field = PolygonField.from_shapely(box(0.5, 0.0, 1.5, 1.0))

    mask = field.evaluate(geometry)

    np.testing.assert_array_equal(mask, np.array([True], dtype=bool))


def test_polygon_field_evaluate_accepts_geometry_owner():
    class GeometryOwner:
        def __init__(self, geometry):
            self.geometry = geometry

    geometry = make_geometry(shape=(1, 1, 1))
    field = PolygonField.from_shapely(box(0.0, 0.0, 1.0, 1.0))

    mask = field.evaluate(GeometryOwner(geometry))

    np.testing.assert_array_equal(mask, np.array([True], dtype=bool))


def test_polygon_field_from_geoparquet_roundtrip(tmp_path):
    gpd = pytest.importorskip("geopandas")

    path = tmp_path / "regions.parquet"
    gdf = gpd.GeoDataFrame(
        {"name": ["lease_a"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)

    field = PolygonField.from_geoparquet(path, name="lease_a")
    assert field.name == "lease_a"
    assert field.geometry.equals(box(0.0, 0.0, 1.0, 1.0))


def test_polygon_field_from_geoparquet_missing_name_raises(tmp_path):
    gpd = pytest.importorskip("geopandas")

    path = tmp_path / "regions.parquet"
    gdf = gpd.GeoDataFrame(
        {"name": ["lease_a"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)

    with pytest.raises(ValueError, match="No polygon found"):
        PolygonField.from_geoparquet(path, name="lease_b")


def test_polygon_field_from_geoparquet_duplicate_name_raises(tmp_path):
    gpd = pytest.importorskip("geopandas")

    path = tmp_path / "regions.parquet"
    gdf = gpd.GeoDataFrame(
        {
            "name": ["lease_a", "lease_a"],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)],
        },
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)

    with pytest.raises(ValueError, match="Multiple polygons found"):
        PolygonField.from_geoparquet(path, name="lease_a")
