from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


def cell_polygon_xy(geometry, i: int, j: int) -> Polygon:
    return cell_run_polygon_xy(geometry, i_start=i, i_stop=i + 1, j=j)


def cell_run_polygon_xy(geometry, *, i_start: int, i_stop: int, j: int) -> Polygon:
    cu, cv, cw = geometry.local.corner
    dx, dy, _ = geometry.local.block_size

    u0 = cu + i_start * dx
    u1 = cu + i_stop * dx
    v0 = cv + j * dy
    v1 = v0 + dy

    local = np.array(
        [
            [u0, u1, u1, u0, u0],
            [v0, v0, v1, v1, v0],
            [cw, cw, cw, cw, cw],
        ],
        dtype=float,
    )
    world = geometry.world.local_to_world(local)
    coords_xy = [(float(world[0, p]), float(world[1, p])) for p in range(local.shape[1])]
    return Polygon(coords_xy)


def iter_polygons(geometry) -> Iterator[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
        return
    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                yield geom


def assert_supported_plan_projection(
    geometry,
    *,
    tolerance: float = 1e-9,
    error_message: str | None = None,
) -> None:
    axis_w = np.asarray(geometry.world.axis_w, dtype=float)
    if np.linalg.norm(axis_w[:2]) > tolerance:
        raise ValueError(
            error_message
            or (
                "Footprint extraction requires projection onto the (i, j) plane. "
                "Current axis_w configuration does not permit this."
            )
        )


def coerce_polygonal_geometry(geometry) -> Polygon | MultiPolygon:
    union = getattr(shapely, "union_all", unary_union)

    if geometry.is_empty:
        return Polygon()

    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    polygon_parts = list(iter_polygons(geometry))
    if not polygon_parts:
        return Polygon()

    polygonal = union(polygon_parts)
    if polygonal.is_empty:
        return Polygon()
    if not isinstance(polygonal, (Polygon, MultiPolygon)):
        raise ValueError("Footprint dissolve must produce polygonal geometry.")
    return polygonal


def dissolve_row_runs(geometry, row_runs: Iterable[tuple[int, int, int]]) -> Polygon | MultiPolygon:
    polygons = [
        cell_run_polygon_xy(geometry, i_start=i_start, i_stop=i_stop, j=j)
        for i_start, i_stop, j in row_runs
    ]
    if not polygons:
        return Polygon()

    union = getattr(shapely, "union_all", unary_union)
    dissolved = union(polygons)
    return coerce_polygonal_geometry(dissolved)
