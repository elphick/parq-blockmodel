from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def _dedupe_non_null(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if pd.isna(value):
            continue
        if value not in result:
            result.append(value)
    return result


def _ordered_present_values(series: pd.Series) -> list[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = _dedupe_non_null(series.tolist())
        return [category for category in series.cat.categories.tolist() if category in present]
    return _dedupe_non_null(pd.unique(series).tolist())


def _resolve_precedence(
    series: pd.Series,
    categories: Sequence[Any] | None,
    precedence: Sequence[Any] | None,
) -> list[Any]:
    present_in_column = _ordered_present_values(series)
    participating = _dedupe_non_null(categories) if categories is not None else present_in_column

    if precedence is not None:
        precedence_order = _dedupe_non_null(precedence)
    elif isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        categorical_order = series.cat.categories.tolist()
        precedence_order = [value for value in categorical_order if value in participating]
    else:
        precedence_order = participating

    # Keep all participating categories even if precedence is partial.
    precedence_with_all_participants = list(precedence_order)
    for value in participating:
        if value not in precedence_with_all_participants:
            precedence_with_all_participants.append(value)
    return precedence_with_all_participants


def _cell_polygon_xy(geometry, i: int, j: int):
    from shapely.geometry import Polygon

    cu, cv, cw = geometry.local.corner
    dx, dy, _ = geometry.local.block_size

    u0 = cu + i * dx
    u1 = u0 + dx
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


def _iter_polygons(geometry):
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

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


def _assert_supported_plan_projection(geometry, tolerance: float = 1e-9) -> None:
    axis_w = np.asarray(geometry.world.axis_w, dtype=float)
    if np.linalg.norm(axis_w[:2]) > tolerance:
        raise ValueError(
            "to_categorical_geodataframe currently supports local-column plan projection only; "
            "geometry with axis_w horizontal components is not supported."
        )


def to_categorical_geodataframe(
    blockmodel,
    *,
    column: str,
    categories: Sequence[Any] | None = None,
    precedence: Sequence[Any] | None = None,
):
    import geopandas as gpd
    from shapely.ops import unary_union

    if column not in blockmodel.available_columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {blockmodel.available_columns}")

    _assert_supported_plan_projection(blockmodel.geometry)

    values = blockmodel.read(columns=[column], index="ijk", dense=True)[column]
    precedence_values = _resolve_precedence(values, categories=categories, precedence=precedence)

    ni, nj, nk = blockmodel.geometry.local.shape
    values_3d = values.to_numpy().reshape((ni, nj, nk), order="C")

    assigned = np.zeros((ni, nj), dtype=bool)
    records: list[dict[str, Any]] = []

    for value in precedence_values:
        category_footprint = np.any(values_3d == value, axis=2)
        visible = category_footprint & ~assigned
        if not np.any(visible):
            continue
        assigned |= visible

        cells = np.argwhere(visible)
        polygons = [_cell_polygon_xy(blockmodel.geometry, int(i), int(j)) for i, j in cells]
        merged = unary_union(polygons)
        for polygon in _iter_polygons(merged):
            if polygon.is_empty:
                continue
            records.append({column: value, "geometry": polygon})

    return gpd.GeoDataFrame(
        records,
        columns=[column, "geometry"],
        geometry="geometry",
        crs=blockmodel.geometry.srs,
    )
