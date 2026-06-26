from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import explain_validity

from parq_blockmodel.geometry import RegularGeometry


PolygonInput = Polygon | MultiPolygon | GeometryCollection | Iterable[Polygon | MultiPolygon]


def _coerce_polygon_geometry(geometry: PolygonInput) -> Polygon | MultiPolygon:
    union = getattr(shapely, "union_all", unary_union)

    if isinstance(geometry, (Polygon, MultiPolygon)):
        polygonal = geometry
    elif isinstance(geometry, GeometryCollection):
        polygon_parts = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polygon_parts:
            raise ValueError("PolygonField requires polygonal geometry; no Polygon/MultiPolygon found.")
        polygonal = union(polygon_parts)
    else:
        try:
            parts = list(geometry)
        except TypeError as exc:
            raise TypeError(
                "geometry must be a shapely Polygon/MultiPolygon, a GeometryCollection, "
                "or an iterable of Polygon/MultiPolygon geometries."
            ) from exc

        if not parts:
            raise ValueError("PolygonField requires at least one polygon geometry.")
        if not all(isinstance(part, (Polygon, MultiPolygon)) for part in parts):
            raise TypeError("All iterable geometry items must be Polygon or MultiPolygon.")
        polygonal = union(parts)

    if polygonal.is_empty:
        raise ValueError("PolygonField geometry cannot be empty.")
    if not polygonal.is_valid:
        raise ValueError(f"Invalid polygon geometry: {explain_validity(polygonal)}")

    if not isinstance(polygonal, (Polygon, MultiPolygon)):
        raise ValueError("PolygonField requires polygonal geometry after union operation.")

    return polygonal


def _resolve_geometry_like(grid: Any) -> RegularGeometry:
    if isinstance(grid, RegularGeometry):
        return grid

    if hasattr(grid, "geometry") and isinstance(grid.geometry, RegularGeometry):
        return grid.geometry

    raise TypeError("evaluate() expects a RegularGeometry or an object exposing .geometry as RegularGeometry.")


@dataclass(frozen=True)
class PolygonField:
    """2D polygon classifier evaluated against block centroids in world XY."""

    geometry: Polygon | MultiPolygon
    name: str | None = None

    @classmethod
    def from_shapely(cls, geometry: PolygonInput, name: str | None = None) -> "PolygonField":
        """Create a PolygonField from shapely polygon geometry."""
        return cls(geometry=_coerce_polygon_geometry(geometry), name=name)

    @classmethod
    def from_geoparquet(
        cls,
        filepath: str | Path,
        *,
        name: str,
        name_column: str = "name",
        geometry_column: str | None = None,
    ) -> "PolygonField":
        """Create a PolygonField from a named polygon row in a GeoParquet file."""
        import geopandas as gpd

        gdf = gpd.read_parquet(filepath)
        if name_column not in gdf.columns:
            raise ValueError(f"GeoParquet file does not contain name column '{name_column}'.")

        if geometry_column is not None:
            if geometry_column not in gdf.columns:
                raise ValueError(f"GeoParquet file does not contain geometry column '{geometry_column}'.")
            gdf = gdf.set_geometry(geometry_column)

        matches = gdf[gdf[name_column] == name]
        if len(matches) == 0:
            raise ValueError(f"No polygon found with {name_column}='{name}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple polygons found with {name_column}='{name}'. "
                "Use unique names for PolygonField.from_geoparquet."
            )

        geometry = matches.geometry.iloc[0]
        return cls.from_shapely(geometry=geometry, name=str(name))

    def evaluate(self, grid: Any) -> np.ndarray:
        """Return a boolean mask where centroid XY points intersect polygon geometry.

        Evaluation is strictly 2D in the world XY plane. Z is ignored.
        Boundary points are classified as inside.
        """
        geometry = _resolve_geometry_like(grid)
        x = np.asarray(geometry.centroid_x, dtype=float)
        y = np.asarray(geometry.centroid_y, dtype=float)

        if hasattr(shapely, "intersects_xy"):
            # intersects(point, polygon) includes interior and boundary.
            mask = shapely.intersects_xy(self.geometry, x, y)
        else:
            points = shapely.points(x, y)
            mask = shapely.intersects(self.geometry, points)
        return np.asarray(mask, dtype=bool)
