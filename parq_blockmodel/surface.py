from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import shapely
from shapely.geometry import Polygon

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.mesh.ply import read_ply
from parq_blockmodel.mesh.types import TriangleMesh


def _resolve_geometry_like(grid: Any) -> RegularGeometry:
    if isinstance(grid, RegularGeometry):
        return grid

    if hasattr(grid, "geometry") and isinstance(grid.geometry, RegularGeometry):
        return grid.geometry

    raise TypeError("evaluate() expects a RegularGeometry or an object exposing .geometry as RegularGeometry.")


def _block_half_z_extent(geometry: RegularGeometry) -> float:
    dx, dy, dz = geometry.local.block_size
    axis_u_z = abs(float(geometry.world.axis_u[2]))
    axis_v_z = abs(float(geometry.world.axis_v[2]))
    axis_w_z = abs(float(geometry.world.axis_w[2]))
    return 0.5 * (dx * axis_u_z + dy * axis_v_z + dz * axis_w_z)


def _intersects_xy(geometry: Polygon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if hasattr(shapely, "intersects_xy"):
        return np.asarray(shapely.intersects_xy(geometry, x, y), dtype=bool)

    points = shapely.points(x, y)
    return np.asarray(shapely.intersects(geometry, points), dtype=bool)


class Surface(Protocol):
    """Protocol for 2.5D elevation surfaces."""

    name: str | None

    def evaluate(self, grid: Any) -> np.ndarray:
        """Return per-block fraction below the surface."""


def _normalise_axes(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("RasterSurface coordinate axes must be 1D.")
    if x.size < 2 or y.size < 2:
        raise ValueError("RasterSurface requires at least two x and y coordinates.")
    if values.ndim != 2:
        raise ValueError("RasterSurface values must be 2D.")
    if values.shape != (y.size, x.size):
        raise ValueError("RasterSurface values must have shape (len(y), len(x)).")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(values)):
        raise ValueError("RasterSurface inputs must be finite.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    x_diff = np.diff(x)
    y_diff = np.diff(y)
    if np.any(x_diff == 0) or np.any(y_diff == 0):
        raise ValueError("RasterSurface coordinate axes must be strictly monotonic.")

    if np.all(x_diff > 0):
        x_ordered = x
        values_ordered = values
    elif np.all(x_diff < 0):
        x_ordered = x[::-1]
        values_ordered = values[:, ::-1]
    else:
        raise ValueError("RasterSurface x coordinates must be strictly monotonic.")

    y_diff = np.diff(y)
    if np.all(y_diff > 0):
        y_ordered = y
        values_ordered = values_ordered
    elif np.all(y_diff < 0):
        y_ordered = y[::-1]
        values_ordered = values_ordered[::-1, :]
    else:
        raise ValueError("RasterSurface y coordinates must be strictly monotonic.")

    return x_ordered, y_ordered, values_ordered


def _bilinear_interpolate(x_coords: np.ndarray, y_coords: np.ndarray, values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if np.any((x < x_coords[0]) | (x > x_coords[-1]) | (y < y_coords[0]) | (y > y_coords[-1])):
        raise ValueError("Surface evaluation points fall outside the raster extent.")

    xi = np.searchsorted(x_coords, x, side="right") - 1
    yi = np.searchsorted(y_coords, y, side="right") - 1
    xi = np.clip(xi, 0, x_coords.size - 2)
    yi = np.clip(yi, 0, y_coords.size - 2)

    x0 = x_coords[xi]
    x1 = x_coords[xi + 1]
    y0 = y_coords[yi]
    y1 = y_coords[yi + 1]

    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x, dtype=float), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y, dtype=float), where=(y1 != y0))

    z00 = values[yi, xi]
    z10 = values[yi, xi + 1]
    z01 = values[yi + 1, xi]
    z11 = values[yi + 1, xi + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )


@dataclass(frozen=True)
class RasterSurface:
    """2.5D surface backed by a regular raster grid."""

    x_coords: np.ndarray
    y_coords: np.ndarray
    values: np.ndarray
    name: str | None = None
    _x: np.ndarray = field(init=False, repr=False, compare=False)
    _y: np.ndarray = field(init=False, repr=False, compare=False)
    _values: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        x, y, values = _normalise_axes(np.asarray(self.x_coords), np.asarray(self.y_coords), np.asarray(self.values))
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_y", y)
        object.__setattr__(self, "_values", values)

    @classmethod
    def from_arrays(
        cls,
        *,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        values: np.ndarray,
        name: str | None = None,
    ) -> "RasterSurface":
        return cls(x_coords=x_coords, y_coords=y_coords, values=values, name=name)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "RasterSurface":
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                x_key = next((key for key in ("x", "x_coords", "xs") if key in keys), None)
                y_key = next((key for key in ("y", "y_coords", "ys") if key in keys), None)
                values_key = next((key for key in ("values", "z", "surface") if key in keys), None)
                if x_key is None or y_key is None or values_key is None:
                    raise ValueError("RasterSurface .npz files must contain x/y coordinates and values.")
                inferred_name = name
                if inferred_name is None and "name" in keys:
                    inferred_name = str(np.asarray(data["name"]).item())
                return cls.from_arrays(
                    x_coords=np.asarray(data[x_key]),
                    y_coords=np.asarray(data[y_key]),
                    values=np.asarray(data[values_key]),
                    name=inferred_name,
                )

        try:
            import rioxarray as rxr
        except ImportError:
            rxr = None

        try:
            import xarray as xr
        except ImportError as exc:
            raise ValueError(
                "Unsupported raster file format. Use a .npz file with x/y/values arrays, or install xarray/rioxarray."
            ) from exc

        if rxr is not None:
            data = rxr.open_rasterio(path).squeeze(drop=True)
            if data.ndim != 2:
                raise ValueError("RasterSurface requires a 2D raster.")
            x_name = next((dim for dim in data.dims if dim.lower().startswith("x")), "x")
            y_name = next((dim for dim in data.dims if dim.lower().startswith("y")), "y")
            return cls.from_arrays(
                x_coords=np.asarray(data.coords[x_name].values),
                y_coords=np.asarray(data.coords[y_name].values),
                values=np.asarray(data.transpose(y_name, x_name).values),
                name=name,
            )

        data = xr.open_dataarray(path)
        if data.ndim != 2:
            raise ValueError("RasterSurface requires a 2D raster.")
        dims = list(data.dims)
        x_dim = next((dim for dim in dims if dim.lower().startswith("x")), dims[-1])
        y_dim = next((dim for dim in dims if dim.lower().startswith("y")), dims[0])
        return cls.from_arrays(
            x_coords=np.asarray(data.coords[x_dim].values),
            y_coords=np.asarray(data.coords[y_dim].values),
            values=np.asarray(data.transpose(y_dim, x_dim).values),
            name=name,
        )

    def surface_z(self, grid: Any) -> np.ndarray:
        geometry = _resolve_geometry_like(grid)
        x = np.asarray(geometry.centroid_x, dtype=float)
        y = np.asarray(geometry.centroid_y, dtype=float)
        return _bilinear_interpolate(self._x, self._y, self._values, x, y)

    def evaluate(self, grid: Any) -> np.ndarray:
        geometry = _resolve_geometry_like(grid)
        surface_z = self.surface_z(geometry)
        centroid_z = np.asarray(geometry.centroid_z, dtype=float)
        half_z = _block_half_z_extent(geometry)
        if half_z <= 0:
            raise ValueError("Geometry block height must be positive.")
        zmin = centroid_z - half_z
        fraction = (surface_z - zmin) / (2.0 * half_z)
        return np.clip(fraction, 0.0, 1.0).astype(float)


def _triangle_barycentric_z(points_xy: np.ndarray, triangle_xy: np.ndarray, triangle_z: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x0, y0 = triangle_xy[0]
    x1, y1 = triangle_xy[1]
    x2, y2 = triangle_xy[2]
    z0, z1, z2 = triangle_z

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if np.isclose(denom, 0.0):
        raise ValueError("MeshSurface requires non-degenerate projected triangles.")

    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    return a * z0 + b * z1 + c * z2


def _triangle_projected_area(triangle_xy: np.ndarray) -> float:
    x = triangle_xy[:, 0]
    y = triangle_xy[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _validate_2p5d_mesh(mesh: TriangleMesh, tol: float = 1e-9) -> None:
    mesh.validate()
    if mesh.n_faces == 0:
        raise ValueError("MeshSurface requires a non-empty mesh.")

    projected_z: dict[tuple[float, float], float] = {}
    for vertex in np.asarray(mesh.vertices, dtype=float):
        key = (float(vertex[0]), float(vertex[1]))
        z = float(vertex[2])
        if key in projected_z and not np.isclose(projected_z[key], z, atol=tol, rtol=0.0):
            raise ValueError("MeshSurface requires a valid 2.5D surface with one Z per (x, y).")
        projected_z[key] = z

    for face in np.asarray(mesh.faces, dtype=np.intp):
        triangle_xy = np.asarray(mesh.vertices[face, :2], dtype=float)
        if _triangle_projected_area(triangle_xy) <= tol:
            raise ValueError("MeshSurface requires triangles with non-zero projected area.")


@dataclass(frozen=True)
class MeshSurface:
    """2.5D surface represented by a triangulated mesh."""

    mesh: TriangleMesh
    name: str | None = None
    _face_xy: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)
    _face_z: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)
    _face_polygons: tuple[Polygon, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_2p5d_mesh(self.mesh)
        face_xy: list[np.ndarray] = []
        face_z: list[np.ndarray] = []
        face_polygons: list[Polygon] = []
        for face in np.asarray(self.mesh.faces, dtype=np.intp):
            tri = np.asarray(self.mesh.vertices[face], dtype=float)
            xy = tri[:, :2]
            face_xy.append(xy)
            face_z.append(tri[:, 2])
            face_polygons.append(Polygon(xy))
        object.__setattr__(self, "_face_xy", tuple(face_xy))
        object.__setattr__(self, "_face_z", tuple(face_z))
        object.__setattr__(self, "_face_polygons", tuple(face_polygons))

    @classmethod
    def from_mesh(cls, mesh: TriangleMesh, name: str | None = None) -> "MeshSurface":
        return cls(mesh=mesh, name=name)

    @classmethod
    def from_ply(cls, path: str | Path, name: str | None = None) -> "MeshSurface":
        mesh = read_ply(path)
        return cls(mesh=mesh, name=name)

    def surface_z(self, grid: Any) -> np.ndarray:
        geometry = _resolve_geometry_like(grid)
        x = np.asarray(geometry.centroid_x, dtype=float)
        y = np.asarray(geometry.centroid_y, dtype=float)
        points_xy = np.column_stack([x, y])
        z = np.full(points_xy.shape[0], np.nan, dtype=float)
        assigned = np.zeros(points_xy.shape[0], dtype=bool)

        for polygon, triangle_xy, triangle_z in zip(self._face_polygons, self._face_xy, self._face_z):
            if hasattr(shapely, "intersects_xy"):
                candidate = np.asarray(shapely.intersects_xy(polygon, x, y), dtype=bool)
            else:
                candidate = np.asarray(shapely.intersects(polygon, shapely.points(x, y)), dtype=bool)
            if not np.any(candidate):
                continue

            candidate_idx = np.flatnonzero(candidate)
            z_candidate = _triangle_barycentric_z(points_xy[candidate_idx], triangle_xy, triangle_z)
            already = assigned[candidate_idx]
            if np.any(already):
                existing = z[candidate_idx[already]]
                current = z_candidate[already]
                if not np.allclose(existing, current, atol=1e-9, rtol=0.0):
                    raise ValueError("MeshSurface is not single-valued in Z over the projected area.")

            new_idx = candidate_idx[~already]
            if new_idx.size:
                z[new_idx] = z_candidate[~already]
                assigned[new_idx] = True

        if not np.all(assigned):
            raise ValueError("MeshSurface does not cover all requested grid centroids.")
        return z

    def evaluate(self, grid: Any) -> np.ndarray:
        geometry = _resolve_geometry_like(grid)
        surface_z = self.surface_z(geometry)
        centroid_z = np.asarray(geometry.centroid_z, dtype=float)
        half_z = _block_half_z_extent(geometry)
        if half_z <= 0:
            raise ValueError("Geometry block height must be positive.")
        zmin = centroid_z - half_z
        fraction = (surface_z - zmin) / (2.0 * half_z)
        return np.clip(fraction, 0.0, 1.0).astype(float)
