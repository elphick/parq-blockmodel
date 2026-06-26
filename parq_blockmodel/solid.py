from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
import pyvista as pv

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.mesh.ply import read_ply
from parq_blockmodel.mesh.types import TriangleMesh


def _resolve_geometry_like(grid: Any) -> RegularGeometry:
    if isinstance(grid, RegularGeometry):
        return grid

    if hasattr(grid, "geometry") and isinstance(grid.geometry, RegularGeometry):
        return grid.geometry

    raise TypeError("evaluate() expects a RegularGeometry or an object exposing .geometry as RegularGeometry.")


def _centroid_points(geometry: RegularGeometry) -> np.ndarray:
    # Compute centroids once and return (N, 3) world-space points.
    return np.asarray(geometry._centroids.T, dtype=float)


class Solid(Protocol):
    """Protocol for volumetric geometry classifiers."""

    def evaluate(self, grid: Any) -> np.ndarray:
        """Return a boolean mask for grid centroids inside the solid."""


def _triangles_from_3dface(entity: Any) -> list[np.ndarray]:
    points = [
        np.asarray(entity.dxf.vtx0, dtype=float),
        np.asarray(entity.dxf.vtx1, dtype=float),
        np.asarray(entity.dxf.vtx2, dtype=float),
        np.asarray(entity.dxf.vtx3, dtype=float),
    ]

    # DXF triangles repeat the last point for vtx3.
    unique: list[np.ndarray] = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, unique[-1]):
            unique.append(point)
    if len(unique) >= 3 and np.allclose(unique[-1], unique[0]):
        unique.pop()

    if len(unique) == 3:
        return [np.vstack(unique)]
    if len(unique) == 4:
        return [np.vstack([unique[0], unique[1], unique[2]]), np.vstack([unique[0], unique[2], unique[3]])]
    return []


def _collect_entity_triangles(entity: Any) -> list[np.ndarray]:
    entity_type = entity.dxftype()
    if entity_type == "3DFACE":
        return _triangles_from_3dface(entity)

    if entity_type in {"INSERT", "POLYLINE", "POLYFACE", "MESH"} and hasattr(entity, "virtual_entities"):
        triangles: list[np.ndarray] = []
        for child in entity.virtual_entities():
            triangles.extend(_collect_entity_triangles(child))
        return triangles

    return []


def _triangle_mesh_from_triangles(triangles: list[np.ndarray]) -> TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []

    for tri in triangles:
        indices: list[int] = []
        for point in tri:
            key = (float(point[0]), float(point[1]), float(point[2]))
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(key)
            indices.append(vertex_index[key])
        faces.append((indices[0], indices[1], indices[2]))

    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.intp),
    )


def _resolve_dxf_triangles(path: str | Path, by: Literal["layer", "block"]) -> dict[str, list[np.ndarray]]:
    try:
        import ezdxf
    except ImportError as exc:
        raise ImportError("MeshSolid.from_dxf() requires optional dependency 'ezdxf'.") from exc

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    groups: dict[str, list[np.ndarray]] = {}

    if by == "layer":
        for entity in msp:
            triangles = _collect_entity_triangles(entity)
            if triangles:
                groups.setdefault(str(entity.dxf.layer), []).extend(triangles)
        return groups

    for insert in msp.query("INSERT"):
        triangles = _collect_entity_triangles(insert)
        if triangles:
            groups.setdefault(str(insert.dxf.name), []).extend(triangles)
    return groups


def _to_polydata(mesh: TriangleMesh) -> pv.PolyData:
    mesh.validate()
    faces = np.hstack(
        [
            np.full((mesh.n_faces, 1), 3, dtype=np.int64),
            np.asarray(mesh.faces, dtype=np.int64),
        ]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)


def _validate_solid_surface(surface: pv.PolyData) -> None:
    if not surface.is_all_triangles:
        raise ValueError("MeshSolid requires a triangulated mesh.")
    if surface.n_open_edges != 0:
        raise ValueError("MeshSolid requires a watertight mesh (n_open_edges must be zero).")
    if not surface.is_manifold:
        raise ValueError("MeshSolid requires a manifold mesh.")


def _triangle_mesh_from_polydata(surface: pv.PolyData) -> TriangleMesh:
    if not isinstance(surface, pv.PolyData):
        raise TypeError("from_pyvista() expects a pyvista.PolyData object.")

    faces_raw = np.asarray(surface.faces, dtype=np.int64)
    if faces_raw.size == 0:
        raise ValueError("MeshSolid requires a non-empty mesh.")
    if faces_raw.size % 4 != 0:
        raise ValueError("MeshSolid requires triangle-only PolyData faces.")

    faces = faces_raw.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("MeshSolid requires triangle-only PolyData faces.")

    return TriangleMesh(
        vertices=np.asarray(surface.points, dtype=float),
        faces=faces[:, 1:].astype(np.intp),
    )


@dataclass(frozen=True)
class MeshSolid:
    """Closed 3D solid represented by a triangulated mesh."""

    EVALUATE_CHUNK_SIZE: ClassVar[int] = 1_000_000

    mesh: TriangleMesh
    name: str | None = None
    _surface: pv.PolyData = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        surface = _to_polydata(self.mesh)
        _validate_solid_surface(surface)
        object.__setattr__(self, "_surface", surface)

    @classmethod
    def from_mesh(cls, mesh: TriangleMesh, name: str | None = None) -> "MeshSolid":
        """Create a MeshSolid from an in-memory triangle mesh."""
        return cls(mesh=mesh, name=name)

    @classmethod
    def from_ply(cls, path: str | Path, name: str | None = None) -> "MeshSolid":
        """Create a MeshSolid from a PLY mesh file."""
        mesh = read_ply(path)
        return cls(mesh=mesh, name=name)

    @classmethod
    def from_pyvista(cls, surface: pv.PolyData, name: str | None = None) -> "MeshSolid":
        """Create a MeshSolid directly from a PyVista PolyData surface."""
        _validate_solid_surface(surface)
        mesh = _triangle_mesh_from_polydata(surface)
        mesh.validate()
        solid = cls.__new__(cls)
        object.__setattr__(solid, "mesh", mesh)
        object.__setattr__(solid, "name", name)
        object.__setattr__(solid, "_surface", surface)
        return solid

    @classmethod
    def from_dxf(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        by: Literal["layer", "block"] = "layer",
    ) -> "MeshSolid":
        """Create a MeshSolid from DXF 3D-face geometry grouped by layer or block."""
        if by not in {"layer", "block"}:
            raise ValueError("from_dxf() argument 'by' must be 'layer' or 'block'.")

        grouped = _resolve_dxf_triangles(path, by=by)
        available = sorted(grouped.keys())
        if not available:
            raise ValueError("No supported DXF solid geometry found. Supported entities include 3DFACE.")

        selected_name = name
        if selected_name is None:
            if len(available) > 1:
                raise ValueError(
                    f"Multiple DXF solid candidates found for by='{by}': {available}. "
                    f"Provide name='<candidate>' to select one."
                )
            selected_name = available[0]

        if selected_name not in grouped:
            raise ValueError(f"DXF solid candidate '{selected_name}' not found for by='{by}'. Available: {available}")

        mesh = _triangle_mesh_from_triangles(grouped[selected_name])
        return cls(mesh=mesh, name=selected_name)

    def evaluate(self, grid: Any) -> np.ndarray:
        """Return boolean mask of centroid points inside the solid.

        Boundary points are classified as inside.
        """
        geometry = _resolve_geometry_like(grid)
        points = _centroid_points(geometry)
        n_points = points.shape[0]
        mask = np.zeros(n_points, dtype=bool)

        xmin, xmax, ymin, ymax, zmin, zmax = self._surface.bounds
        prefiltered = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )
        candidate_idx = np.flatnonzero(prefiltered)
        if candidate_idx.size == 0:
            return mask

        chunk_size = max(1, int(self.EVALUATE_CHUNK_SIZE))
        for start in range(0, candidate_idx.size, chunk_size):
            idx = candidate_idx[start : start + chunk_size]
            cloud = pv.PolyData(points[idx])

            if hasattr(cloud, "select_interior_points"):
                selected = cloud.select_interior_points(
                    self._surface,
                    method="cell_locator",
                    check_surface=False,
                )
                mask[idx] = np.asarray(selected.point_data["selected_points"], dtype=bool)
            else:
                selected = cloud.select_enclosed_points(self._surface, check_surface=False)
                mask[idx] = np.asarray(selected.point_data["SelectedPoints"], dtype=bool)

        return mask
