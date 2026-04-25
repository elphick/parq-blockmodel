"""Type definitions and dataclasses for mesh operations."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class TriangleMesh:
    """Representation of a triangle mesh with optional attributes.
    
    This is the internal mesh format used by the mesh module. It can be
    serialized to PLY (lossless) or GLB (derived) formats.
    
    Attributes
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (n_vertices, 3) in world space.
    faces : np.ndarray
        Triangle face indices, shape (n_faces, 3). Each row is a CCW-ordered
        triangle when viewed from the exterior (right-handed coordinates).
    vertex_attributes : dict[str, np.ndarray]
        Per-vertex scalar attributes, e.g. {"grade": (n_vertices,), "density": (n_vertices,)}.
    face_attributes : dict[str, np.ndarray]
        Per-face scalar attributes, e.g. {"rock_type": (n_faces,)}.
    vertex_ijk : Optional[np.ndarray]
        Logical (i, j, k) indices for each vertex, shape (n_vertices, 3).
        Useful for tracing back to block model.
    face_ijk : Optional[np.ndarray]
        Logical (i, j, k) indices for each face (typically the block containing it),
        shape (n_faces, 3).
    metadata : dict
        Arbitrary metadata, e.g. CRS, units, author, etc.
    """
    
    vertices: NDArray[np.floating]
    faces: NDArray[np.intp]
    vertex_attributes: dict[str, NDArray] = None
    face_attributes: dict[str, NDArray] = None
    vertex_ijk: Optional[NDArray[np.intp]] = None
    face_ijk: Optional[NDArray[np.intp]] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Initialize optional dictionaries if None."""
        if self.vertex_attributes is None:
            self.vertex_attributes = {}
        if self.face_attributes is None:
            self.face_attributes = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)
    
    @property
    def n_faces(self) -> int:
        """Number of faces (triangles)."""
        return len(self.faces)
    
    def validate(self) -> None:
        """Validate mesh integrity.
        
        Raises
        ------
        ValueError
            If mesh is malformed (e.g., vertices not 3D, faces have invalid indices).
        """
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (n, 3).")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Faces must have shape (n, 3).")
        if np.any(self.faces < 0) or np.any(self.faces >= self.n_vertices):
            raise ValueError("Face indices must be in range [0, n_vertices).")
        if self.vertex_ijk is not None:
            if self.vertex_ijk.shape != (self.n_vertices, 3):
                raise ValueError("vertex_ijk must have shape (n_vertices, 3).")
        if self.face_ijk is not None:
            if self.face_ijk.shape != (self.n_faces, 3):
                raise ValueError("face_ijk must have shape (n_faces, 3).")
        for name, attr in self.vertex_attributes.items():
            if len(attr) != self.n_vertices:
                raise ValueError(f"Vertex attribute '{name}' has wrong length ({len(attr)} vs {self.n_vertices}).")
        for name, attr in self.face_attributes.items():
            if len(attr) != self.n_faces:
                raise ValueError(f"Face attribute '{name}' has wrong length ({len(attr)} vs {self.n_faces}).")


MeshMetadata = dict[str, Union[str, float, int, dict]]
"""Type alias for mesh metadata dictionaries."""

