"""Core mesh generation logic for converting block models to triangle meshes.

This module provides the BlockMeshGenerator class that converts a regular
block model (represented by RegularGeometry + sparse/dense block data) into
a triangulated surface mesh with optional per-vertex and per-face attributes.

Key responsibilities:
    - Generate vertex coordinates in world space (with rotation applied).
    - Generate triangle connectivity with consistent winding order.
    - Map block attributes to vertex/face attributes.
    - Handle sparse block models (only vertices for present blocks).
    - Support surface-only vs. full interior mesh generation.
    - Validate right-handed coordinates.
"""

from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.mesh.types import TriangleMesh


class BlockMeshGenerator:
    """Convert a block model to a triangulated surface mesh.
    
    This class generates vertex and face arrays from a RegularGeometry and
    optional block attribute data. Supports both rotated and non-rotated
    geometries, sparse and dense block layouts, and surface-only vs. full
    interior mesh generation.
    
    Coordinates are always in world space (right-handed), with rotation
    applied via the RegularGeometry's axis vectors (axis_u, axis_v, axis_w).
    
    Parameters
    ----------
    geometry : RegularGeometry
        The block model geometry defining corner, block_size, shape, and axis orientation.
    
    Attributes
    ----------
    geometry : RegularGeometry
        The block model geometry.
    
    Notes
    -----
    All generated triangles use counter-clockwise (CCW) winding order when
    viewed from the exterior (right-hand rule). This ensures compatibility
    with standard 3D graphics renderers and PLY/GLB viewers.
    """
    
    def __init__(self, geometry: RegularGeometry):
        """Initialize the mesh generator with a block model geometry.
        
        Parameters
        ----------
        geometry : RegularGeometry
            The block model geometry. Must have orthonormal axis vectors.
        
        Raises
        ------
        ValueError
            If the geometry's axis vectors are not properly oriented.
        """
        self.geometry = geometry
        self._validate_handedness()
    
    def _validate_handedness(self) -> None:
        """Ensure the geometry defines a right-handed coordinate system.
        
        Checks that axis_u × axis_v ≈ axis_w (cross product).
        
        Raises
        ------
        ValueError
            If the axes do not form a right-handed system.
        """
        u = np.array(self.geometry.axis_u, dtype=float)
        v = np.array(self.geometry.axis_v, dtype=float)
        w = np.array(self.geometry.axis_w, dtype=float)
        
        cross = np.cross(u, v)
        if not np.allclose(cross, w, atol=1e-6):
            raise ValueError(
                f"Geometry axes do not form right-handed system. "
                f"axis_u × axis_v = {cross}, but axis_w = {w}."
            )
    
    def block_corners_local(self) -> NDArray[np.floating]:
        """Get the 8 corner offsets for a unit cube in local (i, j, k) space.
        
        Returns a (8, 3) array where each row is a corner offset:
        (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1).
        
        Returns
        -------
        np.ndarray
            Shape (8, 3), corner offsets in local space.
        """
        return np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=float)
    
    def block_vertices_xyz(self, i: int, j: int, k: int) -> NDArray[np.floating]:
        """Get the 8 vertices of a block in world coordinates.
        
        Parameters
        ----------
        i, j, k : int
            Logical block indices.
        
        Returns
        -------
        np.ndarray
            Shape (8, 3), corner coordinates in world space.
        """
        dx, dy, dz = self.geometry.block_size
        cx, cy, cz = self.geometry.corner
        
        # Local coordinates of block corners
        corners_local = self.block_corners_local()
        
        # Scale to block size
        corners_local[:, 0] = corners_local[:, 0] * dx
        corners_local[:, 1] = corners_local[:, 1] * dy
        corners_local[:, 2] = corners_local[:, 2] * dz
        
        # Translate to block position (lower corner)
        corners_local[:, 0] += cx + i * dx
        corners_local[:, 1] += cy + j * dy
        corners_local[:, 2] += cz + k * dz
        
        # Apply rotation (axis transformation)
        u = np.array(self.geometry.axis_u, dtype=float)
        v = np.array(self.geometry.axis_v, dtype=float)
        w = np.array(self.geometry.axis_w, dtype=float)
        R = np.column_stack([u, v, w])  # (3, 3) rotation matrix
        
        corners_world = corners_local @ R.T
        return corners_world
    
    def cube_faces_ccw(self) -> NDArray[np.intp]:
        """Get the 6 faces of a cube with CCW winding (right-hand rule).
        
        Each face is a pair of triangles sharing an edge. Winding order is
        such that the outward normal points away from the cube center.
        
        Local vertex numbering:
            0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)
            4: (0,0,1)  5: (1,0,1)  6: (0,1,1)  7: (1,1,1)
        
        Returns
        -------
        np.ndarray
            Shape (12, 3), 12 triangles (2 per face, 6 faces).
        """
        # Each face: [tri1, tri2]
        # Bottom face (k=0): vertices 0,1,2,3, CCW from below (+z is up)
        # Top face (k=1): vertices 4,5,6,7, CCW from above
        # Front face (j=0): vertices 0,1,4,5, CCW from front (+y is back)
        # Back face (j=1): vertices 2,3,6,7, CCW from back
        # Left face (i=0): vertices 0,2,4,6, CCW from left (+x is right)
        # Right face (i=1): vertices 1,3,5,7, CCW from right
        
        triangles = [
            # Bottom face (k=0)
            [0, 2, 1],  # CCW from below
            [1, 2, 3],
            # Top face (k=1)
            [4, 5, 6],  # CCW from above
            [5, 7, 6],
            # Front face (j=0)
            [0, 1, 4],  # CCW from front
            [1, 5, 4],
            # Back face (j=1)
            [2, 6, 3],  # CCW from back
            [3, 6, 7],
            # Left face (i=0)
            [0, 4, 2],  # CCW from left
            [2, 4, 6],
            # Right face (i=1)
            [1, 3, 5],  # CCW from right
            [3, 7, 5],
        ]
        return np.array(triangles, dtype=np.intp)
    
    def surface_faces_ccw(self) -> NDArray[np.intp]:
        """Get only the exterior surface faces of a cube.
        
        For a sparse block model, we typically want only faces on the
        boundary of the model. This method returns which faces are
        "exterior" in a dense grid context. When sparse blocks are
        present, we'll filter out internal faces via connectivity checks.
        
        Returns the same 12 triangles as cube_faces_ccw(), but this
        is a placeholder for future boundary detection logic.
        
        Returns
        -------
        np.ndarray
            Shape (12, 3), exterior triangles.
        """
        # For now, return all faces. In a sparse model, faces between
        # adjacent blocks will be filtered out by _filter_shared_faces.
        return self.cube_faces_ccw()
    
    def triangulate(
        self,
        block_data: Optional[pd.DataFrame] = None,
        surface_only: bool = True,
        sparse: Optional[bool] = None,
    ) -> TriangleMesh:
        """Generate a triangle mesh from block model data.
        
        Parameters
        ----------
        block_data : pd.DataFrame, optional
            Block model data with columns like "i", "j", "k", "grade", "density", etc.
            If provided, attributes will be mapped to the mesh.
            If None, only geometry is returned with no attribute data.
        sparse : bool, optional
            If True, include only blocks present in block_data.
            If False or None, generate the full dense mesh.
            If sparse=True but block_data is None, only non-empty blocks are included.
        surface_only : bool, default True
            If True, include only exterior surface faces (shared faces with
            adjacent blocks are removed). If False, include all faces
            (interior voids remain empty). Only meaningful for sparse models.
        
        Returns
        -------
        TriangleMesh
            The generated mesh with vertices, faces, and optional attributes.
        
        Raises
        ------
        ValueError
            If block_data is incompatible with the geometry.
        """
        if sparse is None:
            sparse = block_data is None or len(block_data) < int(np.prod(self.geometry.shape))
        
        if sparse and block_data is not None:
            blocks_to_include = self._get_block_indices_from_data(block_data)
        elif sparse:
            blocks_to_include = None  # No data, return empty mesh
        else:
            ni, nj, nk = self.geometry.shape
            blocks_to_include = [
                (i, j, k)
                for i in range(ni)
                for j in range(nj)
                for k in range(nk)
            ]
        
        # Generate vertex and face lists
        vertices, faces, vertex_ijk_list, face_ijk_list = self._generate_vertices_faces(
            blocks_to_include,
            surface_only=surface_only,
            sparse=sparse,
        )
        
        # Map attributes if data is provided
        vertex_attrs = {}
        face_attrs = {}
        if block_data is not None:
            vertex_attrs, face_attrs = self._map_attributes(
                block_data,
                vertex_ijk_list,
                face_ijk_list,
            )
        
        # Create metadata
        metadata = {
            "corner": tuple(self.geometry.corner),
            "block_size": tuple(self.geometry.block_size),
            "shape": tuple(self.geometry.shape),
            "axis_u": tuple(self.geometry.axis_u),
            "axis_v": tuple(self.geometry.axis_v),
            "axis_w": tuple(self.geometry.axis_w),
            "surface_only": surface_only,
            "sparse": sparse,
        }
        if self.geometry.srs:
            metadata["srs"] = self.geometry.srs
        
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            vertex_attributes=vertex_attrs,
            face_attributes=face_attrs,
            vertex_ijk=np.array(vertex_ijk_list, dtype=np.intp) if vertex_ijk_list else None,
            face_ijk=np.array(face_ijk_list, dtype=np.intp) if face_ijk_list else None,
            metadata=metadata,
        )
        mesh.validate()
        return mesh
    
    def _get_block_indices_from_data(self, block_data: pd.DataFrame) -> list[Tuple[int, int, int]]:
        """Extract (i, j, k) block indices from DataFrame.
        
        Parameters
        ----------
        block_data : pd.DataFrame
            Must contain either "i", "j", "k" columns or be indexable by (i,j,k).
        
        Returns
        -------
        list of tuple
            List of (i, j, k) block indices.
        
        Raises
        ------
        ValueError
            If block_data does not contain required positional information.
        """
        if "i" in block_data.columns and "j" in block_data.columns and "k" in block_data.columns:
            return list(zip(
                block_data["i"].astype(int),
                block_data["j"].astype(int),
                block_data["k"].astype(int),
            ))
        elif isinstance(block_data.index, pd.MultiIndex) and block_data.index.names == ["i", "j", "k"]:
            return list(block_data.index)
        else:
            raise ValueError(
                "block_data must contain 'i', 'j', 'k' columns or be indexed by (i, j, k)."
            )
    
    def _generate_vertices_faces(
        self,
        blocks_to_include: Optional[list[Tuple[int, int, int]]],
        surface_only: bool,
        sparse: bool,
    ) -> Tuple[NDArray, NDArray, list, list]:
        """Generate vertex and face arrays.
        
        Returns
        -------
        tuple
            (vertices, faces, vertex_ijk_list, face_ijk_list)
        """
        if not sparse or blocks_to_include is None:
            return self._generate_dense_mesh(surface_only)
        else:
            return self._generate_sparse_mesh(blocks_to_include, surface_only)
    
    def _generate_dense_mesh(
        self,
        surface_only: bool,
    ) -> Tuple[NDArray, NDArray, list, list]:
        """Generate mesh for full dense grid."""
        vertices_list = []
        faces_list = []
        vertex_ijk_list = []
        face_ijk_list = []
        
        vertex_map = {}  # (i, j, k, corner_idx) -> global_vertex_idx
        next_vertex_idx = 0
        
        ni, nj, nk = self.geometry.shape
        
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    # Get 8 corner vertices for this block
                    block_corners = self.block_vertices_xyz(i, j, k)
                    
                    # Map or create vertices
                    block_vertex_indices = []
                    for corner_idx in range(8):
                        key = (i, j, k, corner_idx)
                        if key not in vertex_map:
                            vertex_map[key] = next_vertex_idx
                            vertices_list.append(block_corners[corner_idx])
                            vertex_ijk_list.append((i, j, k))
                            next_vertex_idx += 1
                        block_vertex_indices.append(vertex_map[key])
                    
                    # Add faces
                    base_faces = self.cube_faces_ccw()
                    for face_idx, face in enumerate(base_faces):
                        global_face = [block_vertex_indices[v] for v in face]
                        faces_list.append(global_face)
                        face_ijk_list.append((i, j, k))
        
        vertices = np.array(vertices_list, dtype=float)
        faces = np.array(faces_list, dtype=np.intp)
        
        return vertices, faces, vertex_ijk_list, face_ijk_list
    
    def _generate_sparse_mesh(
        self,
        blocks_to_include: list[Tuple[int, int, int]],
        surface_only: bool,
    ) -> Tuple[NDArray, NDArray, list, list]:
        """Generate mesh for sparse block set."""
        blocks_set = set(blocks_to_include)
        vertices_list = []
        faces_list = []
        vertex_ijk_list = []
        face_ijk_list = []
        
        vertex_map = {}  # (i, j, k, corner_idx) -> global_vertex_idx
        next_vertex_idx = 0
        
        for i, j, k in blocks_to_include:
            # Get 8 corner vertices for this block
            block_corners = self.block_vertices_xyz(i, j, k)
            
            # Map or create vertices (shared with neighbors if they exist)
            block_vertex_indices = []
            for corner_idx in range(8):
                key = (i, j, k, corner_idx)
                if key not in vertex_map:
                    vertex_map[key] = next_vertex_idx
                    vertices_list.append(block_corners[corner_idx])
                    vertex_ijk_list.append((i, j, k))
                    next_vertex_idx += 1
                block_vertex_indices.append(vertex_map[key])
            
            # Add faces, filtering if surface_only
            base_faces = self.cube_faces_ccw()
            for face_idx, face in enumerate(base_faces):
                # Determine which block face this is
                is_exterior = self._is_exterior_face(i, j, k, face_idx, blocks_set)
                
                if not surface_only or is_exterior:
                    global_face = [block_vertex_indices[v] for v in face]
                    faces_list.append(global_face)
                    face_ijk_list.append((i, j, k))
        
        vertices = np.array(vertices_list, dtype=float) if vertices_list else np.empty((0, 3), dtype=float)
        faces = np.array(faces_list, dtype=np.intp) if faces_list else np.empty((0, 3), dtype=np.intp)
        
        return vertices, faces, vertex_ijk_list, face_ijk_list
    
    def _is_exterior_face(
        self,
        i: int,
        j: int,
        k: int,
        face_idx: int,
        blocks_set: Set[Tuple[int, int, int]],
    ) -> bool:
        """Check if a cube face is on the exterior of the sparse model.
        
        face_idx maps to:
            0,1 : bottom (k=0)
            2,3 : top (k=1)
            4,5 : front (j=0)
            6,7 : back (j=1)
            8,9 : left (i=0)
            10,11 : right (i=1)
        """
        face_to_direction = {
            (0, 1): (0, 0, -1),  # bottom
            (2, 3): (0, 0, 1),   # top
            (4, 5): (0, -1, 0),  # front
            (6, 7): (0, 1, 0),   # back
            (8, 9): (-1, 0, 0),  # left
            (10, 11): (1, 0, 0), # right
        }
        
        for (f1, f2), direction in face_to_direction.items():
            if face_idx in (f1, f2):
                ni, nj, nk = direction
                neighbor = (i + ni, j + nj, k + nk)
                return neighbor not in blocks_set
        
        return True  # default: exterior
    
    def _map_attributes(
        self,
        block_data: pd.DataFrame,
        vertex_ijk_list: list[Tuple[int, int, int]],
        face_ijk_list: list[Tuple[int, int, int]],
    ) -> Tuple[dict, dict]:
        """Map block attributes to vertex and face attributes.
        
        For each vertex, we look up the block(s) it belongs to and assign
        attributes. For vertices shared between blocks, we use the first
        block's attributes (or could interpolate in future).
        
        Returns
        -------
        tuple
            (vertex_attributes, face_attributes) as dicts of arrays.
        """
        vertex_attrs = {}
        face_attrs = {}
        
        # Identify which columns are attributes (not positional)
        positional = {"i", "j", "k", "x", "y", "z", "block_id"}
        attribute_cols = [col for col in block_data.columns if col not in positional]
        
        # Create a mapping from (i, j, k) to attributes
        block_to_attrs = {}
        if "i" in block_data.columns and "j" in block_data.columns and "k" in block_data.columns:
            for _, row in block_data.iterrows():
                key = (int(row["i"]), int(row["j"]), int(row["k"]))
                block_to_attrs[key] = row[attribute_cols].to_dict()
        elif isinstance(block_data.index, pd.MultiIndex) and block_data.index.names == ["i", "j", "k"]:
            for key, row in block_data.iterrows():
                block_to_attrs[key] = row[attribute_cols].to_dict()
        
        # Assign vertex attributes (from the block containing the vertex)
        for attr_name in attribute_cols:
            vertex_values = []
            for ijk in vertex_ijk_list:
                if ijk in block_to_attrs:
                    vertex_values.append(block_to_attrs[ijk].get(attr_name, np.nan))
                else:
                    vertex_values.append(np.nan)
            vertex_attrs[attr_name] = np.array(vertex_values, dtype=float)
        
        # Assign face attributes (from the block containing the face)
        for attr_name in attribute_cols:
            face_values = []
            for ijk in face_ijk_list:
                if ijk in block_to_attrs:
                    face_values.append(block_to_attrs[ijk].get(attr_name, np.nan))
                else:
                    face_values.append(np.nan)
            face_attrs[attr_name] = np.array(face_values, dtype=float)
        
        return vertex_attrs, face_attrs

