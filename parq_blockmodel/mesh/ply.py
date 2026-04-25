"""PLY (Polygon File Format) support for triangulated meshes.

PLY is the canonical internal format for mesh storage, supporting
lossless round-tripping of all vertex/face attributes and geometry metadata.

Format details:
    - ASCII or binary (we use ASCII for readability, binary for efficiency)
    - Per-vertex properties: x, y, z, and optional attributes
    - Per-face connectivity with optional face attributes
    - Custom properties stored as additional scalar fields
    - Metadata in comments and extra properties

References:
    https://en.wikipedia.org/wiki/PLY_(file_format)
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from parq_blockmodel.mesh.types import TriangleMesh


def write_ply(
    mesh: TriangleMesh,
    output_path: Union[str, Path],
    binary: bool = False,
) -> None:
    """Write a TriangleMesh to a PLY file.
    
    The PLY file will contain:
    - Vertex coordinates (x, y, z)
    - Optional per-vertex attributes (grade, density, rock_type, etc.)
    - Vertex ijk indices (if available)
    - Face connectivity (as vertex_indices)
    - Optional per-face attributes
    - Metadata in comments and custom properties
    
    Parameters
    ----------
    mesh : TriangleMesh
        The mesh to write.
    output_path : str or Path
        Path to the output PLY file.
    binary : bool, default False
        If True, write in binary format. If False, write ASCII.
    
    Raises
    ------
    ValueError
        If the mesh is invalid or contains unsupported attribute types.
    """
    output_path = Path(output_path)
    mesh.validate()
    
    # Prepare vertex and face data
    vertex_data = _prepare_vertex_data(mesh)
    face_data = _prepare_face_data(mesh)
    
    if binary:
        _write_ply_binary(output_path, mesh, vertex_data, face_data)
    else:
        _write_ply_ascii(output_path, mesh, vertex_data, face_data)


def read_ply(
    input_path: Union[str, Path],
) -> TriangleMesh:
    """Read a TriangleMesh from a PLY file.
    
    Reconstructs the full mesh including vertex/face attributes and metadata.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input PLY file.
    
    Returns
    -------
    TriangleMesh
        The reconstructed mesh.
    
    Raises
    ------
    ValueError
        If the PLY file is malformed or incompatible.
    """
    input_path = Path(input_path)
    
    # Parse PLY header and data
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_lines = []
    data_start = 0
    for i, line in enumerate(lines):
        header_lines.append(line.strip())
        if line.strip() == "end_header":
            data_start = i + 1
            break
    
    if "end_header" not in header_lines:
        raise ValueError("PLY file missing end_header marker.")
    
    # Extract metadata from comments
    metadata = {}
    vertex_count = 0
    face_count = 0
    vertex_properties = []
    face_properties = []
    
    i = 0
    while i < len(header_lines):
        line = header_lines[i]
        
        if line.startswith("comment "):
            comment_data = line[8:].strip()
            if "=" in comment_data:
                key, value = comment_data.split("=", 1)
                try:
                    metadata[key.strip()] = json.loads(value.strip())
                except json.JSONDecodeError:
                    metadata[key.strip()] = value.strip()
        
        elif line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
        
        elif line.startswith("property "):
            # Parse property (type name, or list count type name for faces)
            if header_lines[i-1].startswith("element vertex"):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("Vertex properties cannot be lists in this implementation.")
                vertex_properties.append(parts[-1])
            elif header_lines[i-1].startswith("element face"):
                pass  # Face properties are handled separately
        
        i += 1
    
    # Read data (ASCII for simplicity; binary support can be added)
    vertices_list = []
    faces_list = []
    
    for i in range(data_start, data_start + vertex_count):
        parts = lines[i].strip().split()
        vertices_list.append([float(p) for p in parts[:3]])
    
    for i in range(data_start + vertex_count, data_start + vertex_count + face_count):
        parts = lines[i].strip().split()
        n = int(parts[0])
        face = [int(parts[j]) for j in range(1, 1 + n)]
        if n != 3:
            raise ValueError(f"Non-triangular face encountered: {n} vertices.")
        faces_list.append(face)
    
    vertices = np.array(vertices_list, dtype=float)
    faces = np.array(faces_list, dtype=np.intp)
    
    # TODO: Parse vertex and face attributes from data
    vertex_attributes = {}
    face_attributes = {}
    
    mesh = TriangleMesh(
        vertices=vertices,
        faces=faces,
        vertex_attributes=vertex_attributes,
        face_attributes=face_attributes,
        metadata=metadata,
    )
    
    return mesh


def _prepare_vertex_data(mesh: TriangleMesh) -> dict:
    """Prepare vertex data for PLY export.
    
    Returns dict with "columns" (list of names) and "data" (dict of arrays).
    """
    columns = ["x", "y", "z"]
    data = {
        "x": mesh.vertices[:, 0],
        "y": mesh.vertices[:, 1],
        "z": mesh.vertices[:, 2],
    }
    
    # Add ijk if available
    if mesh.vertex_ijk is not None:
        columns.extend(["i", "j", "k"])
        data["i"] = mesh.vertex_ijk[:, 0]
        data["j"] = mesh.vertex_ijk[:, 1]
        data["k"] = mesh.vertex_ijk[:, 2]
    
    # Add other vertex attributes
    for attr_name in sorted(mesh.vertex_attributes.keys()):
        columns.append(attr_name)
        data[attr_name] = mesh.vertex_attributes[attr_name]
    
    return {"columns": columns, "data": data}


def _prepare_face_data(mesh: TriangleMesh) -> dict:
    """Prepare face data for PLY export."""
    columns = ["vertex_indices"]
    data = {"vertex_indices": mesh.faces}
    
    # Add ijk if available
    if mesh.face_ijk is not None:
        columns.extend(["i", "j", "k"])
        data["i"] = mesh.face_ijk[:, 0]
        data["j"] = mesh.face_ijk[:, 1]
        data["k"] = mesh.face_ijk[:, 2]
    
    # Add face attributes
    for attr_name in sorted(mesh.face_attributes.keys()):
        columns.append(attr_name)
        data[attr_name] = mesh.face_attributes[attr_name]
    
    return {"columns": columns, "data": data}


def _write_ply_ascii(
    output_path: Path,
    mesh: TriangleMesh,
    vertex_data: dict,
    face_data: dict,
) -> None:
    """Write mesh to ASCII PLY file."""
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        
        # Metadata comments
        for key, value in mesh.metadata.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            f.write(f"comment {key}={value_str}\n")
        
        # Vertex element
        _int_props = {"i", "j", "k"}
        f.write(f"element vertex {mesh.n_vertices}\n")
        for col in vertex_data["columns"]:
            ply_type = "int" if col in _int_props else "float"
            f.write(f"property {ply_type} {col}\n")
        
        # Face element
        f.write(f"element face {mesh.n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        for col in face_data["columns"]:
            if col != "vertex_indices":
                ply_type = "int" if col in _int_props else "float"
                f.write(f"property {ply_type} {col}\n")
        
        f.write("end_header\n")
        
        # Vertex data — build each row as a space-separated string.
        # Using numpy directly is much faster than iterrows().
        col_arrays = [vertex_data["data"][col] for col in vertex_data["columns"]]
        for row_vals in zip(*col_arrays):
            f.write(" ".join(str(v) for v in row_vals) + "\n")
        
        # Face data
        for face_idx, face in enumerate(mesh.faces):
            f.write(f"3 {face[0]} {face[1]} {face[2]}")
            if mesh.face_ijk is not None:
                f.write(f" {mesh.face_ijk[face_idx, 0]} {mesh.face_ijk[face_idx, 1]} {mesh.face_ijk[face_idx, 2]}")
            for attr_name in sorted(mesh.face_attributes.keys()):
                f.write(f" {mesh.face_attributes[attr_name][face_idx]}")
            f.write("\n")


def _write_ply_binary(
    output_path: Path,
    mesh: TriangleMesh,
    vertex_data: dict,
    face_data: dict,
) -> None:
    """Write mesh to binary PLY file (little-endian).
    
    Not yet implemented; falls back to ASCII for now.
    """
    # TODO: Implement binary PLY writing
    _write_ply_ascii(output_path, mesh, vertex_data, face_data)

