"""GLB (glTF 2.0 binary) support for mesh export.

GLB is a derived format used for external visualization and exchange.
It supports optional textures and materials, and carries metadata via
the glTF extensions mechanism.

This implementation is a lightweight wrapper that converts TriangleMesh
to GLB format without using heavy dependencies. For advanced features
(animations, skeletal rigs, etc.), consider using pygltf or trimesh.

References:
    https://github.com/KhronosGroup/glTF/tree/main/specification/2.0
"""

import json
import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

from parq_blockmodel.mesh.types import TriangleMesh


def write_glb(
    mesh: TriangleMesh,
    output_path: Union[str, Path],
    texture_attribute: Optional[str] = None,
    colormap: str = "viridis",
    include_metadata: bool = True,
) -> None:
    """Write a TriangleMesh to GLB (glTF 2.0 binary) format.
    
    This generates a single-file GLB suitable for viewing in standard
    3D viewers (Babylon.js, Three.js, etc.). Optionally applies vertex
    colors based on a scalar attribute.
    
    Parameters
    ----------
    mesh : TriangleMesh
        The mesh to write.
    output_path : str or Path
        Path to the output .glb file.
    texture_attribute : str, optional
        If provided, vertex attribute to use for coloring (e.g., "grade").
        Colors are mapped via the specified colormap.
    colormap : str, default "viridis"
        Matplotlib colormap name for texture_attribute mapping.
        Only used if texture_attribute is specified.
    include_metadata : bool, default True
        If True, embed mesh metadata in glTF extras.
    
    Raises
    ------
    ValueError
        If the mesh is invalid or attribute is not found.
    """
    output_path = Path(output_path)
    mesh.validate()
    
    # Create glTF JSON structure
    gltf_data = _create_gltf_data(mesh, texture_attribute, colormap, include_metadata)
    
    # Serialize to GLB
    _write_glb_file(output_path, gltf_data)


def _create_gltf_data(
    mesh: TriangleMesh,
    texture_attribute: Optional[str],
    colormap: str,
    include_metadata: bool,
) -> dict:
    """Create glTF JSON structure for the mesh.
    
    Returns a dict with 'json' (glTF structure) and 'bin' (binary buffer).
    """
    # Allocate buffer for vertices, indices, and optionally colors
    vertex_count = mesh.n_vertices
    index_count = mesh.n_faces * 3
    
    # Build binary buffer
    buffer_views = []
    buffer_data = b""
    
    # Vertex positions (float32)
    vertices_bytes = mesh.vertices.astype(np.float32).tobytes()
    buffer_views.append({
        "buffer": 0,
        "byteOffset": len(buffer_data),
        "byteLength": len(vertices_bytes),
        "target": 34962,  # ARRAY_BUFFER
    })
    buffer_data += vertices_bytes
    vertices_accessor_idx = len(buffer_views) - 1
    
    # Vertex colors (uint8, optional)
    color_accessor_idx = None
    if texture_attribute and texture_attribute in mesh.vertex_attributes:
        colors = _map_attribute_to_colors(
            mesh.vertex_attributes[texture_attribute],
            colormap,
        )
        colors_bytes = colors.astype(np.uint8).tobytes()
        buffer_views.append({
            "buffer": 0,
            "byteOffset": len(buffer_data),
            "byteLength": len(colors_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })
        buffer_data += colors_bytes
        color_accessor_idx = len(buffer_views) - 1
    
    # Face indices (uint32)
    indices_flat = mesh.faces.flatten().astype(np.uint32)
    indices_bytes = indices_flat.tobytes()
    buffer_views.append({
        "buffer": 0,
        "byteOffset": len(buffer_data),
        "byteLength": len(indices_bytes),
        "target": 34963,  # ELEMENT_ARRAY_BUFFER
    })
    buffer_data += indices_bytes
    indices_accessor_idx = len(buffer_views) - 1
    
    # Create accessors
    accessors = [
        {
            "bufferView": vertices_accessor_idx,
            "componentType": 5126,  # FLOAT
            "count": vertex_count,
            "type": "VEC3",
            "min": mesh.vertices.min(axis=0).tolist(),
            "max": mesh.vertices.max(axis=0).tolist(),
        },
    ]
    
    if color_accessor_idx is not None:
        accessors.append({
            "bufferView": color_accessor_idx,
            "componentType": 5121,  # UNSIGNED_BYTE
            "count": vertex_count,
            "type": "VEC4",
            "normalized": True,
        })
    
    accessors.append({
        "bufferView": indices_accessor_idx,
        "componentType": 5125,  # UNSIGNED_INT
        "count": index_count,
        "type": "SCALAR",
    })
    
    # Create mesh primitive
    primitive = {
        "attributes": {
            "POSITION": 0,
        },
        "indices": len(accessors) - 1,
    }
    
    if color_accessor_idx is not None:
        primitive["attributes"]["COLOR_0"] = 1
    
    # Create glTF structure
    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "parq-blockmodel",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{
            "byteLength": len(buffer_data),
        }],
    }
    
    # Add material if colors are used
    if color_accessor_idx is not None:
        gltf_json["materials"] = [{
            "alphaMode": "OPAQUE",
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
        }]
        gltf_json["meshes"][0]["primitives"][0]["material"] = 0
    else:
        # Default gray material
        gltf_json["materials"] = [{
            "alphaMode": "OPAQUE",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
        }]
        gltf_json["meshes"][0]["primitives"][0]["material"] = 0
    
    # Add metadata in extras
    if include_metadata:
        gltf_json["extras"] = mesh.metadata
    
    return {"json": gltf_json, "bin": buffer_data}


def _map_attribute_to_colors(
    attribute: np.ndarray,
    colormap: str,
) -> np.ndarray:
    """Map scalar attribute to RGBA colors.
    
    Uses a matplotlib colormap to map scalar values to colors.
    
    Parameters
    ----------
    attribute : np.ndarray
        1D array of scalar values.
    colormap : str
        Name of a matplotlib colormap (e.g., "viridis", "plasma").
    
    Returns
    -------
    np.ndarray
        Shape (len(attribute), 4), RGBA values in [0, 255] (uint8 range).
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
    except ImportError:
        # Fallback if matplotlib not available: use grayscale
        cmap = None
    
    # Normalize attribute values to [0, 1]
    attr_min = np.nanmin(attribute)
    attr_max = np.nanmax(attribute)
    if attr_min == attr_max:
        normalized = np.zeros_like(attribute)
    else:
        normalized = (attribute - attr_min) / (attr_max - attr_min)
    
    # Map to colors
    if cmap is not None:
        rgba = cmap(normalized)  # Returns (n, 4) with values in [0, 1]
        colors = (rgba * 255).astype(np.uint8)
    else:
        # Grayscale fallback
        gray = (normalized * 255).astype(np.uint8)
        colors = np.column_stack([gray, gray, gray, np.full_like(gray, 255)])
    
    # Replace NaN values with transparent
    nan_mask = np.isnan(attribute)
    colors[nan_mask, 3] = 0  # Transparent
    
    return colors


def _write_glb_file(output_path: Path, gltf_data: dict) -> None:
    """Write glTF data to GLB file.
    
    GLB format:
        - 4 bytes: magic "glTF" (0x46546C67)
        - 4 bytes: version (2)
        - 4 bytes: total file size (bytes)
        - Chunk 0: JSON (type 0x4E4F534A = "JSON")
        - Chunk 1: Binary (type 0x004E4942 = "BIN\0")
    """
    json_text = json.dumps(gltf_data["json"]).encode('utf-8')
    bin_data = gltf_data["bin"]
    
    # Align JSON chunk to 4-byte boundary with spaces
    json_padding = (4 - (len(json_text) % 4)) % 4
    json_chunk_data = json_text + b' ' * json_padding
    
    # Binary chunk (already aligned as needed)
    bin_padding = (4 - (len(bin_data) % 4)) % 4
    bin_chunk_data = bin_data + b'\x00' * bin_padding
    
    # Calculate sizes
    json_chunk_size = len(json_chunk_data)
    bin_chunk_size = len(bin_chunk_data)
    total_size = 28 + json_chunk_size + 8 + bin_chunk_size + 8  # header + json_header + json_data + bin_header + bin_data
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'glTF')  # magic
        f.write(struct.pack('<I', 2))  # version
        f.write(struct.pack('<I', total_size))  # file size
        
        # JSON chunk header
        f.write(struct.pack('<I', json_chunk_size))  # chunk size
        f.write(b'JSON')  # chunk type
        f.write(json_chunk_data)
        
        # Binary chunk header
        f.write(struct.pack('<I', bin_chunk_size))  # chunk size
        f.write(b'BIN\x00')  # chunk type
        f.write(bin_chunk_data)

