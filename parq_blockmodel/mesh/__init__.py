"""Mesh module for triangulated surface/solid representations of block models.

This module provides functionality to convert regular (rotated and non-rotated)
block models into triangulated surface geometry with optional textures.

Key concepts:
    - **PLY (Polygon File Format)** is the canonical internal format for lossless
      round-tripping. Supports per-vertex and per-face attributes.
    - **GLB (glTF 2.0 binary)** is a derived exchange format with optional
      textures and materials for external viewers.
    - Geometry is generated from RegularGeometry metadata + sparse/dense block data.
    - Right-handed coordinates are guaranteed via axis validation.
    - Attribute mapping is explicit and fully customizable.

Units and Coordinates:
    All coordinates are in the units and CRS of the underlying ParquetBlockModel.
    World-space coordinates are computed via RegularGeometry's axis transformation.
    PLY and GLB metadata carry CRS information where available.

Example:
    >>> from pathlib import Path
    >>> from parq_blockmodel import ParquetBlockModel
    >>>
    >>> pbm = ParquetBlockModel(Path("model.pbm"))
    >>>
    >>> # Export to PLY (canonical, lossless)
    >>> pbm.to_ply("model.ply", attributes=["grade", "density"])
    >>>
    >>> # Export to GLB (exchange format with optional texture)
    >>> pbm.to_glb("model.glb", texture_attribute="grade", colormap="viridis")
    >>>
    >>> # Get internal mesh representation
    >>> mesh = pbm.triangulate(surface_only=True, sparse=True)
"""

__version__ = "0.1.0"

from parq_blockmodel.mesh.triangulation import BlockMeshGenerator, TriangleMesh

__all__ = [
    "BlockMeshGenerator",
    "TriangleMesh",
]

