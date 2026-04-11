"""
Mesh Module - Triangulated Surface Representations
===================================================

The ``parq_blockmodel.mesh`` module provides functionality to convert regular
block models into triangulated surface geometry. This enables visualization,
exchange with external tools, and scientific analysis.

Overview
--------

Key concepts:

* **PLY (Polygon File Format)** is the canonical internal format for mesh storage.
  It supports lossless round-tripping of all geometry and attributes.
  PLY files are ASCII by default (readable) and can be binary (compact).

* **GLB (glTF 2.0 binary)** is a derived format for external visualization.
  It supports optional textures and materials, and can be viewed in standard
  3D viewers (Babylon.js, Three.js, etc.).

* Mesh geometry is always generated from ``RegularGeometry`` metadata,
  ensuring consistency between block model and mesh representations.

* Rotated geometries are handled transparently; world-space coordinates
  reflect the rotation via axis vectors.

* Block attributes (grades, rock types, densities) are preserved as
  per-vertex and per-face properties in both PLY and GLB formats.

Coordinates and Units
---------------------

All mesh coordinates are in **world space** with units and CRS inherited
from the underlying ``ParquetBlockModel.geometry``.

For **non-rotated** models, world coordinates align with global X, Y, Z axes.

For **rotated** models, world coordinates are computed by transforming
local block coordinates via the orthonormal basis (axis_u, axis_v, axis_w).

Right-handedness is guaranteed: the cross product of axis_u and axis_v
yields axis_w.

Sparse vs. Dense Meshes
-----------------------

**Sparse meshes** include only blocks that exist in the block model data.
This is useful for geological block models where many cells are empty.
When ``surface_only=True`` (default), only exterior faces are included,
which is typically what you want for visualization.

**Dense meshes** include all blocks in the logical grid defined by geometry.
Use this for models with no missing cells or for analysis that requires
complete coverage.

Attribute Mapping
-----------------

When exporting with attributes:

* **Vertex attributes**: Each vertex inherits attributes from its parent block.
  For sparse models, vertices on block boundaries may be shared; the attribute
  is taken from the block that defines the vertex.

* **Face attributes**: Each face inherits attributes from its parent block.

* **Categorical attributes**: String/category columns are stored as integer
  codes in PLY files, with a metadata mapping for recovery. In GLB, categorical
  attributes are converted to numeric codes for visualization.

* **NaN handling**: Missing values are preserved as NaN in PLY. In GLB with
  texture attributes, NaN values are rendered as transparent.

API Reference
-------------

Main methods on ``ParquetBlockModel``:

* ``triangulate(attributes=None, surface_only=True, sparse=None)``
  → Internal mesh representation

* ``to_ply(output_path, attributes=None, surface_only=True, sparse=None, binary=False)``
  → Export to PLY format

* ``to_glb(output_path, attributes=None, texture_attribute=None, colormap='viridis', surface_only=True, sparse=None)``
  → Export to GLB format

Examples
--------

**Basic export to PLY:**

.. code-block:: python

    from pathlib import Path
    from parq_blockmodel import ParquetBlockModel

    pbm = ParquetBlockModel(Path("model.pbm"))

    # Export to PLY (lossless, includes all attributes)
    pbm.to_ply("model.ply", attributes=["grade", "density"])

**Export to GLB with colored texture:**

.. code-block:: python

    # Export to GLB with grades shown as colors
    pbm.to_glb("model.glb", texture_attribute="grade", colormap="viridis")

**Get internal mesh and inspect:**

.. code-block:: python

    mesh = pbm.triangulate(attributes=["grade"])
    print(f"Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}")
    print(f"Attributes: {list(mesh.vertex_attributes.keys())}")
    print(f"Metadata: {mesh.metadata}")

**Handle sparse model with surface-only:**

.. code-block:: python

    # For a sparse 3D model with ~10% fill, generate surface mesh
    mesh = pbm.triangulate(surface_only=True, sparse=True)

**Dense mesh from sparse model:**

.. code-block:: python

    # Include all blocks in the logical grid (fills empty cells)
    mesh = pbm.triangulate(sparse=False)

Scientific Workflow Notes
-------------------------

This module is designed for scientific and mining workflows:

* **Units are explicit**: All coordinates carry the units of the block model.
  Conversion (e.g., feet to meters) must be done at the block model level.

* **Lossless PLY round-tripping**: PLY export preserves all geometry and
  attribute data. This allows reliable interchange with other tools.

* **No implicit encodings**: Block semantics (rock types, domains) are not
  encoded implicitly in colors. Use explicit attribute columns instead.

* **Rotated geometry support**: Rotations are fully supported and transparent
  to the user; the mesh reflects the rotated orientation in world space.

* **Sparse block models**: The mesh module efficiently handles sparse models
  by generating only surfaces and boundaries, avoiding unnecessary empty space.

See Also
--------

* :class:`parq_blockmodel.ParquetBlockModel` — Main block model class
* :class:`parq_blockmodel.geometry.RegularGeometry` — Geometry metadata
* :class:`parq_blockmodel.mesh.triangulation.BlockMeshGenerator` — Core mesh generation
* :class:`parq_blockmodel.mesh.types.TriangleMesh` — Mesh data structure
"""

