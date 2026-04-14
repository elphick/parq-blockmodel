"""
API Reference: Mesh Module
===========================

.. module:: parq_blockmodel.mesh
   :synopsis: Triangulated surface representations of block models

This module provides classes and functions for converting block models
into triangulated surface geometry and exporting to standard formats.

Classes
-------

TriangleMesh
~~~~~~~~~~~~

.. autoclass:: TriangleMesh
   :members:
   :undoc-members:

   Represents a triangle mesh with optional vertex and face attributes.

   Attributes
   ----------
   vertices : np.ndarray
       Vertex coordinates, shape (n_vertices, 3) in world space.
   faces : np.ndarray
       Triangle face indices, shape (n_faces, 3).
   vertex_attributes : dict[str, np.ndarray]
       Per-vertex attributes (e.g., grades, rock types).
   face_attributes : dict[str, np.ndarray]
       Per-face attributes.
   vertex_ijk : np.ndarray, optional
       Logical (i, j, k) indices for vertices.
   face_ijk : np.ndarray, optional
       Logical (i, j, k) indices for faces.
   metadata : dict
       Geometry and export metadata.

   Methods
   -------
   validate()
       Validate mesh integrity.
   n_vertices
       Number of vertices.
   n_faces
       Number of faces.

BlockMeshGenerator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BlockMeshGenerator
   :members:
   :undoc-members:

   Generates triangle meshes from block model geometry and data.

   Parameters
   ----------
   geometry : RegularGeometry
       Block model geometry.

   Methods
   -------
   triangulate(block_data, surface_only, sparse)
       Generate a triangle mesh.
   block_corners_local()
       Get local cube corner coordinates.
   block_vertices_xyz(i, j, k)
       Get world coordinates of block corners.
   cube_faces_ccw()
       Get cube face triangles with CCW winding.

Functions
---------

PLY Format Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: write_ply
   :noindex:

   Write a TriangleMesh to PLY format.

   Parameters
   ----------
   mesh : TriangleMesh
       Mesh to export.
   output_path : str or Path
       Target PLY file.
   binary : bool, default False
       Use binary format if True.

.. autofunction:: read_ply
   :noindex:

   Read a TriangleMesh from PLY format.

   Parameters
   ----------
   input_path : str or Path
       Source PLY file.

   Returns
   -------
   TriangleMesh
       Reconstructed mesh.

GLB Format Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: write_glb
   :noindex:

   Write a TriangleMesh to GLB (glTF 2.0 binary) format.

   Parameters
   ----------
   mesh : TriangleMesh
       Mesh to export.
   output_path : str or Path
       Target GLB file.
   texture_attribute : str, optional
       Attribute for vertex colors.
   colormap : str, default "viridis"
       Colormap for texture mapping.
   include_metadata : bool, default True
       Include geometry metadata in glTF extras.

ParquetBlockModel Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

These methods are added to ``ParquetBlockModel``:

.. automethod:: ParquetBlockModel.triangulate
   :noindex:

   Generate a triangulated mesh from the block model.

.. automethod:: ParquetBlockModel.to_ply
   :noindex:

   Export block model to PLY format.

.. automethod:: ParquetBlockModel.to_glb
   :noindex:

   Export block model to GLB format.

Examples
--------

**Generate and inspect mesh:**

.. code-block:: python

    from parq_blockmodel import ParquetBlockModel
    from pathlib import Path

    pbm = ParquetBlockModel(Path("model.pbm"))
    mesh = pbm.triangulate(attributes=["grade"])

    print(f"Vertices: {mesh.n_vertices}")
    print(f"Faces: {mesh.n_faces}")
    print(f"Attributes: {list(mesh.vertex_attributes.keys())}")

**Export to PLY (lossless):**

.. code-block:: python

    pbm.to_ply("model.ply", attributes=["grade", "density"])

**Export to GLB with colors:**

.. code-block:: python

    pbm.to_glb("model.glb", texture_attribute="grade", colormap="viridis")

**Handle sparse models:**

.. code-block:: python

    # Surface-only mesh for sparse model (common in mining)
    mesh = pbm.triangulate(surface_only=True, sparse=True)

Notes
-----

Coordinate Systems
    All mesh coordinates are in **world space** with units from the
    underlying block model. Rotations are applied via RegularGeometry's
    axis vectors.

Right-handedness
    Mesh triangles use counter-clockwise (CCW) winding order when viewed
    from the exterior (right-hand rule), ensuring compatibility with
    standard 3D viewers.

Sparse Models
    For block models with missing cells, use ``surface_only=True`` (default)
    to generate only exterior faces, improving performance and visual clarity.

Attribute Preservation
    Block attributes are preserved as per-vertex and per-face properties in
    both PLY and GLB formats. Categorical attributes are encoded as integers
    with metadata mapping.

See Also
--------

* :class:`parq_blockmodel.ParquetBlockModel`
* :class:`parq_blockmodel.geometry.RegularGeometry`
"""

