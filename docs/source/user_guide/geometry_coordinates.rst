.. _geometry-coordinates:

Geometry and Coordinate Transformations
=========================================

Overview
--------

``parq-blockmodel`` uses a precise metadata-driven geometry model to define block model layout.
All spatial coordinates are derived from:

1. **Geometry metadata** (stored in Parquet key-value metadata)
2. **Logical indices** (i, j, k) using canonical C-order
3. **Axis orientation** (rotation transformation to world space)

The diagram below illustrates the complete flow from geometry definition to world coordinates:

.. graphviz::  ../_static/geometry_coordinate_flow.dot
   :align: center
   :caption: Geometry metadata → Logical indices (i,j,k) → Local coordinates → Rotation → World coordinates (x,y,z)

Geometry Definition
-------------------

The :class:`parq_blockmodel.geometry.RegularGeometry` class encapsulates all geometry metadata:

**corner** ``(x₀, y₀, z₀)``
    The world-space position of the (i=0, j=0, k=0) block corner.

**block_size** ``(dx, dy, dz)``
    The size of each block along the logical u, v, w axes.

**shape** ``(nᵢ, nⱼ, nₖ)``
    The number of blocks along each logical axis.

**axis_u, axis_v, axis_w**
    Orthonormal basis vectors that define the orientation of the logical axes in world space.

    Default (unrotated):

    - ``axis_u = (1, 0, 0)`` → u-axis aligned with world X
    - ``axis_v = (0, 1, 0)`` → v-axis aligned with world Y
    - ``axis_w = (0, 0, 1)`` → w-axis aligned with world Z

**srs** (optional)
    Spatial Reference System (CRS) identifier for geographic context.

Logical Indices (i, j, k)
--------------------------

Block indices are stored and processed using **C-order (row-major)** conventions:

- **i** varies fastest (column index)
- **j** varies next (row index)
- **k** varies slowest (depth index)

The flat row index is computed as:

.. math::

    r = i + n_i \cdot (j + n_j \cdot k)

This C-order layout:

    ✓ Matches NumPy's default behavior (``np.ravel``, ``np.unravel_index``)
    ✓ Aligns with Parquet row-oriented storage
    ✓ Enables efficient dense array operations

.. note::

    Do **not** confuse C-order with lexicographic MultiIndex sorting (x, y, z).
    Lexicographic sorting is *not* C-order and should not define canonical storage.

Local Coordinates
-----------------

Given logical indices (i, j, k), the **local grid position** is computed as:

.. math::

    u &= x_0 + (i + 0.5) \cdot dx \\
    v &= y_0 + (j + 0.5) \cdot dy \\
    w &= z_0 + (k + 0.5) \cdot dz

The offset 0.5 positions each coordinate at the **block centroid** (not corner).

Axis Orientation and Rotation
------------------------------

The logical (u, v, w) coordinates are transformed to world-space (x, y, z) via a rotation matrix:

.. math::

    R = \begin{bmatrix}
        \text{axis\_u} \\
        \text{axis\_v} \\
        \text{axis\_w}
    \end{bmatrix}^T

The world coordinates are then:

.. math::

    \begin{bmatrix} x \\ y \\ z \end{bmatrix} = R \cdot \begin{bmatrix} u \\ v \\ w \end{bmatrix}

**Key properties:**

- Since axis vectors are **orthonormal**, the matrix is orthogonal: :math:`R^{-1} = R^T`
- The inverse transformation (world → local) is: :math:`\begin{bmatrix} u \\ v \\ w \end{bmatrix} = R^T \cdot \begin{bmatrix} x \\ y \\ z \end{bmatrix}`
- For unrotated geometries, R is the identity matrix (no rotation applied)

Once centroid coordinates are available in world space, they can also be converted into
``world_id`` using ``parq_blockmodel.utils.spatial_encoding.encode_coordinates``.

``world_id`` is a stable 64-bit positional identifier derived from world-space XYZ
coordinates within a given CRS. It uniquely identifies a block by location and is
intended for joins, tracking, and external references. It is not a spatial index and
should not be used for spatial range queries.

This world_id derivation is shown in the diagram as an optional downstream step from
centroid positions.

Conversion APIs
---------------

The :class:`parq_blockmodel.geometry.RegularGeometry` class provides several conversion methods:

**Forward (logical → world):**

- :meth:`~parq_blockmodel.geometry.RegularGeometry.xyz_from_ijk` - Convert (i, j, k) to (x, y, z)
- :meth:`~parq_blockmodel.geometry.RegularGeometry.xyz_from_row_index` - Convert flat row index to (x, y, z)

**Inverse (world → logical):**

- :meth:`~parq_blockmodel.geometry.RegularGeometry.ijk_from_xyz` - Convert (x, y, z) to (i, j, k)
- :meth:`~parq_blockmodel.geometry.RegularGeometry.row_index_from_xyz` - Convert (x, y, z) to flat row index

**Index conversion:**

- :meth:`~parq_blockmodel.geometry.RegularGeometry.ijk_from_row_index` - Unravel flat index to (i, j, k)
- :meth:`~parq_blockmodel.geometry.RegularGeometry.row_index_from_ijk` - Flatten (i, j, k) to row index

**Export:**

- :meth:`~parq_blockmodel.geometry.RegularGeometry.to_dataframe_xyz` - Export centroid coordinates to DataFrame
- :meth:`~parq_blockmodel.geometry.RegularGeometry.to_multi_index_xyz` - Export as MultiIndex
- :meth:`~parq_blockmodel.geometry.RegularGeometry.to_metadata_dict` - Serialize to dict
- :meth:`~parq_blockmodel.geometry.RegularGeometry.from_metadata` - Reconstruct from dict

Metadata Storage
----------------

All geometry metadata is embedded in Parquet files under a reserved key (default: ``"parq-blockmodel"``).

This approach:

✓ Eliminates the need for external sidecar files
✓ Keeps data and metadata synchronized
✓ Enables seamless data interchange
✓ Supports efficient metadata-driven processing

Metadata is serialized as JSON and stored in the Parquet file's key-value metadata:

.. code-block:: json

    {
      "corner": [100.0, 200.0, 300.0],
      "block_size": [10.0, 10.0, 5.0],
      "shape": [50, 50, 20],
      "axis_u": [1.0, 0.0, 0.0],
      "axis_v": [0.0, 1.0, 0.0],
      "axis_w": [0.0, 0.0, 1.0],
      "srs": null
    }

Examples
--------

Create a regular (unrotated) geometry:

.. code-block:: python

    from parq_blockmodel.geometry import RegularGeometry

    geom = RegularGeometry(
        corner=(100.0, 200.0, 300.0),
        block_size=(10.0, 10.0, 5.0),
        shape=(50, 50, 20),
    )

    # Convert logical index to world coordinates
    x, y, z = geom.xyz_from_ijk(i=10, j=5, k=2)

Create a rotated geometry:

.. code-block:: python

    import numpy as np
    from parq_blockmodel.geometry import RegularGeometry

    # Define 30-degree rotation around Z-axis
    angle = np.radians(30)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(10.0, 10.0, 5.0),
        shape=(50, 50, 20),
        axis_u=(cos_a, sin_a, 0.0),    # Rotated u-axis
        axis_v=(-sin_a, cos_a, 0.0),   # Rotated v-axis
        axis_w=(0.0, 0.0, 1.0),        # Z-axis unchanged
    )

See Also
--------

- :ref:`geometry-metadata-design` - Detailed developer documentation
- :class:`parq_blockmodel.geometry.RegularGeometry` - Full API reference
- :class:`parq_blockmodel.blockmodel.ParquetBlockModel` - Block model container

