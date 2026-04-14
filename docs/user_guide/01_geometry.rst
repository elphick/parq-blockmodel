Geometry
========

What is Geometry?
-----------------

In ``parq-blockmodel``, **geometry** defines the spatial structure of a block model. It answers
fundamental questions:

- What is the shape and size of each block?
- How are blocks arranged in 3D space?
- What are the (x, y, z) coordinates of each block?
- What is the total extent of the model?

Geometry is the **foundation** of every block model. Without it, you only have tables of data
with no spatial meaning. With it, you can:

- Query blocks by location
- Transform between block indices (i, j, k) and world coordinates (x, y, z)
- Export to visualization and analysis tools
- Combine multiple block models using shared references
- Support rotated/angled orientations common in mining and geology

Canonical Representation: RegularGeometry
------------------------------------------

``parq-blockmodel`` supports only **regular** (rectilinear) block models, represented by the
:class:`~parq_blockmodel.geometry.RegularGeometry` class. A regular geometry has:

- Uniform block sizes (all blocks are the same shape)
- Rectangular grid layout (not irregular or unstructured)
- Optional rotation (blocks can be tilted to align with geological features)

The geometry is defined by a small set of metadata parameters:

- **shape**: Number of blocks along each axis (nᵢ, nⱼ, nₖ)
- **block_size**: Size of each block ``(dx, dy, dz)`` along the local logical axes
  ``(i, j, k)`` / ``(u, v, w)``
- **corner**: Local-space corner of block (i=0, j=0, k=0)
- **origin**: World-space position of local (0, 0, 0)
- **axis_u, axis_v, axis_w**: Orthonormal vectors defining orientation (optional rotation)

Creating Geometry
-----------------

The simplest case (unrotated grid at origin):

.. code-block:: python

    from parq_blockmodel import RegularGeometry

    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),        # Grid origin
        block_size=(10.0, 10.0, 5.0),  # Uniform spacing along local i/j/k axes
        shape=(50, 50, 20),            # 50 × 50 × 20 blocks
    )

For rotated geometries, add axis vectors. The ``block_size=(dx, dy, dz)`` tuple still
describes spacing along the local grid axes; the axis vectors define how that grid is
embedded in world space.

.. code-block:: python

    import numpy as np
    from parq_blockmodel import RegularGeometry

    # 30° rotation around Z-axis
    angle = np.radians(30)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    geom = RegularGeometry(
        corner=(100.0, 200.0, 300.0),
        block_size=(10.0, 10.0, 5.0),
        shape=(50, 50, 20),
        axis_u=(cos_a, sin_a, 0.0),    # Rotated u-axis
        axis_v=(-sin_a, cos_a, 0.0),   # Rotated v-axis
        axis_w=(0.0, 0.0, 1.0),        # Z-axis unchanged
    )

Converting Between Indices and Coordinates
-------------------------------------------

The key strength of geometry is converting between:

- **Logical indices** (i, j, k): Block position in the grid (0 ≤ i < nᵢ, etc.)
- **World coordinates** (x, y, z): Position in physical space

Forward transformation (indices → coordinates):

.. code-block:: python

    # Get world-space centroid of block (i=5, j=10, k=2)
    x, y, z = geom.xyz_from_ijk(i=5, j=10, k=2)

Inverse transformation (coordinates → indices):

.. code-block:: python

    # Find block containing world point (105.0, 215.0, 302.5)
    i, j, k = geom.ijk_from_xyz(x=105.0, y=215.0, z=302.5)

You can also work with flat row indices (0 to N-1):

.. code-block:: python

    # Convert flat index to (i, j, k)
    i, j, k = geom.ijk_from_row_index(row=42)

    # Convert (i, j, k) to flat index (C-order)
    row = geom.row_index_from_ijk(i=5, j=10, k=2)

Internal Architecture: LocalGeometry and WorldFrame
---------------------------------------------------

For most users, ``RegularGeometry`` is the only class needed. Internally, it combines two components:

- **LocalGeometry**: Handles logical (i, j, k) indexing and block dimensions. Pure grid math,
  no rotation.
- **WorldFrame**: Handles world embedding, axis orientation, and coordinate transformations.

This separation cleanly isolates concerns:

- LocalGeometry = "What block am I?" (indexing)
- WorldFrame = "Where is that block?" (world embedding)

Users can access these components directly for advanced use cases, but the public API focuses
on the unified ``RegularGeometry``.

See Also
--------

- :ref:`geometry-coordinates` - Detailed coordinate transformation guide with diagrams
- :ref:`geometry-metadata-design` - Developer notes on internal architecture
- :class:`~parq_blockmodel.geometry.RegularGeometry` - Full API reference
- :class:`~parq_blockmodel.blockmodel.ParquetBlockModel` - How to use geometry with block models
