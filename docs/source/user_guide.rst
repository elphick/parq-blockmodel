User Guide
==========

The purpose of this guide is to walk the user through how-to use the package.
It is complemented by the examples.

.. note::

   This is a work in progress and not all features are documented yet.
   Please check back later for updates.

Geometry, ijk indexing, and ``.pbm`` files
-----------------------------------------

``parq-blockmodel`` works with regular 3D block models that are stored in
Parquet files and wrapped by the :class:`parq_blockmodel.blockmodel.ParquetBlockModel`
class.

The **canonical on-disk container** for a block model is a file with a
``.pbm`` extension. A ``.pbm`` file is just a Parquet table that:

* stores block attributes (grades, density, rock type, ...),
* may include centroid columns ``x``, ``y``, ``z`` for convenience and
  interoperability, and
* always carries embedded geometry metadata under a reserved key
  (``"parq-blockmodel"``).

The geometry metadata encodes a regular logical grid in terms of
``(i, j, k)`` indices via :class:`parq_blockmodel.geometry.RegularGeometry`.
This ijk grid is the **single source of truth** for the model layout:

* ``shape`` gives the number of blocks along each axis,
* ``block_size`` and ``corner`` describe where the grid sits in world
  coordinates, and
* optional axis vectors (``axis_u``, ``axis_v``, ``axis_w``) define the
  orientation of the logical axes in 3D space.

World coordinates ``(x, y, z)`` are therefore a **derived view** of
``(i, j, k) + geometry`` rather than the primary index. Existing
xyz-centric Parquet files remain supported, and ``ParquetBlockModel`` can
promote them into canonical ``.pbm`` containers with embedded geometry.

For a deeper, developer-oriented discussion of how geometry and metadata
are encoded, see :ref:`geometry-metadata-design`.

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   user_guide/geometry_coordinates
   user_guide/*