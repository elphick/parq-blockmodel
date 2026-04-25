.. _geometry-metadata-design:

=====================================
Developer notes: geometry & metadata
=====================================

Overview
========

This document describes the design for moving ``parq-blockmodel`` to a
canonical ``(i, j, k)`` indexing scheme, with geometry stored as metadata
in Parquet files and attached to in-memory DataFrames via ``df.attrs``.

The goals are:

1. Make ``(i, j, k)`` the canonical, internal indexing for all block models.
2. Treat world coordinates ``(x, y, z)`` as *derived* from geometry.
3. Store block model geometry as a compact, versioned metadata payload
   ("geometry metadata") under a single reserved key, shared between
   Parquet key-value metadata and ``DataFrame.attrs``.
4. Keep a single dense geometry concept (``RegularGeometry``) and handle
   sparse vs dense *data* purely at the block model / DataFrame level.
5. Maintain backward compatibility with existing xyz-centric Parquet
   files that do not yet carry geometry metadata.


Design Steps
============

1. Canonical indexing and ordering
----------------------------------

- **Canonical index**: ``(i, j, k)`` block indices.
- **Canonical ordering**: NumPy **C-order** (row-major), where:

  - ``i`` varies fastest,
  - ``j`` next,
  - ``k`` slowest.

  The flattened row index ``r`` is given by::

      r = i + ni * (j + nj * k)

  where ``shape = (ni, nj, nk)``.

- ``RegularGeometry`` is responsible for converting between
  ``(i, j, k)`` and ``(x, y, z)`` using geometry metadata:

  - ``corner``: local corner of block ``(i=0, j=0, k=0)`` in ``(u, v, w)``
  - ``origin``: world position of local ``(0, 0, 0)``
  - ``block_size``: block dimensions ``(dx, dy, dz)`` along the local
    ``i/j/k`` (``u/v/w``) axes
  - ``shape``: number of blocks ``(ni, nj, nk)``
  - ``axis_u``, ``axis_v``, ``axis_w``: orthonormal basis vectors
  - ``srs``: optional spatial reference system identifier

- ``(x, y, z)`` centroids are derived from geometry; they are not
  considered the primary key for block identity.


2. Internal architecture: LocalGeometry and WorldFrame
-----------------------------------------------------

The **public** :class:`~parq_blockmodel.geometry.RegularGeometry` is composed of two
internal components that separate concerns:

**LocalGeometry** (local lattice, no rotation/CRS)

- Represents the logical (i, j, k) indexing grid
- Stores: ``corner``, ``block_size``, ``shape``
- Provides: C-order conversion methods (``ijk_from_row_index``, ``row_index_from_ijk``)
- Computes: local centroid coordinates (u, v, w)
- **No rotation, no world frame, no CRS**

**WorldFrame** (world embedding)

- Represents the spatial reference system and orientation
- Stores: ``origin``, ``axis_u``, ``axis_v``, ``axis_w``, ``srs``
- Provides: orthonormal rotation matrix and transform methods
- Maps: local (u, v, w) ↔ world (x, y, z) coordinates
- Ensures: axes are orthonormal (enforced in ``__post_init__``)
- Handles: inverse transformations via transpose (since axes are orthonormal)

**RegularGeometry** (public composite)

- Combines ``LocalGeometry`` and ``WorldFrame``
- Provides the complete grid definition and coordinate transformation API
- Delegates to components internally
- Exposes high-level methods like ``xyz_from_ijk``, ``ijk_from_xyz``, etc.
- Users typically interact only with ``RegularGeometry``; ``LocalGeometry`` and
  ``WorldFrame`` are available for advanced use cases but not required

This separation enables:

- Clean separation of concerns (local indexing vs. world embedding)
- Simpler testing and validation of each component
- Clear data flow through coordinate transformations
- Transparent internal refactoring without affecting the public API

3. Geometry metadata layout
---------------------------

``RegularGeometry`` defines a compact, JSON-serialisable metadata payload
via ``to_metadata_dict`` and ``from_metadata``.

- The **canonical geometry payload** is a plain ``dict``::

    {
        "schema_version": "1.0",
        "corner": [u0, v0, w0],
        "origin": [x0, y0, z0],
        "block_size": [dx, dy, dz],
        "shape": [ni, nj, nk],
        "axis_u": [ux, uy, uz],
        "axis_v": [vx, vy, vz],
        "axis_w": [wx, wy, wz],
        "srs": "EPSG:XXXX" or None,
    }

- The payload includes a schema version and may include optional
  ``world_id`` encoding metadata::

    {
        "schema_version": "1.0",
        ...,
        "world_id_encoding": {
            "enabled": true,
            "column": "world_id",
            "frame": "world_xyz",
            "axis_order": ["x", "y", "z"],
            "quantization": {"scale": 10.0},
            "offset": {"x": 500000.0, "y": 7000000.0, "z": 0.0}
        }
    }

  ``world_id_encoding`` is optional and used only when ``world_id``
  columns are present/required.

- This payload is designed to be stored:

  - Directly in ``df.attrs["parq-blockmodel"]`` (as a Python ``dict``),
  - JSON-encoded under a reserved key in Parquet file metadata.


4. Reserved key: ``"parq-blockmodel"``
--------------------------------------

A single, namespaced key is used to avoid collisions with other tools
and to keep related metadata grouped logically:

- **Reserved key**: ``"parq-blockmodel"``

Usage:

- In a :class:`pandas.DataFrame`::

    df.attrs["parq-blockmodel"] = geometry.to_metadata_dict()

- In a Parquet file's key-value metadata:

  - Key: ``"parq-blockmodel"``
  - Value: ``json.dumps(geometry.to_metadata_dict())`` (UTF-8 string)

Future versions may choose to wrap this payload with additional fields,
for example::

    {
        "version": 1,
        "geometry": { ... },
        "extras": { ... },
    }

In that case, the new class methods (see below) will be responsible for
unpacking the appropriate sub-dict before delegating to
``RegularGeometry.from_metadata``.


5. New constructors on ``RegularGeometry``
-----------------------------------------

Two new class methods are introduced to reconstruct a
:class:`~parq_blockmodel.geometry.RegularGeometry` instance from
metadata stored either on a DataFrame or in a Parquet file:

1. ``from_attrs``
~~~~~~~~~~~~~~~~~

Signature::

    @classmethod
    def from_attrs(
        cls,
        attrs: dict,
        key: str = "parq-blockmodel",
    ) -> "RegularGeometry":
        """Reconstruct geometry from a DataFrame.attrs mapping.

        Expects attrs[key] to contain the dict produced by
        RegularGeometry.to_metadata_dict().
        """

Behaviour:

- Looks up ``attrs[key]``.
- Expects this value to be a ``dict`` compatible with
  ``to_metadata_dict``.
- Calls ``RegularGeometry.from_metadata`` on that dict.
- Raises a clear ``KeyError`` if the key is missing.
- Raises ``TypeError`` / ``ValueError`` for incompatible payloads.

This defines a simple round-trip for in-memory objects:

- ``geometry -> df.attrs -> geometry``

without needing to read/write Parquet.

2. ``from_parquet_metadata``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signature::

    @classmethod
    def from_parquet_metadata(
        cls,
        metadata,
        key: str = "parq-blockmodel",
    ) -> "RegularGeometry":
        """Reconstruct geometry from Parquet file metadata.

        Expects key_value_metadata to contain a JSON-encoded dict
        produced by RegularGeometry.to_metadata_dict() under the
        given key.
        """

Behaviour:

- Accepts either:

  - a :class:`pyarrow.parquet.FileMetaData` instance, or
  - a mapping ``dict[str, str]`` of key-value metadata.

- Normalises to a simple ``dict[str, str]``.
- Extracts the value for ``key`` (``"parq-blockmodel"`` by default).
- If missing: raises ``KeyError``.
- If present:

  - If the value is a JSON string, ``json.loads`` is used to decode it
    into a dict.
  - If the value is already a dict, it is used directly.
  - The resulting dict is passed to ``RegularGeometry.from_metadata``.

- Invalid JSON or incompatible dicts raise a clear ``ValueError``.

This provides a single, well-defined way for all consumers to recreate
geometry from a Parquet file that carries compliant metadata.


5. Behaviour of ``ParquetBlockModel``
--------------------------------------

The :class:`~parq_blockmodel.blockmodel.ParquetBlockModel` class is
responsible for reading/writing block model data in Parquet format while
using ``RegularGeometry`` as the authoritative description of the grid.

Dense geometry, sparse vs dense data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``RegularGeometry`` always represents the full dense grid of blocks
  with shape ``(ni, nj, nk)`` in C-order.
- A ParquetBlockModel's **data** may be:

  - dense on disk: every possible block is present as a row, or
  - sparse on disk: only some subset of blocks are present.

- Sparsity is a property of the stored data, not of the geometry:

  - ``geometry`` always knows the full dense grid.
  - ``ParquetBlockModel.read(..., dense=False)`` returns exactly the
    rows stored in the file (sparse view).
  - ``ParquetBlockModel.read(..., dense=True)`` reindexes onto the full
    dense grid implied by ``geometry``, filling missing blocks.

- The ``to_dense_parquet`` method remains the explicit, opt-in way to
  write a physically dense Parquet file from a (possibly) sparse pbm.


Reading geometry
~~~~~~~~~~~~~~~~

When constructing a :class:`ParquetBlockModel` from a ``.pbm.parquet``
file, geometry is resolved as follows (target design):

1. Try to read geometry from Parquet metadata using
   ``RegularGeometry.from_parquet_metadata`` and the ``"parq-blockmodel"``
   key.
2. If the key is absent (for example in a source parquet without geometry metadata), fall back to centroid-based
   inference using the existing ``RegularGeometry.from_parquet`` logic
   (which inspects ``x, y, z`` values and optional rotation angles).
3. If the key is present but invalid (malformed JSON or incompatible
   dict), raise a clear ``ValueError`` rather than silently inferring
   from centroids.

This makes metadata-based reconstruction the preferred path, while
keeping backward compatibility for older xyz-centric files.


Writing geometry
~~~~~~~~~~~~~~~~

When creating a pbm file from an input Parquet or DataFrame, the
following invariants apply:

- The resulting pbm file must have a valid ``RegularGeometry``.
- That geometry is written as metadata under ``"parq-blockmodel"``.
- The file may remain sparse or be densified depending on the specific
  API used (e.g. ``to_dense_parquet``).

Examples:

- ``ParquetBlockModel.from_parquet``:

  - Ingests an arbitrary Parquet file (often sparse, xyz-centric).
  - Infers geometry from centroids if metadata is missing.
  - Writes a new ``.pbm.parquet`` file with geometry metadata attached
    under ``"parq-blockmodel"``.

- ``ParquetBlockModel.from_dataframe``:

  - Accepts a DataFrame with an xyz MultiIndex *or* one that
    already carries geometry metadata in ``df.attrs["parq-blockmodel"]``.
  - If geometry is not provided explicitly, it is inferred from the
    index or attrs.
  - Writes a ``.pbm.parquet`` file and attaches geometry metadata.

- ``ParquetBlockModel.from_geometry``:

  - Creates an empty pbm file (no attributes) from a
    ``RegularGeometry``.
  - The resulting Parquet file includes geometry metadata and may
    store centroids for convenience / interop.


6. Backward compatibility
--------------------------

- Existing pbm files and raw Parquet files that:

  - lack ``"parq-blockmodel"`` metadata, but
  - contain xyz centroid columns (``x``, ``y``, ``z``),

  will continue to be supported via the existing
  ``RegularGeometry.from_parquet`` centroid-based inference.

- New code paths will prefer metadata-based reconstruction when
  available, but will fall back gracefully for old files.

- ``RegularGeometry`` constructor semantics intentionally changed to
  separate local and world frames (breaking change).

- ``WorldFrame`` now uses ``origin`` instead of ``world_origin``
  (breaking rename).

- Existing files without ``origin`` in geometry metadata are still
  readable; ``from_metadata`` falls back to ``origin = (0, 0, 0)``.

- New behaviour is introduced via:

  - the additional class methods on ``RegularGeometry``, and
  - internal changes to how ``ParquetBlockModel`` resolves geometry.


7. Geometry schema versioning
-----------------------------

To allow future evolution of geometry encoding without tying it
rigidly to the top-level package version, geometry metadata can carry
its own schema version (e.g. ``"schema_version": 1``).

- ``to_metadata_dict`` writes this field once introduced.
- ``from_metadata`` reads it and dispatches to version-specific
  decoding logic if necessary.
- Old files without a version field can be treated as version 0 or 1 by
  default.

This keeps geometry I/O stable and makes it possible to introduce
additional fields (e.g. alternative rotation encodings, anisotropic
cells) in a controlled way.


8. Terminology: block_id, world_id, and ijk
--------------------------------------------

To avoid confusion, terminology is used precisely:

**ijk (or i, j, k)**
    - Logical block indices: ``0 ≤ i < nᵢ, 0 ≤ j < nⱼ, 0 ≤ k < nₖ``
    - Canonical indexing scheme for ``parq-blockmodel``
    - **No separate "block_id" column** – ijk is the primary key
    - Can be reconstructed from row index using C-order formula

**block_id** (if used externally)
    - May refer to ijk-based encoding in external systems
    - In ``parq-blockmodel``, we use ijk directly, not a composite "block_id"
    - Row index (flat, C-order) is used internally for DataFrame operations

**world_id**
    - Stable 64-bit positional identifier derived from world (x, y, z) centroids
    - Unique within a CRS and encoding contract
    - Useful for cross-model joins when models share the same SRS
    - Generated by ``encode_world_coordinates`` using quantization + offset
    - **Not a spatial index** – should not be used for range queries

The distinction is important:

- Use **ijk** for logical block queries and geometric operations
- Use **world_id** for stable cross-model references and external joins
- Use row index (flat, C-order) for internal DataFrame/Parquet operations
Summary
=======

Key concepts to remember:

- ``RegularGeometry`` is the public interface; ``LocalGeometry`` + ``WorldFrame``
  are internal components that handle the coordinate transformation pipeline.
- **LocalGeometry** = indexing + local grid (no rotation/CRS)
- **WorldFrame** = world embedding + axis orientation + optional SRS
- **ijk** is canonical indexing (not "block_id"); row index is the flat C-order version
- **world_id** is a stable 64-bit position key, useful for cross-model joins
- Geometry metadata is compact, versioned, and stored in Parquet key-value pairs

Decode ``world_id`` from ``DataFrame.attrs``
--------------------------------------------

Consumers that only have a DataFrame (no ``ParquetBlockModel`` instance)
can still decode ``world_id`` values using the attached metadata payload:

.. code-block:: python

    from parq_blockmodel.utils import get_id_encoding_params, decode_frame_coordinates

    meta = df.attrs["parq-blockmodel"]
    encoding = meta["world_id_encoding"]

    offset, scale = get_id_encoding_params(encoding)
    x, y, z = decode_frame_coordinates(
        df["world_id"].to_numpy(dtype="int64"),
        offset=offset,
        scale=scale,
    )

Cross-model uniqueness depends on shared encoding policy and non-overlapping
centroid positions within the same SRS.

- ``RegularGeometry`` is the single, dense description of block model
  geometry using C-order and ijk as canonical indexing.
- Geometry metadata is serialised by ``to_metadata_dict`` and
  deserialised by ``from_metadata``.
- Two new helpers, ``from_attrs`` and ``from_parquet_metadata``, provide
  a consistent way to rebuild geometry from DataFrame attrs and Parquet
  file metadata respectively, using a reserved key
  ``"parq-blockmodel"``.
- ``ParquetBlockModel`` always uses a dense geometry, while allowing the
  underlying data to be sparse or dense on disk; density is controlled
  by read/write options such as ``dense=True`` and
  ``to_dense_parquet``.
- Existing xyz-centric files remain supported through centroid-based
  geometry inference, with metadata-based reconstruction becoming the
  preferred path for new, canonical pbm files.

