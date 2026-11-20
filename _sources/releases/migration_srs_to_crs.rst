Migration: srs -> crs
=======================

Overview
--------

parq-blockmodel replaces the legacy metadata string field ``srs`` with a structured
``crs`` representation to improve portability and unambiguous identification of coordinate
reference systems. The change is intentionally backward-compatible: the serializer writes
both a structured ``crs`` object and the legacy ``srs`` string when possible, and
WorldFrame exposes a ``srs`` property for callers that still expect the string.

What changed
------------

- New type: parq_blockmodel.crs.CRSDef(authority, code, name=None, wkt=None). Identity = (authority, code).
- RegularGeometry and WorldFrame now accept the ``crs`` keyword (string like ``"EPSG:4326"``,
  a dict ``{"authority":...,"code":...}`` or a CRSDef instance).
- The old ``srs`` constructor kwarg is removed (breaking). Most callers should pass ``crs`` instead.

Migration steps
---------------

- For simple cases (EPSG): replace keyword ``srs="EPSG:4326"`` with ``crs="EPSG:4326"``.

  Example:

  .. code-block:: python

      geom = RegularGeometry.create(crs="EPSG:4326")

- For local/unknown CRS where full WKT is needed, construct a CRSDef with wkt:

  .. code-block:: python

      from parq_blockmodel.crs import CRSDef
      local_crs = CRSDef("LOCAL", "site-123", name="Local site", wkt=my_wkt_text)
      geom = RegularGeometry(world=WorldFrame(crs=local_crs))

Notes
-----

- Parquet metadata written by the library will include both the structured ``crs`` payload
  and the legacy ``srs`` string (when available) to help older consumers remain compatible.
- ParquetBlockModel exposes convenience helpers: ``bm.crs`` (CRSDef), ``bm.crs_key`` (authority,code)
  and ``bm.pyproj_crs`` (pyproj.CRS instance) to simplify integrations.

If you maintain downstream code that relied on the ``srs`` constructor keyword, update it to use ``crs``.
