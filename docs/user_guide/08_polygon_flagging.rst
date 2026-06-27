Polygon Flagging (2D XY)
========================

Use :class:`parq_blockmodel.polygon_field.PolygonField` to classify blocks by
whether their centroids fall inside polygon regions in the XY plane.

Key points
----------

* Evaluation is centroid-based and strictly 2D (XY only; Z ignored).
* Boundary points are classified as inside.
* Geometry is validated on construction (invalid/empty polygons raise).
* ``PolygonField`` supports optional ``name`` metadata for downstream labeling.

Create from Shapely geometry
----------------------------

.. code-block:: python

    import pandas as pd
    from shapely.geometry import Polygon
    from parq_blockmodel import PolygonField

    polygon = Polygon(
        [(3.0, 2.0), (15.0, 2.5), (17.0, 8.0), (13.0, 13.5), (6.0, 12.0), (2.5, 7.0)]
    )
    field = PolygonField.from_shapely(polygon, name="lease_a")

Load named polygons from GeoParquet
-----------------------------------

Persisted named polygon layers can be stored in GeoParquet and ingested by name:

.. code-block:: python

    from parq_blockmodel import PolygonField

    field = PolygonField.from_geoparquet(
        "lease_regions.parquet",
        name="lease_a",
        name_column="name",
    )

The GeoParquet file should contain:

* a geometry column with Polygon/MultiPolygon features, and
* a name column with unique polygon identifiers.

Apply to a block model
----------------------

.. code-block:: python

    # convenience method: evaluates and writes the column in one step
    mask = pbm.flag_polygon(field, column="lease_domain")

    # equivalent low-level form:
    # mask = field.evaluate(pbm.geometry)
    # ... map mask by block_id ...
    # pbm.write(..., merge=True)

Notes on persisted domain values
--------------------------------

Recommended persisted pattern for class flags:

* inside class: string label (for example ``"lease_a"``)
* outside/unassigned: ``pd.NA``

This keeps the stored categorical semantically clean and lets rendering layers
decide how missing values are displayed.

See also
--------

* Gallery example: :doc:`/auto_examples/13_polygon_flagging`
* Next: :doc:`/user_guide/09_surface_flagging`
* API: :class:`parq_blockmodel.polygon_field.PolygonField`
