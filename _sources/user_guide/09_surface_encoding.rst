Surface Encoding (2.5D Elevation)
=================================

Use :class:`parq_blockmodel.surface.RasterSurface` and
:class:`parq_blockmodel.surface.MeshSurface` to evaluate single-valued
elevation surfaces against a regular block model grid.

Key points
----------

* Evaluation is geometry-first and vectorised against block centroids.
* Output is a fraction in ``[0, 1]`` representing how much of each block lies below the surface.
* Raster inputs must provide a regular X/Y grid with finite values.
* Mesh inputs must be valid 2.5D surfaces with one Z value per ``(x, y)`` location.
* Invalid inputs fail fast; v1 does not repair or infer missing geometry.

Create from raster or mesh
--------------------------

.. code-block:: python

    import numpy as np
    from parq_blockmodel import MeshSurface, RasterSurface

    raster = RasterSurface.from_arrays(
        x_coords=np.array([0.0, 1.0, 2.0]),
        y_coords=np.array([0.0, 1.0, 2.0]),
        values=np.full((3, 3), 1.5),
        name="topography",
    )

    mesh = MeshSurface.from_ply("horizon.ply", name="horizon_a")

Evaluate against a block model
------------------------------

.. code-block:: python

    # convenience method: evaluates and writes the column in one step
    values = pbm.evaluate_surface(raster, column="topo_fraction")

    # equivalent low-level form:
    # values = raster.evaluate(pbm.geometry)
    # ... map values by block_id ...
    # pbm.write(..., merge=True)

Notes
-----

Current v1 scope excludes:

* multi-surface stacking or precedence rules,
* overhangs, vertical faces, or other multi-valued geometry,
* sub-block sampling or high-resolution integration, and
* CRS handling or reprojection workflows.

See also
--------

* Gallery example: :doc:`/auto_examples/14_surface_encoding`
* Previous: :doc:`/user_guide/08_polygon_flagging`
* Next: :doc:`/user_guide/10_solid_flagging`
* :class:`parq_blockmodel.surface.RasterSurface`
* :class:`parq_blockmodel.surface.MeshSurface`
