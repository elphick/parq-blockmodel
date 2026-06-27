Solid Flagging (3D Volumes)
===========================

Use :class:`parq_blockmodel.solid.MeshSolid` to classify blocks by whether
their centroids are inside a closed 3D mesh volume.

Key points
----------

* Evaluation is centroid-based in full 3D (X, Y, Z).
* Boundary points are classified as inside.
* v1 supports triangulated, watertight PLY meshes.
* Invalid meshes fail fast (no automatic mesh repair).

Create from PyVista or PLY mesh
-------------------------------

.. code-block:: python

    import pyvista as pv
    from parq_blockmodel import MeshSolid

    surface = pv.Sphere(radius=10.0).triangulate()
    solid = MeshSolid.from_pyvista(surface, name="ore_shell")

    # or load from persisted PLY
    solid = MeshSolid.from_ply("ore_shell.ply", name="ore_shell")

Load from DXF (named object selection)
--------------------------------------

DXF support requires the optional CAD dependency:

.. code-block:: bash

    pip install "parq-blockmodel[cad]"

Then load a solid by named layer or block reference:

.. code-block:: python

    from parq_blockmodel import MeshSolid

    solid = MeshSolid.from_dxf("solids.dxf", name="ore_shell", by="layer")
    # or: by="block"

If ``name`` is omitted and multiple solid candidates exist, ``from_dxf`` raises
and lists the available names so selection stays explicit.

Apply to a block model
----------------------

.. code-block:: python

    # convenience method: evaluates and writes the column in one step
    mask = pbm.flag_solid(solid, column="ore")

    # equivalent low-level form:
    # mask = solid.evaluate(pbm.geometry)
    # ... map mask by block_id ...
    # pbm.write(..., merge=True)

Notes
-----

Current v1 scope excludes:

* partial/sub-block volume fractions,
* multiple solids with precedence/stacking rules,
* mesh repair workflows, and
* distance outputs or advanced metrics.

See also
--------

* Gallery example: :doc:`/auto_examples/14_solid_flagging`
* :class:`parq_blockmodel.solid.MeshSolid`
* :class:`parq_blockmodel.polygon_field.PolygonField`
