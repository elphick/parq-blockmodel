Trame visualization
===================

The Trame visualization path provides a read-only viewer for a loaded
:class:`parq_blockmodel.blockmodel.ParquetBlockModel`.

Phase 1 focuses on a single PBM at a time:

* load a PBM from a file path,
* choose one attribute,
* adjust a threshold slider,
* and refresh the scene without mutating the source model.

The implementation is split into a reusable plotting module and a small Trame
session wrapper:

* :mod:`parq_blockmodel.visualization.blockmodel_plot`
* :mod:`parq_blockmodel.visualization.trame_app`

Example
-------

The example script ``examples/15_trame_threshold_viewer.py`` shows the
standalone entry point. If the sample PBM is missing, the script creates a toy
model in the system temporary directory and logs a warning before launching the
viewer.

Gallery-safe pattern
--------------------

For interactive examples that should appear in Sphinx-Gallery but not start a
live server during docs builds, keep the example executable and gate the final
``app.launch()`` call on ``pyvista.BUILDING_GALLERY`` (or a similar docs-only
flag). Pair that with ``# sphinx_gallery_thumbnail_path = ...`` so the gallery
still gets a stable thumbnail.

Future work
-----------

Hive-directory browsing and multi-PBM filtering are intentionally deferred to a
later phase.
