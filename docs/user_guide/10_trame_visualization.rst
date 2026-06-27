Trame visualization
===================

The Trame viewer provides a read-only interactive view for
:class:`parq_blockmodel.blockmodel.ParquetBlockModel`.

Supported workflows
-------------------

The viewer supports both:

* single-PBM launch from a specific file path,
* hive-style drill-down selection using ``key=value`` directories plus PBM name.

In both modes you can:

* choose an attribute,
* adjust a threshold slider,
* refresh the rendered scene without mutating source PBM files.

Data source loader
------------------

The left drawer now starts with a collapsible **Data source** panel containing
a server path field and a **Load** button.

* If the path is a ``.pbm`` file, the app switches to single-model mode.
* If the path is a directory, the app treats it as a hive root and enables the
  hive selector panel.

Hive-style selector
-------------------

When launched from a hive root, the left drawer shows an **Asset selector**
panel. The selector exposes:

* one dropdown per hive level key (in discovered order),
* a ``PBM name`` dropdown (``pbm.name`` / file stem),
* the resolved PBM file path.

Changing selection loads the matching PBM in-place in the same app session.

API
---

Use :meth:`parq_blockmodel.visualization.trame_app.BlockModelTrameApp.from_source_path`
to launch from either a PBM file or a hive directory:

.. code-block:: python

   from parq_blockmodel.visualization import BlockModelTrameApp

   app = BlockModelTrameApp.from_source_path("path/to/source")
   app.launch()

Examples
--------

One combined example is provided:

* ``examples/15_trame_threshold_viewer.py``: combined file/hive demo. The
  ``DEMO_SOURCE_KIND`` constant in the script switches between:
  * file startup from a single ``.pbm`` path,
  * hive startup from a directory path with selector drill-down.

For interactive examples included in Sphinx-Gallery, gate ``app.launch()`` with
``pyvista.BUILDING_GALLERY`` so docs builds do not start a live Trame server.
