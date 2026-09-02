Trame visualization
===================

The Trame viewer provides a read-only interactive view for
:class:`parq_blockmodel.blockmodel.ParquetBlockModel`.

Install the server visualization extra first:

.. code-block::

   pip install "parq-blockmodel[server-viz]"

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

Navigation and hotkeys
----------------------

The Trame view uses VTK-style camera interactions. Common controls are:

* **Left drag**: rotate/orbit camera
* **Shift + left drag**: translate (pan)
* **Middle drag**: translate (pan)
* **Right drag**: dolly/zoom
* **Mouse wheel**: zoom

When ``z_up_lock=True`` is enabled, hold ``z`` while rotating to keep +Z
vertical on screen (turntable-style orbit with no roll).

Notes:

* In browser-based Trame sessions, click inside the 3D view first so keyboard
  events are captured by the app.
* Exact interaction feel can vary slightly by platform/browser.

API
---

Use explicit startup entry points:

.. code-block:: python

   from parq_blockmodel.visualization import BlockModelTrameApp

   file_app = BlockModelTrameApp.from_pbm_file("path/to/model.pbm")
   hive_app = BlockModelTrameApp.from_hive_directory("path/to/hive_root")

   # Scaffold-only API for a future query-backed startup workflow.
   query_app = BlockModelTrameApp.from_duckdb_query(
       "SELECT * FROM read_parquet('path/to/hive/**/*.parquet')"
   )

   file_app.launch()

Examples
--------

One combined example is provided:

* ``examples/16_trame_threshold_viewer.py``: combined file/hive demo. The
  ``DEMO_SOURCE_KIND`` constant in the script switches between:
  * file startup from a single ``.pbm`` path,
  * hive startup from a directory path with selector drill-down.

For interactive examples included in Sphinx-Gallery, gate ``app.launch()`` with
``pyvista.BUILDING_GALLERY`` so docs builds do not start a live Trame server.
