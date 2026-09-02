Installation
============

The package is pip installable from PyPI.

Base Install
------------

For data I/O and basic operations:

.. code-block::

    pip install parq-blockmodel

3-Step Workflow (Full Install)
------------------------------

For the complete experience—validate, profile, visualize, and enable GIS/server workflows:

.. code-block::

    pip install "parq-blockmodel[all]"

Install by Workflow Step
------------------------

**Step 1: Validate** — Schema validation with Pandera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[schema]"

Enables:

* Pandera ``DataFrameSchema`` definitions and validation
* YAML schema loading via ``df-eval``
* See :doc:`../user_guide/03_blockmodels` for usage

**Step 2: Review** — Profiling reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[profiling]"

Enables:

* HTML profile report generation with ``ydata-profiling``
* Batch-wise profiling for large block models
* See :doc:`../user_guide/05_reports` for usage

**Step 3: View** — Local visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[viz]"

Enables:

* 3D visualization with PyVista
* Plotly-based heatmaps
* Block model rendering and interactive exploration
* Terrain context with rasters
* See :doc:`../usage/quickstart` for local plotting usage

**Optional: Spatial / GIS**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[spatial]"

Enables:

* GeoDataFrame / GeoParquet workflows with GeoPandas
* Polygon-based spatial workflows backed by Shapely
* GIS-facing exports such as categorical footprint mapping

**Optional: Server visualization**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[server-viz]"

Enables:

* Trame-based browser/server viewer
* Web app wrappers built on top of the PyVista visualization path
* See :doc:`../user_guide/11_trame_visualization` for usage

For Editable Development
------------------------

Install from a local clone with all development and optional dependencies:

.. code-block::

    pip install -e ".[all]"

Troubleshooting: ImportError for Optional Features
---------------------------------------------------

If you see an error like ``ModuleNotFoundError: No module named 'pyvista'`` when trying to use local visualization,
you need to install the visualization extra:

.. code-block::

    pip install "parq-blockmodel[viz]"

Similarly:

* For schema validation errors: ``pip install "parq-blockmodel[schema]"``
* For profiling errors: ``pip install "parq-blockmodel[profiling]"``
* For GIS / GeoDataFrame features: ``pip install "parq-blockmodel[spatial]"``
* For Trame server viewer features: ``pip install "parq-blockmodel[server-viz]"``

Or install everything at once:

.. code-block::

    pip install "parq-blockmodel[all]"
