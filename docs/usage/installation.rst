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

For the complete experience—validate, profile, and visualize:

.. code-block::

    pip install "parq-blockmodel[schema,profiling,viz]"

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

**Step 3: View** — Interactive visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pip install "parq-blockmodel[viz]"

Enables:

* 3D visualization with PyVista and Trame
* Block model rendering and interactive exploration
* Terrain context with rasters
* See :doc:`../user_guide/11_trame_visualization` for usage

For Editable Development
------------------------

Install from a local clone with all development and optional dependencies:

.. code-block::

    pip install -e ".[schema,profiling,viz]"

Troubleshooting: ImportError for Optional Features
---------------------------------------------------

If you see an error like ``ModuleNotFoundError: No module named 'pyvista'`` when trying to use visualization, 
you need to install the visualization extra:

.. code-block::

    pip install "parq-blockmodel[viz]"

Similarly:

* For schema validation errors: ``pip install "parq-blockmodel[schema]"``
* For profiling errors: ``pip install "parq-blockmodel[profiling]"``

Or install all three at once:

.. code-block::

    pip install "parq-blockmodel[schema,profiling,viz]"
