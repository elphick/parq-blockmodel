Installation
============

.. toctree::
   :maxdepth: 2
   :caption: Installation:


The package is pip installable.

..  code-block::

    pip install parq-blockmodel

Common optional extras can be installed from PyPI as needed.

.. code-block::

    pip install "parq-blockmodel[profiling,progress,viz]"

To enable Pandera-backed schema validation and YAML schema loading support,
install the ``schema`` extra:

.. code-block::

    pip install "parq-blockmodel[schema]"

The ``schema`` extra installs:

* ``pandera`` for defining and applying ``DataFrameSchema`` objects, and
* ``df-eval`` for loading schema definitions from YAML files via
  ``df_eval.utils.pandera_io_compat``.

For editable local development, install from a clone instead:

.. code-block::

    pip install -e ".[profiling,progress,viz,schema]"

Or, if poetry is more your flavour.

..  code-block::

    poetry add parq-blockmodel

or with extras...

..  code-block::

    poetry add "parq-blockmodel[profiling,progress,viz,schema]"
