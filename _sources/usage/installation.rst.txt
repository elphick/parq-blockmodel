Installation
============

.. toctree::
   :maxdepth: 2
   :caption: Installation:


The package is pip installable.

..  code-block::

    pip install parq-blockmodel

If you want the extras (for visualisation and networks of objects) you'll install like this with pip.

.. code-block::

    pip install parq-blockmodel -e .[profiling,progress,viz]

Or, if poetry is more your flavour.

..  code-block::

    poetry add parq-blockmodel

or with extras...

..  code-block::

    poetry add "parq-blockmodel[profiling,progress,viz]"
