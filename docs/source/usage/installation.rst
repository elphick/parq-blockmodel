Installation
============

.. toctree::
   :maxdepth: 2
   :caption: Installation:


The package is pip installable.

..  code-block::

    pip install mass-composition

If you want the extras (for visualisation and networks of objects) you'll install like this with pip.

.. code-block::

    pip install mass-composition -e .[viz,network]

Or, if poetry is more your flavour.

..  code-block::

    poetry add mass-composition

or with extras...

..  code-block::

    poetry add "mass-composition[viz,network]"
