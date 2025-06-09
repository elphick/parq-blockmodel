Quick Start Guide
=================

This page will describe the basic steps to use the package.

.. note::

   This is not an example, but simple static code blocks.


You'll probably first demonstrate imports...

..  code-block:: python

    import xarray as xr
    from elphick.mc.mass_composition import MassComposition

You may describe some prerequisites or requirements, and then the key command to instantiate.

..  code-block:: python

    obj_mc: MassComposition = MassComposition(df_data)

And then demonstrate a common method.

..  code-block:: python

    obj_mc.aggregate()


For examples that demonstrate a range of use cases, see the :doc:`/auto_examples/index`.
