Quick Start Guide
=================

This page will describe the basic steps to use the package.

First we define the imports...

..  code-block:: python

    from parq_blockmodel import ParquetBlockModel

Create an instance of the block model, which will read the specified parquet file.

..  code-block:: python

    pbm: ParquetBlockModel = ParquetBlockModel.from_parquet("path/to/your/parquet_file.parquet")

If the `viz` extra is installed, you can visualize the block model using the `plot` method:

..  code-block:: python

    import pyvista as pv

    p: pv.Plotter = pbm.plot()
    p.show()


For examples that demonstrate a range of use cases, see the :doc:`/auto_examples/index`.
