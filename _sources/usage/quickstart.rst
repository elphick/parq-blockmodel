Quick Start Guide
=================

This page will describe the basic steps to use the package.

First we define the imports...

..  code-block:: python

    from pathlib import Path

    from parq_blockmodel import ParquetBlockModel

Create an instance of the block model, which will read the specified parquet file.

..  code-block:: python

    pbm: ParquetBlockModel = ParquetBlockModel.from_parquet(
        Path("path/to/your/parquet_file.parquet")
    )

If you have installed ``parq-blockmodel[schema]``, you can attach a Pandera
schema while loading the model and then validate the canonical ``.pbm`` file in
chunks:

..  code-block:: python

    pbm = ParquetBlockModel.from_parquet(
        Path("path/to/your/parquet_file.parquet"),
        schema=Path("schemas/blockmodel.schema.yaml"),
    )

    pbm.validate()
    pbm.validate(sample_chunks=1)  # quick spot-check on large models

If the `viz` extra is installed, you can visualize the block model using the `plot` method:

..  code-block:: python

    import pyvista as pv

    p: pv.Plotter = pbm.plot(z_up_lock=True, z_up_hotkey="z")
    p.show()

To use the Trame viewer from ``pbm.plot(...)``:

..  code-block:: python

    from parq_blockmodel.visualization import TrameBlockModelPlotEngine

    app = pbm.plot(
        scalar=pbm.available_attributes[0],
        engine=TrameBlockModelPlotEngine(),
        z_up_lock=True,
        z_up_hotkey="z",
    )
    app.launch(port=8080)

You can also construct the app directly:

..  code-block:: python

    from parq_blockmodel.visualization import BlockModelTrameApp

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], z_up_lock=True, z_up_hotkey="z")
    app.launch(port=8080)

With ``z_up_lock=True``, hold ``z`` for turntable-style orbit (yaw/pitch, no roll) with camera up aligned to +Z.


For examples that demonstrate a range of use cases, see the :doc:`/auto_examples/index`.
