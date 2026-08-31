Quick Start Guide
=================

This page describes the three-step workflow designed into parq-blockmodel:

1. **Validate** — Check data against a schema
2. **Review** — Profile data quality with an HTML report
3. **View** — Visualize the model interactively

For installation prerequisites, see :doc:`./installation`.

Base Setup
----------

First we import and load a block model:

..  code-block:: python

    from pathlib import Path
    from parq_blockmodel import ParquetBlockModel

    pbm: ParquetBlockModel = ParquetBlockModel.from_parquet(
        Path("path/to/your/parquet_file.parquet")
    )

1) Validate
-----------

**Prerequisite:** ``pip install "parq-blockmodel[schema]"``

Attach a Pandera schema while loading and validate the model in chunks:

..  code-block:: python

    pbm = ParquetBlockModel.from_parquet(
        Path("path/to/your/parquet_file.parquet"),
        schema=Path("schemas/blockmodel.schema.yaml"),
    )

    pbm.validate()
    pbm.validate(sample_chunks=1)  # quick spot-check on large models

See :doc:`../user_guide/03_blockmodels` for detailed schema usage.

2) Review (profile)
-------------------

**Prerequisite:** ``pip install "parq-blockmodel[profiling]"``

Generate an interactive HTML profile report:

..  code-block:: python

    report = pbm.create_report(columns_per_batch=None)
    print(report.output_path)

When a schema includes column ``title`` and ``description`` values, those values
are shown in the report.

See :doc:`../user_guide/05_reports` for advanced profiling options.

3) View
-------

**Prerequisite:** ``pip install "parq-blockmodel[viz]"``

Visualize the block model with PyVista or Trame.

**PyVista (default):**

..  code-block:: python

    p = pbm.plot(z_up_lock=True, z_up_hotkey="z")
    p.show()

With ``z_up_lock=True``, press ``z`` to enter turntable-style orbit (yaw/pitch, no roll).

**Trame (web-based interactive):**

Requires ``pip install "parq-blockmodel[server-viz]"`` in addition to the local visualization extra.

..  code-block:: python

    from parq_blockmodel.visualization import TrameBlockModelPlotEngine

    app = pbm.plot(
        scalar=pbm.available_attributes[0],
        engine=TrameBlockModelPlotEngine(),
        z_up_lock=True,
        z_up_hotkey="z",
    )
    app.launch(port=8080)

Or construct the app directly:

..  code-block:: python

    from parq_blockmodel.visualization import BlockModelTrameApp

    app = BlockModelTrameApp(
        pbm,
        scalar=pbm.available_attributes[0],
        z_up_lock=True,
        z_up_hotkey="z"
    )
    app.launch(port=8080)

See :doc:`../user_guide/11_trame_visualization` for full documentation.

Next Steps
----------

For more examples and workflows, see the :doc:`/auto_examples/index`.
