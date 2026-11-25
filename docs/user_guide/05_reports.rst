Profiling Reports
=================

``ParquetBlockModel.create_report`` builds a profiling report for a block model
using the optional ``profiling`` extra.

Install profiling support
-------------------------

Install the extra dependencies before creating reports:

.. code-block:: bash

    pip install "parq-blockmodel[profiling]"

Create and save a report
------------------------

``create_report`` now returns a :class:`parq_blockmodel.reporting.BlockModelReport`
object. By default, the HTML report is written beside the backing ``.pbm`` file
with the same stem and an ``.html`` suffix.

.. code-block:: python

    from pathlib import Path

    from parq_blockmodel import ParquetBlockModel

    pbm = ParquetBlockModel.from_parquet(Path("orebody.parquet"))

    report = pbm.create_report()
    print(report.output_path)

The returned report object exposes the rendered HTML and can be written again:

.. code-block:: python

    html = report.to_html()
    report.save(Path("reports/orebody_profile.html"))

Include calculated columns
--------------------------

By default, reports use the persisted columns in the block model. To include
available calculated columns as well, opt in:

.. code-block:: python

    report = pbm.create_report(include_calculated=True)

With this option and no explicit ``columns`` list, the report includes
persisted and calculated columns, including the built-in ``volume``, but
excludes positional and identity columns (``block_id``, ``world_id``, ``i``,
``j``, ``k``, ``x``, ``y``, and ``z``). Pass ``include_special=True`` to
include those columns too. An explicit ``columns`` list controls the selection
and can include special columns regardless of ``include_special``:

.. code-block:: python

    report = pbm.create_report(
        columns=["block_id", "volume"],
        include_calculated=True,
    )

Descriptions of materialised calculated columns are prefixed with ``(𝑓𝑥)``.
Persisted columns are not marked, even if a calculation with the same name is
defined. Schema expressions, lookups, and functions use the model's configured
engine, just as they do for ``read()``.

.. warning::

   When calculated columns are needed, the model is loaded into memory and the
   selected columns are written to a temporary Parquet file for profiling.
   The temporary file is removed after report creation, including on failure.
   ``memory_budget`` affects profiling batches, not this full-memory
   materialisation. When no calculated columns are needed, the report uses the
   backing file directly.

Run the full report without column chunking
-------------------------------------------

To process all selected columns in a single pass, set
``columns_per_batch=None``:

.. code-block:: python

    report = pbm.create_report(columns_per_batch=None)

This is the clearest option when you explicitly want the full report without
column chunking.

Automatic chunking from a memory budget
---------------------------------------

For larger models, you can ask ``create_report`` to infer a suitable column
batch size from Parquet metadata:

.. code-block:: python

    report = pbm.create_report(memory_budget="512MB")
    print(report.columns_per_batch)

``memory_budget`` accepts:

* an integer byte count,
* a floating-point byte count, or
* strings such as ``"512MB"``, ``"1GiB"``, or ``"250_000_000"``.

If the estimated size of the requested columns fits within the budget,
``parq-blockmodel`` disables column chunking automatically and profiles the full
selection in one pass.

.. note::

   The memory-budget calculation is heuristic. It uses Parquet column-chunk
   metadata, which is a useful guide, but it cannot perfectly predict the final
   in-memory expansion during profiling, especially for compressed, string-heavy,
   or categorical datasets.

Control when the HTML file is written
-------------------------------------

If you want the report object first and prefer to save later, disable the
initial write:

.. code-block:: python

    report = pbm.create_report(write_html=False)

    # decide on a destination later
    report.save(Path("reports/orebody_profile.html"))

Choose a custom output path
---------------------------

Use ``report_path`` to control where the initial HTML file is written:

.. code-block:: python

    report = pbm.create_report(
        report_path=Path("reports/orebody_profile.html"),
    )

Open the report in a browser
----------------------------

To open the generated report immediately after profiling:

.. code-block:: python

    report = pbm.create_report(open_in_browser=True)

Schema metadata in profile reports
----------------------------------

When a :class:`parq_blockmodel.blockmodel.ParquetBlockModel` has an attached
Pandera schema, ``create_report()`` uses schema metadata to enrich report
metadata:

* Per selected column, report variable descriptions are built by concatenating
  schema column ``title`` and ``description`` values (when present).
* At dataset level, report metadata includes ``dataset["description"]`` as a
  dict-like string containing ``DataFrameSchema`` object properties
  ``name``, ``title``, and ``description``, with null or missing values
  removed.

Column descriptions use ``Column`` object properties (``title`` and
``description``), not ``Column.metadata`` keys.

.. code-block:: python

    from pandera import Column, DataFrameSchema
    from parq_blockmodel import ParquetBlockModel

    schema = DataFrameSchema(
        columns={
            "depth": Column(float)
        },
        strict=False,
    )
    schema.columns["depth"].title = "Depth"
    schema.columns["depth"].description = "Vertical distance"
    schema.name = "example_model"
    schema.title = "Example Block Model"
    schema.description = "Demonstration dataset"

    pbm = ParquetBlockModel.from_parquet("example.parquet", schema=schema)
    report = pbm.create_report()

Summary of the reporting API
----------------------------

``ParquetBlockModel.create_report`` accepts these main arguments:

* ``columns``: restrict profiling to selected columns,
* ``include_calculated``: include available calculated columns (default ``False``),
* ``include_special``: include positional and identity columns in the automatic
  selection when ``include_calculated=True`` (default ``False``),
* ``columns_per_batch``: explicit column chunk size, or ``None`` for no chunking,
* ``memory_budget``: derive a chunk size automatically,
* ``report_path``: choose where the initial HTML file is written,
* ``write_html``: control whether the initial HTML file is written immediately,
* ``open_in_browser``: display the report after generation.

See also: :doc:`03_blockmodels` and the :doc:`/auto_examples/index` gallery
(including ``18_profile_report_with_schema_descriptions.py``).
