Calculated attributes
=====================

Calculated attributes are schema-defined values derived from other block-model
attributes. In ``parq-blockmodel`` they are expressed as calculated columns in
Pandera metadata and evaluated with ``df-eval`` when you ask for a materialized
read.

For regular models, ``volume`` is also available as a built-in calculated column
derived from geometry (``pbm.geometry.block_volume``), even when it is not
persisted in parquet.

Install the optional schema dependencies first:

.. code-block:: bash

   uv add parq-blockmodel[schema]

The example below derives ``tonnes`` from ``density * volume`` and then derives
``contained_metal`` from ``tonnes * grade``.

.. code-block:: python

   from pandera import Column, DataFrameSchema

   from parq_blockmodel import ParquetBlockModel
   from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel

   df = create_demo_blockmodel(
       shape=(2, 2, 2),
       block_size=(10.0, 10.0, 5.0),
       corner=(0.0, 0.0, 0.0),
       index_type="world_centroids",
   )
   df["density"] = 2.4 + 0.05 * df["depth"]
   df["grade"] = 0.1 + 0.01 * df["depth"]

   schema = DataFrameSchema(
       columns={
           "density": Column(float, coerce=True, nullable=True),
           "grade": Column(float, coerce=True, nullable=True),
           "tonnes": Column(
               float,
               coerce=True,
               nullable=True,
               required=False,
               metadata={"df-eval": {"expr": "density * volume"}},
           ),
           "contained_metal": Column(
               float,
               coerce=True,
               nullable=True,
               required=False,
               metadata={"df-eval": {"expr": "tonnes * grade"}},
           ),
       },
       strict=False,
   )

   pbm = ParquetBlockModel.from_dataframe(
       df[["density", "grade"]],
       filename=Path("orebody.parquet"),
       schema=schema,
   )

   result = pbm.read(index="ijk", dense=True, include_calculated=True)

``ParquetBlockModel.read()`` stays raw by default. Pass
``include_calculated=True`` when you want the schema-derived columns included in
the returned DataFrame.

Schema-calculated columns can also be consumed by reblocking. See
:doc:`06_reblocking` for weighted downsampling examples where ``basis`` and/or
aggregated targets are materialized from schema metadata before aggregation.
