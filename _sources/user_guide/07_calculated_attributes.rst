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

Custom lookups and functions
=============================

When your schema uses df-eval **functions** that require registered resolvers
(e.g., lookup functions with DictResolver), you need to register them with the 
Engine before evaluating. Two APIs are provided:

**Option 1: Constructor-time registration** (eager)
   Register resolvers when creating the PBM.

**Option 2: Post-load registration** (lazy)
   Register resolvers after loading the PBM, via ``configure_engine()``.

Example: DictResolver-based lookup in calculated column
---------------------------------------------------------

Suppose you have a block model with rock codes (1, 2, 3) and want to look up
the rock type names using a DictResolver:

.. code-block:: python

    import pandas as pd
    from pathlib import Path
    from pandera import Column, DataFrameSchema
    from df_eval import DictResolver

    from parq_blockmodel import ParquetBlockModel

    # Create a DictResolver for rock type lookup
    rock_type_resolver = DictResolver({
        1: "Granite",
        2: "Diorite",
        3: "Gabbro",
    }, default="Unknown")

    # Define schema with lookup operation using the resolver
    schema = DataFrameSchema(
        columns={
            "rock_code": Column(int, nullable=False),
            "rock_name": Column(
                str,
                required=False,
                metadata={"df-eval": {
                    "lookup": {
                        "resolver": "rock_type",
                        "key": "rock_code",
                        "on_missing": "default"
                    }
                }},
            ),
        },
        strict=False,
    )

**Approach 1: Register at construction**

.. code-block:: python

    def setup_engine(engine):
        engine.register_resolver("rock_type", rock_type_resolver)
        return engine

    pbm = ParquetBlockModel.from_parquet(
        Path("model.pbm"),
        schema=schema,
        engine_initializer=setup_engine,  # ← Register here
    )

    result = pbm.read(columns=["rock_code", "rock_name"], dense=True)

**Approach 2: Register after loading**

.. code-block:: python

    pbm = ParquetBlockModel.from_parquet(
        Path("model.pbm"),
        schema=schema,
    )

    # Configure engine before reading calculated columns
    pbm.configure_engine(setup_engine)

    result = pbm.read(columns=["rock_code", "rock_name"], dense=True)

Combining lookups with expressions
-----------------------------------

You can combine lookup functions with other calculated columns. For example:

.. code-block:: python

    density_class_resolver = DictResolver({
        2.0: "Low",
        2.5: "Medium",
        3.0: "High",
    }, default="Unknown")

    schema = DataFrameSchema(
        columns={
            "density": Column(float, nullable=False),
            "rock_code": Column(int, nullable=False),
            # Calculated column 1: simple expression
            "tonnes": Column(
                float,
                required=False,
                metadata={"df-eval": {"expr": "density * volume"}},
            ),
     # Calculated column 2: lookup using rock_code
            "rock_name": Column(
               str,
               required=False,
               metadata={"df-eval": {
                   "lookup": {
                       "resolver": "rock_type",
                       "key": "rock_code",
                       "on_missing": "default"
                   }
               }},
            ),
            # Calculated column 3: lookup using density
            "density_class": Column(
               str,
               required=False,
               metadata={"df-eval": {
                   "lookup": {
                       "resolver": "density_class",
                       "key": "density",
                       "on_missing": "default"
                   }
               }},
            ),
        },
        strict=False,
    )

    def setup_engine(engine):
        engine.register_resolver("rock_type", rock_type_resolver)
        engine.register_resolver("density_class", density_class_resolver)
        return engine

    pbm = ParquetBlockModel.from_dataframe(
        df[["density", "rock_code"]],
        filename=Path("model.parquet"),
        schema=schema,
        engine_initializer=setup_engine,
    )

    # Read multiple calculated columns at once
    result = pbm.read(
        columns=["tonnes", "rock_name", "density_class"],
        index="ijk",
        dense=True,
    )

See `the custom lookup example <../auto_examples/12_calculated_with_custom_lookups.html>`_ for a complete working demonstration.

For further details on custom resolvers and functions, refer to the `df-eval documentation <https://elphick.github.io/df-eval/index.html>`_.