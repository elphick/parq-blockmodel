"""
Calculated Attributes
=====================

Schema-backed calculated attributes keep derived values close to the block
model schema. This example derives ``tonnes`` from ``density * volume`` and
then derives ``contained_metal`` from ``tonnes * grade``.

The ``volume`` is available as a column from the block model geometry.
"""

import tempfile
from pathlib import Path

import pandas as pd

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel

# %%
# Create a small block model with base attributes.

temp_dir = Path(tempfile.gettempdir()) / "calculated_attributes_example"
temp_dir.mkdir(parents=True, exist_ok=True)

df = create_demo_blockmodel(
    shape=(2, 2, 2),
    block_size=(10.0, 10.0, 5.0),
    corner=(0.0, 0.0, 0.0),
    index_type="world_centroids",
)
df["density"] = 2.4 + 0.05 * df["depth"]
df["grade"] = 0.1 + 0.01 * df["depth"]

df[["density", "grade"]].head()

# %%
try:
    import df_eval  # noqa: F401
    from pandera import Column, DataFrameSchema
except ImportError:
    print(
        "Install parq-blockmodel[schema] to run the schema-backed part of this example."
    )

# %%
# Build the schema-backed model when the optional schema dependencies are
# installed. The schema stores the calculated-column expressions in Pandera
# metadata under ``df-eval``.

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
    filename=temp_dir / "calculated_attributes.parquet",
    schema=schema,
    overwrite=True,
)

# %%
# We can read only the calculated columns...

calculated: pd.DataFrame = pbm.read(columns=["tonnes", "contained_metal"], index="ijk", dense=True)
calculated.head()

# %%
# Or we can read all columns, including the calculated attributes

calculated: pd.DataFrame = pbm.read(index="ijk", dense=True, include_calculated=True)
calculated.head()
