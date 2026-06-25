"""
Reblocking with Calculated Aggregation Inputs
=============================================

Use schema-defined calculated attributes in reblocking configuration. In this
example ``tonnes`` and ``contained_metal`` are defined in Pandera ``df-eval``
metadata and materialized for downsampling.
"""

import tempfile
from pathlib import Path

import numpy as np

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel

try:
    import df_eval  # noqa: F401
    from pandera import Column, DataFrameSchema
except ImportError:
    print("Install parq-blockmodel[schema] to run this example.")
    raise SystemExit(0)


# %%
# Build a small demo model with persisted base attributes.

temp_dir = Path(tempfile.gettempdir()) / "reblocking_calculated_example"
temp_dir.mkdir(parents=True, exist_ok=True)

df = create_demo_blockmodel(
    shape=(4, 4, 4),
    block_size=(10.0, 10.0, 5.0),
    corner=(0.0, 0.0, 0.0),
    index_type="world_centroids",
)

df["grade"] = np.linspace(0.2, 1.2, len(df))
df["density"] = np.linspace(2.0, 3.0, len(df))


# %%
# Define calculated attributes in schema metadata.

schema = DataFrameSchema(
    columns={
        "grade": Column(float, coerce=True, nullable=True),
        "density": Column(float, coerce=True, nullable=True),
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
    df[["grade", "density"]],
    filename=temp_dir / "reblocking_calculated_inputs.parquet",
    schema=schema,
    overwrite=True,
)


# %%
# Downsample using calculated attributes as aggregation inputs.

downsampled = pbm.downsample(
    new_block_size=(20.0, 20.0, 10.0),
    aggregation_config={
        "grade": {"method": "mean"},
        "contained_metal": {"method": "weighted_mean", "basis": "tonnes"},
    },
)

result = downsampled.read(columns=["grade", "contained_metal"], index="ijk", dense=True)
result.head()
