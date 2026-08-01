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
from df_eval import Engine

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel

# sphinx_gallery_thumbnail_path = "../docs/_static/branding/parq-blockmodel-gallery-thumbnail.svg"

# %%
# Instantiate
# -----------
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
# Evaluation
# ----------
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

# %%
# Registered Functions
# --------------------
# While expressions can be defined in the schema directly, applying custom functions requires
# that they be registered with the engine. The function registration must be provided when
# creating the PBM via the ``engine_initializer`` parameter.

def calculate_metal_pct(data: pd.DataFrame) -> pd.Series:
    """Calculate metal percentage from contained_metal and tonnes."""
    return data["contained_metal"] / data["tonnes"] * 100

def setup_engine(engine):
    """Register custom pipeline functions with the engine."""
    engine.register_pipeline_function("metal_pct", calculate_metal_pct)
    return engine

# Add the calculated column that uses the registered function to the existing schema
schema.columns["metal_pct"] = Column(
    float,
    coerce=True,
    nullable=True,
    required=False,
    metadata={"df-eval": {"function":
                              {"name": "metal_pct",
                               "inputs": ["contained_metal", "tonnes"],
                               "outputs": ["metal_pct"],
                               "params": {}}}},
)

# Create PBM with engine_initializer to register the function at creation time
pbm_with_fn = ParquetBlockModel.from_dataframe(
    df[["density", "grade"]],
    filename=temp_dir / "calculated_attributes_with_fn.parquet",
    schema=schema,
    engine_initializer=setup_engine,  # Register the function here
    overwrite=True,
)

# Now we can read the calculated column that depends on the registered function
calculated_via_fn: pd.DataFrame = pbm_with_fn.read(
    columns=["tonnes", "contained_metal", "metal_pct"],
    index="ijk",
    dense=True,
)
calculated_via_fn.head()