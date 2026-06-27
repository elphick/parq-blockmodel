"""
Calculated Columns with Custom DictResolver Lookups
===================================================

This example demonstrates how to register custom resolvers with the df-eval
engine when working with schema-defined calculated columns. Two approaches
are shown: constructor-time registration (eager) and post-load registration (lazy).

DictResolver is used for in-memory lookups that require registration with the
Engine before evaluation.
"""

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
from pathlib import Path
from pandera import DataFrameSchema, Column
from df_eval import DictResolver

from parq_blockmodel import ParquetBlockModel, RegularGeometry, LocalGeometry, WorldFrame

# sphinx_gallery_thumbnail_path = "../docs/_static/branding/parq-blockmodel-gallery-thumbnail.svg"
# %%
# Setup: Create sample block model geometry and data
geometry = RegularGeometry(
    local=LocalGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(10.0, 10.0, 5.0),
        shape=(5, 5, 3),
    ),
    world=WorldFrame(),
)

# Create sample block model data
n_blocks = int(np.prod(geometry.local.shape))
df = pd.DataFrame({
    "density": np.random.uniform(2.0, 3.5, n_blocks),
    "rock_code": np.random.randint(1, 4, n_blocks, dtype=np.int64),  # Codes 1, 2, 3
})

print(f"Block model shape: {geometry.local.shape}")
print(f"Total blocks: {n_blocks}")
print(f"\nSample data:\n{df.head()}")

# %%
# Create DictResolvers for lookups
# =========================================================================
# These resolvers map rock codes and density values to descriptions.
# They will be registered with the engine for use in calculated columns.

rock_type_resolver = DictResolver({
    1: "Granite",
    2: "Diorite",
    3: "Gabbro",
}, default="Unknown")

density_class_resolver = DictResolver({
    2.0: "Low",
    2.5: "Medium",
    3.0: "High",
    3.5: "Very High",
}, default="Unknown")

print("Rock type resolver mapping: 1->Granite, 2->Diorite, 3->Gabbro")
print("Density class resolver mapping: 2.0->Low, 2.5->Medium, 3.0->High, 3.5->Very High")

# %%
# Define schema with calculated columns using resolvers
# =========================================================================
# The schema defines calculated columns that use the ``lookup`` function
# with registered resolvers.

schema = DataFrameSchema(
    columns={
        "density": Column(float, nullable=False),
        "rock_code": Column(int, nullable=False),
        # Calculated column 1: lookup rock type from code
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
        # Calculated column 2: lookup density class
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

print("Schema defined with 2 calculated columns using resolvers")

# %%
# Approach 1: Register resolvers at construction time (eager)
# =========================================================================
# This approach registers the resolvers when creating the PBM instance.

def setup_engine(engine):
    """Register resolvers with the engine."""
    engine.register_resolver("rock_type", rock_type_resolver)
    engine.register_resolver("density_class", density_class_resolver)
    return engine

# Add block_id and positional columns
df["block_id"] = np.arange(n_blocks, dtype=np.uint32)
df["i"], df["j"], df["k"] = geometry.ijk_from_row_index(df["block_id"].to_numpy())
df["x"], df["y"], df["z"] = geometry.xyz_from_row_index(df["block_id"].to_numpy())

# Create PBM with engine_initializer at construction
pbm = ParquetBlockModel.from_dataframe(
    df.set_index(["x", "y", "z"])[["density", "rock_code"]],
    filename=Path("./example_blocks_constructor.parquet"),
    geometry=geometry,
    schema=schema,
    engine_initializer=setup_engine,  # <- Register at construction
    overwrite=True,
)

print("PBM created with engine_initializer at construction")

# Read calculated columns
result = pbm.read(columns=["density", "rock_code", "rock_name", "density_class"], dense=True)
print("\nResult (first 3 rows):")
print(result.head(3))

# %%
# Approach 2: Register resolvers after loading (lazy)
# =========================================================================
# This approach registers the resolvers after the PBM is created,
# using the configure_engine() method.

# Create a fresh PBM without engine_initializer
pbm2 = ParquetBlockModel.from_dataframe(
    df.set_index(["x", "y", "z"])[["density", "rock_code"]],
    filename=Path("./example_blocks_post_load.parquet"),
    geometry=geometry,
    schema=schema,
    # No engine_initializer here
    overwrite=True,
)

print("\nPBM created without engine_initializer")

# Now configure engine after loading
pbm2.configure_engine(setup_engine)
print("Engine configured using configure_engine() method")

# Read calculated columns
result2 = pbm2.read(columns=["density", "rock_code", "rock_name", "density_class"], dense=True)
print("\nResult (first 3 rows):")
print(result2.head(3))

# %%
# Cleanup
# =========================================================================
# Remove temporary files (may require a delay on Windows due to file locks)
import time
for path in [
    Path("./example_blocks_constructor.parquet"),
    Path("./example_blocks_constructor.pbm"),
    Path("./example_blocks_post_load.parquet"),
    Path("./example_blocks_post_load.pbm")
]:
    try:
        time.sleep(0.1)
        path.unlink(missing_ok=True)
    except PermissionError:
        pass  # File may still be locked on some systems

print("\nTemporary files cleaned up")
