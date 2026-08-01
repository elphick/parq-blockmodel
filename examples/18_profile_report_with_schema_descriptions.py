"""
Profile Reports with Schema Descriptions
========================================

This example demonstrates how Pandera schema column ``title`` and
``description`` metadata improve the variable descriptions shown in
``create_report`` output.
"""

import tempfile
from pathlib import Path

from pandera import Check, Column, DataFrameSchema

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel

# %%
# Build a small demo model and attach schema metadata
# ---------------------------------------------------

temp_dir = Path(tempfile.gettempdir()) / "profiling_with_schema_descriptions"
temp_dir.mkdir(parents=True, exist_ok=True)

df = create_demo_blockmodel(shape=(3, 3, 3))
df["cu_pct"] = 0.15 + 0.01 * df["depth"]

schema = DataFrameSchema(
    columns={
        "depth": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "cu_pct": Column(float, nullable=True),
    },
    strict=False,
)
schema.columns["depth"].title = "Depth"
schema.columns["depth"].description = "Vertical distance below the model top surface."
schema.columns["cu_pct"].title = "Copper grade (%)"
schema.columns["cu_pct"].description = "Synthetic copper grade for demonstration."
schema.name = "demo_profile_model"
schema.title = "Demo Profile Model"
schema.description = "Synthetic dataset used to demonstrate schema-enriched profiling reports."

pbm = ParquetBlockModel.from_dataframe(
    df[["depth", "cu_pct"]],
    filename=temp_dir / "schema_profile.parquet",
    schema=schema,
    overwrite=True,
)

# %%
# Generate the report
# -------------------
# Use ``columns_per_batch=None`` to profile all selected columns in one pass.

report = pbm.create_report(
    columns=["depth", "cu_pct"],
    columns_per_batch=None,
    show_progress=True,
)
print(f"Report saved to: {report.output_path}")
