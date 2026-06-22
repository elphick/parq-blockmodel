"""
Block Models
============

Block models represent 3D data, typically via a 3D array.  3D arrays can be flattened into a 2D tabular
representation that can be stored in a parquet file.

"""
import tempfile

from pathlib import Path

from pandera import Column, DataFrameSchema

from parq_blockmodel import ParquetBlockModel

# %%
# Create a Parquet Block Model
# ----------------------------
# We create demo data then load it with a schema so profile report metadata is enriched.

temp_dir = Path(tempfile.gettempdir()) / "block_model_example"
temp_dir.mkdir(parents=True, exist_ok=True)

source_path = temp_dir / "demo_block_model.parquet"
ParquetBlockModel.create_demo_block_model(filename=source_path)

schema = DataFrameSchema(
    columns={
        "depth": Column(float, nullable=True)
    },
    strict=False,
)
schema.columns["depth"].title = "Depth"
schema.columns["depth"].description = "Vertical distance from model top."
schema.name = "demo_block_model"
schema.title = "Demo Block Model"
schema.description = "Synthetic block model used to demonstrate reporting metadata."

pbm: ParquetBlockModel = ParquetBlockModel.from_parquet(source_path, schema=schema)
pbm

# %%
# Create Report
# -------------
# We'll create a report for the Parquet Block Model.
# Use ``columns_per_batch=None`` to profile all requested columns in one pass.
# Set ``open_in_browser=True`` in interactive use if you want it displayed immediately.

report = pbm.create_report(columns_per_batch=None, show_progress=True)
print(f"Report saved to: {report.output_path}")
# report.show()

# %%
# Visualise the Model
# -------------------

p = pbm.plot(scalar='depth', threshold=False, enable_picking=True)
p.show()
