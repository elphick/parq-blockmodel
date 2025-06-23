"""
Block Models
============

Block models represent 3D data, typically via a 3D array.  3D arrays can be flattened into a 2D tabular
representation that can be stored in a parquet file.

"""
import tempfile

import pandas as pd
from pathlib import Path

from parq_blockmodel import ParquetBlockModel

# %%
# Create a Parquet Block Model
# ----------------------------
# We leverage the create_demo_block_model class method to create a Parquet Block Model.

temp_dir = Path(tempfile.gettempdir()) / "block_model_example"
temp_dir.mkdir(parents=True, exist_ok=True)

pbm: ParquetBlockModel = ParquetBlockModel.create_demo_block_model(
    filename=temp_dir / "demo_block_model.parquet")
pbm

# %%
# Create Report
# -------------
# We'll create a report for the Parquet Block Model.

pbm.create_report(open_in_browser=True, show_progress=True)

# %%
# Visualise the Model
# -------------------

p = pbm.plot(scalar='depth', threshold=True)
p.show()
