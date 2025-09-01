"""
Reblocking
==========


"""

import tempfile

import pandas as pd
from pathlib import Path
import pyvista as pv

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
# Visualise the Model
# -------------------
# A continuous attribute.
pbm.plot(scalar='depth', threshold=False, enable_picking=True).show()

# %%
# A categorical attribute.

pbm.plot(scalar='depth_category', threshold=False, enable_picking=True).show()

# %%
# Upsampling
# -------
# Create a new grid with smaller blocks, and visualise.
reblocked_pbm: ParquetBlockModel = pbm.upsample(new_block_size=(0.5, 0.5, 0.5),
                                                interpolation_config={
                                                    'depth': 'linear',
                                                    'depth_category': 'nearest'
                                                })

reblocked_pbm.plot(scalar='depth', threshold=False, enable_picking=True).show()

# %%
# Visualise a reblocked categorical attribute.
reblocked_pbm.plot(scalar='depth_category', threshold=False, enable_picking=True).show()

# %%
# Downsample
# ----------
# Downsample the upsampled model back to the original block size.

downsampled_pbm: ParquetBlockModel = reblocked_pbm.downsample(new_block_size=(1.0, 1.0, 1.0),
                                                              aggregation_config={
                                                                  'depth': {'method': 'mean'},
                                                                  'depth_category': {'method': 'mode'}
                                                              })

reblocked_pbm.plot(scalar='depth', threshold=False, enable_picking=True).show()
