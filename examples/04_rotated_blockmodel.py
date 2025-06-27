"""
Rotated Block Model
===================

This example demonstrates how to create a rotated block model.

"""
import tempfile

from pathlib import Path
from parq_blockmodel import ParquetBlockModel
import pyvista as pv

# %%
# Create a parquet block model
# ----------------------------
# We will create a block model with a shape of 3x3x3 blocks, each block having a size of 1.0 in all dimensions.
# We will apply rotation.

corner = (0.0, 0.0, 0.0)
block_size = (1.0, 1.0, 1.0)
shape = (2, 2, 2)

temp_dir: Path = Path(tempfile.gettempdir()) / "block_model_example"

temp_dir.mkdir(parents=True, exist_ok=True)
# create a temporary file path for the block model

pbm: ParquetBlockModel = ParquetBlockModel.create_demo_block_model(filename=temp_dir / "demo_block_model.parquet",
                                                                   block_size=block_size, corner=corner, shape=shape,
                                                                   axis_azimuth=30, axis_dip=0, axis_plunge=0,
                                                                   )

pbm

# %%
# Check the block model
# ---------------------
print("Block Model Path:", pbm.blockmodel_path)
print("Name:", pbm.name)
print("Axis Angles:", pbm.geometry.axis_angles)
print("Data Shape:", pbm.data.shape)
print("Data Head:\n", pbm.data.head())
print("Model Attributes:", pbm.attributes)

# %%
# Visualise

p: pv.Plotter = pbm.plot(scalar='depth', grid_type="image")
p.show()
