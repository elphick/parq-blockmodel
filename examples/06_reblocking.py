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

p = pbm.plot(scalar='depth', threshold=False, enable_picking=True)
p.show()

# %%
# Create a new grid with smaller blocks
reblocked_mesh: pv.ImageData = pbm.reblock(block_size=(0.5, 0.5, 0.5), return_pyvista=True)

reblocked_mesh = reblocked_mesh.point_data_to_cell_data()

p: pv.Plotter = pv.Plotter()
p.add_mesh(reblocked_mesh, scalars='depth', show_edges=True)
p.show()

print(reblocked_mesh)