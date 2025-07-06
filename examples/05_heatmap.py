"""
Heatmap
=======

A heatmap allows you to visualize the distribution of a specific attribute across a 3D block model from
a 2D perspective.

"""
import tempfile

import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go

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
# Create Heatmap
# --------------
# We'll create a heatmap plot in plan view.

fig: go.Figure = pbm.plot_heatmap(attribute='c_index',
                                  threshold=4,
                                  axis='z')

fig

# %%
# view the heatmap matrix
heatmap_matrix: np.ndarray = pbm.create_heatmap_from_threshold(attribute='c_index',
                                                                 threshold=4,
                                                                 axis='z',
                                                                 return_array=True)

heatmap_matrix
