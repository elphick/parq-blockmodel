"""
Heatmap
=======

A heatmap allows you to visualize the distribution of a specific attribute across a 3D block model from
a 2D perspective.

"""

import tempfile
from typing import cast

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.geometry_utils import rotate_points

# %%
# Create a Toy Parquet Block Model
# --------------------------------
# We leverage the create_toy_blockmodel class method to create a model with
# a synthetic Fe distribution, which makes threshold heatmaps more informative.
#
# Use a finer grid for a smoother heatmap while preserving the same overall
# model extent. ``axis_azimuth`` is measured from +x (east), so 30° from north
# corresponds to 60° from east.

temp_dir = Path(tempfile.gettempdir()) / "block_model_example"
temp_dir.mkdir(parents=True, exist_ok=True)

axis_azimuth = 60.0  # 30° east of north
deposit_center = tuple(
    rotate_points(
        np.array([[10.0, 7.5, 5.0]]),
        azimuth=axis_azimuth,
        dip=0.0,
        plunge=0.0,
    )[0]
)

pbm: ParquetBlockModel = ParquetBlockModel.create_toy_blockmodel(
    filename=temp_dir / "toy_block_model.parquet",
    shape=(40, 30, 20),
    block_size=(0.5, 0.5, 0.5),
    corner=(0.0, 0.0, 0.0),
    axis_azimuth=axis_azimuth,
    axis_dip=0.0,
    axis_plunge=0.0,
    grade_name="fe",
    grade_min=50.0,
    grade_max=65.0,
    deposit_center=deposit_center,
    deposit_radii=(8.0, 5.0, 3.0),
    noise_std=0.0,
)
pbm

# %%
# Create Heatmap
# --------------
# We'll create a plan-view heatmap showing blocks above a high-Fe threshold.

fig: go.Figure = pbm.plot_heatmap(attribute='fe',
                                  threshold=57.0,
                                  axis='z')
pio.show(fig)  # type: ignore[arg-type]

# %%
# View the underlying heatmap matrix.
heatmap_matrix = cast(
    np.ndarray,
    pbm.create_heatmap_from_threshold(attribute='fe',
                                      threshold=57.0,
                                      axis='z',
                                      return_array=True),
)

heatmap_matrix

# %%
# Plot the returned matrix directly with Plotly as well.
matrix_fig: go.Figure = px.imshow(
    heatmap_matrix,
    origin="lower",
    color_continuous_scale="Viridis",
    labels={"x": "i", "y": "j", "color": "count above Fe threshold"},
    title="Heatmap matrix derived from thresholded Fe",
)
pio.show(matrix_fig)  # type: ignore[arg-type]
