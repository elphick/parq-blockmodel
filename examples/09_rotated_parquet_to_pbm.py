"""
Rotated XYZ Parquet to PBM
==========================

This example demonstrates a rotated centroid-parquet workflow where:

1. Local corner is kept at ``(0, 0, 0)``.
2. World placement is controlled by ``origin``.
3. Rotation is defined by ``axis_u``, ``axis_v``, ``axis_w``.
4. The parquet is promoted to PBM.
5. Recovered geometry is validated and visualized in both local and world frames.

"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvista as pv

from parq_blockmodel import ParquetBlockModel, RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.utils.demo_block_model import create_toy_blockmodel
from parq_blockmodel.utils.geometry_utils import angles_to_axes, axes_to_angles, rotate_points

# %%
# Create Rotated Source Parquet
# -----------------------------
# Keep local corner at (0, 0, 0), then shift centroids into world space via
# explicit world origin before writing parquet.

temp_dir = Path(tempfile.gettempdir()) / "example_09_rotated_parquet_to_pbm"
temp_dir.mkdir(parents=True, exist_ok=True)

axis_azimuth = 35.0
axis_dip = 12.0
axis_plunge = 8.0
world_origin = (100.0, 200.0, 50.0)

source_parquet = temp_dir / "rotated_source_xyz.parquet"

shape = (20, 15, 10)
block_size = (1.0, 1.0, 1.0)
corner = (0.0, 0.0, 0.0)

# Keep the synthetic deposit centered in local ijk space so the local panel
# shows the same coherent blob as the non-rotated examples.
extent_center = (
    corner[0] + 0.5 * shape[0] * block_size[0],
    corner[1] + 0.5 * shape[1] * block_size[1],
    corner[2] + 0.5 * shape[2] * block_size[2],
)
deposit_radii = (
    0.35 * shape[0] * block_size[0],
    0.30 * shape[1] * block_size[1],
    0.35 * shape[2] * block_size[2],
)

toy_df = create_toy_blockmodel(
    shape=shape,
    block_size=block_size,
    corner=corner,
    # Generate grade on an unrotated local grid; we rotate only xyz positions
    # below so scalar connectivity is preserved across local/world views.
    axis_azimuth=0.0,
    axis_dip=0.0,
    axis_plunge=0.0,
    deposit_bearing=0.0,
    deposit_dip=0.0,
    deposit_plunge=0.0,
    grade_name="fe",
    grade_min=45.0,
    grade_max=70.0,
    deposit_center=extent_center,
    deposit_radii=deposit_radii,
    noise_rel=1e-3,
    noise_seed=42,
)

source_df = toy_df.reset_index(drop=False)
source_df = source_df.drop(columns=["i", "j", "k"], errors="ignore")
rotated_xyz = rotate_points(
    source_df[["x", "y", "z"]].to_numpy(dtype=float),
    azimuth=axis_azimuth,
    dip=axis_dip,
    plunge=axis_plunge,
)
source_df["x"] = rotated_xyz[:, 0] + world_origin[0]
source_df["y"] = rotated_xyz[:, 1] + world_origin[1]
source_df["z"] = rotated_xyz[:, 2] + world_origin[2]
source_df = source_df.set_index(["x", "y", "z"]).sort_index()
source_df.to_parquet(source_parquet)

expected_u, expected_v, expected_w = angles_to_axes(
    axis_azimuth=axis_azimuth,
    axis_dip=axis_dip,
    axis_plunge=axis_plunge,
)
geometry = RegularGeometry(
    local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
    world=WorldFrame(origin=world_origin, axis_u=expected_u, axis_v=expected_v, axis_w=expected_w),
)

table = pq.read_table(source_parquet)
meta = dict(table.schema.metadata or {})
meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
table = table.replace_schema_metadata(meta)
pq.write_table(table, source_parquet)

loaded_source_df = pd.read_parquet(source_parquet)
if loaded_source_df.index.names != ["x", "y", "z"]:
    raise RuntimeError("Expected source parquet to load with MultiIndex ['x', 'y', 'z'].")
if any(col in loaded_source_df.columns for col in ["i", "j", "k"]):
    raise RuntimeError("Source parquet must not contain i, j, k columns.")

print(f"Source rows: {len(loaded_source_df):,}")

# %%
# Promote to PBM
# --------------

pbm = ParquetBlockModel.from_parquet(source_parquet)
pbm

# %%
# Validate Geometry
# -----------------

if not np.allclose(pbm.corner, (0.0, 0.0, 0.0), atol=1e-6):
    raise RuntimeError("Recovered local corner is not (0, 0, 0).")
if not np.allclose(pbm.origin, world_origin, atol=1e-6):
    raise RuntimeError("Recovered world origin does not match expected origin.")
if not np.allclose(pbm.geometry.axis_u, expected_u, atol=1e-6):
    raise RuntimeError("Recovered axis_u does not match expected basis.")
if not np.allclose(pbm.geometry.axis_v, expected_v, atol=1e-6):
    raise RuntimeError("Recovered axis_v does not match expected basis.")
if not np.allclose(pbm.geometry.axis_w, expected_w, atol=1e-6):
    raise RuntimeError("Recovered axis_w does not match expected basis.")

azimuth_out, dip_out, plunge_out = axes_to_angles(pbm.geometry.axis_u, pbm.geometry.axis_v, pbm.geometry.axis_w)

print(f"corner (local) = {pbm.corner}")
print(f"origin (world) = {pbm.origin}")
print(f"axis_u = {pbm.geometry.axis_u}")
print(f"axis_v = {pbm.geometry.axis_v}")
print(f"axis_w = {pbm.geometry.axis_w}")
print(f"angles = ({azimuth_out:.3f}, {dip_out:.3f}, {plunge_out:.3f})")

# %%
# Visualize Rotation
# ------------------

local_grid = pbm.to_pyvista(grid_type="image", attributes=["fe"], frame="local")
world_grid = pbm.to_pyvista(grid_type="image", attributes=["fe"], frame="world")

if not np.array_equal(local_grid.cell_data["fe"], world_grid.cell_data["fe"]):
    raise RuntimeError("Cell values must remain identical across local/world image frames.")

fe_min = float(loaded_source_df["fe"].min())
fe_max = float(loaded_source_df["fe"].max())
if np.isclose(fe_min, fe_max):
    fe_max = fe_min + 1e-6

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_text("Local frame (axis-aligned)", font_size=11)
plotter.add_mesh_threshold(local_grid, scalars="fe", clim=[fe_min, fe_max], show_edges=True)
plotter.show_axes()

plotter.subplot(0, 1)
plotter.add_text("World frame (rotated)", font_size=11)
plotter.add_mesh_threshold(world_grid, scalars="fe", clim=[fe_min, fe_max], show_edges=True)
# Do NOT call link_views(): the two grids sit at different world positions
# (local at origin, world at world_origin).  Linking would copy the local
# camera and leave the world viewport looking at empty space.
plotter.show_axes()
plotter.show()

