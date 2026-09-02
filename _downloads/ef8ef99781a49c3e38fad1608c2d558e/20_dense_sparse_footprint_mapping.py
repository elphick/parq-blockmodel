"""
Dense and Sparse Footprint Mapping
==================================

This example demonstrates dense and sparse plan-view footprint extraction
from a :class:`parq_blockmodel.blockmodel.ParquetBlockModel`.

The workflow is:

1. Build a rotated geometry-backed block model.
2. Rewrite it as a sparse populated model with large gaps.
3. Generate dense and sparse footprint polygons in one GeoDataFrame.
4. Attach custom attributes to the output rows.
5. Plot and summarise the two footprint concepts side by side.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel import LocalGeometry, ParquetBlockModel, RegularGeometry, WorldFrame
from parq_blockmodel.utils.geometry_utils import angles_to_axes


# %%
# Create a rotated geometry-backed block model
# --------------------------------------------
# The footprint method works in model coordinates and supports rotated block
# models provided the logical columns remain vertical in plan view.

temp_dir = Path(tempfile.gettempdir()) / "dense_sparse_footprint_mapping_example"
temp_dir.mkdir(parents=True, exist_ok=True)
pbm_path = temp_dir / "dense_sparse_footprint_mapping.pbm"

axis_u, axis_v, axis_w = angles_to_axes(axis_azimuth=30.0, axis_dip=0.0, axis_plunge=0.0)
geometry = RegularGeometry(
    local=LocalGeometry(
        corner=(100.0, 200.0, 0.0),
        block_size=(20.0, 20.0, 10.0),
        shape=(6, 5, 4),
    ),
    world=WorldFrame(
        axis_u=axis_u,
        axis_v=axis_v,
        axis_w=axis_w,
        crs="EPSG:28350",
    ),
)
pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=pbm_path)


# %%
# Rewrite the dense model as a sparse populated model
# ---------------------------------------------------
# Keep only selected logical columns so the sparse populated footprint differs
# materially from the full dense model footprint.

blocks = pbm.read(columns=["block_id"], index="ijk", dense=True).reset_index()
populated_mask = (
    ((blocks["i"] <= 2) & (blocks["j"] <= 3) & (blocks["k"] <= 1))
    | ((blocks["i"] == 4) & (blocks["j"].isin([1, 2])) & (blocks["k"] == 0))
    | ((blocks["i"] == 5) & (blocks["j"] == 4) & (blocks["k"] == 2))
)

subset_indices = np.flatnonzero(populated_mask.to_numpy(dtype=bool))
table = pq.read_table(pbm.blockmodel_path)
sparse_table = table.take(pa.array(subset_indices, type=pa.int64()))
pq.write_table(sparse_table, pbm.blockmodel_path)

pbm = ParquetBlockModel(pbm_path)
print("Sparse model rows:", pbm.pf.metadata.num_rows)
print("Model sparsity:", f"{pbm.sparsity:.1%}")


# %%
# Generate dense and sparse footprints together
# ---------------------------------------------
# The dense footprint is the full `(i, j)` model domain. The sparse footprint
# is built from occupied `(i, j)` cells only.

gdf = pbm.to_footprint_geodataframe(
    attributes={
        "footprint_type": {
            "dense": "model_extent",
            "sparse": "populated_extent",
        },
        "color": {
            "dense": "#4C78A8",
            "sparse": "#54A24B",
        },
    }
)

print("Footprint rows:")
print(gdf[["footprint_type", "color"]])
print("CRS:", gdf.crs)


# %%
# Plot dense and sparse footprint concepts
# ----------------------------------------
# Plot the dense domain and the populated extent side by side so the difference
# between potential coverage and actual occupancy is easy to see.

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, footprint_type in zip(axes, ["model_extent", "populated_extent"]):
    subset = gdf.loc[gdf["footprint_type"] == footprint_type]
    color = subset["color"].iloc[0]
    subset.plot(
        ax=ax,
        facecolor=color,
        edgecolor="#222222",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.set_title(footprint_type.replace("_", " ").title())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("Dense vs sparse footprint mapping on a rotated block model", fontsize=14)


# %%
# Summarise the footprint areas
# -----------------------------
# Area is reported in model XY units. The sparse footprint is smaller because
# only occupied footprint cells contribute.

summary = gdf.assign(area=gdf.geometry.area)[["footprint_type", "area"]]
print("Footprint area summary:")
print(summary.to_string(index=False))
