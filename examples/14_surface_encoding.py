"""
Surface Encoding (2.5D Elevation)
=================================

This example demonstrates end-to-end surface-based encoding:

1. Build a regular block model from geometry.
2. Create a raster-backed elevation surface.
3. Evaluate a fraction-below-surface value for all block centroids.
4. Persist the result as a block-model attribute.
5. Visualise the resulting fractions in 3D.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parq_blockmodel import ParquetBlockModel, RasterSurface, RegularGeometry


# %%
# Create a geometry-backed block model
# ------------------------------------

temp_dir = Path(tempfile.gettempdir()) / "surface_encoding_example"
temp_dir.mkdir(parents=True, exist_ok=True)
pbm_path = temp_dir / "surface_encoding.pbm"

geometry = RegularGeometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(20, 16, 6),
)
pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=pbm_path)
blocks = pbm.read(index=None)


# %%
# Build and ingest a raster surface
# ---------------------------------

x_coords = np.linspace(-0.5, 19.5, 9)
y_coords = np.linspace(-0.5, 15.5, 7)
xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
values = 3.5 + 0.08 * xx - 0.04 * yy
surface = RasterSurface.from_arrays(x_coords=x_coords, y_coords=y_coords, values=values, name="topography")

print("Surface name:", surface.name)


# %%
# Evaluate the surface and visualise in 3D
# ----------------------------------------

values = pbm.evaluate_surface(surface, column="topo_fraction")
block_ids = blocks["block_id"].to_numpy(dtype=np.int64)
blocks["topo_fraction"] = values[block_ids]

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    blocks["x"],
    blocks["y"],
    blocks["z"],
    c=blocks["topo_fraction"],
    cmap="viridis",
    s=18,
    edgecolors="none",
)
ax.set_title("Surface fraction below topography")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.colorbar(scatter, ax=ax, label="Fraction below surface", shrink=0.75)
plt.show()


# %%
print("topo_fraction:")
print(pbm.read(columns=["topo_fraction"], index=None)["topo_fraction"].describe())
