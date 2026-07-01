"""
Rotated Blockmodel
==================

Minimal example of a **rotated** block model using the new ijk‑first
geometry design.

The key idea is:

* ijk indices and ``shape`` define the canonical dense grid.
* Rotation is represented via the orthonormal basis vectors
  ``axis_u, axis_v, axis_w`` on :class:`parq_blockmodel.geometry.RegularGeometry`.
* World coordinates ``(x, y, z)`` are a *derived view* of
  ``(i, j, k) + geometry``. Changing the axes rotates the xyz centroids
  without changing ijk.

This example:

1. Constructs a small ``RegularGeometry`` with a 30° azimuth rotation.
2. Builds a canonical ``.pbm`` file from that geometry using
   :meth:`ParquetBlockModel.from_geometry`.
3. Derives a simple scalar ``z_centroid`` from the rotated xyz and
   visualises it using :meth:`ParquetBlockModel.plot`.
"""

import tempfile

from pathlib import Path

import pyvista as pv

from parq_blockmodel import ParquetBlockModel, RegularGeometry
from parq_blockmodel.utils.geometry_utils import angles_to_axes


# %%
# Define a small rotated RegularGeometry
# --------------------------------------
corner = (0.0, 0.0, 0.0)
block_size = (1.0, 1.0, 1.0)
shape = (3, 3, 3)

axis_u, axis_v, axis_w = angles_to_axes(
    axis_azimuth=30.0,
    axis_dip=0.0,
    axis_plunge=0.0,
)

geometry = RegularGeometry(
    corner=corner,
    block_size=block_size,
    shape=shape,
    axis_u=axis_u,
    axis_v=axis_v,
    axis_w=axis_w,
)

# %%
# Materialise a canonical .pbm from geometry only
# -----------------------------------------------
temp_dir: Path = Path(tempfile.gettempdir()) / "rotated_block_model_example"
temp_dir.mkdir(parents=True, exist_ok=True)
pbm_path = temp_dir / "rotated_block_model.pbm"

pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=pbm_path)

print("Block Model Path:", pbm.blockmodel_path)
print("Name:", pbm.name)
print("Shape (ijk):", pbm.geometry.shape)
print("Corner:", pbm.geometry.corner)
print("Block size (local i/j/k spacing):", pbm.geometry.block_size)
print("Axis U:", pbm.geometry.axis_u)
print("Axis V:", pbm.geometry.axis_v)
print("Axis W:", pbm.geometry.axis_w)
print("Rotated centroids (head):\n", pbm.geometry.to_dataframe_xyz().head())

# %%
# Derive a simple scalar from rotated xyz centroids
# -------------------------------------------------
# For visualisation we use the rotated Z centroid as a scalar field.

df = pbm.read(index=None)  # raw read; block_id available as column for alignment
df["z_centroid"] = df["z"]

# Persist the derived scalar back using the proper merge write API.
# merge=True appends the new column to existing rows, aligned on block_id.
pbm.write(df[["block_id", "z_centroid"]], merge=True)

# %%
# Plot
# ----

plotter: pv.Plotter = pbm.plot(scalar="z_centroid", grid_type="image")
plotter.show()


