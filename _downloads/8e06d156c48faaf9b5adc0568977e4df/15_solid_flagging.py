"""
Solid Flagging (3D Volume)
==========================

This example demonstrates end-to-end solid-based flagging:

1. Build a regular block model from geometry.
2. Create a closed triangulated ellipsoid mesh in PyVista.
3. Load the mesh as a ``MeshSolid``.
4. Evaluate a boolean inside/outside mask for all block centroids.
5. Visualise centroid classification in 3D with a mesh wireframe.
6. Encode the result as a categorical block-model attribute.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from parq_blockmodel import MeshSolid, ParquetBlockModel, RegularGeometry


# %%
# Create a geometry-backed block model
# ------------------------------------

temp_dir = Path(tempfile.gettempdir()) / "solid_flagging_example"
temp_dir.mkdir(parents=True, exist_ok=True)
pbm_path = temp_dir / "solid_flagging.pbm"

geometry = RegularGeometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(20, 16, 6),
)
pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=pbm_path)
blocks = pbm.read(index=None)

# %%
# Build and ingest a named ellipsoid solid from PyVista
# -----------------------------------------------------

ellipsoid_surface = pv.Sphere(
    radius=1.0,
    center=(0.0, 0.0, 0.0),
    theta_resolution=40,
    phi_resolution=28,
).triangulate()
ellipsoid_surface.points[:, 0] = 9.0 + 5.0 * ellipsoid_surface.points[:, 0]
ellipsoid_surface.points[:, 1] = 7.5 + 4.0 * ellipsoid_surface.points[:, 1]
ellipsoid_surface.points[:, 2] = 2.5 + 2.0 * ellipsoid_surface.points[:, 2]

solid = MeshSolid.from_pyvista(ellipsoid_surface, name="ore_shell")

print("Solid name:", solid.name)

# %%
# Evaluate centroid-in-solid and visualise in 3D
# ----------------------------------------------
# We plot all block centroids and overlay the ellipsoid as a wireframe.
# Inside and outside centroids use distinct colors.

mask = solid.evaluate(pbm.geometry)
block_ids = blocks["block_id"].to_numpy(dtype=np.int64)
centroids = blocks[["x", "y", "z"]].to_numpy(dtype=float)
inside_points = centroids[mask[block_ids]]
outside_points = centroids[~mask[block_ids]]

plotter = pv.Plotter()
plotter.add_points(
    outside_points,
    color="#bdbdbd",
    point_size=5,
    render_points_as_spheres=True,
    label="outside",
)
plotter.add_points(
    inside_points,
    color="#2c7fb8",
    point_size=12,
    render_points_as_spheres=True,
    label="inside",
)
plotter.add_mesh(
    ellipsoid_surface,
    color="#6baed6",
    opacity=0.14,
    smooth_shading=True,
)
plotter.add_mesh(
    ellipsoid_surface,
    style="wireframe",
    color="#222222",
    line_width=0.9,
    opacity=0.9,
    label=f"{solid.name} wireframe",
)
plotter.add_legend()
plotter.show_grid()
plotter.show_axes()
plotter.show()

# %%
# Encode the mask as a categorical block-model attribute
# ------------------------------------------------------

inside_label = solid.name or "inside"
domain = np.full(len(blocks), pd.NA, dtype=object)
domain[mask[blocks["block_id"].to_numpy(dtype=np.int64)]] = inside_label
blocks["solid_domain"] = pd.Categorical(domain, categories=[inside_label], ordered=False)

pbm.write(blocks[["block_id", "solid_domain"]], merge=True)
print("solid_domain (NA-aware):")
print(pbm.read(columns=["solid_domain"], index=None)["solid_domain"].value_counts(dropna=False))

# %%
# Plot the final 3D block model using the encoded categorical domain
# ------------------------------------------------------------------

plotter: pv.Plotter = pbm.plot(scalar="solid_domain", threshold=True, enable_picking=True, grid_type="image")
plotter.show()
