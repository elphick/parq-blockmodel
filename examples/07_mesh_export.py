"""
Mesh Export
===========

Block models can be exported as triangulated surface meshes.

PLY is the canonical format for lossless storage — it preserves all vertex
and face attributes together with geometry metadata. GLB (glTF 2.0 binary)
is a derived format suited for exchange with 3D viewers and optionally
carries vertex colours derived from a scalar attribute.

"""

import tempfile
from pathlib import Path

from parq_blockmodel import ParquetBlockModel

# %%
# Create a Block Model
# --------------------
# Use the toy block-model helper to create a 5 × 5 × 5 dense model with a
# synthetic ellipsoidal grade distribution.

temp_dir = Path(tempfile.gettempdir()) / "mesh_export_example"
temp_dir.mkdir(parents=True, exist_ok=True)

pbm: ParquetBlockModel = ParquetBlockModel.create_toy_blockmodel(
    filename=temp_dir / "sample.parquet",
    shape=(5, 5, 5),
    block_size=(10.0, 10.0, 10.0),
    corner=(0.0, 0.0, 0.0),
    grade_name="grade",
    grade_min=50.0,
    grade_max=100.0,
)
pbm

# %%
# Triangulate the Mesh
# --------------------
# :meth:`~parq_blockmodel.ParquetBlockModel.triangulate` returns a
# :class:`~parq_blockmodel.mesh.TriangleMesh` containing vertices, triangle
# faces, and per-vertex/face attribute arrays.  By default only the exterior
# surface is generated (``surface_only=True``).

mesh = pbm.triangulate(attributes=["grade"], surface_only=True)

print(f"Vertices : {mesh.n_vertices}")
print(f"Faces    : {mesh.n_faces}")
print(f"Attributes: {list(mesh.vertex_attributes.keys())}")

# %%
# Export to PLY
# -------------
# PLY is the canonical, lossless format.  The file embeds geometry metadata
# as comments and writes per-vertex ``i``, ``j``, ``k`` indices alongside
# ``x``, ``y``, ``z`` so that every vertex can be traced back to its source
# block.

ply_path = temp_dir / "sample.ply"
pbm.to_ply(ply_path, attributes=["grade"])

# Preview the header of the written file.
with open(ply_path) as fh:
    for line in fh:
        print(line, end="")
        if line.strip() == "end_header":
            break

# %%
# Export to GLB
# -------------
# GLB (glTF 2.0 binary) is a single-file exchange format suitable for web
# viewers such as Babylon.js, Three.js and Cesium.  Passing
# ``texture_attribute`` maps that scalar to vertex colours using the
# requested Matplotlib colormap.

glb_path = temp_dir / "sample.glb"
pbm.to_glb(glb_path, texture_attribute="grade", colormap="viridis")

print(f"GLB file size: {glb_path.stat().st_size:,} bytes")

# Verify the GLB magic number.
with open(glb_path, "rb") as fh:
    print(f"GLB magic   : {fh.read(4)}")

# %%
# Rotated Geometry
# ----------------
# Rotation is handled transparently.  The axis vectors stored in
# :attr:`~parq_blockmodel.geometry.RegularGeometry.axis_u`,
# :attr:`~parq_blockmodel.geometry.RegularGeometry.axis_v` and
# :attr:`~parq_blockmodel.geometry.RegularGeometry.axis_w` are applied when
# computing world-space vertex coordinates, so no special handling is
# required by the caller.

pbm_rot: ParquetBlockModel = ParquetBlockModel.create_toy_blockmodel(
    filename=temp_dir / "rotated.parquet",
    shape=(3, 3, 3),
    block_size=(5.0, 5.0, 5.0),
    corner=(0.0, 0.0, 0.0),
    axis_azimuth=30.0,
    axis_dip=15.0,
    axis_plunge=0.0,
)

print(f"axis_u: {pbm_rot.geometry.axis_u}")
print(f"axis_v: {pbm_rot.geometry.axis_v}")
print(f"axis_w: {pbm_rot.geometry.axis_w}")

glb_rot_path = temp_dir / "rotated.glb"
pbm_rot.to_glb(glb_rot_path, texture_attribute="grade")

print(f"Rotated GLB file size: {glb_rot_path.stat().st_size:,} bytes")

