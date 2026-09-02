"""
Block Models
============

Block models represent 3D data, typically via a 3D array.  3D arrays can be flattened into a 2D tabular
representation that can be stored in a parquet file.

"""
import tempfile

from pathlib import Path

import pyvista as pv
from pandera import Column, DataFrameSchema

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp, TrameBlockModelPlotEngine

# %%
# Create a Parquet Block Model
# ----------------------------
# We create demo data then load it with a schema so profile report metadata is enriched.

temp_dir = Path(tempfile.gettempdir()) / "block_model_example"
temp_dir.mkdir(parents=True, exist_ok=True)

source_path = temp_dir / "demo_block_model.parquet"
ParquetBlockModel.create_demo_block_model(filename=source_path)

schema = DataFrameSchema(
    columns={
        "depth": Column(float, nullable=True)
    },
    strict=False,
)
schema.columns["depth"].title = "Depth"
schema.columns["depth"].description = "Vertical distance from model top."
schema.name = "demo_block_model"
schema.title = "Demo Block Model"
schema.description = "Synthetic block model used to demonstrate reporting metadata."

pbm: ParquetBlockModel = ParquetBlockModel.from_parquet(source_path, schema=schema)
pbm

# %%
# Create Report
# -------------
# We'll create a report for the Parquet Block Model.
# Use ``columns_per_batch=None`` to profile all requested columns in one pass.
# Set ``open_in_browser=True`` in interactive use if you want it displayed immediately.

report = pbm.create_report(columns_per_batch=None, show_progress=True)
print(f"Report saved to: {report.output_path}")
# report.show()

# %%
# Visualise the Model
# -------------------
# Hold "z" while rotating for turntable orbit with +Z staying up on screen.
p = pbm.plot(
    scalar="depth",
    threshold=False,
    enable_picking=True,
    z_up_lock=True,
    z_up_hotkey="z",
)
p.show()

# %%
# Trame backend from pbm.plot
# ---------------------------
# ``TrameBlockModelPlotEngine`` lets you request a Trame app via ``pbm.plot(...)``.

trame_app_from_plot = pbm.plot(
    scalar="depth",
    threshold=False,
    engine=TrameBlockModelPlotEngine(),
    z_up_lock=True,
    z_up_hotkey="z",
)

# %%
# Direct Trame app
# ----------------
# ``BlockModelTrameApp`` provides the same viewer path directly.

trame_app = BlockModelTrameApp(
    pbm,
    scalar="depth",
    show_edges=True,
    z_up_lock=True,
    z_up_hotkey="z",
)

# %%
# Optional launch
# ---------------
# Trame launch is skipped during gallery/doc builds.

if not getattr(pv, "BUILDING_GALLERY", False):
    # Launch one of these:
    # trame_app_from_plot.launch(port=8080)
    # trame_app.launch(port=8080)
    pass
