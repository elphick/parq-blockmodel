import numpy as np
import pyvista as pv

from parq_blockmodel import RegularGeometry
from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils.pyvista.custom_plotter import CustomPlotter
from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data

# grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1, 1, 1), origin=(0, 0, 0))
# grid.cell_data["block_id"] = np.arange(grid.n_cells)

df = create_demo_blockmodel()  # includes a categorical called 'depth_category'
geometry = RegularGeometry.from_multi_index(df.index)
grid = df_to_pv_image_data(df, geometry, categorical_encode=True)

plotter = CustomPlotter()
plotter.add_mesh_threshold(grid, scalars='depth_category', show_edges=True)
plotter.set_directional_view(direction='WSW', elevation_deg=30)
plotter.add_axes()
plotter.show()
