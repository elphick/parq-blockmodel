import numpy as np
import pyvista as pv

from parq_blockmodel.utils.pyvista_utils import CustomPlotter

grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1, 1, 1), origin=(0, 0, 0))
grid.cell_data["block_id"] = np.arange(grid.n_cells)

plotter = CustomPlotter()
plotter.add_mesh(grid, show_edges=True)
plotter.set_directional_view(direction='WSW', elevation_deg=30)
plotter.add_axes()
plotter.show()
