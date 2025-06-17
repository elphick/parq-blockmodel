import pandas as pd
import numpy as np


def df_to_pv_structured_grid(df: pd.DataFrame) -> 'pv.StructuredGrid':
    import pyvista as pv

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    df = df.sort_index(level=['z', 'y', 'x'])

    # Get the unique x, y, z coordinates (centroids)
    x_centroids = df.index.get_level_values('x').unique()
    y_centroids = df.index.get_level_values('y').unique()
    z_centroids = df.index.get_level_values('z').unique()

    # Calculate the cell size (assuming all cells are of equal size)
    dx = np.diff(x_centroids)[0]
    dy = np.diff(y_centroids)[0]
    dz = np.diff(z_centroids)[0]

    # Calculate the grid points
    x_points = np.concatenate([x_centroids - dx / 2, x_centroids[-1:] + dx / 2])
    y_points = np.concatenate([y_centroids - dy / 2, y_centroids[-1:] + dy / 2])
    z_points = np.concatenate([z_centroids - dz / 2, z_centroids[-1:] + dz / 2])

    # Create the 3D grid of points
    x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')

    # Create a StructuredGrid object
    grid = pv.StructuredGrid(x, y, z)

    # Add the data from the DataFrame to the grid
    for column in df.columns:
        grid.cell_data[column] = df[column].values

    return grid


def df_to_pv_unstructured_grid(df: pd.DataFrame, block_size) -> 'pv.UnstructuredGrid':
    """
    Requires the index to be a pd.MultiIndex with names ['x', 'y', 'z', 'dx', 'dy', 'dz'].
    :return:
    """

    import pyvista as pv

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    blocks = df.reset_index().sort_values(['z', 'y', 'x'])

    # Get the x, y, z coordinates and cell dimensions
    # if no dims are passed, estimate them
    if 'dx' not in blocks.columns:
        dx, dy, dz = block_size[0], block_size[1], block_size[2]
        blocks['dx'] = dx
        blocks['dy'] = dy
        blocks['dz'] = dz

    x, y, z, dx, dy, dz = (blocks[col].values for col in blocks.columns if col in ['x', 'y', 'z', 'dx', 'dy', 'dz'])
    blocks.set_index(['x', 'y', 'z', 'dx', 'dy', 'dz'], inplace=True)
    # Create the cell points/vertices
    # REF: https://github.com/OpenGeoVis/PVGeo/blob/main/PVGeo/filters/voxelize.py

    n_cells = len(x)

    # Generate cell nodes for all points in data set
    # - Bottom
    c_n1 = np.stack(((x - dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
    c_n2 = np.stack(((x + dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
    c_n3 = np.stack(((x - dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
    c_n4 = np.stack(((x + dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
    # - Top
    c_n5 = np.stack(((x - dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
    c_n6 = np.stack(((x + dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
    c_n7 = np.stack(((x - dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)
    c_n8 = np.stack(((x + dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)

    # - Concatenate
    # nodes = np.concatenate((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0)
    nodes = np.hstack((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8)).ravel().reshape(n_cells * 8, 3)

    # create the cells
    # REF: https://docs/pyvista.org/examples/00-load/create-unstructured-surface.html
    cells_hex = np.arange(n_cells * 8).reshape(n_cells, 8)

    grid = pv.UnstructuredGrid({pv.CellType.VOXEL: cells_hex}, nodes)

    # add the attributes (column) data
    for col in blocks.columns:
        grid.cell_data[col] = blocks[col].values

    return grid
