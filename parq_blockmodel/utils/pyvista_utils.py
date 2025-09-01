from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
import numpy as np
from scipy.stats import mode

from parq_blockmodel import RegularGeometry

import pyvista as pv


def df_to_pv_image_data(df: pd.DataFrame,
                        geometry: RegularGeometry,
                        fill_value=np.nan) -> pv.ImageData:
    """
    Convert a DataFrame to a PyVista ImageData object for a dense regular grid.

    Args:
        df: DataFrame with MultiIndex (x, y, z) or columns x, y, z.
        geometry: RegularGeometry instance (provides shape, spacing, origin).
        fill_value: Value to use for missing cells.

    Returns:
        pv.ImageData: PyVista ImageData object with cell data.
    """

    # Ensure index is MultiIndex (x, y, z)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['x', 'y', 'z'])

    # Sort index to match PyVista's Fortran order (z, y, x)
    df = df.sort_index(level=['z', 'y', 'x'])

    # Create dense index and reindex
    dense_index = geometry.to_multi_index()
    dense_df = df.reindex(dense_index)
    shape = geometry.shape

    grid: pv.ImageData = geometry.to_pyvista()

    for attr in df.columns:
        arr = dense_df[attr].to_numpy().reshape(shape, order='C').ravel(order='F')
        if dense_df[attr].hasnans:
            arr = np.where(np.isnan(arr), fill_value, arr)
        grid.cell_data[attr] = arr

    return grid


def pv_image_data_to_df(image_data: pv.ImageData) -> pd.DataFrame:
    """
    Convert a PyVista ImageData object to a DataFrame using cell centroids.

    Args:
        image_data (pv.ImageData): The input PyVista ImageData object.

    Returns:
        pd.DataFrame: A DataFrame with columns x, y, z (centroids) and cell data attributes.
    """
    centroids = image_data.cell_centers().points
    df = pd.DataFrame({k: np.asarray(v) for k, v in image_data.cell_data.items()})
    df['x'] = centroids[:, 0]
    df['y'] = centroids[:, 1]
    df['z'] = centroids[:, 2]
    # set and sort index
    df.set_index(['x', 'y', 'z'], inplace=True)
    df.sort_index(inplace=True, ascending=True)
    return df


def df_to_pv_structured_grid(df: pd.DataFrame,
                             block_size: Optional[tuple[float, float, float]] = None,
                             validate_block_size: bool = True
                             ) -> pv.StructuredGrid:
    """Convert a DataFrame into a PyVista StructuredGrid.

    This function is for the full grid dense block model.

    The DataFrame should have a MultiIndex of coordinates (x, y, z) and data columns.
    The grid is created assuming uniform block sizes in the x, y, z directions.
    The grid points are calculated based on the centroids of the blocks, and the data is added to the cell
    data of the grid.

    Args:
        df: pd.DataFrame with a MultiIndex of coordinates (x, y, z) and data columns.
        block_size: tuple of floats (dx, dy, dz), optional.  Not used if geometry is provided.
        validate_block_size: bool, optional.  Not needed if geometry is provided.

    Returns:
        pv.StructuredGrid: A PyVista StructuredGrid object.
    """

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    df = df.sort_index(level=['z', 'y', 'x'])

    # Get the unique x, y, z coordinates (centroids)
    x_centroids = df.index.get_level_values('x').unique()
    y_centroids = df.index.get_level_values('y').unique()
    z_centroids = df.index.get_level_values('z').unique()

    if block_size is None:
        # Calculate the cell size (assuming all cells are of equal size)
        dx = np.diff(x_centroids)[0]
        dy = np.diff(y_centroids)[0]
        dz = np.diff(z_centroids)[0]
    else:
        dx, dy, dz = block_size[0], block_size[1], block_size[2]

    if validate_block_size:
        # Check all diffs are the same (within tolerance)
        tol = 1e-8
        if (np.any(np.abs(np.diff(x_centroids) - dx) > tol) or
                np.any(np.abs(np.diff(y_centroids) - dy) > tol) or
                np.any(np.abs(np.diff(z_centroids) - dz) > tol)):
            raise ValueError("Block sizes are not uniform in the structured grid.")

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


def df_to_pv_unstructured_grid(df: pd.DataFrame, block_size: tuple[float, float, float],
                               validate_block_size: bool = True) -> pv.UnstructuredGrid:
    """Convert a DataFrame into a PyVista UnstructuredGrid.

    This function is for the unstructured grid block model, which is typically used for sparse or
    irregular block models.

    The DataFrame should have a MultiIndex of coordinates (x, y, z) and block sizes (dx, dy, dz).
    The grid is created based on the centroids of the blocks, and the data is added to the cell
    data of the grid.
    The block sizes (dx, dy, dz) can be provided or estimated from the DataFrame.


    Args:
        df: pd.DataFrame with a MultiIndex of coordinates (x, y, z) and block sizes (dx, dy, dz).
        block_size: tuple of floats, optional
        validate_block_size: bool, optional

    Returns:
        pv.UnstructuredGrid: A PyVista UnstructuredGrid object.
    """

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    blocks = df.reset_index().sort_values(['z', 'y', 'x'])

    # Get the x, y, z coordinates and cell dimensions
    # if no dims are passed, estimate them
    if 'dx' not in blocks.columns:
        dx, dy, dz = block_size[0], block_size[1], block_size[2]
        blocks['dx'] = dx
        blocks['dy'] = dy
        blocks['dz'] = dz

    if validate_block_size:
        tol = 1e-8
        if blocks[['dx', 'dy', 'dz']].std().max() > tol:
            raise ValueError("Block sizes are not uniform in the unstructured grid.")

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


def df_to_poly_data(df: pd.DataFrame) -> pv.PolyData:
    """
    Convert a DataFrame to a PyVista PolyData object for a sparse grid.

    Args:
        df: DataFrame with MultiIndex (x, y, z) or columns x, y, z.
        geometry: RegularGeometry instance (provides shape, spacing, origin).

    Returns:
        pv.PolyData: PyVista PolyData object with point data.
    """

    # Ensure index is MultiIndex (x, y, z)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['x', 'y', 'z'])

    # Use only the present centroids
    points = df.index.to_frame(index=False).values

    poly_data = pv.PolyData(points)
    for attr in df.columns:
        poly_data.point_data[attr] = df[attr].values

    return poly_data


def poly_data_to_df(poly_data: pv.PolyData) -> pd.DataFrame:
    """
    Convert a PyVista PolyData object to a DataFrame using point coordinates.

    Args:
        poly_data (pv.PolyData): The input PyVista PolyData object.

    Returns:
        pd.DataFrame: A DataFrame with columns x, y, z (coordinates) and point data attributes.
    """
    points = poly_data.points
    df = pd.DataFrame({k: np.asarray(v) for k, v in poly_data.point_data.items()})
    df['x'] = points[:, 0]
    df['y'] = points[:, 1]
    df['z'] = points[:, 2]
    # set and sort index
    df.set_index(['x', 'y', 'z'], inplace=True)
    df.sort_index(inplace=True, ascending=True)
    return df


def calculate_spacing(grid: pv.UnstructuredGrid) -> tuple[float, float, float]:
    """
    Calculate the spacing of an UnstructuredGrid by finding the mode of unique differences.

    Args:
        grid (pv.UnstructuredGrid): The input PyVista UnstructuredGrid.

    Returns:
        tuple[float, float, float]: The spacing in x, y, and z directions.
    """
    # Extract unique x, y, z coordinates
    x_coords = np.unique(grid.points[:, 0])
    y_coords = np.unique(grid.points[:, 1])
    z_coords = np.unique(grid.points[:, 2])

    # Calculate differences and find the mode
    dx = mode(np.diff(x_coords)).mode
    dy = mode(np.diff(y_coords)).mode
    dz = mode(np.diff(z_coords)).mode

    return dx, dy, dz


def infer_regular_geometry_from_df(df: pd.DataFrame):
    """
    Infer RegularGeometry from DataFrame centroids (x, y, z columns or index).
    Returns a RegularGeometry instance.
    """
    # Try to get x, y, z from index or columns
    if isinstance(df.index, pd.MultiIndex):
        x = df.index.get_level_values('x').unique()
        y = df.index.get_level_values('y').unique()
        z = df.index.get_level_values('z').unique()
    else:
        x = df['x'].unique()
        y = df['y'].unique()
        z = df['z'].unique()
    x = np.sort(x)
    y = np.sort(y)
    z = np.sort(z)
    shape = (len(x), len(y), len(z))
    spacing = (
        np.diff(x).mean() if len(x) > 1 else 1.0,
        np.diff(y).mean() if len(y) > 1 else 1.0,
        np.diff(z).mean() if len(z) > 1 else 1.0,
    )
    origin = (x[0] - spacing[0] / 2, y[0] - spacing[1] / 2, z[0] - spacing[2] / 2)
    return RegularGeometry(shape=shape, block_size=spacing, corner=origin)


def _get_geometry_from_df(df):
    # Try to get geometry from df.geometry if present, else infer from centroids
    if hasattr(df, 'geometry'):
        return df.geometry
    return infer_regular_geometry_from_df(df)


import math

import numpy as np
import pyvista as pv


class CustomPlotter(pv.Plotter):
    """
    A custom PyVista Plotter with Z-up enforcement, picking, and directional camera view.

    Examples
    --------
    >>> grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1, 1, 1), origin=(0, 0, 0))
    >>> grid.cell_data["block_id"] = np.arange(grid.n_cells)
    >>>
    >>> plotter = CustomPlotter()
    >>> plotter.add_mesh(grid, show_edges=True)
    >>> plotter.set_directional_view(direction='WSW', elevation_deg=30)
    >>> plotter.add_axes()
    >>> plotter.show()
    """

    HELP_TEXT_NAME = "help_overlay"

    HOTKEYS = {
        "h": "Show/hide this help",
        "p": "Toggle cell picking",
        "z": "Z-up rotation (hold)",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hotkey_pressed = {'z': False}
        self.picking_enabled = False
        self.slicer_enabled = False
        self.help_visible = False
        self._show_help_overlay()
        self._setup_callbacks()

    def _key_press_callback(self, obj, event):
        key = obj.GetKeySym()
        if key == 'z':
            self.hotkey_pressed['z'] = True
        if key == 'p':
            if not self.picking_enabled:
                self.enable_general_picking()
                self.picking_enabled = True
            else:
                self.disable_picking()
                self.remove_actor("cell_info_text")
                self.picking_enabled = False
        if key == 'h':
            if not self.help_visible:
                self._show_help_overlay()
                self.help_visible = True
            else:
                self.remove_actor(self.HELP_TEXT_NAME)
                self.help_visible = False

    def _show_help_overlay(self):
        lines = []
        for k, v in self.HOTKEYS.items():
            lines.append(f"[{k.upper()}]  {v}")
        help_text = "\n".join(lines)
        self.add_text(
            help_text,
            position="right_edge",
            font_size=10,
            name=self.HELP_TEXT_NAME,
            color="white",
            shadow=True,
            viewport=True,
        )

    def set_directional_view(
            self,
            direction='WSW',
            radius_factor=4.0,
            elevation_deg=30,
            azimuth_deg=None
    ):
        # Map compass directions to azimuth angles (degrees)
        direction_azimuth = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        if azimuth_deg is None:
            azimuth_deg = direction_azimuth.get(direction.upper(), 247.5)  # Default to WSW

        bounds = self.bounds
        center = [
            (bounds[1] + bounds[0]) / 2,
            (bounds[3] + bounds[2]) / 2,
            (bounds[5] + bounds[4]) / 2
        ]
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        r = max_dim * radius_factor

        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        x = center[0] + r * math.cos(elevation) * math.cos(azimuth)
        y = center[1] + r * math.cos(elevation) * math.sin(azimuth)
        z = center[2] + r * math.sin(elevation)
        self.camera_position = [(x, y, z), center, (0, 0, 1)]

    def _setup_callbacks(self):
        iren = self.iren
        iren.add_observer("KeyPressEvent", self._key_press_callback)
        iren.add_observer("KeyReleaseEvent", self._key_release_callback)
        iren.add_observer("InteractionEvent", self._enforce_z_up_during_interaction)

    def _key_release_callback(self, obj, event):
        key = obj.GetKeySym()
        if key == 'z':
            self.hotkey_pressed['z'] = False

    def _enforce_z_up_during_interaction(self, obj, event):
        if self.hotkey_pressed['z']:
            self.camera.SetViewUp(0, 0, 1)
            self.render()

    def enable_general_picking(self):
        def cell_callback(picked_cell):
            text_name = "cell_info_text"
            if text_name in self.actors:
                self.remove_actor(text_name)
            if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
                for mesh in self.meshes:  # self.meshes is a list
                    if "vtkOriginalCellIds" in picked_cell.cell_data:
                        cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                        centroid = mesh.cell_centers().points[cell_id]
                        centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                        values = {attr: mesh.cell_data[attr][cell_id] for attr in mesh.cell_data}
                        msg = f"Cell ID: {cell_id}, {centroid_str}, " + ", ".join(
                            f"{k}: {v}" for k, v in values.items())
                        break
                else:
                    msg = "Picked cell, but could not resolve mesh/cell data."
                self.add_text(msg, position="upper_left", font_size=12, name=text_name)
            else:
                self.add_text("No valid cell picked.", position="upper_left", font_size=12, name=text_name)

        self.disable_picking()  # Always disable before enabling
        self.enable_cell_picking(callback=cell_callback, show_message=False, through=False)
