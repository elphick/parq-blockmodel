"""
blockmodel.py

This module defines the ParquetBlockModel class, which represents a block model stored in a Parquet file.

Main API:

- ParquetBlockModel: Class for representing a block model stored in a Parquet file.

"""
import logging
import math
import shutil
import typing
import warnings
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel.types import Shape3D, Point, BlockSize
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel, create_toy_blockmodel
from parq_blockmodel.utils.geometry_utils import rotation_to_axis_orientation
from parq_tools.lazy_parquet import LazyParquetDataFrame
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from parq_tools import ParquetProfileReport, filter_parquet_file
from parq_tools.utils import atomic_output_file

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.reblocking.reblocking import downsample_blockmodel, upsample_blockmodel

if typing.TYPE_CHECKING:
    import pyvista as pv  # type: ignore[import]
    import plotly.graph_objects as go


class ParquetBlockModel:
    """
    A class to represent a **regular** Parquet block model.

    Block ordering is c-style, ordered by x, y, z coordinates.

    Attributes:
        blockmodel_path (Path): The file path to the blockmodel Parquet file.  This file is the source of the 
            block model data.  Consider a .pbm.parquet extension to imply a ParquetBlockModel file.
        name (str): The name of the block model, derived from the file name.
        geometry (RegularGeometry): The geometry of the block model, derived from the Parquet file.
    """

    def __init__(self, blockmodel_path: Path, name: Optional[str] = None, geometry: Optional[RegularGeometry] = None):
        if blockmodel_path.suffixes[-2:] != [".pbm", ".parquet"]:
            raise ValueError("The provided file must have a '.pbm.parquet' extension.")
        self.blockmodel_path = blockmodel_path
        self.name = name or blockmodel_path.stem.strip('.pbm')
        self.geometry = geometry or RegularGeometry.from_parquet(self.blockmodel_path)
        self.pf: ParquetFile = ParquetFile(blockmodel_path)
        self.data: LazyParquetDataFrame = LazyParquetDataFrame(self.blockmodel_path)
        self.columns: list[str] = pq.read_schema(self.blockmodel_path).names
        self._centroid_index: Optional[pd.MultiIndex] = None
        self.attributes: list[str] = [col for col in self.columns if col not in ["x", "y", "z"]]
        self._extract_column_dtypes()
        self._logger = logging.getLogger(__name__)

        if self.is_sparse:
            if not self.validate_sparse():
                raise ValueError("The sparse ParquetBlockModel is invalid. "
                                 "Sparse centroids must be a subset of the dense grid.")

    def __repr__(self):
        return f"ParquetBlockModel(name={self.name}, path={self.blockmodel_path})"

    def _extract_column_dtypes(self):
        self.column_dtypes: dict[str, np.dtype] = {}
        self._column_categorical_ordered: dict[str, bool] = {}
        schema = pq.read_schema(self.blockmodel_path)
        for col in self.columns:
            if col in ["x", "y", "z"]:
                continue
            field_type = schema.field(col).type
            if pa.types.is_dictionary(field_type):
                self.column_dtypes[col] = pd.CategoricalDtype(ordered=field_type.ordered)
                self._column_categorical_ordered[col] = field_type.ordered
            else:
                self.column_dtypes[col] = field_type.to_pandas_dtype()

    @property
    def column_categorical_ordered(self) -> dict[str, bool]:
        return self._column_categorical_ordered.copy()

    @property
    def centroid_index(self) -> pd.MultiIndex:
        """
        Get the centroid index of the block model.

        Returns:
            pd.MultiIndex: The MultiIndex representing the centroid coordinates (x, y, z).
        """

        if self._centroid_index is None:
            centroid_cols = ["x", "y", "z"]
            centroids: pd.DataFrame = pq.read_table(self.blockmodel_path, columns=centroid_cols).to_pandas()

            if centroids.index.names == centroid_cols:
                index = centroids.index
            else:
                if centroids.empty:
                    raise ValueError("Parquet file is empty or does not contain valid centroid data.")
                index = centroids.set_index(["x", "y", "z"]).index
            if not index.is_unique:
                raise ValueError("The index of the Parquet file is not unique. "
                                 "Ensure that the centroid coordinates (x, y, z) are unique.")

            # Only check monotonicity if axes are aligned (not rotated)
            if not self.geometry.is_rotated and not index.is_monotonic_increasing:
                raise ValueError("The index of the Parquet file is not sorted in ascending order. "
                                 "Ensure that the centroid coordinates (x, y, z) are sorted.")
            self._centroid_index = index
        return self._centroid_index

    @property
    def is_sparse(self) -> bool:
        dense_index = self.geometry.to_multi_index()
        return len(self.centroid_index) < len(dense_index)

    @property
    def sparsity(self) -> float:
        dense_index = self.geometry.to_multi_index()
        return 1.0 - (len(self.centroid_index) / len(dense_index))

    @property
    def index_c(self) -> np.ndarray:
        """Zero-based C-order (x, y, z) indices for the dense grid."""
        shape = self.geometry.shape
        return np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='C')

    @property
    def index_f(self) -> np.ndarray:
        """Zero-based F-order (z, y, x) indices for the dense grid."""
        shape = self.geometry.shape
        return np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='F')

    def validate_sparse(self) -> bool:
        dense_index = self.geometry.to_multi_index()
        # All sparse centroids must be in the dense grid
        return self.centroid_index.isin(dense_index).all()

    @classmethod
    def from_parquet(cls, parquet_path: Path,
                     columns: Optional[list[str]] = None,
                     overwrite: bool = False,
                     axis_azimuth: float = 0.0,
                     axis_dip: float = 0.0,
                     axis_plunge: float = 0.0
                     ) -> "ParquetBlockModel":
        """ Create a ParquetBlockModel instance from a Parquet file.

        Args:
            parquet_path (Path): The path to the Parquet file.
            columns (Optional[list[str]]): The list of columns to extract from the Parquet file.
            overwrite (bool): If True, allows overwriting an existing ParquetBlockModel file. Defaults to False.
            axis_azimuth (float): The azimuth angle in degrees for rotation. Defaults to 0.0.
            axis_dip (float): The dip angle in degrees for rotation. Defaults to 0.0.
            axis_plunge (float): The plunge angle in degrees for rotation. Defaults to 0.0.

        """
        if parquet_path.suffixes[-2:] == [".pbm", ".parquet"]:
            if not overwrite:
                raise ValueError(
                    f"File {parquet_path} appears to be a compliant ParquetBlockModel file. "
                    f"Use the constructor directly, or pass overwrite=True to allow mutation."
                )

        geometry = RegularGeometry.from_parquet(filepath=parquet_path, axis_azimuth=axis_azimuth, axis_dip=axis_dip,
                                                axis_plunge=axis_plunge)

        cls._validate_geometry(parquet_path)

        new_filepath: Path = parquet_path.resolve().with_suffix(".pbm.parquet")
        if columns is None:
            new_filepath = shutil.copy(parquet_path, new_filepath)
        else:
            filter_parquet_file(input_path=parquet_path,
                                output_path=new_filepath,
                                columns=columns)
        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame,
                       filename: Path,
                       geometry: Optional[RegularGeometry] = None,
                       name: Optional[str] = None,
                       overwrite: bool = False,
                       axis_azimuth: float = 0.0,
                       axis_dip: float = 0.0,
                       axis_plunge: float = 0.0
                       ) -> "ParquetBlockModel":
        """Create a ParquetBlockModel from a Pandas DataFrame.
        Args:
            dataframe (pd.DataFrame): The DataFrame containing block model data.
            filename (Path): The file path where the Parquet file will be saved.
            geometry (Optional[RegularGeometry]): The geometry of the block model. If None, it will be inferred from the DataFrame.
            name (Optional[str]): The name of the block model. If None, the name will be derived from the filename.
            overwrite (bool): If True, allows overwriting an existing ParquetBlockModel file. Defaults to False.
            axis_azimuth (float): The azimuth angle in degrees for rotation. Defaults to 0.0.
            axis_dip (float): The dip angle in degrees for rotation. Defaults to 0.0.
            axis_plunge (float): The plunge angle in degrees for rotation. Defaults to 0.0.
        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel created from the DataFrame.
        """
        # Ensure MultiIndex
        if dataframe.index.names != ["x", "y", "z"]:
            raise ValueError("DataFrame index must be a MultiIndex with names ['x', 'y', 'z'].")

        # Warn if not sorted
        if not dataframe.index.is_monotonic_increasing:
            warnings.warn("DataFrame index is not sorted in ascending order.")

        # Ensure correct filename extension
        if not filename.suffix == ".parquet":
            raise ValueError(f"Filename {filename} must have a '.parquet' extension.")
        pbm_path = filename.with_suffix(".pbm.parquet")
        if pbm_path.exists() and not overwrite:
            raise FileExistsError(f"File {pbm_path} already exists. Use overwrite=True to allow mutation.")

        # Infer geometry if needed
        geometry = geometry or RegularGeometry.from_multi_index(
            dataframe.index, axis_azimuth=axis_azimuth, axis_dip=axis_dip, axis_plunge=axis_plunge)
        if not isinstance(geometry, RegularGeometry):
            raise TypeError("geometry must be a RegularGeometry instance.")

        # Save DataFrame to Parquet
        dataframe.to_parquet(pbm_path, index=True)

        # Validate geometry
        cls._validate_geometry(pbm_path, geometry)

        return cls(blockmodel_path=pbm_path, name=name, geometry=geometry)


    @classmethod
    def create_demo_block_model(cls, filename: Path,
                                shape: Shape3D = (3, 3, 3),
                                block_size: BlockSize = (1, 1, 1),
                                corner: Point = (-0.5, -0.5, -0.5),
                                axis_azimuth: float = 0.0,
                                axis_dip: float = 0.0,
                                axis_plunge: float = 0.0
                                ) -> "ParquetBlockModel":
        """
        Create a demo block model with specified parameters.

        Args:
            filename (Path): The file path where the Parquet file will be saved.
            shape (tuple): The shape of the block model.
            block_size (tuple): The size of each block.
            corner (tuple): The coordinates of the corner of the block model.
            axis_azimuth (float): The azimuth angle in degrees for rotation.
            axis_dip (float): The dip angle in degrees for rotation.
            axis_plunge (float): The plunge angle in degrees for rotation.

        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel with demo data.
        """
        create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner,
                               azimuth=axis_azimuth, dip=axis_dip, plunge=axis_plunge,
                               parquet_filepath=filename)

        # get the orientation of the axes
        axis_u, axis_v, axis_w = rotation_to_axis_orientation(axis_azimuth=axis_azimuth, axis_dip=axis_dip,
                                                              axis_plunge=axis_plunge)
        # create geometry that aligns with the demo block model
        geometry = RegularGeometry(block_size=block_size, corner=corner, shape=shape,
                                   axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

        if not geometry.is_rotated:
            cls._validate_geometry(filename)

        new_filepath = shutil.copy(filename, filename.resolve().with_suffix(".pbm.parquet"))

        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def create_toy_blockmodel(cls, filename: Path,
                              shape: Shape3D = (3, 3, 3),
                              block_size: BlockSize = (1, 1, 1),
                              corner: Point = (-0.5, -0.5, -0.5),
                              axis_azimuth: float = 0.0,
                              axis_dip: float = 0.0,
                              axis_plunge: float = 0.0,
                              deposit_bearing: float = 20.0,
                              deposit_dip: float = 30.0,
                              deposit_plunge: float = 10.0,
                              grade_name: str = 'grade',
                              grade_min: float = 50.0,
                              grade_max: float = 65.0,
                              deposit_center=(10.0, 7.5, 5.0),
                              deposit_radii=(8.0, 5.0, 3.0),
                              noise_std: float = 0.2,
                              ) -> "ParquetBlockModel":
        """
        Create a toy block model with specified parameters.

        Args:
            filename (Path): The file path where the Parquet file will be saved.
            shape (tuple): The shape of the block model.
            block_size (tuple): The size of each block.
            corner (tuple): The coordinates of the corner of the block model.
            axis_azimuth (float): The azimuth angle in degrees for rotation.
            axis_dip (float): The dip angle in degrees for rotation.
            axis_plunge (float): The plunge angle in degrees for rotation.
            deposit_bearing (float): The azimuth angle of the deposit in degrees.
            deposit_dip (float): The dip angle of the deposit in degrees.
            deposit_plunge (float): The plunge angle of the deposit in degrees.
            grade_name (str): The name of the column to store the grade values.
            grade_min (float): The minimum grade value.
            grade_max (float): The maximum grade value.
            deposit_center (tuple): The center of the deposit (x, y, z).
            deposit_radii (tuple): The radii of the deposit (rx, ry, rz).
            noise_std: (float):
        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel with toy data.
        """

        create_toy_blockmodel(shape=shape, block_size=block_size, corner=corner,
                              axis_azimuth=axis_azimuth, axis_dip=axis_dip, axis_plunge=axis_plunge,
                              deposit_bearing=deposit_bearing, deposit_dip=deposit_dip, deposit_plunge=deposit_plunge,
                              grade_name=grade_name, grade_min=grade_min, grade_max=grade_max,
                              deposit_center=deposit_center, deposit_radii=deposit_radii,
                              noise_std=noise_std, parquet_filepath=filename,
                              )
        # get the orientation of the axes
        axis_u, axis_v, axis_w = rotation_to_axis_orientation(
            axis_azimuth=axis_azimuth, axis_dip=axis_dip,
            axis_plunge=axis_plunge)
        # create geometry that aligns with the demo block model
        geometry = RegularGeometry(block_size=block_size, corner=corner, shape=shape,
                                   axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

        if not geometry.is_rotated:
            cls._validate_geometry(filename)

        new_filepath = shutil.copy(filename, filename.resolve().with_suffix(".pbm.parquet"))

        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_geometry(cls, geometry: RegularGeometry,
                      path: Path,
                      name: Optional[str] = None
                      ) -> "ParquetBlockModel":
        """Create a ParquetBlockModel from a RegularGeometry object.

        The model will have no attributes.

        Args:
            geometry (RegularGeometry): The geometry of the block model.
            path (Path): The file path where the Parquet file will be saved.
            name (Optional[str]): The name of the block model. If None, the name will be derived from the path.
        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel with the specified geometry.
        """
        centroids_df = geometry.to_dataframe()
        centroids_df.to_parquet(path, index=False)
        return cls(blockmodel_path=path, name=name, geometry=geometry)

    def create_report(self, columns: Optional[list[str]] = None,
                      column_batch_size: int = 10,
                      show_progress: bool = True,
                      open_in_browser: bool = False
                      ) -> Path:
        """
        Create a ydata-profiling report for the block model.
        The report will be of the same name as the block model, with a '.html' extension.

        Args:
            columns: List of column names to include in the profile. If None, all columns are used.
            column_batch_size: The number of columns to process in each batch. If None, processes all columns at once.
            show_progress: bool: If True, displays a progress bar during profiling.
            open_in_browser: bool: If True, opens the report in a web browser after generation.

        Returns
            Path: The path to the generated profile report.

        """
        report: ParquetProfileReport = ParquetProfileReport(self.blockmodel_path, columns=columns,
                                                            batch_size=column_batch_size,
                                                            show_progress=show_progress).profile()
        if open_in_browser:
            report.show(notebook=False)
        if not columns:
            self.report_path = self.blockmodel_path.with_suffix('.html')
        return self.report_path

    def downsample(self, new_block_size, aggregation_config) -> "ParquetBlockModel":
        """
        Downsample the block model to a coarser grid with specified aggregation methods for each attribute.
        This function supports downsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            aggregation_config: dict mapping attribute names to aggregation methods.

        Example:
            aggregation_config = {
                'grade': {'method': 'weighted_mean', 'weight': 'dry_mass'},
                'density': {'method': 'weighted_mean', 'weight': 'volume'},
                'dry_mass': {'method': 'sum'},
                'volume': {'method': 'sum'},
                'rock_type': {'method': 'mode'}
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the downsampled grid.
        """
        return downsample_blockmodel(self, new_block_size, aggregation_config)

    def upsample(self, new_block_size, interpolation_config) -> "ParquetBlockModel":
        """
        Upsample the block model to a finer grid with specified interpolation methods for each attribute.
        This function supports upsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            interpolation_config: dict mapping attribute names to interpolation methods.

        Example:
            interpolation_config = {
                'grade': {'method': 'linear'},
                'density': {'method': 'nearest'},
                'dry_mass': {'method': 'linear'},
                'volume': {'method': 'linear'},
                'rock_type': {'method': 'nearest'}
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the upsampled grid.
        """
        return upsample_blockmodel(self, new_block_size, interpolation_config)

    def create_heatmap_from_threshold(self, attribute: str, threshold: float, axis: str = "z",
                                      return_array: bool = False) -> Union['pv.ImageData', np.ndarray]:
        """
        Create a 2D heatmap from a 3D block model at a specified attribute threshold.

        Args:
            attribute (str): The name of the attribute to threshold.
            threshold (float): The threshold value for the attribute.
            axis (str): The axis to view from ('x', 'y', or 'z'). Defaults to 'z'.
            return_array (bool): If True, returns the heatmap as a NumPy array. Defaults to False.

        Returns:
            Union[pv.ImageData, np.ndarray]: A 2D heatmap as a PyVista ImageData object or a NumPy array.
            """
        import pyvista as pv
        import numpy as np

        if attribute not in self.attributes:
            raise ValueError(f"Attribute '{attribute}' not found in the block model.")
        if axis not in {"x", "y", "z"}:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

        # Load the block model as PyVista ImageData
        mesh: pv.ImageData = self.to_pyvista(grid_type="image", attributes=[attribute])

        # Apply the threshold
        mesh.cell_data['count'] = mesh.cell_data[attribute] > threshold
        mesh.cell_data['count'] = mesh.cell_data['count'].astype(np.int8)

        # Reshape and sum along the specified axis, using Fortran order to match (x, y, z)
        reshaped_data = mesh.cell_data['count'].reshape(
            (mesh.dimensions[0] - 1, mesh.dimensions[1] - 1, mesh.dimensions[2] - 1), order='F')
        summed_data: Optional[np.ndarray] = None
        if axis == "z":
            summed_data = np.sum(reshaped_data, axis=2)
            new_dimensions = (mesh.dimensions[0], mesh.dimensions[1], 1)
        elif axis == "x":
            summed_data = np.sum(reshaped_data, axis=0)
            new_dimensions = (1, mesh.dimensions[1], mesh.dimensions[2])
        elif axis == "y":
            summed_data = np.sum(reshaped_data, axis=1)
            new_dimensions = (mesh.dimensions[0], 1, mesh.dimensions[2])
        if return_array:
            return summed_data.T  # Flip for correct orientation
        # Create a new ImageData object with correct dimensions
        new_mesh = pv.ImageData(dimensions=(summed_data.shape[0] + 1, summed_data.shape[1] + 1, 1),
                                spacing=self.geometry.block_size,
                                origin=self.geometry.corner)
        new_mesh.cell_data[attribute] = summed_data.ravel(order='F')
        return new_mesh

    def plot_heatmap(self, attribute: str, threshold: float, axis: str = "z",
                     title: Optional[str] = None) -> 'go.Figure':
        """
        Create a 2D heatmap plotly figure from a 3D block model at a specified attribute threshold.

        Args:
            attribute (str): The name of the attribute to threshold.
            threshold (float): The threshold value for the attribute.
            axis (str): The axis to view from ('x', 'y', or 'z'). Defaults to 'z'.
            title (Optional[str]): The title of the heatmap. If None, a default title is used.

        Returns:
            go.Figure: A Plotly figure containing the heatmap.
        """
        import plotly.express as px

        summed_data = self.create_heatmap_from_threshold(attribute, threshold, axis, return_array=True)

        data, labels, extents = self._get_heatmap_data(axis, summed_data)
        x_extent, y_extent = extents
        nx, ny = summed_data.shape[1], summed_data.shape[0]
        x_edges = np.linspace(x_extent[0], x_extent[1], nx + 1)
        y_edges = np.linspace(y_extent[0], y_extent[1], ny + 1)
        x_ticks = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_ticks = 0.5 * (y_edges[:-1] + y_edges[1:])
        fig = px.imshow(summed_data, origin="lower", x=x_ticks, y=y_ticks, color_continuous_scale='Viridis')

        fig.update_layout(title=f"Heatmap of {attribute}, thresholded at {threshold}",
                          xaxis_title=labels[0],
                          yaxis_title=labels[1], )
        fig.update_yaxes(range=[0, 1])
        if title is not None:
            fig.update_layout(title=title)
        return fig

    def _get_heatmap_data(self, axis, summed_data
                          ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], tuple[float, ...]]:
        if axis == "z":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[1])
            z = summed_data
            labels = "Easting", "Northing"
            extents = self.geometry.extents[0:2]  # x, y extents
        elif axis == "x":
            x = np.arange(summed_data.shape[1])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Northing", "RL"
            extents = self.geometry.extents[1], self.geometry.extents[2]
        elif axis == "y":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Easting", "RL"
            extents = self.geometry.extents[0], self.geometry.extents[1]
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
        return tuple([x, y, z]), labels, extents

    def plot(self, scalar: str,
             grid_type: typing.Literal["image", "structured", "unstructured"] = "image",
             threshold: bool = True, show_edges: bool = True,
             show_axes: bool = True, enable_picking: bool = False,
             picked_attributes: Optional[list[str]] = None) -> 'pv.Plotter':
        """Plot the block model using PyVista.

        Args:
            scalar: The name of the scalar attribute to visualize.
            grid_type: The type of grid to use for plotting. Options are "image", "structured", or "unstructured".
            threshold: The thresholding option for the mesh. If True, applies a threshold to the scalar values.
            show_edges: Show edges of the mesh.
            show_axes: Show the axes in the plot.
            enable_picking: If True, enables picking mode to interactively select cells in the plot.
            picked_attributes: A list of attributes that will be returned in picking mode. If None, all attributes are returned.

        Returns:

        """

        import pyvista as pv
        if scalar not in self.attributes:
            raise ValueError(f"Column '{scalar}' not found in the ParquetBlockModel.")

        # Create a PyVista plotter
        plotter = pv.Plotter()

        attributes = [scalar]
        if enable_picking:
            if picked_attributes is None:
                attributes = self.attributes
            else:
                attributes = picked_attributes
            if scalar not in attributes:
                attributes.append(scalar)
        mesh = self.to_pyvista(grid_type=grid_type, attributes=attributes)

        # Add a thresholded mesh to the plotter
        if threshold:
            plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, scalars=scalar, show_edges=show_edges)

        plotter.title = self.name
        if show_axes:
            plotter.show_axes()

        text_name = "cell_info_text"
        plotter.add_text("", position="upper_left", font_size=12, name=text_name)
        cell_centers = mesh.cell_centers().points  # shape: (n_cells, 3)

        if enable_picking:
            def cell_callback(picked_cell):
                if text_name in plotter.actors:
                    plotter.remove_actor(text_name)
                if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
                    if "vtkOriginalCellIds" in picked_cell.cell_data:
                        cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                        centroid = cell_centers[cell_id]  # numpy array of (x, y, z)
                        centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                        values = {attr: mesh.cell_data[attr][cell_id] for attr in attributes}
                        msg = f"Cell ID: {cell_id}, {centroid_str}, " + ", ".join(
                            f"{k}: {v}" for k, v in values.items())
                    else:
                        value = picked_cell.cell_data[scalar][0]
                        msg = f"Picked cell value: {scalar}: {value}"
                    plotter.add_text(msg, position="upper_left", font_size=12, name=text_name)
                else:
                    plotter.add_text("No valid cell picked.", position="upper_left", font_size=12, name=text_name)

            plotter.enable_cell_picking(callback=cell_callback, show_message=False, through=False)
            plotter.title = f"{self.name} - Press R and select a cell for attribute data"

        return plotter

    def read(self, columns: Optional[list[str]] = None,
             index: typing.Literal["xyz", "ijk", None] = "xyz",
             dense: bool = False) -> pd.DataFrame:
        """
        Read the Parquet file and return a DataFrame.

        Args:
            columns: List of column names to read. If None, all columns are read.
            index: The index type to use for the DataFrame. Options are "xyz" for centroid coordinates,
                "ijk" for block indices, or None for no index.
            dense: If True, reads the data as a dense grid. If False, reads the data as a sparse grid.

        Returns:
            pd.DataFrame: The DataFrame containing the block model data.
        """
        if columns is None:
            columns = self.columns
        df = pq.read_table(self.blockmodel_path, columns=columns).to_pandas()
        if index == "xyz":
            df.index = self.centroid_index
        elif index == "ijk":
            dense_index = self.geometry.to_ijk_multi_index()
            df.index = dense_index
        else:
            raise ValueError("index_type must be 'xyz' or 'ijk'")
        if dense:
            dense_index = self.geometry.to_multi_index() if index == "xyz" else self.geometry.to_ijk_multi_index()
            df = df.reindex(dense_index)

        # if index:
        #     df.index = self.centroid_index
        #     if dense:
        #         dense_index = self.geometry.to_multi_index()
        #         if len(df) == len(dense_index):
        #             assert df.index.equals(dense_index)
        #         df = df.reindex(dense_index)
        return df

    def to_pyvista(self, grid_type: typing.Literal["image", "structured", "unstructured"] = "structured",
                   attributes: Optional[list[str]] = None
                   ) -> Union['pv.ImageData', 'pv.StructuredGrid', 'pv.UnstructuredGrid']:

        if attributes is None:
            attributes = self.attributes

        if grid_type == "image":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
            grid = df_to_pv_image_data(df=self.read(columns=attributes, dense=False),
                                       geometry=self.geometry)
        elif grid_type == "structured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_structured_grid
            grid = df_to_pv_structured_grid(df=self.read(columns=attributes, dense=False),
                                            validate_block_size=True)
        elif grid_type == "unstructured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_unstructured_grid
            grid = df_to_pv_unstructured_grid(df=self.read(columns=attributes, dense=False),
                                              block_size=self.geometry.block_size,
                                              validate_block_size=True)
        else:
            raise ValueError(f"Invalid grid type: {grid_type}. "
                             "Choose from 'image', 'structured', or 'unstructured'.")

        return grid

    @staticmethod
    def _validate_geometry(filepath: Path, geometry: Optional[RegularGeometry] = None) -> None:
        """
        Validates the geometry of a Parquet file by checking if the index (centroid) columns are present
        and have valid values. For sparse models, ensures centroids are a subset of the dense grid.

        Args:
            filepath (Path): Path to the Parquet file.
            geometry (RegularGeometry, optional): The geometry of the block model. If None, it will be derived from
             the Parquet file.

        Raises:
            ValueError: If any index column is missing, contains invalid values, or centroids are not valid.
        """
        index_columns = ['x', 'y', 'z']
        columns = pq.read_schema(filepath).names
        if not all(col in columns for col in index_columns):
            raise ValueError(f"Missing index columns in the dataset: {', '.join(index_columns)}")

        table = pq.read_table(filepath, columns=index_columns)
        for col in index_columns:
            if table[col].null_count > 0:
                raise ValueError(f"Column '{col}' contains NaN values, which is not allowed in the index columns.")

        # Ensure arrays are of the same length
        centroids = table.to_pandas()
        if isinstance(centroids.index, pd.MultiIndex) and centroids.index.names == ['x', 'y', 'z']:
            x_values = centroids.index.get_level_values('x').values
            y_values = centroids.index.get_level_values('y').values
            z_values = centroids.index.get_level_values('z').values
        else:
            x_values = centroids['x'].values
            y_values = centroids['y'].values
            z_values = centroids['z'].values

        if len(x_values) != len(y_values) or len(y_values) != len(z_values):
            raise ValueError("Centroid arrays (x, y, z) must have the same length.")

        if geometry is None:
            geometry = RegularGeometry.from_parquet(filepath)

        dense_index = geometry.to_multi_index()
        sparse_index = pd.MultiIndex.from_arrays([x_values, y_values, z_values])

        # For sparse models, ensure centroids are a subset of the dense grid
        if not sparse_index.isin(dense_index).all():
            raise ValueError("Sparse centroids must be a subset of the dense grid.")

        logging.info(f"Geometry validation completed successfully for {filepath}.")

    @staticmethod
    def _validate_and_load_data(df, expected_num_blocks):
        required_cols = {'x', 'y', 'z'}
        if not required_cols.issubset(df.columns):
            if len(df) == expected_num_blocks:
                warnings.warn("Data loaded without x, y, z columns. "
                              "Order is assumed to match the block model geometry.")
            else:
                raise ValueError("Data missing x, y, z and row count does not match block model.")
        return df

    def to_dense_parquet(self, filepath: Path,
                         chunk_size: int = 100_000, show_progress: bool = False) -> None:
        """
        Save the block model to a Parquet file.

        This method saves the block model as a Parquet file by chunk. If `dense` is True, it saves the block model as a dense grid,
        Args:
            filepath (Path): The file path where the Parquet file will be saved.
            chunk_size (int): The number of blocks to save in each chunk. Defaults to 100_000.
            show_progress (bool): If True, show a progress bar. Defaults to False.
        """
        columns = self.columns
        dense_index = self.geometry.to_multi_index()
        parquet_file = pq.ParquetFile(self.blockmodel_path)
        total_rows = parquet_file.metadata.num_rows
        total_batches = max(math.ceil(total_rows / chunk_size), 1)

        progress = tqdm(total=total_batches, desc="Exporting", disable=not show_progress) if show_progress else None

        with atomic_output_file(filepath) as tmp_path:
            writer = None
            try:
                for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
                    df = pa.Table.from_batches([batch]).to_pandas()
                    df = df.reindex(dense_index)
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
                    if progress:
                        progress.update(1)
            finally:
                if writer is not None:
                    writer.close()
                if progress:
                    progress.close()
