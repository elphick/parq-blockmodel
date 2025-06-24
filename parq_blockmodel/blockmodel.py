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

from parq_blockmodel.utils import create_demo_blockmodel, rotation_to_axis_orientation
from parq_blockmodel.utils.pyvista_utils import df_to_pv_structured_grid, df_to_pv_unstructured_grid
from parq_tools.lazy_parquet import LazyParquetDataFrame
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from parq_tools import ParquetProfileReport
from parq_tools.utils import atomic_output_file

from parq_blockmodel.geometry import RegularGeometry

Point = Union[tuple[float, float, float], list[float, float, float]]
Triple = Union[tuple[float, float, float], list[float, float, float]]

if typing.TYPE_CHECKING:
    import pyvista as pv  # type: ignore[import]

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
    def from_parquet(cls, parquet_path: Path, overwrite: bool = False,
                     azimuth: float = 0.0, dip: float = 0.0, plunge: float = 0.0) -> "ParquetBlockModel":
        """ Create a ParquetBlockModel instance from a Parquet file.

        Args:
            parquet_path (Path): The path to the Parquet file.
            overwrite (bool): If True, allows overwriting an existing ParquetBlockModel file. Defaults to False.
            azimuth (float): The azimuth angle in degrees for rotation. Defaults to 0.0.
            dip (float): The dip angle in degrees for rotation. Defaults to 0.0.
            plunge (float): The plunge angle in degrees for rotation. Defaults to 0.0.

        """
        if parquet_path.suffixes[-2:] == [".pbm", ".parquet"]:
            if not overwrite:
                raise ValueError(
                    f"File {parquet_path} appears to be a compliant ParquetBlockModel file. "
                    f"Use the constructor directly, or pass overwrite=True to allow mutation."
                )

        geometry: Optional[RegularGeometry] = None
        if azimuth != 0.0 or dip != 0.0 or plunge != 0.0:
            # If rotation angles are provided, create a geometry with the specified orientation
            axis_u, axis_v, axis_w = rotation_to_axis_orientation(azimuth=azimuth, dip=dip, plunge=plunge)
            geometry = RegularGeometry.from_parquet(filepath=parquet_path)
            geometry.axis_u = axis_u
            geometry.axis_v = axis_v
            geometry.axis_w = axis_w

        cls._validate_geometry(parquet_path)

        new_filepath = shutil.copy(parquet_path, parquet_path.resolve().with_suffix(".pbm.parquet"))
        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def create_demo_block_model(cls, filename: Path,
                                shape=(3, 3, 3),
                                block_size=(1, 1, 1),
                                corner=(-0.5, -0.5, -0.5),
                                azimuth: float = 0.0,
                                dip: float = 0.0,
                                plunge: float = 0.0) -> "ParquetBlockModel":
        """
        Create a demo block model with specified parameters.

        Args:
            filename (Path): The file path where the Parquet file will be saved.
            shape (tuple): The shape of the block model.
            block_size (tuple): The size of each block.
            corner (tuple): The coordinates of the corner of the block model.
            azimuth (float): The azimuth angle in degrees for rotation.
            dip (float): The dip angle in degrees for rotation.
            plunge (float): The plunge angle in degrees for rotation.

        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel with demo data.
        """
        create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner,
                               azimuth=azimuth, dip=dip, plunge=plunge,
                               parquet_filepath=filename)
        # get the orientation of the axes
        axis_u, axis_v, axis_w = rotation_to_axis_orientation(azimuth=azimuth, dip=dip, plunge=plunge)
        # create geometry that aligns with the demo block model
        geometry = RegularGeometry(block_size=block_size, corner=corner, shape=shape,
                                   axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

        if not geometry.is_rotated:
            cls._validate_geometry(filename)

        new_filepath = shutil.copy(filename, filename.resolve().with_suffix(".pbm.parquet"))

        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_geometry(cls, geometry: RegularGeometry, path: Path, name: Optional[str] = None) -> "ParquetBlockModel":
        centroids_df = geometry.to_dataframe()
        centroids_df.to_parquet(path, index=False)
        return cls(blockmodel_path=path, name=name, geometry=geometry)

    def create_report(self, columns: Optional[list[str]] = None,
                      column_batch_size: int = 10,
                      show_progress: bool = True, open_in_browser: bool = False) -> Path:
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

    def plot(self, scalar: str, threshold: bool = True, show_edges: bool = True,
             show_axes: bool = True, enable_picking: bool = False, picked_attributes: Optional[list[str]] = None) -> 'pv.Plotter':
        """Plot the block model using PyVista.

        Args:
            scalar: The name of the scalar attribute to visualize.
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
        mesh = self.to_pyvista(attributes=attributes)

        # Add a thresholded mesh to the plotter
        if threshold:
            plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, scalars=scalar, show_edges=show_edges)

        plotter.title = self.name
        if show_axes:
            plotter.show_axes()

        text_name = "cell_info_text"
        plotter.add_text("", position="lower_left", font_size=12, name=text_name)

        if enable_picking:
            def cell_callback(picked_cell):
                if text_name in plotter.actors:
                    plotter.remove_actor(text_name)
                if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
                    if "vtkOriginalCellIds" in picked_cell.cell_data:
                        cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                        values = {attr: mesh.cell_data[attr][cell_id] for attr in attributes}
                        msg = f"Cell ID: {cell_id}, " + ", ".join(f"{k}: {v}" for k, v in values.items())
                    else:
                        value = picked_cell.cell_data[scalar][0]
                        msg = f"Picked cell value: {scalar}: {value}"
                    plotter.add_text(msg, position="lower_left", font_size=12, name=text_name)
                else:
                    plotter.add_text("No valid cell picked.", position="lower_left", font_size=12, name=text_name)

            plotter.enable_cell_picking(callback=cell_callback, show_message=True, through=False)

        return plotter

    def read(self, columns: Optional[list[str]] = None,
             with_index: bool = True, dense: bool = False) -> pd.DataFrame:
        """
        Read the Parquet file and return a DataFrame.

        Args:
            columns: List of column names to read. If None, all columns are read.
            with_index: If True, includes the index ('x', 'y', 'z') in the DataFrame.
            dense: If True, reads the data as a dense grid. If False, reads the data as a sparse grid.

        Returns:
            pd.DataFrame: The DataFrame containing the block model data.
        """
        if columns is None:
            columns = self.columns
        df = pq.read_table(self.blockmodel_path, columns=columns).to_pandas()
        if with_index:
            df.index = self.centroid_index
            if dense:
                dense_index = self.geometry.to_multi_index()
                if len(df) == len(dense_index):
                    assert df.index.equals(dense_index)
                df = df.reindex(dense_index)
        return df

    def to_pyvista(self, attributes: Optional[list[str]] = None) -> 'pv.ImageData':

        if attributes is None:
            attributes = self.attributes

        grid = self.geometry.to_pyvista()
        df = self.read(columns=attributes, with_index=False, dense=True)
        df['f_order'] = self.index_f
        df = df.sort_values('f_order')
        df = df.drop(columns='f_order')

        for col in attributes:
            grid.cell_data[col] = df[col].values

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
