"""
blockmodel.py

This module defines the ParquetBlockModel class, which represents a block model stored in a Parquet file.

Main API:

- ParquetBlockModel: Class for representing a block model stored in a Parquet file.

"""
import logging
import math
import shutil
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.core.dtypes.common import is_sparse
from parq_tools.lazy_parquet import LazyParquetDataFrame
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from parq_tools import ParquetProfileReport
from parq_tools.block_models.geometry import RegularGeometry
from parq_tools.block_models.utils.geometry import validate_geometry
from parq_tools.block_models.utils.pyvista_utils import df_to_pv_structured_grid, df_to_pv_unstructured_grid
from parq_tools.utils import atomic_output_file
from parq_tools.utils.progress import get_batch_progress_bar

Point = Union[tuple[float, float, float], list[float, float, float]]
Triple = Union[tuple[float, float, float], list[float, float, float]]


class ParquetBlockModel:
    """
    A class to represent a **regular** Parquet block model.

    Block ordering is c-style, ordered by x, y, z coordinates.

    Attributes:
        name (str): The name of the block model.
        path (Path): The file path to the Parquet file.
    """

    def __init__(self, name: str, path: Path):
        self.name: str = name
        self.path: Path = path
        self.pf: ParquetFile = ParquetFile(path)
        self.report_path: Optional[Path] = None
        self.geometry: Optional[RegularGeometry] = None
        if path.exists():
            # validate_geometry(path)  # TODO: reinstate this validation
            self.geometry = RegularGeometry.from_parquet(self.path)
        self.data: LazyParquetDataFrame = LazyParquetDataFrame(self.path)
        self.columns: list[str] = pq.read_schema(self.path).names
        self._centroid_index: Optional[pd.MultiIndex] = None
        self.attributes: list[str] = [col for col in self.columns if col not in ["x", "y", "z"]]
        self._extract_column_dtypes()
        self._logger = logging.getLogger(__name__)

        if self.is_sparse:
            if not self.validate_sparse():
                raise ValueError("The sparse ParquetBlockModel is invalid. "
                                 "Sparse centroids must be a subset of the dense grid.")

    def __repr__(self):
        return f"ParquetBlockModel(name={self.name}, path={self.path})"

    def _extract_column_dtypes(self):
        self.column_dtypes: dict[str, np.dtype] = {}
        self._column_categorical_ordered: dict[str, bool] = {}
        schema = pq.read_schema(self.path)
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
            # Read the Parquet file to get the index, whether file was written by pandas or not
            centroid_cols = ["x", "y", "z"]
            centroids: pd.DataFrame = pq.read_table(self.path, columns=centroid_cols).to_pandas()

            if centroids.index.names == centroid_cols:
                index = centroids.index
            else:
                if centroids.empty:
                    raise ValueError("Parquet file is empty or does not contain valid centroid data.")
                index = centroids.set_index(["x", "y", "z"]).index
            if not index.is_unique:
                raise ValueError("The index of the Parquet file is not unique. "
                                 "Ensure that the centroid coordinates (x, y, z) are unique.")
            if not index.is_monotonic_increasing:
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

    def validate_sparse(self) -> bool:
        dense_index = self.geometry.to_multi_index()
        # All sparse centroids must be in the dense grid
        return self.centroid_index.isin(dense_index).all()

    @classmethod
    def from_parquet(cls, parquet_path: Path) -> "ParquetBlockModel":
        """
        Create a ParquetBlockModel instance from a Parquet file path.
        The file must contain columns 'x', 'y', 'z', representing the cell centroids.

        Args:
            parquet_path (Path): The file path to the Parquet file.

        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel.
        """

        # create a copy and rename the file to avoid issues with the original file
        new_filepath: Path = shutil.copy(parquet_path, parquet_path.resolve().with_suffix(".pbm.parquet"))
        return cls(name=parquet_path.stem, path=new_filepath)

    @classmethod
    def create_demo_block_model(cls, filename: Path,
                                shape=(3, 3, 3),
                                block_size=(1, 1, 1),
                                corner=(-0.5, -0.5, -0.5)) -> "ParquetBlockModel":
        """
        Create a demo block model with specified parameters.

        Args:
            filename (Path): The file path where the Parquet file will be saved.
            shape (tuple): The shape of the block model.
            block_size (tuple): The size of each block.
            corner (tuple): The coordinates of the corner of the block model.

        Returns:
            ParquetBlockModel: An instance of ParquetBlockModel with demo data.
        """
        from parq_tools.block_models.utils.demo_block_model import create_demo_blockmodel
        create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner,
                               parquet_filepath=filename)

        return cls.from_parquet(filename)

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
        report: ParquetProfileReport = ParquetProfileReport(self.path, columns=columns,
                                                            batch_size=column_batch_size,
                                                            show_progress=show_progress).profile()
        if open_in_browser:
            report.show(notebook=False)
        if not columns:
            self.report_path = self.path.with_suffix('.html')
        return self.report_path

    def plot(self, scalar: str, threshold: bool = True, show_edges: bool = True,
             show_axes: bool = True) -> 'pv.Plotter':
        import pyvista as pv
        if scalar not in self.attributes:
            raise ValueError(f"Column '{scalar}' not found in the ParquetBlockModel.")

        # Create a PyVista plotter
        plotter = pv.Plotter()

        mesh = self.get_blocks(attributes=[scalar])

        # Add a thresholded mesh to the plotter
        if threshold:
            plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, scalars=scalar, show_edges=show_edges)

        plotter.title = self.name
        if show_axes:
            plotter.show_axes()

        return plotter

    def read(self, columns: Optional[list[str]] = None, with_index: bool=True) -> pd.DataFrame:
        """
        Read the Parquet file and return a DataFrame.

        Args:
            columns: List of column names to read. If None, all columns are read.
            with_index: If True, includes the index ('x', 'y', 'z') in the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the block model data.
        """
        if columns is None:
            columns = self.columns
        df = pq.read_table(self.path, columns=columns).to_pandas()
        if with_index:
            df.index = self.centroid_index
        return df

    def get_blocks(self, attributes: Optional[list[str]] = None) -> Union['pv.StructuredGrid', 'pv.UnstructuredGrid']:

        if attributes is None:
            attributes = self.attributes
        df = self.read(columns=attributes)
        if self.is_sparse:
            df = df.reindex(self.centroid_index)

        try:
            # Attempt to create a regular grid
            grid = df_to_pv_structured_grid(df)
            self._logger.debug("Created a pv.StructuredGrid.")
        except ValueError:
            # If it fails, create an irregular grid
            grid = df_to_pv_unstructured_grid(df, block_size=self.geometry.block_size)
            self._logger.debug("Created a pv.UnstructuredGrid.")
        return grid

    def _validate_geometry(filepath: Path) -> RegularGeometry:
        """
        Validates the geometry of a Parquet file by checking if the index (centroid) columns are present
        and have valid values.

        Args:
            filepath (Path): Path to the Parquet file.

        Raises:
            ValueError: If any index column is missing or contains invalid values.
        """
        index_columns = ['x', 'y', 'z']

        columns = pq.read_schema(filepath).names
        if not all(col in columns for col in index_columns):
            raise ValueError(f"Missing index columns in the dataset: {', '.join(index_columns)}")

        # Read the Parquet file to check for NaN values in index columns
        table = pq.read_table(filepath, columns=index_columns)
        for col in index_columns:
            if table[col].null_count > 0:
                raise ValueError(f"Column '{col}' contains NaN values, which is not allowed in the index columns.")

        # check the geometry is regular
        x_values = np.sort(table['x'].to_pandas().unique())
        y_values = np.sort(table['y'].to_pandas().unique())
        z_values = np.sort(table['z'].to_pandas().unique())
        if len(x_values) < 2 or len(y_values) < 2 or len(z_values) < 2:
            raise ValueError(
                "The geometry is not regular. At least two unique values are required in each index column.")

        def is_regular_spacing(values, tol=1e-8):
            diffs = np.diff(values)
            return np.all(np.abs(diffs - diffs[0]) < tol)

        if not (is_regular_spacing(x_values) and is_regular_spacing(y_values) and is_regular_spacing(z_values)):
            raise ValueError(
                "The geometry is not regular. The index columns must be evenly spaced (regular grid) in x, y, and z.")

        logging.info(f"Geometry validation completed successfully for {filepath}.")

    import math
    from tqdm import tqdm  # Add this import at the top

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
        parquet_file = pq.ParquetFile(self.path)
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