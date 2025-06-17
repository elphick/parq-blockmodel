import logging
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np

def validate_geometry(filepath: Path) -> None:
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
        raise ValueError("The geometry is not regular. At least two unique values are required in each index column.")

    def is_regular_spacing(values, tol=1e-8):
        diffs = np.diff(values)
        return np.all(np.abs(diffs - diffs[0]) < tol)

    if not (is_regular_spacing(x_values) and is_regular_spacing(y_values) and is_regular_spacing(z_values)):
        raise ValueError("The geometry is not regular. The index columns must be evenly spaced (regular grid) in x, y, and z.")

    logging.info(f"Geometry validation completed successfully for {filepath}.")