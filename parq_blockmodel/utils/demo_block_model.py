from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from parq_blockmodel.utils.geometry_utils import rotate_points


def create_demo_blockmodel(shape: tuple[int, int, int] = (3, 3, 3),
                           block_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
                           corner: tuple[float, float, float] = (0.0, 0.0, 0.0),
                           azimuth: float = 0.0,
                           dip: float = 0.0,
                           plunge: float = 0.0,
                           parquet_filepath: Path = None
                           ) -> pd.DataFrame | Path:
    """Create a demo blockmodel DataFrame or Parquet file.

    The model contains block coordinates, indices, and depth information.

    - index_c: A zero based index in C-style order (row-major). The order returned when sorting by x, y, z.
    - index_f: A zero based index in Fortran-style order (column-major). The order returned when sorting by z, y, x.
    - depth: The depth of each block, calculated as the distance from the surface (maximum z coordinate).
    - depth_category: A categorical attribute that cuts the depth into two bins: 'shallow' and 'deep'.

    Args:
        shape: Shape of the block model (nx, ny, nz).
        block_size: Size of each block (dx, dy, dz).
        corner: The lower left (minimum) corner of the block model.
        azimuth: Azimuth angle in degrees.
        dip: Dip angle in degrees.
        plunge: Plunge angle in degrees.
        parquet_filepath: If provided, save the DataFrame to this Parquet file and return the file path.

    Returns:
        pd.DataFrame if parquet_filepath is None, else Path to the Parquet file.
    """
    num_blocks = np.prod(shape)

    # Generate the coordinates for the block model
    x_coords = np.arange(corner[0] + block_size[0] / 2, corner[0] + shape[0] * block_size[0], block_size[0])
    y_coords = np.arange(corner[1] + block_size[1] / 2, corner[1] + shape[1] * block_size[1], block_size[1])
    z_coords = np.arange(corner[2] + block_size[2] / 2, corner[2] + shape[2] * block_size[2], block_size[2])

    # Create a meshgrid of coordinates
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    coords = np.stack([xx.ravel(order='C'), yy.ravel(order='C'), zz.ravel(order='C')], axis=-1)

    index_c = np.arange(num_blocks)
    index_f = np.arange(num_blocks).reshape(shape, order='C').ravel(order='F')

    if any(angle != 0.0 for angle in (azimuth, dip, plunge)):
        rotated = rotate_points(points=coords, azimuth=azimuth, dip=dip, plunge=plunge)
        xx_flat_c, yy_flat_c, zz_flat_c = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    else:
        xx_flat_c, yy_flat_c, zz_flat_c = coords[:, 0], coords[:, 1], coords[:, 2]

    surface_rl = np.max(zz_flat_c) + block_size[2] / 2

    df = pd.DataFrame({
        'x': xx_flat_c,
        'y': yy_flat_c,
        'z': zz_flat_c,
        'index_c': index_c
    })

    df.set_index(keys=['x', 'y', 'z'], inplace=True)
    df['index_f'] = index_f
    df['depth'] = surface_rl - zz_flat_c
    df['depth_category'] = pd.cut(
        df['depth'],
        bins=2,
        labels=['shallow', 'deep'],
        include_lowest=True
    ).astype('category')
    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath
    return df


def add_gradient_ellipsoid_grade(
        df: pd.DataFrame,
        center: Iterable,
        radii: Iterable,
        grade_min: float = 5.0,
        grade_max: float = 65.0,
        bearing: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0,
        column_name: str = 'grade',
        noise_std: float = 0.1
) -> pd.DataFrame:
    """Add a gradient ellipsoid grade to the block model DataFrame."""
    if df.index.names == ['x', 'y', 'z']:
        coords = np.array(df.index.tolist()) - np.array(center)
    else:
        coords = df[['x', 'y', 'z']].values - np.array(center)
    if any([bearing, dip, plunge]):
        from parq_blockmodel.utils.geometry_utils import rotate_points
        coords = rotate_points(coords, bearing, dip, plunge)
    norm = (
            (coords[:, 0] / radii[0]) ** 2 +
            (coords[:, 1] / radii[1]) ** 2 +
            (coords[:, 2] / radii[2]) ** 2
    )
    inside = norm <= 1
    grad = np.full_like(norm, grade_min, dtype=float)
    grad[inside] = grade_max - (grade_max - grade_min) * np.sqrt(norm[inside])
    if noise_std > 0.0:
        grad += np.random.normal(0, noise_std, size=grad.shape)
    df[column_name] = grad
    return df


def create_toy_blockmodel(
        shape=(20, 15, 10),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
        axis_azimuth=0.0,
        axis_dip=0.0,
        axis_plunge=0.0,
        deposit_bearing=20.0,
        deposit_dip=30.0,
        deposit_plunge=10.0,
        grade_name='grade',
        grade_min=50.0,
        grade_max=65.0,
        deposit_center=(10.0, 7.5, 5.0),
        deposit_radii=(8.0, 5.0, 3.0),
        noise_std: float = 0.0,
        parquet_filepath: Path = None
) -> pd.DataFrame | Path:
    """Create a toy blockmodel with a gradient ellipsoid grade.
    Args:
        shape: Shape of the block model (nx, ny, nz).
        block_size: Size of each block (dx, dy, dz).
        corner: The lower left (minimum) corner of the block model.
        axis_azimuth: The azimuth angle of the block model axis in degrees.
        axis_dip: The dip angle of the block model axis in degrees.
        axis_plunge: The plunge angle of the block model axis in degrees.
        deposit_bearing: The azimuth angle of the deposit in degrees.
        deposit_dip: The dip angle of the deposit in degrees.
        deposit_plunge: The plunge angle of the deposit in degrees.
        grade_name: The name of the column to store the grade values.
        grade_min: The minimum grade value.
        grade_max: The maximum grade value.
        deposit_center: The center of the deposit (x, y, z).
        deposit_radii: The radii of the deposit (rx, ry, rz).
        noise_std: Standard deviation of the noise to add to the grade values.
        parquet_filepath: The file path to save the DataFrame as a Parquet file. If None, returns a DataFrame.

    Returns:
        pd.DataFrame if parquet_filepath is None, else Path to the Parquet file.
    """
    df = create_demo_blockmodel(shape, block_size, corner, axis_azimuth, axis_dip, axis_plunge)
    df = add_gradient_ellipsoid_grade(df=df,
                                      center=deposit_center,
                                      radii=deposit_radii,
                                      grade_min=grade_min,
                                      grade_max=grade_max,
                                      bearing=deposit_bearing,
                                      dip=deposit_dip,
                                      plunge=deposit_plunge,
                                      column_name=grade_name,
                                      noise_std=noise_std)
    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath
    return df
