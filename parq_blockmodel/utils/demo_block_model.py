from pathlib import Path
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

    - c_index: A zero based index in C-style order (row-major). The order returned when sorting by x, y, z.
    - f_index: A zero based index in Fortran-style order (column-major). The order returned when sorting by z, y, x.
    - depth: The depth of each block, calculated as the distance from the surface (maximum z coordinate).

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

    c_index = np.arange(num_blocks)
    f_index = np.arange(num_blocks).reshape(shape, order='C').ravel(order='F')

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
        'c_index': c_index
    })

    df.set_index(keys=['x', 'y', 'z'], inplace=True)
    df['f_index'] = f_index
    df['depth'] = surface_rl - zz_flat_c

    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath
    return df
