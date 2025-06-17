from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_blockmodel(shape: tuple[int, int, int] = (3, 3, 3),
                           block_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
                           corner: tuple[float, float, float] = (0.0, 0.0, 0.0),
                           parquet_filepath: Path = None
                           ) -> pd.DataFrame | Path:
    """
    Create a demo blockmodel DataFrame or Parquet file.

    Args:
        shape: Shape of the block model (x, y, z).
        block_size: Size of each block (x, y, z).
        corner: The lower left (minimum) corner of the block model.
        parquet_filepath: If provided, save the DataFrame to this Parquet file and return the file path.
    Returns:
        DataFrame or Parquet file path (Path)
    """

    num_blocks = np.prod(shape)

    # Generate the coordinates for the block model
    x_coords = np.arange(corner[0] + block_size[0] / 2, corner[0] + shape[0] * block_size[0], block_size[0])
    y_coords = np.arange(corner[1] + block_size[1] / 2, corner[1] + shape[1] * block_size[1], block_size[1])
    z_coords = np.arange(corner[2] + block_size[2] / 2, corner[2] + shape[2] * block_size[2], block_size[2])

    # Create a meshgrid of coordinates
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Flatten the coordinates
    xx_flat_c = xx.ravel(order='C')
    yy_flat_c = yy.ravel(order='C')
    zz_flat_c = zz.ravel(order='C')

    # Create the attributes
    c_order_xyz = np.arange(num_blocks)

    # assume the surface of the highest block is the topo surface
    surface_rl = np.max(zz_flat_c) + block_size[2] / 2

    # Create the DataFrame
    df = pd.DataFrame({
        'x': xx_flat_c,
        'y': yy_flat_c,
        'z': zz_flat_c,
        'c_style_xyz': c_order_xyz})

    # Set the index to x, y, z
    df.set_index(keys=['x', 'y', 'z'], inplace=True)
    df.sort_index(level=['x', 'y', 'z'], inplace=True)
    df.sort_index(level=['z', 'y', 'x'], inplace=True)
    df['f_style_zyx'] = c_order_xyz
    df.sort_index(level=['x', 'y', 'z'], inplace=True)

    df['depth'] = surface_rl - zz_flat_c

    # Check the ordering - confirm that the c_order_xyz and f_order_zyx columns are in the correct order
    assert np.array_equal(df.sort_index(level=['x', 'y', 'z'])['c_style_xyz'].values, np.arange(num_blocks))
    assert np.array_equal(df.sort_index(level=['z', 'y', 'x'])['f_style_zyx'].values, np.arange(num_blocks))

    # TODO: remove this temp code
    # drop a single record to test sparse input
    # df = df.drop(df.index[-1])

    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath
    return df
