
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interpn

from parq_blockmodel.reblocking.conversion import to_numeric


def upsample_attributes(attributes, fx, fy, fz, interpolation_config):
    """
    Upsample a 3D block model to a finer grid with multiple attributes using specified interpolation methods.

    Parameters:
    - attributes: dict of 3D arrays (NumPy or Pandas extension types)
    - fx, fy, fz: upsampling factors along x, y, z axes
    - interpolation_config: dict specifying interpolation method for each attribute

    Example:
        attributes = {
            'grade': grade,
            'density': density,
            'dry_mass': dry_mass,
            'volume': volume,
            'rock_type': rock_types
        }

        interpolation_config = {
            'grade': 'linear',
            'density': 'linear',
            'dry_mass': 'linear',
            'volume': 'linear',
            'rock_type': 'nearest'
        }

        upsampled = upsample_attributes(attributes, fx=2, fy=2, fz=2, interpolation_config=interpolation_config)

    Returns:
    - dict of upsampled 3D arrays
    """

    first_arr = next(iter(attributes.values()))
    nx, ny, nz = first_arr.shape
    result = {}

    for attr, arr in attributes.items():
        method = interpolation_config.get(attr, 'linear')
        new_nx, new_ny, new_nz = nx * fx, ny * fy, nz * fz

        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        new_x = np.linspace(0, nx - 1, new_nx)
        new_y = np.linspace(0, ny - 1, new_ny)
        new_z = np.linspace(0, nz - 1, new_nz)
        grid = np.meshgrid(new_x, new_y, new_z, indexing='ij')
        points = np.stack([g.ravel() for g in grid], axis=-1)

        # Use conversion utilities
        arr_numeric, restore_fn = to_numeric(arr)
        interpolator = RegularGridInterpolator((x, y, z), arr_numeric, method=method if arr_numeric.dtype.kind == 'f' else 'nearest', bounds_error=False, fill_value=np.nan)
        upsampled_numeric = interpolator(points).reshape(new_nx, new_ny, new_nz)
        result[attr] = restore_fn(upsampled_numeric)

    return result