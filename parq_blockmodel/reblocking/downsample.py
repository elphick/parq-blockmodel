import numpy as np
from collections import Counter

import pandas as pd

from parq_blockmodel.reblocking.conversion import to_numeric


def downsample_attributes(attributes, fx, fy, fz, aggregation_config):
    """
    Downsample a 3D block model to a coarser grid with multiple attributes using specified aggregation methods.

    Parameters:
    - attributes: dict of 3D numpy arrays, e.g., {'grade': ..., 'density': ..., 'rock_type': ...}
    - fx, fy, fz: downsampling factors along x, y, z axes
    - aggregation_config: dict specifying aggregation method and optional basis for each attribute

    The `fill_ratio` key in aggregation_config indicates that the aggregation should be normalized
    by the fill ratio attribute specified.  This is useful when some blocks may be partially filled after reblocking.

    The `replace` key is used in cased where the reblocking return 0 for missing data, but the user wants to convert
    these back to NaN or another missing value indicator.

    Example:
        attributes = {
            'grade': grade,
            'density': density,
            'dry_mass': dry_mass,
            'volume': volume,
            'rock_type': rock_types
            'mass_with_missing': mass_with_missing
        }

        aggregation_config = {
            'grade': {'method': 'weighted_mean', 'basis': 'dry_mass'},
            'density': {'method': 'weighted_mean', 'basis': 'volume', fill_ratio: fill_ratio},
            'dry_mass': {'method': 'sum', fill_ratio: fill_ratio},
            'volume': {'method': 'sum'},
            'rock_type': {'method': 'mode'}
            'mass_with_missing': {'method': 'sum', 'replace': { 0: nan }}
        }

        downsampled = downsample_attributes(attributes, fx=2, fy=2, fz=2, aggregation_config=aggregation_config)

    NOTE: for regular models only the averaging f density can be simplified:
    From: 'density': {'method': 'weighted_mean', 'basis': 'volume', fill_ratio: fill_ratio},
    To: 'density': {'method': 'mean'},


    Returns:
    - dict of downsampled 3D arrays
    """

    # Validate downsampling factors
    if not all(isinstance(f, int) and f > 0 for f in (fx, fy, fz)):
        raise ValueError("Downsampling factors fx, fy, fz must be positive integers.")

    # Get original shape from one of the attributes
    shape = next(iter(attributes.values())).shape
    sx, sy, sz = shape

    # Validate divisibility
    if sx % fx != 0 or sy % fy != 0 or sz % fz != 0:
        raise ValueError("Input dimensions must be divisible by the downsampling factors.")

    # Prepare output dictionary
    result = {}

    # Precompute reshaping and transposing axes
    new_shape = (sx // fx, fx, sy // fy, fy, sz // fz, fz)
    transpose_axes = (0, 2, 4, 1, 3, 5)

    # Manage fill_ratio if needed
    fill_ratio_vars: set[str] = set()
    fill_ratio_var: str | None = None
    fill_ratio = None
    for attr, config in aggregation_config.items():
        if config.get('fill_ratio'): # if the fill_ratio key exists...
            fill_ratio_vars.add(config.get('fill_ratio'))
    if fill_ratio_vars:
        if len(fill_ratio_vars) > 1:
            raise ValueError(f"Only one fill_ratio is supported, but multiple are supplied: {fill_ratio_vars}")
        fill_ratio_var = fill_ratio_vars.pop()
        reshaped_fill_ratio = attributes[fill_ratio_var].reshape(new_shape).transpose(transpose_axes)
        fill_ratio = np.nanmean(reshaped_fill_ratio, axis=(3, 4, 5))

    # Precompute basis arrays if needed
    basis_arrays = {}
    for attr, config in aggregation_config.items():
        if config.get('method') != 'weighted_mean':
            continue
        basis_name = config.get('basis', config.get('weight'))
        if basis_name is not None:
            basis_arrays[attr] = attributes[basis_name]

    for attr, data in attributes.items():
        config = aggregation_config.get(attr, {'method': 'mean'})
        method = config['method']
        fill = config.get('fill_ratio')
        replacement = config.get('replace', {})
        if not isinstance(replacement, dict):
            raise ValueError("The replace value must be a dictionary of the form {old_value: new_value}")

        if method == 'sum':
            data_numeric, restore_fn = to_numeric(data, operation="downsampling", attribute=attr)
            reshaped = data_numeric.reshape(new_shape).transpose(transpose_axes)
            if fill:
                if fill_ratio is None:
                    raise ValueError(f"fill_ratio '{fill}' was requested but is not configured")
                aggregated = np.nansum(reshaped, axis=(3, 4, 5), dtype=np.float64) / fill_ratio
            else:
                aggregated = np.nansum(reshaped, axis=(3, 4, 5), dtype=np.float64)
            result[attr] = restore_fn(aggregated)

        elif method == 'mean':
            data_numeric, restore_fn = to_numeric(data, operation="downsampling", attribute=attr)
            reshaped = data_numeric.reshape(new_shape).transpose(transpose_axes)
            aggregated = np.nanmean(reshaped, axis=(3, 4, 5), dtype=np.float64)
            result[attr] = restore_fn(aggregated)

        elif method == 'weighted_mean':
            data_numeric, restore_fn = to_numeric(data, operation="downsampling", attribute=attr)
            reshaped = data_numeric.reshape(new_shape).transpose(transpose_axes)
            basis = basis_arrays.get(attr)
            if basis is None:
                raise ValueError(f"Missing basis array for attribute '{attr}'")
            basis_name = config.get('basis', config.get('weight'))
            basis_numeric, _ = to_numeric(basis, operation="downsampling", attribute=basis_name)
            basis_reshaped = basis_numeric.reshape(new_shape).transpose(transpose_axes)
            if fill:
                if fill_ratio is None:
                    raise ValueError(f"fill_ratio '{fill}' was requested but is not configured")
                weighted_sum = np.nansum(reshaped * basis_reshaped, axis=(3, 4, 5), dtype=np.float64) / fill_ratio
            else:
                weighted_sum = np.nansum(reshaped * basis_reshaped, axis=(3, 4, 5), dtype=np.float64)
            total_basis = np.nansum(basis_reshaped, axis=(3, 4, 5), dtype=np.float64)
            aggregated = np.divide(
                weighted_sum,
                total_basis,
                out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
                where=total_basis != 0,
            )
            result[attr] = restore_fn(aggregated)

        elif method == 'mode':
            # Mode aggregation for categorical data
            reshaped = data.reshape(new_shape).transpose(transpose_axes)
            nx, ny, nz = reshaped.shape[:3]
            mode_array = np.empty((nx, ny, nz), dtype=object)
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        block = reshaped[i, j, k].ravel()
                        block = block[~pd.isnull(block)]
                        if len(block) == 0:
                            mode_array[i, j, k] = None
                        else:
                            counts = Counter(block)
                            mode_array[i, j, k] = counts.most_common(1)[0][0]
            result[attr] = mode_array

        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        if replacement:
            arr = result[attr]
            replace_map = {k: normalise_missing_value(v) for k, v in replacement.items()}
            for old_value, new_value in replace_map.items():
                arr[arr == old_value] = new_value
            result[attr] = arr

    return result

def normalise_missing_value(val):
    """Convert common missing value indicators to np.nan."""
    if val is None or str(val).lower() in {'nan', 'none', 'na', 'n/a', 'null'}:
        return np.nan
    return val

