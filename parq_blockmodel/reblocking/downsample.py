import numpy as np
from collections import Counter

import pandas as pd


def downsample_attributes(attributes, fx, fy, fz, aggregation_config):
    """
    Downsample a 3D block model to a coarser grid with multiple attributes using specified aggregation methods.

    Parameters:
    - attributes: dict of 3D numpy arrays, e.g., {'grade': ..., 'density': ..., 'rock_type': ...}
    - fx, fy, fz: downsampling factors along x, y, z axes
    - aggregation_config: dict specifying aggregation method and optional weight for each attribute

    Example:
        attributes = {
            'grade': grade,
            'density': density,
            'dry_mass': dry_mass,
            'volume': volume,
            'rock_type': rock_types
        }

        aggregation_config = {
            'grade': {'method': 'weighted_mean', 'weight': 'dry_mass'},
            'density': {'method': 'weighted_mean', 'weight': 'volume', fill_ratio: fill_ratio},
            'dry_mass': {'method': 'sum', fill_ratio: fill_ratio},
            'volume': {'method': 'sum'},
            'rock_type': {'method': 'mode'}
        }

        downsampled = downsample_attributes(attributes, fx=2, fy=2, fz=2, aggregation_config=aggregation_config)

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
    for attr, config in aggregation_config.items():
        if config.get('fill_ratio'): # if the fill_ratio key exists...
            fill_ratio_vars.add(config.get('fill_ratio'))
    if fill_ratio_vars:
        if len(fill_ratio_vars) > 1:
            raise ValueError(f"Only one fill_ratio is supported, but multiple are supplied: {fill_ratio_vars}")
        fill_ratio_var = fill_ratio_vars.pop()
        reshaped_fill_ratio = attributes[fill_ratio_var].reshape(new_shape).transpose(transpose_axes)
        fill_ratio = np.nanmean(reshaped_fill_ratio, axis=(3, 4, 5))

    # Precompute weights if needed
    weight_arrays = {}
    for attr, config in aggregation_config.items():
        if config.get('method') == 'weighted_mean' and 'weight' in config:
            weight_name = config['weight']
            weight_arrays[attr] = attributes[weight_name]

    for attr, data in attributes.items():
        config = aggregation_config.get(attr, {'method': 'mean'})
        method = config['method']
        fill = config.get('fill_ratio')

        # Reshape and transpose
        reshaped = data.reshape(new_shape).transpose(transpose_axes)

        if method == 'sum':
            if fill:
                result[attr] = np.nansum(reshaped, axis=(3, 4, 5)) / fill_ratio
            else:
                result[attr] = np.nansum(reshaped, axis=(3, 4, 5))

        elif method == 'mean':
            result[attr] = np.nanmean(reshaped, axis=(3, 4, 5))

        elif method == 'weighted_mean':
            weight = weight_arrays.get(attr)
            if weight is None:
                raise ValueError(f"Missing weight array for attribute '{attr}'")
            weight_reshaped = weight.reshape(new_shape).transpose(transpose_axes)
            if fill:
                weighted_sum = np.nansum(reshaped * weight_reshaped, axis=(3, 4, 5)) / fill_ratio
            else:
                weighted_sum = np.nansum(reshaped * weight_reshaped, axis=(3, 4, 5))
            total_weight = np.nansum(weight_reshaped, axis=(3, 4, 5))
            result[attr] = np.divide(weighted_sum, total_weight, out=np.full_like(weighted_sum, np.nan), where=total_weight != 0)

        elif method == 'mode':
            # Mode aggregation for categorical data
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

    return result

