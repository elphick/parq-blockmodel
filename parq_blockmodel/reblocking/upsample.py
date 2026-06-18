
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from parq_blockmodel.reblocking.conversion import to_numeric


_UPSAMPLE_METHODS = {"linear", "nearest", "mode", "parent"}


def _validate_upsample_config(attributes, upsample_config):
    if not isinstance(upsample_config, dict):
        raise ValueError("Upsampling config must be a dictionary of interpolation methods per attribute.")

    missing = sorted(set(attributes) - set(upsample_config))
    extra = sorted(set(upsample_config) - set(attributes))
    if missing:
        raise ValueError(
            "Upsampling config must specify a method for every attribute. "
            f"Missing: {missing}"
        )
    if extra:
        raise ValueError(f"Upsampling config contains unknown attributes: {extra}")

    invalid = {k: v for k, v in upsample_config.items() if not isinstance(v, str)}
    if invalid:
        raise ValueError("Upsampling config must be a dict of interpolation methods per attribute.")

    unsupported = sorted({v for v in upsample_config.values() if v not in _UPSAMPLE_METHODS})
    if unsupported:
        raise ValueError(
            f"Unsupported upsampling method(s): {unsupported}. "
            f"Supported methods are: {sorted(_UPSAMPLE_METHODS)}"
        )


def _nearest_lower_index(coords: np.ndarray, max_index: int) -> np.ndarray:
    # Midpoint ties round down to the lower index for deterministic behavior.
    idx = np.floor(coords + 0.5 - 1e-12).astype(np.int64)
    return np.clip(idx, 0, max_index)


def _upsample_nearest(arr: np.ndarray, new_x: np.ndarray, new_y: np.ndarray, new_z: np.ndarray) -> np.ndarray:
    xi = _nearest_lower_index(new_x, arr.shape[0] - 1)
    yi = _nearest_lower_index(new_y, arr.shape[1] - 1)
    zi = _nearest_lower_index(new_z, arr.shape[2] - 1)
    return arr[np.ix_(xi, yi, zi)]


def _upsample_parent(arr: np.ndarray, fx: int, fy: int, fz: int) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(arr, fx, axis=0), fy, axis=1), fz, axis=2)


def _lowest_mode_value(values) -> object:
    non_null = [v for v in values if not pd.isna(v)]
    if not non_null:
        return np.nan
    counts = {}
    for v in non_null:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    try:
        return min(candidates)
    except TypeError:
        return min(candidates, key=lambda v: repr(v))


def _mode_neighbors(arr: np.ndarray, x: float, y: float, z: float) -> object:
    def _bounds(coord: float, n: int) -> list[int]:
        lo = int(np.floor(coord))
        hi = int(np.ceil(coord))
        lo = max(0, min(lo, n - 1))
        hi = max(0, min(hi, n - 1))
        return [lo] if lo == hi else [lo, hi]

    xs = _bounds(x, arr.shape[0])
    ys = _bounds(y, arr.shape[1])
    zs = _bounds(z, arr.shape[2])
    neighborhood = [arr[ix, iy, iz] for ix in xs for iy in ys for iz in zs]
    return _lowest_mode_value(neighborhood)


def _upsample_mode(arr: np.ndarray, new_x: np.ndarray, new_y: np.ndarray, new_z: np.ndarray) -> np.ndarray:
    out = np.empty((len(new_x), len(new_y), len(new_z)), dtype=object)
    for ix, x in enumerate(new_x):
        for iy, y in enumerate(new_y):
            for iz, z in enumerate(new_z):
                out[ix, iy, iz] = _mode_neighbors(arr, float(x), float(y), float(z))
    return out


def upsample_attributes(attributes, fx, fy, fz, upsample_config=None, interpolation_config=None):
    """
    Upsample a 3D block model to a finer grid with explicit per-attribute methods.

    Parameters:
    - attributes: dict of 3D arrays (NumPy or Pandas extension types)
    - fx, fy, fz: upsampling factors along x, y, z axes
    - upsample_config: dict specifying one method per attribute
      ("linear", "nearest", "mode", or "parent")

    Example:
        attributes = {
            'grade': grade,
            'density': density,
            'dry_mass': dry_mass,
            'volume': volume,
            'rock_type': rock_types
        }

        upsample_config = {
            'grade': 'linear',
            'density': 'linear',
            'dry_mass': 'linear',
            'volume': 'linear',
            'rock_type': 'mode'
        }

        upsampled = upsample_attributes(attributes, fx=2, fy=2, fz=2, upsample_config=upsample_config)

    Returns:
    - dict of upsampled 3D arrays
    """

    if upsample_config is not None and interpolation_config is not None:
        raise ValueError("Provide only one of 'upsample_config' or deprecated 'interpolation_config'.")
    effective_config = upsample_config if upsample_config is not None else interpolation_config

    _validate_upsample_config(attributes, effective_config)

    first_arr = next(iter(attributes.values()))
    nx, ny, nz = first_arr.shape
    result = {}

    for attr, arr in attributes.items():
        method = effective_config[attr]
        new_nx, new_ny, new_nz = nx * fx, ny * fy, nz * fz

        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        new_x = np.linspace(0, nx - 1, new_nx)
        new_y = np.linspace(0, ny - 1, new_ny)
        new_z = np.linspace(0, nz - 1, new_nz)
        if method == "nearest":
            result[attr] = _upsample_nearest(np.asarray(arr), new_x, new_y, new_z)
            continue

        if method == "parent":
            result[attr] = _upsample_parent(np.asarray(arr), fx, fy, fz)
            continue

        if method == "mode":
            mode_values = _upsample_mode(np.asarray(arr), new_x, new_y, new_z)
            arr_numeric, restore_fn = to_numeric(arr, operation="upsampling", attribute=attr)
            mode_values = np.asarray(mode_values, dtype=arr_numeric.dtype if arr_numeric.dtype.kind in {"f", "i", "u"} else object)
            result[attr] = restore_fn(mode_values)
            continue

        grid = np.meshgrid(new_x, new_y, new_z, indexing='ij')
        points = np.stack([g.ravel() for g in grid], axis=-1)
        arr_numeric, restore_fn = to_numeric(arr, operation="upsampling", attribute=attr)
        interpolator = RegularGridInterpolator(
            (x, y, z),
            arr_numeric,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        upsampled_numeric = interpolator(points).reshape(new_nx, new_ny, new_nz)
        result[attr] = restore_fn(upsampled_numeric)

    return result