import numpy as np
from typing import Tuple, Union

import pandas as pd

Point = Tuple[float, float, float]
BlockDimension = Tuple[float, float, float]
ArrayOrFloat = Union[np.ndarray, float]

MAX_XY_VALUE = 1677721.5  # Maximum value for x and y (2^24 - 1) / 10
MAX_Z_VALUE = 6553.5  # Maximum value for z (2^16 - 1) / 10


def is_integer(value):
    return np.floor(value) == value


def encode_coordinates(x: ArrayOrFloat, y: ArrayOrFloat, z: ArrayOrFloat) -> Union[np.ndarray, int]:
    """Encode the coordinates into a 64-bit integer or an array of 64-bit integers."""

    def check_value(value, max_value):
        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum supported value of {max_value}")
        if not is_integer(value * 10):
            raise ValueError(f"Value {value} has more than 1 decimal place")
        return value

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray):
        x = np.vectorize(check_value)(x, MAX_XY_VALUE)
        y = np.vectorize(check_value)(y, MAX_XY_VALUE)
        z = np.vectorize(check_value)(z, MAX_Z_VALUE)
        x_int = (x * 10).astype(np.int64) & 0xFFFFFF
        y_int = (y * 10).astype(np.int64) & 0xFFFFFF
        z_int = (z * 10).astype(np.int64) & 0xFFFF
        encoded = (x_int << 40) | (y_int << 16) | z_int
        return encoded
    else:
        x = check_value(x, MAX_XY_VALUE)
        y = check_value(y, MAX_XY_VALUE)
        z = check_value(z, MAX_Z_VALUE)
        x_int = int(x * 10) & 0xFFFFFF
        y_int = int(y * 10) & 0xFFFFFF
        z_int = int(z * 10) & 0xFFFF
        encoded = (x_int << 40) | (y_int << 16) | z_int
        return encoded


def decode_coordinates(encoded: Union[np.ndarray, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Point]:
    """Decode the 64-bit integer or array of 64-bit integers back to the original coordinates."""
    x_int = (encoded >> 40) & 0xFFFFFF
    y_int = (encoded >> 16) & 0xFFFFFF
    z_int = encoded & 0xFFFF
    x = x_int / 10.0
    y = y_int / 10.0
    z = z_int / 10.0
    return x, y, z


def multiindex_to_encoded_index(multi_index: pd.MultiIndex) -> pd.Index:
    """Convert a MultiIndex to an encoded integer Index."""
    encoded_indices = [
        encode_coordinates(x, y, z)
        for x, y, z in zip(
            multi_index.get_level_values("x"),
            multi_index.get_level_values("y"),
            multi_index.get_level_values("z"),
        )
    ]
    return pd.Index(encoded_indices, name='encoded_xyz')


def encoded_index_to_multiindex(encoded_index: pd.Index) -> pd.MultiIndex:
    """Convert an encoded integer Index back to a MultiIndex."""
    decoded_coords = [decode_coordinates(encoded) for encoded in encoded_index]
    x, y, z = zip(*decoded_coords)
    return pd.MultiIndex.from_arrays([x, y, z], names=["x", "y", "z"])
