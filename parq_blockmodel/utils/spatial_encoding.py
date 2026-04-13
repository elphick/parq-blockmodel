import numpy as np
from typing import Tuple, Union

import pandas as pd

Point = Tuple[float, float, float]
BlockDimension = Tuple[float, float, float]
ArrayOrFloat = Union[np.ndarray, float]

MAX_XY_VALUE = 1677721.5  # Maximum value for x and y (2^24 - 1) / 10
MAX_Z_VALUE = 6553.5  # Maximum value for z (2^16 - 1) / 10

WORLD_ID_X_BITS = 24
WORLD_ID_Y_BITS = 24
WORLD_ID_Z_BITS = 16
WORLD_ID_X_SHIFT = 40
WORLD_ID_Y_SHIFT = 16
WORLD_ID_Z_SHIFT = 0
WORLD_ID_X_MAX_INT = (1 << WORLD_ID_X_BITS) - 1
WORLD_ID_Y_MAX_INT = (1 << WORLD_ID_Y_BITS) - 1
WORLD_ID_Z_MAX_INT = (1 << WORLD_ID_Z_BITS) - 1


def get_id_encoding_params(
    id_encoding: dict | None,
) -> tuple[tuple[float, float, float], float]:
    """Return (offset_xyz, scale) from a metadata-like encoding payload."""
    if not id_encoding:
        return (0.0, 0.0, 0.0), 10.0

    offset_payload = id_encoding.get("offset", {})
    offset = (
        float(offset_payload.get("x", 0.0)),
        float(offset_payload.get("y", 0.0)),
        float(offset_payload.get("z", 0.0)),
    )

    quant = id_encoding.get("quantization", {})
    scale_payload = quant.get("scale", 10.0)
    if isinstance(scale_payload, (list, tuple)):
        # Current implementation uses a shared scale for all axes.
        scale = float(scale_payload[0])
    else:
        scale = float(scale_payload)

    return offset, scale


def get_world_id_encoding_params(
    world_id_encoding: dict | None,
) -> tuple[tuple[float, float, float], float]:
    """Backward-compatible alias for :func:`get_id_encoding_params`."""
    return get_id_encoding_params(world_id_encoding)


def encode_frame_coordinates(
    x: ArrayOrFloat,
    y: ArrayOrFloat,
    z: ArrayOrFloat,
    offset: Point = (0.0, 0.0, 0.0),
    scale: float = 10.0,
) -> Union[np.ndarray, int]:
    """Encode xyz-like coordinates into uint64-compatible integers using offset+scale quantization."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    ox, oy, oz = offset
    x_q = np.rint((x_arr - ox) * scale).astype(np.int64)
    y_q = np.rint((y_arr - oy) * scale).astype(np.int64)
    z_q = np.rint((z_arr - oz) * scale).astype(np.int64)

    if np.any(x_q < 0) or np.any(y_q < 0) or np.any(z_q < 0):
        raise ValueError("Encoding produced negative quantized values. Check offsets.")
    if np.any(x_q > WORLD_ID_X_MAX_INT) or np.any(y_q > WORLD_ID_Y_MAX_INT) or np.any(z_q > WORLD_ID_Z_MAX_INT):
        raise ValueError("Encoding overflow. Check offsets, scale, or coordinate extents.")

    encoded = (x_q << WORLD_ID_X_SHIFT) | (y_q << WORLD_ID_Y_SHIFT) | z_q
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        return int(np.asarray(encoded).item())
    return encoded


def encode_world_coordinates(
    x: ArrayOrFloat,
    y: ArrayOrFloat,
    z: ArrayOrFloat,
    offset: Point = (0.0, 0.0, 0.0),
    scale: float = 10.0,
) -> Union[np.ndarray, int]:
    """Backward-compatible alias for :func:`encode_frame_coordinates`."""
    return encode_frame_coordinates(x=x, y=y, z=z, offset=offset, scale=scale)


def decode_frame_coordinates(
    encoded: Union[np.ndarray, int],
    offset: Point = (0.0, 0.0, 0.0),
    scale: float = 10.0,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Point]:
    """Decode uint64-compatible ids back to xyz-like coordinates using offset+scale."""
    x_int = (encoded >> WORLD_ID_X_SHIFT) & WORLD_ID_X_MAX_INT
    y_int = (encoded >> WORLD_ID_Y_SHIFT) & WORLD_ID_Y_MAX_INT
    z_int = encoded & WORLD_ID_Z_MAX_INT

    ox, oy, oz = offset
    x = x_int / scale + ox
    y = y_int / scale + oy
    z = z_int / scale + oz
    return x, y, z


def decode_world_coordinates(
    encoded: Union[np.ndarray, int],
    offset: Point = (0.0, 0.0, 0.0),
    scale: float = 10.0,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Point]:
    """Backward-compatible alias for :func:`decode_frame_coordinates`."""
    return decode_frame_coordinates(encoded=encoded, offset=offset, scale=scale)


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
