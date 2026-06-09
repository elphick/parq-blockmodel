import numpy as np
from typing import Tuple, Union

import pandas as pd

Point = Tuple[float, float, float]
BlockDimension = Tuple[float, float, float]
ArrayOrFloat = Union[np.ndarray, float]

MAX_XY_VALUE = 1677721.5  # Maximum value for x and y (2^24 - 1) / 10
MAX_Z_VALUE = 6553.5  # Maximum value for z (2^16 - 1) / 10

ENCODED_X_BITS = 24
ENCODED_Y_BITS = 24
ENCODED_Z_BITS = 16
ENCODED_X_SHIFT = 40
ENCODED_Y_SHIFT = 16
ENCODED_Z_SHIFT = 0
ENCODED_X_MAX_INT = (1 << ENCODED_X_BITS) - 1
ENCODED_Y_MAX_INT = (1 << ENCODED_Y_BITS) - 1
ENCODED_Z_MAX_INT = (1 << ENCODED_Z_BITS) - 1

DEFAULT_AXIS_BITS = (ENCODED_X_BITS, ENCODED_Y_BITS, ENCODED_Z_BITS)


def _normalize_axis_bits(
    bits_per_axis: int | tuple[int, int, int] | list[int] | dict[str, int]
) -> tuple[int, int, int]:
    if isinstance(bits_per_axis, dict):
        x_bits = int(bits_per_axis.get("x", ENCODED_X_BITS))
        y_bits = int(bits_per_axis.get("y", ENCODED_Y_BITS))
        z_bits = int(bits_per_axis.get("z", ENCODED_Z_BITS))
    elif isinstance(bits_per_axis, (tuple, list)):
        if len(bits_per_axis) != 3:
            raise ValueError("axis bit configuration must provide exactly three values (x, y, z).")
        x_bits = int(bits_per_axis[0])
        y_bits = int(bits_per_axis[1])
        z_bits = int(bits_per_axis[2])
    else:
        scalar = int(bits_per_axis)
        x_bits = y_bits = z_bits = scalar

    if x_bits <= 0 or y_bits <= 0 or z_bits <= 0:
        raise ValueError("axis bit counts must be positive integers.")
    if x_bits > ENCODED_X_BITS or y_bits > ENCODED_Y_BITS or z_bits > ENCODED_Z_BITS:
        raise ValueError(
            f"axis bit counts exceed supported maxima x/y/z={ENCODED_X_BITS}/{ENCODED_Y_BITS}/{ENCODED_Z_BITS}."
        )
    if (x_bits + y_bits + z_bits) > 64:
        raise ValueError("axis bit counts must total <= 64 for 64-bit storage.")
    return x_bits, y_bits, z_bits


def get_id_encoding_params(
    id_encoding: dict | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]:
    """Return ``(offset_xyz, scale_xyz, axis_bits_xyz)`` from encoding metadata.

    ``offset_xyz`` and ``scale_xyz`` are always returned as three-float tuples ordered as ``(x, y, z)``.
    """
    if not id_encoding:
        return (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), DEFAULT_AXIS_BITS

    offset_payload = id_encoding.get("offset", {})
    offset = (
        float(offset_payload.get("x", 0.0)),
        float(offset_payload.get("y", 0.0)),
        float(offset_payload.get("z", 0.0)),
    )

    quant = id_encoding.get("quantization", {})
    scale_payload = quant.get("scale", 10.0)
    scale_xyz: tuple[float, float, float]
    if isinstance(scale_payload, (list, tuple)):
        if len(scale_payload) != 3:
            raise ValueError(
                f"quantization.scale list/tuple must provide three values (x, y, z), got {len(scale_payload)}."
            )
        scale_xyz = (
            float(scale_payload[0]),
            float(scale_payload[1]),
            float(scale_payload[2]),
        )
    elif isinstance(scale_payload, dict):
        scale_xyz = (
            float(scale_payload.get("x", 10.0)),
            float(scale_payload.get("y", 10.0)),
            float(scale_payload.get("z", 10.0)),
        )
    else:
        shared_scale = float(scale_payload)
        scale_xyz = (shared_scale, shared_scale, shared_scale)

    # Support asymmetric axis bits in the new contract (`encoding.axis_bits`), while
    # preserving fallback compatibility with older payloads.
    encoding_payload = id_encoding.get("encoding", {})
    axis_bits_payload = (
        encoding_payload.get("axis_bits")
        or id_encoding.get("axis_bits")
        or encoding_payload.get("bits_per_axis")
        or id_encoding.get("bits_per_axis")
        or DEFAULT_AXIS_BITS
    )
    axis_bits = _normalize_axis_bits(axis_bits_payload)

    return offset, scale_xyz, axis_bits


def get_world_id_encoding_params(
    world_id_encoding: dict | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]:
    """Alias for :func:`get_id_encoding_params`."""
    return get_id_encoding_params(world_id_encoding)


def _morton_encode_3d(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, axis_bits: tuple[int, int, int]
) -> np.ndarray:
    x_bits, y_bits, z_bits = axis_bits
    encoded = np.zeros_like(ix, dtype=np.uint64)
    bit_position = 0
    max_bits = max(x_bits, y_bits, z_bits)
    for bit in range(max_bits):
        if bit < x_bits:
            encoded |= ((ix >> bit) & 1).astype(np.uint64) << bit_position
            bit_position += 1
        if bit < y_bits:
            encoded |= ((iy >> bit) & 1).astype(np.uint64) << bit_position
            bit_position += 1
        if bit < z_bits:
            encoded |= ((iz >> bit) & 1).astype(np.uint64) << bit_position
            bit_position += 1
    return encoded


def _morton_decode_3d(
    encoded: np.ndarray, axis_bits: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bits, y_bits, z_bits = axis_bits
    ix = np.zeros_like(encoded, dtype=np.uint64)
    iy = np.zeros_like(encoded, dtype=np.uint64)
    iz = np.zeros_like(encoded, dtype=np.uint64)
    bit_position = 0
    max_bits = max(x_bits, y_bits, z_bits)
    for bit in range(max_bits):
        if bit < x_bits:
            ix |= ((encoded >> bit_position) & 1).astype(np.uint64) << bit
            bit_position += 1
        if bit < y_bits:
            iy |= ((encoded >> bit_position) & 1).astype(np.uint64) << bit
            bit_position += 1
        if bit < z_bits:
            iz |= ((encoded >> bit_position) & 1).astype(np.uint64) << bit
            bit_position += 1
    return ix, iy, iz


def encode_frame_coordinates(
    x: ArrayOrFloat,
    y: ArrayOrFloat,
    z: ArrayOrFloat,
    offset: Point = (0.0, 0.0, 0.0),
    scale: float | tuple[float, float, float] = 10.0,
    bits_per_axis: int | tuple[int, int, int] | list[int] | dict[str, int] = DEFAULT_AXIS_BITS,
) -> Union[np.ndarray, int]:
    """Encode xyz-like coordinates with quantization + 3D Morton/Z-order bit interleaving.

    Returns an ``int`` for scalar inputs and an ``np.ndarray[int64]`` for array-like inputs.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    axis_bits = _normalize_axis_bits(bits_per_axis)

    ox, oy, oz = offset
    if isinstance(scale, (tuple, list)):
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    else:
        sx = sy = sz = float(scale)

    x_q = np.rint((x_arr - ox) * sx).astype(np.int64)
    y_q = np.rint((y_arr - oy) * sy).astype(np.int64)
    z_q = np.rint((z_arr - oz) * sz).astype(np.int64)

    if np.any(x_q < 0) or np.any(y_q < 0) or np.any(z_q < 0):
        raise ValueError("Encoding produced negative quantized values. Check offsets.")
    x_max_int = (1 << axis_bits[0]) - 1
    y_max_int = (1 << axis_bits[1]) - 1
    z_max_int = (1 << axis_bits[2]) - 1
    if np.any(x_q > x_max_int) or np.any(y_q > y_max_int) or np.any(z_q > z_max_int):
        raise ValueError("Encoding overflow. Check offsets, scale, or coordinate extents.")

    encoded = _morton_encode_3d(
        x_q.astype(np.uint64),
        y_q.astype(np.uint64),
        z_q.astype(np.uint64),
        axis_bits=axis_bits,
    )

    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        return int(np.asarray(encoded, dtype=np.int64).item())
    return encoded.astype(np.int64)


def encode_world_coordinates(
    x: ArrayOrFloat,
    y: ArrayOrFloat,
    z: ArrayOrFloat,
    offset: Point = (0.0, 0.0, 0.0),
    scale: float | tuple[float, float, float] = 10.0,
    bits_per_axis: int | tuple[int, int, int] | list[int] | dict[str, int] = DEFAULT_AXIS_BITS,
) -> Union[np.ndarray, int]:
    """Alias for :func:`encode_frame_coordinates` for global spatial identifiers."""
    return encode_frame_coordinates(x=x, y=y, z=z, offset=offset, scale=scale, bits_per_axis=bits_per_axis)


def decode_frame_coordinates(
    encoded: Union[np.ndarray, int],
    offset: Point = (0.0, 0.0, 0.0),
    scale: float | tuple[float, float, float] = 10.0,
    bits_per_axis: int | tuple[int, int, int] | list[int] | dict[str, int] = DEFAULT_AXIS_BITS,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Point]:
    """Decode Morton/Z-order ids back to xyz-like coordinates using offset+scale.

    Returns a ``(x, y, z)`` tuple of floats for scalar input ids and arrays for vectorized input.
    """
    encoded_arr = np.asarray(encoded, dtype=np.int64).astype(np.uint64)
    axis_bits = _normalize_axis_bits(bits_per_axis)
    x_int, y_int, z_int = _morton_decode_3d(encoded_arr, axis_bits=axis_bits)

    ox, oy, oz = offset
    if isinstance(scale, (tuple, list)):
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    else:
        sx = sy = sz = float(scale)

    x = x_int.astype(np.float64) / sx + ox
    y = y_int.astype(np.float64) / sy + oy
    z = z_int.astype(np.float64) / sz + oz

    if np.isscalar(encoded):
        return float(np.asarray(x).item()), float(np.asarray(y).item()), float(np.asarray(z).item())
    return x, y, z


def decode_world_coordinates(
    encoded: Union[np.ndarray, int],
    offset: Point = (0.0, 0.0, 0.0),
    scale: float | tuple[float, float, float] = 10.0,
    bits_per_axis: int | tuple[int, int, int] | list[int] | dict[str, int] = DEFAULT_AXIS_BITS,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Point]:
    """Alias for :func:`decode_frame_coordinates` for global spatial identifiers."""
    return decode_frame_coordinates(encoded=encoded, offset=offset, scale=scale, bits_per_axis=bits_per_axis)


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
