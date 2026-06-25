"""Pure utility functions for data ingestion and canonicalization.

This module provides stateless validation and encoding helpers for
the data ingestion pipeline. All functions are pure (no side effects)
and can be reused independently.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.utils.spatial_encoding import (
    DEFAULT_AXIS_BITS,
    ENCODED_X_BITS,
    ENCODED_Y_BITS,
    ENCODED_Z_BITS,
    decode_world_coordinates,
    encode_world_coordinates,
    get_world_id_encoding_params,
)

logger = logging.getLogger(__name__)


def assert_block_id_xyz_consistent(
    block_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: RegularGeometry,
    tol: float,
    context: str,
) -> None:
    """Validate that provided block_id values match xyz-derived ids.

    Parameters
    ----------
    block_ids : np.ndarray
        Provided block_id column values.
    x, y, z : np.ndarray
        Centroid coordinates.
    geometry : RegularGeometry
        Geometry used to derive expected block_ids from xyz.
    tol : float
        Tolerance for xyz to ijk conversion.
    context : str
        Context string for error messages (e.g., filename).

    Raises
    ------
    ValueError
        If any block_id does not match the xyz-derived value.
    """
    expected = geometry.row_index_from_xyz(x, y, z, tol=tol).astype(np.uint32)
    actual = np.asarray(block_ids, dtype=np.uint32)
    mismatch = actual != expected
    if np.any(mismatch):
        first = int(np.flatnonzero(mismatch)[0])
        raise ValueError(
            "Inconsistent positional columns: provided block_id does not match "
            f"xyz-derived block_id ({context}, first mismatch at batch offset {first}: "
            f"provided={int(actual[first])}, expected={int(expected[first])})."
        )


def validate_geometry(
    filepath: Path,
    geometry: Optional[RegularGeometry] = None,
    chunk_size: int = 1_000_000,
    tol: float = 1e-6,
) -> None:
    """Validate centroid columns against a :class:`RegularGeometry`.

    This helper enforces that a Parquet file has well-formed centroid
    columns ``x``, ``y``, ``z`` and that these centroids lie on the
    dense grid implied by a :class:`RegularGeometry` instance.

    Parameters
    ----------
    filepath : Path
        Path to the Parquet file to validate (typically a ``.pbm``
        container or its source ``.parquet`` file).
    geometry : RegularGeometry, optional
        Geometry of the logical ijk grid. If omitted, it is inferred
        from ``filepath`` via :meth:`RegularGeometry.from_parquet`.
    chunk_size : int, default 1_000_000
        Batch size for reading Parquet data.
    tol : float, default 1e-6
        Tolerance for xyz to ijk conversion.

    Raises
    ------
    ValueError
        If any centroid column is missing or contains null values,
        if their lengths differ, or if the resulting sparse centroids
        are not a subset of the dense grid implied by ``geometry``.
    """
    columns = pq.read_schema(filepath).names

    has_block_id = "block_id" in columns
    has_world_id = "world_id" in columns
    has_ijk = all(col in columns for col in ["i", "j", "k"])
    has_xyz = all(col in columns for col in ["x", "y", "z"])

    if not any([has_block_id, has_world_id, has_ijk, has_xyz]):
        raise ValueError(
            "Dataset must contain at least one positional representation: "
            "block_id, world_id, ijk, or xyz."
        )

    if geometry is None:
        geometry = RegularGeometry.from_parquet(filepath)

    dense_count = int(np.prod(geometry.local.shape))
    seen: set[int] = set()

    pf = pq.ParquetFile(filepath)
    if has_block_id and has_xyz:
        columns_to_read = ["block_id", "x", "y", "z"]
    elif has_block_id:
        columns_to_read = ["block_id"]
    elif has_world_id:
        columns_to_read = ["world_id"]
    elif has_xyz:
        columns_to_read = ["x", "y", "z"]
    else:
        columns_to_read = ["i", "j", "k"]

    for batch in pf.iter_batches(columns=columns_to_read, batch_size=chunk_size):
        if has_block_id and has_xyz:
            block_ids = np.asarray(batch.column(0), dtype=np.uint32)
            x = np.asarray(batch.column(1), dtype=float)
            y = np.asarray(batch.column(2), dtype=float)
            z = np.asarray(batch.column(3), dtype=float)
            assert_block_id_xyz_consistent(
                block_ids=block_ids,
                x=x,
                y=y,
                z=z,
                geometry=geometry,
                tol=tol,
                context=f"validation for {filepath}",
            )
        elif has_block_id:
            block_ids = np.asarray(batch.column(0), dtype=np.uint32)
        elif has_world_id:
            if not geometry.world_id_encoding:
                raise ValueError(
                    "world_id column present but metadata has no world_id_encoding payload."
                )
            offset, scale, bits_per_axis = get_world_id_encoding_params(
                geometry.world_id_encoding
            )
            world_ids = np.asarray(batch.column(0), dtype=np.int64)
            x, y, z = decode_world_coordinates(
                world_ids, offset=offset, scale=scale, bits_per_axis=bits_per_axis
            )
            block_ids = geometry.row_index_from_xyz(x, y, z, tol=tol).astype(np.uint32)
        elif has_xyz:
            x = np.asarray(batch.column(0), dtype=float)
            y = np.asarray(batch.column(1), dtype=float)
            z = np.asarray(batch.column(2), dtype=float)
            block_ids = geometry.row_index_from_xyz(x, y, z, tol=tol).astype(np.uint32)
        else:
            i = np.asarray(batch.column(0), dtype=np.int64)
            j = np.asarray(batch.column(1), dtype=np.int64)
            k = np.asarray(batch.column(2), dtype=np.int64)
            block_ids = geometry.row_index_from_ijk(i, j, k).astype(np.uint32)

        if np.any(block_ids < 0) or np.any(block_ids >= dense_count):
            raise ValueError("Sparse positions must be a subset of the dense geometry grid.")

        seen_before = len(seen)
        seen.update(block_ids.tolist())
        if len(seen) - seen_before != len(block_ids):
            raise ValueError("Duplicate block positions detected in dataset.")

    logger.info(f"Geometry validation completed successfully for {filepath}.")


def validate_xyz_parquet(
    parquet_path: Path,
    axis_azimuth: float = 0.0,
    axis_dip: float = 0.0,
    axis_plunge: float = 0.0,
    chunk_size: int = 1_000_000,
    tol: float = 1e-6,
) -> RegularGeometry:
    """Validate xyz-defined Parquet input and return inferred geometry.

    Parameters
    ----------
    parquet_path : Path
        Path to the Parquet file to validate.
    axis_azimuth : float, default 0.0
        Rotation angle for axis orientation.
    axis_dip : float, default 0.0
        Rotation angle for axis orientation.
    axis_plunge : float, default 0.0
        Rotation angle for axis orientation.
    chunk_size : int, default 1_000_000
        Batch size for reading Parquet data.
    tol : float, default 1e-6
        Tolerance for xyz to ijk conversion.

    Returns
    -------
    RegularGeometry
        Inferred geometry from the Parquet file.
    """
    geometry = RegularGeometry.from_parquet(
        filepath=parquet_path,
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
        chunk_size=chunk_size,
    )
    validate_geometry(filepath=parquet_path, geometry=geometry, chunk_size=chunk_size, tol=tol)
    return geometry


def build_world_id_encoding_from_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    scale: float = 10.0,
    bits_per_axis: tuple[int, int, int] = DEFAULT_AXIS_BITS,
) -> dict[str, object]:
    """Build default world_id encoding metadata from xyz ranges.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays (may contain min/max values).
    scale : float, default 10.0
        Quantization scale factor.
    bits_per_axis : tuple[int, int, int], default DEFAULT_AXIS_BITS
        Number of bits per axis for Morton encoding.

    Returns
    -------
    dict[str, object]
        Encoding metadata payload ready for geometry.world_id_encoding.

    Raises
    ------
    ValueError
        If axis bit counts are invalid or exceed supported limits.
    """
    x_bits, y_bits, z_bits = tuple(int(v) for v in bits_per_axis)
    if x_bits <= 0 or y_bits <= 0 or z_bits <= 0:
        raise ValueError("axis bit counts must be positive integers.")
    if x_bits > ENCODED_X_BITS or y_bits > ENCODED_Y_BITS or z_bits > ENCODED_Z_BITS:
        raise ValueError(
            f"axis bit counts exceed supported maxima x/y/z={ENCODED_X_BITS}/{ENCODED_Y_BITS}/{ENCODED_Z_BITS}."
        )
    if (x_bits + y_bits + z_bits) > 64:
        raise ValueError("axis bit counts must total <= 64 for 64-bit storage.")

    ox = np.floor(float(np.min(x)) * scale) / scale
    oy = np.floor(float(np.min(y)) * scale) / scale
    oz = np.floor(float(np.min(z)) * scale) / scale

    x_span = float(np.max(x) - ox)
    y_span = float(np.max(y) - oy)
    z_span = float(np.max(z) - oz)
    x_axis_max = ((1 << x_bits) - 1) / scale
    y_axis_max = ((1 << y_bits) - 1) / scale
    z_axis_max = ((1 << z_bits) - 1) / scale
    if x_span > x_axis_max or y_span > y_axis_max or z_span > z_axis_max:
        raise ValueError(
            f"Coordinates exceed world_id span limits (x/y/z bits={x_bits}/{y_bits}/{z_bits} at scale={scale}). "
            "Provide a custom encoding strategy."
        )

    return {
        "enabled": True,
        "column": "world_id",
        "frame": "world_xyz",
        "axis_order": ["x", "y", "z"],
        "encoding": {
            "type": "morton_zorder",
            "version": "1.0",
            "bits_per_axis": x_bits,
            "axis_bits": {"x": x_bits, "y": y_bits, "z": z_bits},
            "signed": False,
        },
        "quantization": {
            "scale": scale,
            "rounding": "nearest",
            "precision_decimals": 1,
        },
        "offset": {"x": ox, "y": oy, "z": oz, "units": "world"},
        "range_after_offset": {
            "x_min": 0.0,
            "x_max": x_axis_max,
            "y_min": 0.0,
            "y_max": y_axis_max,
            "z_min": 0.0,
            "z_max": z_axis_max,
        },
        "overflow_policy": "error",
        "null_policy": "no_nulls",
    }
