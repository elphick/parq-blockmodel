import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from parq_blockmodel.types import Vector


def dense_ijk_multiindex(shape) -> pd.MultiIndex:
    # shape = (nx, ny, nz)
    i, j, k = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    # Flatten and stack for MultiIndex
    return pd.MultiIndex.from_arrays(
        [i.ravel(order='C'), j.ravel(order='C'), k.ravel(order='C')],
        names=['i', 'j', 'k']
    )


def validate_geometry(filepath: Path) -> None:
    """
    Validates the geometry of a Parquet file by checking if the index (centroid) columns are present
    and have valid values.

    Args:
        filepath (Path): Path to the Parquet file.

    Raises:
        ValueError: If any index column is missing or contains invalid values.
    """
    index_columns = ['x', 'y', 'z']

    columns = pq.read_schema(filepath).names
    if not all(col in columns for col in index_columns):
        raise ValueError(f"Missing index columns in the dataset: {', '.join(index_columns)}")

    # Read the Parquet file to check for NaN values in index columns
    table = pq.read_table(filepath, columns=index_columns)
    for col in index_columns:
        if table[col].null_count > 0:
            raise ValueError(f"Column '{col}' contains NaN values, which is not allowed in the index columns.")

    # check the geometry is regular
    x_values = np.sort(table['x'].to_pandas().unique())
    y_values = np.sort(table['y'].to_pandas().unique())
    z_values = np.sort(table['z'].to_pandas().unique())
    if len(x_values) < 2 or len(y_values) < 2 or len(z_values) < 2:
        raise ValueError("The geometry is not regular. At least two unique values are required in each index column.")

    def is_regular_spacing(values, tol=1e-8):
        diffs = np.diff(values)
        return np.all(np.abs(diffs - diffs[0]) < tol)

    if not (is_regular_spacing(x_values) and is_regular_spacing(y_values) and is_regular_spacing(z_values)):
        raise ValueError(
            "The geometry is not regular. The index columns must be evenly spaced (regular grid) in x, y, and z.")

    logging.info(f"Geometry validation completed successfully for {filepath}.")


def validate_axes_orthonormal(u, v, w, tol=1e-8):
    """ Validate if three vectors are orthonormal.

    Args:
        u (array-like): First vector.
        v (array-like): Second vector.
        w (array-like): Third vector.
        tol (float): Tolerance for checking orthonormality.

    Returns:
        bool: True if the vectors are orthonormal, False otherwise.
    """

    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    w = np.array(w, dtype=float)
    # Check normalization
    if not (np.isclose(np.linalg.norm(u), 1.0, atol=tol) and
            np.isclose(np.linalg.norm(v), 1.0, atol=tol) and
            np.isclose(np.linalg.norm(w), 1.0, atol=tol)):
        return False
    # Check orthogonality
    if not (np.isclose(np.dot(u, v), 0.0, atol=tol) and
            np.isclose(np.dot(u, w), 0.0, atol=tol) and
            np.isclose(np.dot(v, w), 0.0, atol=tol)):
        return False
    return True


def rotation_to_axis_orientation(axis_azimuth: float = 0,
                                 axis_dip: float = 0,
                                 axis_plunge: float = 0
                                 ) -> tuple[Vector, Vector, Vector]:
    """
    Convert azimuth, dip, and plunge angles to orthonormal axes.

    Args:
        axis_azimuth (float): Azimuth angle in degrees.
        axis_dip (float): Angle from horizontal down (degrees)
        axis_plunge (float): Rotation around the u-axis (degrees, optional, often 0 for planar features)

    Returns:
        tuple: Three orthonormal vectors representing the axes.
    """

    if axis_azimuth != 0.0 or axis_dip != 0.0 or axis_plunge != 0.0:

        azimuth_rad = np.radians(axis_azimuth)
        dip_rad = np.radians(axis_dip)
        plunge_rad = np.radians(axis_plunge)

        # Initial axes
        u = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
        v = np.array([-np.sin(azimuth_rad) * np.cos(dip_rad),
                      np.cos(azimuth_rad) * np.cos(dip_rad),
                      np.sin(dip_rad)])
        w = np.cross(u, v)

        # Rotation matrix around u by plunge angle
        def rotate_around_axis(vec, axis, angle):
            axis = axis / np.linalg.norm(axis)
            return (vec * np.cos(angle) +
                    np.cross(axis, vec) * np.sin(angle) +
                    axis * np.dot(axis, vec) * (1 - np.cos(angle)))

        v_rot = rotate_around_axis(v, u, plunge_rad)
        w_rot = rotate_around_axis(w, u, plunge_rad)

        if not validate_axes_orthonormal(u, v_rot, w_rot):
            raise ValueError("The provided angles do not yield orthonormal axes.")

        return ((float(u[0]), float(u[1]), float(u[2])),
                (float(v_rot[0]), float(v_rot[1]), float(v_rot[2])),
                (float(w_rot[0]), float(w_rot[1]), float(w_rot[2])),
                )
    else:
        # If no rotation is specified, return the standard axes
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def axis_orientation_to_rotation(axis_u, axis_v, axis_w):
    """
    Convert orthonormal axes to azimuth, dip, and plunge angles (in degrees).
    Returns (azimuth, dip, plunge).
    """

    # Bypass calculation for identity axes
    if (np.allclose(axis_u, (1, 0, 0), atol=1e-8) and
            np.allclose(axis_v, (0, 1, 0), atol=1e-8) and
            np.allclose(axis_w, (0, 0, 1), atol=1e-8)):
        return 0.0, 0.0, 0.0

    u = np.array(axis_u)
    v = np.array(axis_v)
    # Azimuth: angle from x-axis in xy-plane
    azimuth = np.degrees(np.arctan2(u[1], u[0]))
    # Dip: angle from horizontal down (z component of v)
    dip = np.degrees(np.arcsin(v[2]))
    # Plunge: rotation of v around u (project v onto plane normal to u)
    # For most block models, plunge is 0, but you can compute it if needed.
    # Here, set to 0 for simplicity.
    plunge = 0.0
    return float(azimuth), float(dip), float(plunge)


def rotate_points(points: np.ndarray,
                  azimuth: float = 0,
                  dip: float = 0,
                  plunge: float = 0
                  ) -> np.ndarray:
    """Rotate points in 3D space based on azimuth, dip, and plunge angles.

    Args:
        points (np.ndarray): Array of shape (n, 3) representing the points to be rotated.
        azimuth (float): Azimuth angle in degrees.
        dip (float): Dip angle in degrees.
        plunge (float): Plunge angle in degrees.

    Returns:
        np.ndarray: Rotated points.
    """
    u, v, w = rotation_to_axis_orientation(azimuth, dip, plunge)
    rotation_matrix = np.array([u, v, w])
    return points @ rotation_matrix
