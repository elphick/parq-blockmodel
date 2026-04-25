from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from parq_blockmodel.utils.geometry_utils import rotate_points


def create_demo_blockmodel(shape: tuple[int, int, int] = (3, 3, 3),
                           block_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
                           corner: tuple[float, float, float] = (0.0, 0.0, 0.0),
                           azimuth: float = 0.0,
                           dip: float = 0.0,
                           plunge: float = 0.0,
                           parquet_filepath: Path = None,
                           index_type: Literal["block_index", "world_centroids", "block_id"] = "block_index",
                           ) -> pd.DataFrame | Path:

    """
    Create a synthetic block model DataFrame or Parquet file.

    This function generates a rectilinear block model defined in logical
    index space (i, j, k), together with world–space centroids and optional
    rotation. The function is intended for testing, exploration, and examples.

    ─────────────────────────────────────────────────────────────────────────────
    BLOCK MODEL INDEXING
    ─────────────────────────────────────────────────────────────────────────────
    Logical indices:
        - (i, j, k) represent block indices along the x, y, and z axes.
        - Sorting by (i, j, k) gives the canonical logical block‑model order:
              i varies slowest
              j varies next
              k varies fastest
        This ordering is *purely logical* and independent of how NumPy stores
        the data in memory.

    Canonical block id:
        - block_id:
            Canonical linear id obtained by flattening logical (i,j,k)
            coordinates in NumPy C-order. This is frame-invariant and remains
            stable regardless of world-space rotation.

    ─────────────────────────────────────────────────────────────────────────────
    WORLD COORDINATES
    ─────────────────────────────────────────────────────────────────────────────
    World centroids are computed using:
        x = corner_x + (i + 0.5) * dx
        y = corner_y + (j + 0.5) * dy
        z = corner_z + (k + 0.5) * dz

    If azimuth, dip, or plunge are non‑zero, the world positions are rotated
    about the origin after centroid generation.

    ─────────────────────────────────────────────────────────────────────────────
    DEPTH INFORMATION
    ─────────────────────────────────────────────────────────────────────────────
    - depth:
        Computed as (maximum z centroid + dz/2) − z.
    - depth_category:
        Categorical split into 'shallow' and 'deep'.

    ─────────────────────────────────────────────────────────────────────────────
    DATAFRAME INDEXING OPTION
    ─────────────────────────────────────────────────────────────────────────────
    index_type:
        - "block_index": the DataFrame is indexed by (i, j, k).
        - "world_centroids": indexed by (x, y, z).
        - "block_id": indexed by canonical block_id.

    ─────────────────────────────────────────────────────────────────────────────
    ATTRIBUTES
    ─────────────────────────────────────────────────────────────────────────────
    The DataFrame is assigned lightweight geometry metadata under:
        df.attrs["geometry"] = {
            "shape": (nx, ny, nz),
            "block_size": (dx, dy, dz),
            "corner": (corner_x, corner_y, corner_z),
            "rotation": { "azimuth": ..., "dip": ..., "plunge": ... },
            "geometry_type": "rectilinear_blockmodel",
            "canonical_identity": "block_id",
            "index_type": index_type
        }

    These attrs are intentionally simpler than the full "parq‑blockmodel"
    production schema, but pseudo‑compatible and useful for testing.

    ─────────────────────────────────────────────────────────────────────────────
    PARAMETERS
    ─────────────────────────────────────────────────────────────────────────────
    shape : tuple[int, int, int]
        Logical block model size (nx, ny, nz).

    block_size : tuple[float, float, float]
        Block dimensions (dx, dy, dz).

    corner : tuple[float, float, float]
        World‑space minimum corner of the unrotated model.

    azimuth, dip, plunge : float
        Rotation angles applied to world centroids.

    parquet_filepath : Path | None
        If supplied, the DataFrame is written to Parquet.

    index_type : {"block_index", "world_centroids", "block_id"}
        Specifies which index is assigned to the returned DataFrame.

    ─────────────────────────────────────────────────────────────────────────────
    RETURNS
    ─────────────────────────────────────────────────────────────────────────────
    DataFrame or Path
        The block model DataFrame, or the Parquet path if saved.
    """

    # ------------------------------------------------------------------
    # Canonical logical indices (i, j, k) in C-order
    # ------------------------------------------------------------------
    ni, nj, nk = shape
    num_blocks = int(np.prod(shape))

    # Use NumPy's C-order unravel to generate (i, j, k)
    rows = np.arange(num_blocks, dtype=int)
    i, j, k = np.unravel_index(rows, shape, order="C")

    # ------------------------------------------------------------------
    # World-space centroids (x, y, z), optionally rotated
    # ------------------------------------------------------------------
    dx, dy, dz = block_size
    cx, cy, cz = corner

    x = cx + (i + 0.5) * dx
    y = cy + (j + 0.5) * dy
    z = cz + (k + 0.5) * dz

    coords = np.column_stack([x, y, z])

    if any(angle != 0.0 for angle in (azimuth, dip, plunge)):
        rotated = rotate_points(points=coords, azimuth=azimuth, dip=dip, plunge=plunge)
        x, y, z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

    # Canonical block id from logical indices in C-order.
    block_id = rows

    # ------------------------------------------------------------------
    # Build DataFrame with canonical logical + world coordinates
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "block_id": block_id,
        "i": i,
        "j": j,
        "k": k,
        "x": x,
        "y": y,
        "z": z,
    })

    # Depth information (using world Z)
    surface_rl = float(np.max(z) + dz / 2.0)
    df["depth"] = surface_rl - z
    df["depth_category"] = pd.cut(
        df["depth"],
        bins=2,
        labels=["shallow", "deep"],
        include_lowest=True,
    ).astype("category")

    # ------------------------------------------------------------------
    # Assign index according to index_type
    # ------------------------------------------------------------------
    if index_type == "block_index":
        df.set_index(["i", "j", "k"], inplace=True)
    elif index_type == "world_centroids":
        df.set_index(["x", "y", "z"], inplace=True)
    elif index_type == "block_id":
        df.set_index("block_id", inplace=True)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    # Lightweight geometry metadata
    df.attrs.setdefault("geometry", {})
    df.attrs["geometry"].update({
        "shape": tuple(shape),
        "block_size": tuple(block_size),
        "corner": tuple(corner),
        "rotation": {"azimuth": azimuth, "dip": dip, "plunge": plunge},
        "geometry_type": "rectilinear_blockmodel",
        "canonical_identity": "block_id",
        "index_type": index_type,
    })

    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath

    return df


def add_gradient_ellipsoid_grade(
        df: pd.DataFrame,
        center: Tuple[float, float, float],
        radii: Tuple[float, float, float],
        grade_min: float = 5.0,
        grade_max: float = 65.0,
        bearing: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0,
        column_name: str = 'grade',
        noise_std: float = 0.1,
        noise_seed: int | None = None,
) -> pd.DataFrame:
    """Add a gradient ellipsoid grade to the block model DataFrame."""
    if df.index.names == ['x', 'y', 'z']:
        coords = np.array(df.index.tolist()) - np.array(center)
    else:
        coords = df[['x', 'y', 'z']].values - np.array(center)
    if any([bearing, dip, plunge]):
        from parq_blockmodel.utils.geometry_utils import rotate_points
        coords = rotate_points(coords, bearing, dip, plunge)
    norm = (
            (coords[:, 0] / radii[0]) ** 2 +
            (coords[:, 1] / radii[1]) ** 2 +
            (coords[:, 2] / radii[2]) ** 2
    )
    inside = norm <= 1
    grad = np.full_like(norm, grade_min, dtype=float)
    grad[inside] = grade_max - (grade_max - grade_min) * np.sqrt(norm[inside])
    if noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        grad += rng.normal(0, noise_std, size=grad.shape)
    df[column_name] = grad
    return df


def create_toy_blockmodel(
        shape=(20, 15, 10),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
        axis_azimuth=0.0,
        axis_dip=0.0,
        axis_plunge=0.0,
        deposit_bearing=20.0,
        deposit_dip=30.0,
        deposit_plunge=10.0,
        grade_name='grade',
        grade_min=50.0,
        grade_max=65.0,
        deposit_center=(10.0, 7.5, 5.0),
        deposit_radii=(8.0, 5.0, 3.0),
        noise_std: float = 0.0,
        noise_rel: float | None = None,
        noise_seed: int | None = None,
        parquet_filepath: Path = None
) -> pd.DataFrame | Path:
    """Create a toy blockmodel with a gradient ellipsoid grade.
    Args:
        shape: Shape of the block model (nx, ny, nz).
        block_size: Size of each block (dx, dy, dz).
        corner: The lower left (minimum) corner of the block model.
        axis_azimuth: The azimuth angle of the block model axis in degrees.
        axis_dip: The dip angle of the block model axis in degrees.
        axis_plunge: The plunge angle of the block model axis in degrees.
        deposit_bearing: The azimuth angle of the deposit in degrees.
        deposit_dip: The dip angle of the deposit in degrees.
        deposit_plunge: The plunge angle of the deposit in degrees.
        grade_name: The name of the column to store the grade values.
        grade_min: The minimum grade value.
        grade_max: The maximum grade value.
        deposit_center: The center of the deposit (x, y, z).
        deposit_radii: The radii of the deposit (rx, ry, rz).
        noise_std: Absolute standard deviation of Gaussian noise added to grades.
        noise_rel: Relative standard deviation expressed as a fraction of
            ``(grade_max - grade_min)``. Mutually exclusive with ``noise_std``.
        noise_seed: Optional random seed for reproducible Gaussian noise.
        parquet_filepath: The file path to save the DataFrame as a Parquet file. If None, returns a DataFrame.

    Returns:
        pd.DataFrame if parquet_filepath is None, else Path to the Parquet file.
    """
    if noise_rel is not None:
        if noise_rel < 0.0:
            raise ValueError("noise_rel must be >= 0.0")
        if noise_std not in (0.0, 0):
            raise ValueError("Specify only one of noise_std or noise_rel")
        noise_std = float(noise_rel) * float(grade_max - grade_min)

    df = create_demo_blockmodel(shape, block_size, corner, axis_azimuth, axis_dip, axis_plunge)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("create_demo_blockmodel returned a path unexpectedly")
    df = add_gradient_ellipsoid_grade(df=df,
                                      center=deposit_center,
                                      radii=deposit_radii,
                                      grade_min=grade_min,
                                      grade_max=grade_max,
                                      bearing=deposit_bearing,
                                      dip=deposit_dip,
                                      plunge=deposit_plunge,
                                      column_name=grade_name,
                                      noise_std=noise_std,
                                      noise_seed=noise_seed)
    if parquet_filepath is not None:
        df.to_parquet(parquet_filepath)
        return parquet_filepath
    return df
