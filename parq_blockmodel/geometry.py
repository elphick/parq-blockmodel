from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Mapping, Any

import json
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parq_blockmodel.types import Point, Vector, Shape3D, BlockSize
from parq_blockmodel.utils.geometry_utils import (
    validate_axes_orthonormal,
)


@dataclass
class RegularGeometry:
    """Dense, metadata‑defined block model geometry (C‑order canonical).

    This class describes *all* block model geometry in parq‑blockmodel.
    It does **not** store x, y, z or i, j, k per block. All coordinates
    are derived from metadata + implicit row ordering.

    ---------------------------------------
    🧭 IMPORTANT, INTERNAL STORAGE ORDERING
    ---------------------------------------

    parq‑blockmodel uses **NumPy C‑order** as its canonical definition of
    block ordering.

    ✔ **C‑order flattening (NumPy default):**
        i varies fastest  
        j varies next  
        k varies slowest

        r = i + ni * (j + nj * k)

    ✔ This matches:
        - Parquet row‑oriented storage naturally  
        - NumPy operations (ravel, reshape)  
        - Efficient dense arrays  
        - Our new metadata‑based geometry model  

    ✔ This does *NOT* match lexicographic MultiIndex sorting:
        .sort_index(["x", "y", "z"])
        which yields:
            x slowest  
            y middle  
            z fastest  
        ❌ This is **NOT** C‑order.
        ❌ This must NOT define canonical storage.

    ❌ **F‑order NOT used** (even though OMF/VTK are sometimes described
       this way). PyVista accepts C‑order 1D arrays perfectly fine, and
       VTK data is exposed to Python as C‑order NumPy arrays anyway.

    Summary:
        **Canonical = C‑order. Do not switch to F‑order.**
        **Do not rely on MultiIndex lexicographic XYZ sorting.**

    ----------------------------------------------------------------------

    Geometry metadata:
        corner      (x0, y0, z0) of the (i=0, j=0, k=0) block *corner*
        block_size  (dx, dy, dz)
        shape       (ni, nj, nk)
        axis_u/v/w  orthonormal basis vectors (rotation)
        srs         optional CRS

    """

    corner: Point
    block_size: BlockSize
    shape: Shape3D
    axis_u: Vector = (1.0, 0.0, 0.0)
    axis_v: Vector = (0.0, 1.0, 0.0)
    axis_w: Vector = (0.0, 0.0, 1.0)
    srs: Optional[str] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not validate_axes_orthonormal(self.axis_u, self.axis_v, self.axis_w):
            raise ValueError("Axis vectors must be orthogonal and normalized.")

        self.corner = tuple(float(v) for v in self.corner)
        self.block_size = tuple(float(v) for v in self.block_size)
        self.shape = tuple(int(v) for v in self.shape)

    # ----------------------------------------------------------------------
    # Centroid calculation (C‑order)
    # ----------------------------------------------------------------------

    @property
    def _centroids(self) -> np.ndarray:
        """Compute XYZ centroids as a (3, N) array in **C‑order**.

        Returns:
            np.ndarray: Array of shape (3, N), where N = ni*nj*nk.
                        First row = X, second = Y, third = Z.
        """
        dx, dy, dz = self.block_size
        ni, nj, nk = self.shape
        cx, cy, cz = self.corner

        u = np.arange(cx + dx/2, cx + dx*ni, dx)
        v = np.arange(cy + dy/2, cy + dy*nj, dy)
        w = np.arange(cz + dz/2, cz + dz*nk, dz)

        uu, vv, ww = np.meshgrid(u, v, w, indexing="ij")
        local = np.vstack([uu.ravel("C"), vv.ravel("C"), ww.ravel("C")])

        R = np.array([self.axis_u, self.axis_v, self.axis_w]).T
        return R @ local

    @property
    def centroid_x(self) -> NDArray[np.floating]:
        return self._centroids[0]

    @property
    def centroid_y(self) -> NDArray[np.floating]:
        return self._centroids[1]

    @property
    def centroid_z(self) -> NDArray[np.floating]:
        return self._centroids[2]

    # ----------------------------------------------------------------------
    # Index conversion API (C‑order guaranteed)
    # ----------------------------------------------------------------------

    def ijk_from_row_index(self, rows):
        """Convert row index/indices into (i, j, k) using **C‑order**."""
        return np.unravel_index(rows, self.shape, order="C")

    def row_index_from_ijk(self, i, j, k):
        """Convert (i, j, k) into row index using **C‑order**."""
        return np.ravel_multi_index((i, j, k), self.shape, order="C")

    def xyz_from_ijk(self, i, j, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert logical indices to world-space centroid coordinates."""
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        k = np.asarray(k, dtype=float)

        dx, dy, dz = self.block_size
        cx, cy, cz = self.corner

        local = np.vstack([
            cx + (i + 0.5) * dx,
            cy + (j + 0.5) * dy,
            cz + (k + 0.5) * dz,
        ])
        R = np.array([self.axis_u, self.axis_v, self.axis_w], dtype=float).T
        world = R @ local
        return world[0], world[1], world[2]

    def xyz_from_row_index(self, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert C-order row index/indices to world-space centroid coordinates."""
        i, j, k = self.ijk_from_row_index(rows)
        return self.xyz_from_ijk(i, j, k)

    def ijk_from_xyz(self, x, y, z, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map world-space centroid coordinates to integer logical ijk indices."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        pts = np.vstack([x, y, z])
        R = np.array([self.axis_u, self.axis_v, self.axis_w], dtype=float).T
        # Axes are orthonormal, so inverse is transpose.
        local = R.T @ pts

        dx, dy, dz = self.block_size
        cx, cy, cz = self.corner

        fi = (local[0] - (cx + 0.5 * dx)) / dx
        fj = (local[1] - (cy + 0.5 * dy)) / dy
        fk = (local[2] - (cz + 0.5 * dz)) / dz

        i = np.rint(fi).astype(np.int64)
        j = np.rint(fj).astype(np.int64)
        k = np.rint(fk).astype(np.int64)

        if not (
            np.all(np.abs(fi - i) <= tol)
            and np.all(np.abs(fj - j) <= tol)
            and np.all(np.abs(fk - k) <= tol)
        ):
            raise ValueError("Centroid coordinates are not aligned to geometry centroid lattice.")

        ni, nj, nk = self.shape
        if not (
            np.all((i >= 0) & (i < ni))
            and np.all((j >= 0) & (j < nj))
            and np.all((k >= 0) & (k < nk))
        ):
            raise ValueError("Centroid coordinates map outside geometry bounds.")

        return i, j, k

    def row_index_from_xyz(self, x, y, z, tol: float = 1e-6) -> np.ndarray:
        """Map world-space centroid coordinates directly to C-order row indices."""
        i, j, k = self.ijk_from_xyz(x, y, z, tol=tol)
        return self.row_index_from_ijk(i, j, k)

    # ----------------------------------------------------------------------
    # Convenience exports (not used internally)
    # ----------------------------------------------------------------------

    def to_multi_index_xyz(self) -> pd.MultiIndex:
        """Return XYZ centroid coordinates as a MultiIndex.

        NOTE:
            This MultiIndex preserves **C‑order row layout**.
            If you want lexicographic ('x slowest, z fastest'), sort it
            explicitly:

                mi = geom.to_multi_index_xyz().sort_index()
        """
        coords = np.column_stack([self.centroid_x, self.centroid_y, self.centroid_z])
        return pd.MultiIndex.from_arrays(coords.T, names=["x", "y", "z"])

    def to_multi_index_ijk(self) -> pd.MultiIndex:
        """Export (i, j, k) indices as a MultiIndex in **C‑order**."""
        N = np.prod(self.shape)
        rows = np.arange(N)
        i, j, k = self.ijk_from_row_index(rows)
        return pd.MultiIndex.from_arrays([i, j, k], names=["i", "j", "k"])

    def to_dataframe_xyz(self) -> pd.DataFrame:
        """Return a DataFrame of XYZ centroids for export/inspection."""
        return pd.DataFrame({
            "x": self.centroid_x,
            "y": self.centroid_y,
            "z": self.centroid_z,
        })

    # Convenience alias for older callers expecting ``to_dataframe``.

    def to_dataframe(self) -> pd.DataFrame:
        """Backward-compatible alias for :meth:`to_dataframe_xyz`."""
        return self.to_dataframe_xyz()

    # ----------------------------------------------------------------------
    # Metadata I/O
    # ----------------------------------------------------------------------

    def to_metadata_dict(self) -> dict:
        """Serialize geometry metadata for Parquet storage."""
        return {
            "corner": list(self.corner),
            "block_size": list(self.block_size),
            "shape": list(self.shape),
            "axis_u": list(self.axis_u),
            "axis_v": list(self.axis_v),
            "axis_w": list(self.axis_w),
            "srs": self.srs,
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "RegularGeometry":
        """Reconstruct geometry from stored metadata."""
        return cls(
            corner=tuple(meta["corner"]),
            block_size=tuple(meta["block_size"]),
            shape=tuple(meta["shape"]),
            axis_u=tuple(meta["axis_u"]),
            axis_v=tuple(meta["axis_v"]),
            axis_w=tuple(meta["axis_w"]),
            srs=meta.get("srs"),
        )

    # ------------------------------------------------------------------
    # Metadata helpers for DataFrame.attrs and Parquet key_value_metadata
    # ------------------------------------------------------------------

    @classmethod
    def from_attrs(
        cls,
        attrs: Mapping[str, Any],
        key: str = "parq-blockmodel",
    ) -> "RegularGeometry":
        """Reconstruct geometry from a ``DataFrame.attrs``-style mapping.

        ``attrs[key]`` is expected to contain the dict produced by
        :meth:`to_metadata_dict` (or a compatible future schema that
        :meth:`from_metadata` can consume).
        """

        if key not in attrs:
            raise KeyError(f"Geometry metadata not found in attrs under key {key!r}.")

        payload = attrs[key]
        if not isinstance(payload, dict):
            raise TypeError(
                "Expected attrs[%r] to be a dict compatible with RegularGeometry.to_metadata_dict(), "
                "got %s." % (key, type(payload).__name__)
            )

        return cls.from_metadata(payload)

    @classmethod
    def from_parquet_metadata(
        cls,
        metadata: Any,
        key: str = "parq-blockmodel",
    ) -> "RegularGeometry":
        """Reconstruct geometry from Parquet file metadata.

        Parameters
        ----------
        metadata:
            Either a ``pyarrow.parquet.FileMetaData`` instance or a
            mapping of key-value metadata (usually ``dict[str, str]``).
        key:
            Metadata key under which the geometry payload is stored.

        Returns
        -------
        RegularGeometry

        Raises
        ------
        KeyError
            If the key is not present in the provided metadata.
        TypeError, ValueError
            If the payload cannot be interpreted as valid geometry
            metadata.
        """

        # Normalise to a simple dict[str, Any]
        meta_dict: dict[str, Any]
        if hasattr(metadata, "metadata"):
            # Likely a pyarrow.parquet.FileMetaData
            raw = metadata.metadata or {}
            meta_dict = {
                (k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)):
                (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
                for k, v in raw.items()
            }
        elif isinstance(metadata, Mapping):
            meta_dict = dict(metadata)
        else:
            raise TypeError(
                "metadata must be a pyarrow.parquet.FileMetaData or a mapping of key-value "
                f"strings, got {type(metadata).__name__}."
            )

        if key not in meta_dict:
            raise KeyError(f"Geometry metadata key {key!r} not found in Parquet metadata.")

        raw_value = meta_dict[key]

        # Allow already-decoded dict payloads for convenience/testing.
        if isinstance(raw_value, dict):
            return cls.from_metadata(raw_value)

        if not isinstance(raw_value, str):
            raise TypeError(
                f"Expected Parquet metadata value for key {key!r} to be a JSON string or dict, "
                f"got {type(raw_value).__name__}."
            )

        try:
            payload = json.loads(raw_value)
        except Exception as exc:  # pragma: no cover - error path
            raise ValueError(
                f"Failed to parse geometry metadata JSON in key {key!r}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"Geometry metadata under key {key!r} must decode to a dict, got {type(payload).__name__}."
            )

        return cls.from_metadata(payload)

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    @property
    def is_rotated(self) -> bool:
        """Return True if axes differ from the identity orientation."""
        return not (
            np.allclose(self.axis_u, (1.0, 0.0, 0.0))
            and np.allclose(self.axis_v, (0.0, 1.0, 0.0))
            and np.allclose(self.axis_w, (0.0, 0.0, 1.0))
        )

    @property
    def extents(self) -> tuple[float, float, float, float, float, float]:
        """Return the axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax).

        For unrotated geometries this is derived directly from corner, block_size,
        and shape. For rotated geometries, we conservatively compute extents from
        the centroid cloud.
        """

        if not self.is_rotated:
            cx, cy, cz = self.corner
            dx, dy, dz = self.block_size
            ni, nj, nk = self.shape
            xmin, ymin, zmin = cx, cy, cz
            xmax = cx + ni * dx
            ymax = cy + nj * dy
            zmax = cz + nk * dz
            return xmin, xmax, ymin, ymax, zmin, zmax

        # Rotated: fall back to centroid cloud bounds
        xs = self.centroid_x
        ys = self.centroid_y
        zs = self.centroid_z
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.min()), float(zs.max())

    # ------------------------------------------------------------------
    # PyVista helper
    # ------------------------------------------------------------------

    def to_pyvista(self):
        """Return a ``pyvista.ImageData`` representing the dense grid.

        This constructs a uniform rectilinear grid whose cell centers match
        the XYZ centroids defined by this geometry. Rotation (axis_u/v/w) is
        applied to the local grid axes.
        """

        import pyvista as pv

        dx, dy, dz = self.block_size
        ni, nj, nk = self.shape
        cx, cy, cz = self.corner

        # Compute origin at the minimum corner of the grid
        origin = (cx, cy, cz)
        spacing = (dx, dy, dz)

        # Voxel dimensions: number of points = cells + 1 along each axis
        dimensions = (ni + 1, nj + 1, nk + 1)

        grid = pv.ImageData()
        grid.origin = origin
        grid.spacing = spacing
        grid.dimensions = dimensions
        return grid

    @classmethod
    def from_parquet(
        cls,
        filepath,
        axis_azimuth: float = 0.0,
        axis_dip: float = 0.0,
        axis_plunge: float = 0.0,
        chunk_size: int = 1_000_000,
    ) -> "RegularGeometry":
        """Reconstruct geometry from a Parquet file.

        Preferred path is to read geometry metadata from the Parquet
        key_value_metadata under the reserved key ``"parq-blockmodel"``.
        If that key is missing, fall back to centroid-based inference
        using ``x, y, z`` columns and provided rotation angles.

        .. todo:: Consider efficiency gains by staying in numpy versus using python sets.
        """

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filepath)
        try:
            return cls.from_parquet_metadata(pf.metadata)
        except KeyError:
            pass

        # Fallback: infer from centroids and rotation angles.
        columns = set(pf.schema.names)
        if not {"x", "y", "z"}.issubset(columns):
            raise ValueError("Parquet file must contain x, y, z columns to infer geometry.")

        from parq_blockmodel.utils.geometry_utils import angles_to_axes

        axis_u, axis_v, axis_w = angles_to_axes(
            axis_azimuth=axis_azimuth,
            axis_dip=axis_dip,
            axis_plunge=axis_plunge,
        )

        R = np.array([axis_u, axis_v, axis_w], dtype=float).T

        local_x_values: set[float] = set()
        local_y_values: set[float] = set()
        local_z_values: set[float] = set()

        for batch in pf.iter_batches(columns=["x", "y", "z"], batch_size=chunk_size):
            table = batch.to_pydict()
            x = np.asarray(table["x"], dtype=float)
            y = np.asarray(table["y"], dtype=float)
            z = np.asarray(table["z"], dtype=float)
            pts = np.vstack([x, y, z])
            local = R.T @ pts

            local_x_values.update(np.unique(local[0]).tolist())
            local_y_values.update(np.unique(local[1]).tolist())
            local_z_values.update(np.unique(local[2]).tolist())

        ux = np.array(sorted(local_x_values), dtype=float)
        uy = np.array(sorted(local_y_values), dtype=float)
        uz = np.array(sorted(local_z_values), dtype=float)

        if ux.size < 2 or uy.size < 2 or uz.size < 2:
            raise ValueError("Cannot infer block size/shape from degenerate centroid coordinates.")

        dx = float(np.diff(ux).min())
        dy = float(np.diff(uy).min())
        dz = float(np.diff(uz).min())

        ni = int(round((ux.max() - ux.min()) / dx)) + 1
        nj = int(round((uy.max() - uy.min()) / dy)) + 1
        nk = int(round((uz.max() - uz.min()) / dz)) + 1

        # Corner is half a block before the minimum centroid along each axis.
        corner = (float(ux.min() - dx / 2), float(uy.min() - dy / 2), float(uz.min() - dz / 2))

        block_size: BlockSize = (dx, dy, dz)
        shape: Shape3D = (ni, nj, nk)


        return cls(
            corner=corner,
            block_size=block_size,
            shape=shape,
            axis_u=axis_u,
            axis_v=axis_v,
            axis_w=axis_w,
        )

