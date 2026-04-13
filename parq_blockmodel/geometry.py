from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Mapping, Any

import json
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parq_blockmodel.types import Point, Vector, Shape3D, BlockSize
from parq_blockmodel.utils.geometry_utils import (
    validate_axes_orthonormal,
)


@dataclass
class LocalGeometry:
    """Local dense grid geometry with C-order indexing.

    This class models the logical/local lattice only:
    - local corner/origin
    - block size
    - shape

    It has no rotation, CRS, or world-frame meaning.
    """

    corner: Point
    block_size: BlockSize
    shape: Shape3D

    def __post_init__(self):
        corner = [float(v) for v in self.corner]
        block_size = [float(v) for v in self.block_size]
        shape = [int(v) for v in self.shape]

        self.corner = (corner[0], corner[1], corner[2])
        self.block_size = (block_size[0], block_size[1], block_size[2])
        self.shape = (shape[0], shape[1], shape[2])

    @property
    def centroids_uvw(self) -> np.ndarray:
        """Compute local centroids as a (3, N) array in C-order."""
        dx, dy, dz = self.block_size
        ni, nj, nk = self.shape
        cu, cv, cw = self.corner

        u = np.arange(cu + dx / 2, cu + dx * ni, dx)
        v = np.arange(cv + dy / 2, cv + dy * nj, dy)
        w = np.arange(cw + dz / 2, cw + dz * nk, dz)

        uu, vv, ww = np.meshgrid(u, v, w, indexing="ij")
        return np.vstack([uu.ravel("C"), vv.ravel("C"), ww.ravel("C")])

    def ijk_from_row_index(self, rows):
        """Convert row index/indices into (i, j, k) using C-order."""
        return np.unravel_index(rows, self.shape, order="C")

    def row_index_from_ijk(self, i, j, k):
        """Convert (i, j, k) into row index using C-order."""
        return np.ravel_multi_index((i, j, k), self.shape, order="C")

    def uvw_from_ijk(self, i, j, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert logical indices to local centroid coordinates."""
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        k = np.asarray(k, dtype=float)

        dx, dy, dz = self.block_size
        cu, cv, cw = self.corner
        u = cu + (i + 0.5) * dx
        v = cv + (j + 0.5) * dy
        w = cw + (k + 0.5) * dz
        return u, v, w

    def ijk_from_uvw(self, u, v, w, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map local centroid coordinates to integer logical ijk indices."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        w = np.asarray(w, dtype=float)

        dx, dy, dz = self.block_size
        cu, cv, cw = self.corner

        fi = (u - (cu + 0.5 * dx)) / dx
        fj = (v - (cv + 0.5 * dy)) / dy
        fk = (w - (cw + 0.5 * dz)) / dz

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


@dataclass
class WorldFrame:
    """World embedding for local geometry.

    Holds world origin, orthonormal axes, and optional CRS. It is
    responsible for local<->world coordinate transforms.
    """

    world_origin: Point = (0.0, 0.0, 0.0)
    axis_u: Vector = (1.0, 0.0, 0.0)
    axis_v: Vector = (0.0, 1.0, 0.0)
    axis_w: Vector = (0.0, 0.0, 1.0)
    srs: Optional[str] = None

    def __post_init__(self):
        if not validate_axes_orthonormal(self.axis_u, self.axis_v, self.axis_w):
            raise ValueError("Axis vectors must be orthogonal and normalized.")

        world_origin = [float(v) for v in self.world_origin]
        axis_u = [float(v) for v in self.axis_u]
        axis_v = [float(v) for v in self.axis_v]
        axis_w = [float(v) for v in self.axis_w]

        self.world_origin = (world_origin[0], world_origin[1], world_origin[2])
        self.axis_u = (axis_u[0], axis_u[1], axis_u[2])
        self.axis_v = (axis_v[0], axis_v[1], axis_v[2])
        self.axis_w = (axis_w[0], axis_w[1], axis_w[2])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return matrix whose columns are world axes (u, v, w)."""
        return np.array([self.axis_u, self.axis_v, self.axis_w], dtype=float).T

    def local_to_world(self, local_points: np.ndarray) -> np.ndarray:
        """Map local 3xN points into world coordinates."""
        origin = np.asarray(self.world_origin, dtype=float).reshape(3, 1)
        return origin + self.rotation_matrix @ local_points

    def world_to_local(self, world_points: np.ndarray) -> np.ndarray:
        """Map world 3xN points into local coordinates."""
        origin = np.asarray(self.world_origin, dtype=float).reshape(3, 1)
        # Axes are orthonormal, so inverse is transpose.
        return self.rotation_matrix.T @ (world_points - origin)


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

    Internal frame split:
        LocalGeometry: local lattice (corner, block_size, shape, C-order)
        WorldFrame:    world embedding (origin, axis_u/v/w, srs)

    Geometry metadata (unchanged schema):
        corner      local (u0, v0, w0) corner of the (i=0, j=0, k=0) block
        block_size  (du, dv, dw)
        shape       (ni, nj, nk)
        axis_u/v/w  orthonormal world basis vectors
        srs         optional CRS for world coordinates

    """

    local: LocalGeometry
    world: WorldFrame
    schema_version: str = "1.0"
    world_id_encoding: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """Validate and initialize geometry components."""
        # No additional initialization needed; LocalGeometry and WorldFrame
        # validate themselves via their own __post_init__ methods.
        pass

    @classmethod
    def create(
        cls,
        corner: Point = (0.0, 0.0, 0.0),
        block_size: BlockSize = (1.0, 1.0, 1.0),
        shape: Shape3D = (1, 1, 1),
        axis_u: Vector = (1.0, 0.0, 0.0),
        axis_v: Vector = (0.0, 1.0, 0.0),
        axis_w: Vector = (0.0, 0.0, 1.0),
        srs: Optional[str] = None,
    ) -> "RegularGeometry":
        """Convenience factory for creating RegularGeometry with keyword arguments.

        This is the recommended way to construct geometries when you have
        individual parameters rather than pre-built LocalGeometry and WorldFrame.

        Parameters
        ----------
        corner : Point, optional
            Grid origin (x₀, y₀, z₀). Default: (0, 0, 0)
        block_size : BlockSize, optional
            Block dimensions (dx, dy, dz). Default: (1, 1, 1)
        shape : Shape3D, optional
            Number of blocks (nᵢ, nⱼ, nₖ). Default: (1, 1, 1)
        axis_u, axis_v, axis_w : Vector, optional
            Orthonormal basis vectors for world embedding. Default: identity orientation
        srs : str, optional
            Spatial reference system identifier

        Returns
        -------
        RegularGeometry
        """
        return cls(
            local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
            world=WorldFrame(world_origin=corner, axis_u=axis_u, axis_v=axis_v, axis_w=axis_w, srs=srs),
        )

    def __init__(
        self,
        local: Optional[LocalGeometry] = None,
        world: Optional[WorldFrame] = None,
        schema_version: str = "1.0",
        world_id_encoding: Optional[dict[str, Any]] = None,
        # Backward-compatible keyword arguments
        corner: Optional[Point] = None,
        block_size: Optional[BlockSize] = None,
        shape: Optional[Shape3D] = None,
        axis_u: Optional[Vector] = None,
        axis_v: Optional[Vector] = None,
        axis_w: Optional[Vector] = None,
        srs: Optional[str] = None,
    ):
        """Initialize RegularGeometry.

        Can be called with either:
        1. New style: RegularGeometry(local=..., world=...)
        2. Old style: RegularGeometry(corner=..., block_size=..., shape=..., axis_u=...)
        3. Or mix default parameters with keyword overrides
        """
        # If old-style parameters are provided, use them to construct local/world
        if any(p is not None for p in [corner, block_size, shape, axis_u, axis_v, axis_w, srs]):
            if local is not None or world is not None:
                raise ValueError(
                    "Cannot specify both new-style (local/world) and old-style "
                    "(corner/block_size/shape/axis_*) parameters"
                )
            # Use old-style parameters
            local = LocalGeometry(
                corner=corner or (0.0, 0.0, 0.0),
                block_size=block_size or (1.0, 1.0, 1.0),
                shape=shape or (1, 1, 1),
            )
            world = WorldFrame(
                world_origin=corner or (0.0, 0.0, 0.0),
                axis_u=axis_u or (1.0, 0.0, 0.0),
                axis_v=axis_v or (0.0, 1.0, 0.0),
                axis_w=axis_w or (0.0, 0.0, 1.0),
                srs=srs,
            )
        elif local is None or world is None:
            # If neither new-style nor old-style provided, use defaults
            local = local or LocalGeometry(
                corner=(0.0, 0.0, 0.0),
                block_size=(1.0, 1.0, 1.0),
                shape=(1, 1, 1),
            )
            world = world or WorldFrame()

        self.local = local
        self.world = world
        self.schema_version = schema_version
        self.world_id_encoding = world_id_encoding

    # ----------------------------------------------------------------------
    # Centroid calculation (C‑order)
    # ----------------------------------------------------------------------

    @property
    def _centroids(self) -> np.ndarray:
        """Compute world XYZ centroids as a (3, N) array in **C‑order**.

        Returns:
            np.ndarray: Array of shape (3, N), where N = ni*nj*nk.
                        First row = X, second = Y, third = Z.
        """
        local = self.local.centroids_uvw
        return self.world.local_to_world(local)

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
        return self.local.ijk_from_row_index(rows)

    def row_index_from_ijk(self, i, j, k):
        """Convert (i, j, k) into row index using **C‑order**."""
        return self.local.row_index_from_ijk(i, j, k)

    def xyz_from_ijk(self, i, j, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert logical indices to world-space centroid coordinates."""
        u, v, w = self.local.uvw_from_ijk(i, j, k)
        local = np.vstack([u, v, w])
        world = self.world.local_to_world(local)
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

        world = np.vstack([x, y, z])
        local = self.world.world_to_local(world)
        return self.local.ijk_from_uvw(local[0], local[1], local[2], tol=tol)

    def row_index_from_xyz(self, x, y, z, tol: float = 1e-6) -> np.ndarray:
        """Map world-space centroid coordinates directly to C-order row indices."""
        i, j, k = self.ijk_from_xyz(x, y, z, tol=tol)
        return np.asarray(self.row_index_from_ijk(i, j, k), dtype=np.int64)

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
        N = np.prod(self.local.shape)
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
        metadata = {
            "schema_version": self.schema_version,
            "corner": list(self.local.corner),
            "block_size": list(self.local.block_size),
            "shape": list(self.local.shape),
            "axis_u": list(self.world.axis_u),
            "axis_v": list(self.world.axis_v),
            "axis_w": list(self.world.axis_w),
            "srs": self.world.srs,
        }
        if self.world_id_encoding is not None:
            metadata["world_id_encoding"] = self.world_id_encoding
        return metadata

    @classmethod
    def from_metadata(cls, meta: dict) -> "RegularGeometry":
        """Reconstruct geometry from stored metadata."""
        return cls(
            local=LocalGeometry(
                corner=tuple(meta["corner"]),
                block_size=tuple(meta["block_size"]),
                shape=tuple(meta["shape"]),
            ),
            world=WorldFrame(
                axis_u=tuple(meta["axis_u"]),
                axis_v=tuple(meta["axis_v"]),
                axis_w=tuple(meta["axis_w"]),
                srs=meta.get("srs"),
            ),
            schema_version=str(meta.get("schema_version", "1.0")),
            world_id_encoding=meta.get("world_id_encoding"),
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
            np.allclose(self.world.axis_u, (1.0, 0.0, 0.0))
            and np.allclose(self.world.axis_v, (0.0, 1.0, 0.0))
            and np.allclose(self.world.axis_w, (0.0, 0.0, 1.0))
        )

    # Backward-compatible property accessors
    @property
    def corner(self) -> Point:
        """Backward-compatible access to local corner."""
        return self.local.corner

    @property
    def block_size(self) -> BlockSize:
        """Backward-compatible access to block size."""
        return self.local.block_size

    @property
    def shape(self) -> Shape3D:
        """Backward-compatible access to grid shape."""
        return self.local.shape

    @property
    def axis_u(self) -> Vector:
        """Backward-compatible access to U-axis."""
        return self.world.axis_u

    @property
    def axis_v(self) -> Vector:
        """Backward-compatible access to V-axis."""
        return self.world.axis_v

    @property
    def axis_w(self) -> Vector:
        """Backward-compatible access to W-axis."""
        return self.world.axis_w

    @property
    def srs(self) -> Optional[str]:
        """Backward-compatible access to spatial reference system."""
        return self.world.srs

    @property
    def extents(self) -> tuple[float, float, float, float, float, float]:
        """Return the axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax).

        For unrotated geometries this is derived directly from corner, block_size,
        and shape. For rotated geometries, we conservatively compute extents from
        the centroid cloud.
        """

        if not self.is_rotated:
            cx, cy, cz = self.local.corner
            dx, dy, dz = self.local.block_size
            ni, nj, nk = self.local.shape
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

    def to_pyvista(self, *, frame: str = "world"):
        """Return a ``pyvista.ImageData`` representing the dense grid.

        Parameters
        ----------
        frame : {"world", "local"}, optional
            Coordinate frame used for the returned grid.
            - ``"world"`` (default): apply world orientation from ``axis_u/v/w``.
            - ``"local"``: return axis-aligned local grid (no rotation).

        Notes
        -----
        In ``"world"`` mode, rotation is encoded using the ImageData
        direction matrix, so plotting reflects geometry orientation.
        """

        import pyvista as pv

        if frame not in {"world", "local"}:
            raise ValueError("frame must be one of {'world', 'local'}")

        dx, dy, dz = self.local.block_size
        ni, nj, nk = self.local.shape

        spacing = (dx, dy, dz)

        # Voxel dimensions: number of points = cells + 1 along each axis
        dimensions = (ni + 1, nj + 1, nk + 1)

        grid = pv.ImageData()
        if frame == "world":
            local_origin = np.asarray(self.local.corner, dtype=float).reshape(3, 1)
            origin = tuple(self.world.local_to_world(local_origin).ravel())
            grid.origin = origin
            grid.direction_matrix = self.world.rotation_matrix
        else:
            grid.origin = tuple(self.local.corner)
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
            local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
            world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
        )
