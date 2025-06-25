"""
geometry.py

This module defines the Geometry class and its subclasses for handling block model geometries.

Main API:

- Geometry: Abstract base class for block model geometries.
- RegularGeometry: Concrete class for regular block model geometries.

"""

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

from numpy.typing import NDArray

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import centroid

from parq_blockmodel.types import Point, Vector, Shape3D, MinMax, BlockSize
from parq_blockmodel.utils.geometry_utils import validate_axes_orthonormal, rotation_to_axis_orientation, rotate_points
from parq_blockmodel.utils.spatial_encoding import multiindex_to_encoded_index

if TYPE_CHECKING:
    import pyvista as pv


class Geometry(ABC):
    """Base class for geometry objects.

    The geometry associated with omf block models are not defined by block centroids, and vary by block model type.
    In the pandas representation, the geometry is defined by the block centroids, so this class is used to
    define the geometry in terms of block centroids.
    Additionally, other properties of the geometry are defined here, such as the shape of the geometry.

    Attributes (in omf and pyvista) are stored in Fortran 'F' order, meaning that the last index changes the fastest.
    Hence, the MultiIndex levels need to be sorted by 'z', 'y', 'x', to align with the Fortran order.
    This has x changing fastest, z changing slowest.

    """

    corner: Point
    axis_u: Vector = (1, 0, 0)
    axis_v: Vector = (0, 1, 0)
    axis_w: Vector = (0, 0, 1)
    srs: Optional[str] = None  # Spatial Reference System, e.g. EPSG code

    _shape: Optional[Shape3D] = None
    _is_regular: Optional[bool] = None
    _logger: logging.Logger = logging.getLogger(__name__)

    def to_summary_json(self) -> str:
        """Convert the geometry to a JSON string.

        Returns:
            str: The JSON string representing the geometry.
        """
        return json.dumps(self.summary)

    def to_json_file(self, json_filepath: Path) -> Path:
        """Write the Geometry to a JSON file.

        Args:
            json_filepath (Path): The path to write the JSON file.

        Returns:
            Path to the json file.
        """
        json_filepath.write_text(self.to_json())
        return json_filepath

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def is_regular(self) -> bool:
        pass

    @property
    def is_rotated(self) -> bool:
        """Check if the geometry is rotated."""
        axes = np.array([self.axis_u, self.axis_v, self.axis_w])
        return not np.allclose(axes, np.eye(3), atol=1e-8)

    @property
    @abstractmethod
    def _centroids(self) -> np.ndarray:
        """Return the centroids as a (3, N) array."""
        pass

    @property
    @abstractmethod
    def centroid_u(self) -> NDArray[np.floating]:
        """Return the unique u coordinates of the centroids."""
        pass

    @property
    @abstractmethod
    def centroid_v(self) -> NDArray[np.floating]:
        """Return the unique v coordinates of the centroids."""
        pass

    @property
    @abstractmethod
    def centroid_w(self) -> NDArray[np.floating]:
        """Return the unique w coordinates of the centroids."""
        pass

    @property
    @abstractmethod
    def centroid_x(self) -> NDArray[np.floating]:
        """Return the x coordinates of the centroids."""
        pass

    @property
    @abstractmethod
    def centroid_y(self) -> NDArray[np.floating]:
        """Return the y coordinates of the centroids."""
        pass

    @property
    @abstractmethod
    def centroid_z(self) -> NDArray[np.floating]:
        """Return the z coordinates of the centroids."""
        pass

    @property
    def num_blocks(self) -> int:
        return int(np.prod(self.shape))

    @property
    def shape(self) -> Shape3D:
        if self._shape is None:
            self._shape = (
                len(self.centroid_x),
                len(self.centroid_y),
                len(self.centroid_z),
            )
        return self._shape

    @property
    @abstractmethod
    def extents(self) -> tuple[MinMax, MinMax, MinMax]:
        pass

    @property
    def bounding_box(self) -> tuple[MinMax, MinMax]:
        return self.extents[0], self.extents[1]

    @property
    @abstractmethod
    def summary(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_multi_index(cls, index: pd.MultiIndex):
        pass

    @abstractmethod
    def to_multi_index(self) -> pd.MultiIndex:
        pass

    @abstractmethod
    def nearest_centroid_lookup(self, x: float, y: float, z: float) -> Point:
        pass


@dataclass
class RegularGeometry(Geometry):
    """Regular geometry data class.

    """

    corner: Point
    block_size: BlockSize
    shape: Shape3D
    axis_u: Vector = (1, 0, 0)
    axis_v: Vector = (0, 1, 0)
    axis_w: Vector = (0, 0, 1)
    srs: Optional[str] = None  # Spatial Reference System, e.g. EPSG code

    def __post_init__(self):
        if not validate_axes_orthonormal(self.axis_u, self.axis_v, self.axis_w):
            raise ValueError("Axis vectors must be orthogonal and normalized.")

    def __repr__(self):
        return f"RegularGeometry: {self.summary}"

    def __str__(self):
        return f"RegularGeometry: {self.summary}"

    @property
    def is_regular(self) -> bool:
        return True

    @cached_property
    def _centroids(self) -> np.ndarray:
        # Compute axis-aligned centroids
        x = np.arange(self.corner[0] + self.block_size[0] / 2,
                      self.corner[0] + self.block_size[0] * self.shape[0],
                      self.block_size[0])
        y = np.arange(self.corner[1] + self.block_size[1] / 2,
                      self.corner[1] + self.block_size[1] * self.shape[1],
                      self.block_size[1])
        z = np.arange(self.corner[2] + self.block_size[2] / 2,
                      self.corner[2] + self.block_size[2] * self.shape[2],
                      self.block_size[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        centroids = np.vstack([xx.ravel(order='C'), yy.ravel(order='C'), zz.ravel(order='C')])
        # Apply rotation
        rotation_matrix = np.array([self.axis_u, self.axis_v, self.axis_w]).T
        rotated = rotation_matrix @ centroids
        return rotated  # shape: (3, N)

    @property
    def centroid_u(self) -> np.ndarray:
        return np.unique(self._centroids[0])

    @property
    def centroid_v(self) -> np.ndarray:
        return np.unique(self._centroids[1])

    @property
    def centroid_w(self) -> np.ndarray:
        return np.unique(self._centroids[2])

    @property
    def centroid_x(self) -> np.ndarray:
        return self._centroids[0]

    @property
    def centroid_y(self) -> np.ndarray:
        return self._centroids[1]

    @property
    def centroid_z(self) -> np.ndarray:
        return self._centroids[2]

    @property
    def shape(self) -> Shape3D:
        return self._shape

    @shape.setter
    def shape(self, value: Shape3D) -> None:
        self._shape = value

    @property
    def extents(self) -> tuple[MinMax, MinMax, MinMax]:
        return (
            (
                float(self.centroid_x[0] - self.block_size[0] / 2),
                float(self.centroid_x[-1] + self.block_size[0] / 2),
            ),
            (
                float(self.centroid_y[0] - self.block_size[1] / 2),
                float(self.centroid_y[-1] + self.block_size[1] / 2),
            ),
            (
                float(self.centroid_z[0] - self.block_size[2] / 2),
                float(self.centroid_z[-1] + self.block_size[2] / 2),
            ),
        )

    @property
    def axis_angles(self):
        """Return (azimuth, dip, plunge) corresponding to axis_u, axis_v, axis_w."""
        from parq_blockmodel.utils.geometry_utils import axis_orientation_to_rotation
        return axis_orientation_to_rotation(self.axis_u, self.axis_v, self.axis_w)

    @property
    def summary(self) -> dict:
        return {
            "corner": tuple(self.corner),
            "axis_angles": tuple(self.axis_angles),
            "block_size": self.block_size,
            "block_count": self.num_blocks,
            "shape": self.shape,
            "is_regular": self.is_regular,
            "extents": self.extents,
            "bounding_box": self.bounding_box,
        }

    @classmethod
    def from_parquet(cls, filepath: Path,
                     axis_azimuth: float = 0.0,
                     axis_dip: float = 0.0,
                     axis_plunge: float = 0.0
                     ) -> "RegularGeometry":
        import pyarrow.parquet as pq
        columns = pq.ParquetFile(filepath).schema.names
        if not {"x", "y", "z"}.issubset(columns):
            raise ValueError("Parquet file must contain 'x', 'y', 'z' columns.")

        # Read the Parquet file to get the index, whether file was written by pandas or not
        centroid_cols = ["x", "y", "z"]
        centroids: pd.DataFrame = pq.read_table(filepath, columns=centroid_cols).to_pandas()

        if centroids.index.names == centroid_cols:
            index = centroids.index
        else:
            if centroids.empty:
                raise ValueError("Parquet file is empty or does not contain valid centroid data.")
            index = centroids.set_index(["x", "y", "z"]).index
        # Create a RegularGeometry from the MultiIndex
        return cls.from_multi_index(index, axis_azimuth=axis_azimuth, axis_dip=axis_dip, axis_plunge=axis_plunge)

    @classmethod
    def from_multi_index(cls, index: pd.MultiIndex,
                         axis_azimuth: float = 0.0,
                         axis_dip: float = 0.0,
                         axis_plunge: float = 0.0
                         ) -> "RegularGeometry":
        if not {"x", "y", "z"}.issubset(index.names):
            raise ValueError("Index must contain the levels 'x', 'y', 'z'.")

        axis_u, axis_v, axis_w = rotation_to_axis_orientation(axis_azimuth, axis_dip, axis_plunge)

        x = np.sort(index.get_level_values("x").unique())
        y = np.sort(index.get_level_values("y").unique())
        z = np.sort(index.get_level_values("z").unique())

        dx = np.unique(np.diff(x))
        dy = np.unique(np.diff(y))
        dz = np.unique(np.diff(z))

        block_size = float(dx.min()), float(dy.min()), float(dz.min())

        # Compute the expected corner as if the minimum centroid is the first block
        corner_x = float(x[0] - block_size[0] / 2)
        corner_y = float(y[0] - block_size[1] / 2)
        corner_z = float(z[0] - block_size[2] / 2)

        # Compute the shape, accounting for sparsity in the index
        shape = (int((max(x) - min(x)) / block_size[0]) + 1,
                 int((max(y) - min(y)) / block_size[1]) + 1,
                 int((max(z) - min(z)) / block_size[2]) + 1)

        return cls(corner=(corner_x, corner_y, corner_z),
                   axis_u=axis_u,
                   axis_v=axis_v,
                   axis_w=axis_w,
                   block_size=block_size,
                   shape=shape)

    @classmethod
    def from_extents(cls, extents: tuple[MinMax, MinMax, MinMax],
                     block_size: BlockSize,
                     axis_u: Vector = (1, 0, 0),
                     axis_v: Vector = (0, 1, 0),
                     axis_w: Vector = (0, 0, 1),
                     ) -> "RegularGeometry":
        """Create a RegularGeometry from extents."""
        min_x, max_x = extents[0]
        min_y, max_y = extents[1]
        min_z, max_z = extents[2]

        corner = (min_x, min_y, min_z)
        shape = (
            int(math.ceil((max_x - min_x) / block_size[0])),
            int(math.ceil((max_y - min_y) / block_size[1])),
            int(math.ceil((max_z - min_z) / block_size[2])),
        )

        return cls(corner=corner,
                   block_size=block_size,
                   shape=shape,
                   axis_u=axis_u,
                   axis_v=axis_v,
                   axis_w=axis_w)

    def to_json(self) -> str:
        """Convert the full geometry to a JSON string."""
        data = {
            "corner": list(self.corner),
            "axis_u": list(self.axis_u),
            "axis_v": list(self.axis_v),
            "axis_w": list(self.axis_w),
            "block_size": list(self.block_size),
            "shape": list(self.shape),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "RegularGeometry":
        """Deserialize a JSON string to a full geometry object."""
        data = json.loads(json_str)
        return cls(
            corner=list(data["corner"]),
            axis_u=list(data["axis_u"]),
            axis_v=list(data["axis_v"]),
            axis_w=list(data["axis_w"]),
            block_size=list(data["block_size"]),
            shape=list(data["shape"]),
        )

    def to_multi_index(self) -> pd.MultiIndex:
        """Convert a RegularGeometry to a MultiIndex.

        The MultiIndex will have the following levels:
        - x: The x coordinates of the cell centres
        - y: The y coordinates of the cell centres
        - z: The z coordinates of the cell centres

        Returns:
            pd.MultiIndex: The MultiIndex representing the blockmodel element geometry.
        """
        ox, oy, oz = self.corner
        dx, dy, dz = self.block_size
        nx, ny, nz = self.shape

        # Calculate the coordinates of the block centers
        x = ox + (np.arange(nx) + 0.5) * dx
        y = oy + (np.arange(ny) + 0.5) * dy
        z = oz + (np.arange(nz) + 0.5) * dz

        # Create a meshgrid and flatten in C order, then stack as (N, 3)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        coords = np.stack([xx.ravel(order='C'), yy.ravel(order='C'), zz.ravel(order='C')], axis=-1)  # (N, 3)

        # Apply rotation if needed
        rotation_matrix = np.array([self.axis_u, self.axis_v, self.axis_w]).T
        rotated_coords = coords @ rotation_matrix  # (N, 3)

        # Create MultiIndex from rotated coordinates
        index = pd.MultiIndex.from_arrays(
            [rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]],
            names=["x", "y", "z"]
        )
        return index

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert a RegularGeometry to a DataFrame using the cached centroids.

        Returns:
            pd.DataFrame: The DataFrame representing the blockmodel element geometry.
        """
        centroids = self._centroids  # shape: (3, N)
        df = pd.DataFrame({
            "x": centroids[0],
            "y": centroids[1],
            "z": centroids[2],
        })
        df.attrs["geometry"] = self.summary
        return df

    def to_spatial_index(self) -> pd.Index:
        """Convert a RegularGeometry to an encoded integer index

        The integer index is encoded to preserve the spatial position.

        Use the coordinate_hashing.hashed_index_to_multiindex function to convert it back to x, y, z pd.MultiIndex

        Returns:

        """
        return multiindex_to_encoded_index(self.to_multi_index())

    def to_pyvista(self) -> 'pv.ImageData':
        import pyvista as pv

        # PyVista expects dimensions as (nx, ny, nz) + 1
        nx, ny, nz = self.shape
        dims = (nx, ny, nz)
        origin = self.corner
        spacing = self.block_size

        # ImageData expects dimensions as number of points, so add 1 to each
        dims = tuple(d + 1 for d in dims)

        grid = pv.ImageData(dimensions=dims,
                            spacing=spacing,
                            origin=origin)
        grid.direction_matrix = np.array([self.axis_u, self.axis_v, self.axis_w]).T
        return grid

    def nearest_centroid_lookup(self, x: float, y: float, z: float) -> Point:
        """Find the nearest centroid for provided x, y, z points.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.

        Returns:
            Point3: The coordinates of the nearest centroid.
        """

        reference_centroid: Point = (float(self.centroid_x[0]),
                                     float(self.centroid_y[0]),
                                     float(self.centroid_z[0]),
                                     )
        dx, dy, dz = self.block_size
        ref_x, ref_y, ref_z = reference_centroid

        nearest_x = round((x - ref_x) / dx) * dx + ref_x
        nearest_y = round((y - ref_y) / dy) * dy + ref_y
        nearest_z = round((z - ref_z) / dz) * dz + ref_z

        return nearest_x, nearest_y, nearest_z

    def is_compatible(self, other: 'RegularGeometry') -> True:
        """Check if the geometry is compatible with another RegularGeometry.

        Args:
            other: The other RegularGeometry to check compatibility with.

        Returns:
            bool: True if the geometries are compatible, False otherwise.

        """

        if self.srs != other.srs:
            self._logger.warning(f"SRS {self.srs} != {other.srs}.")
            return False
        if self.block_size != other.block_size:
            self._logger.warning(f"Block size {self.block_size} != {other.block_size}.")
            return False
        if self.shape != other.shape:
            self._logger.warning(f"Shape {self.shape} != {other.shape}.")
            return False
        if self.axis_u != other.axis_u:
            self._logger.warning(f"Axis {self.axis_u} != {other.axis_u}.")
            return False
        if self.axis_v != other.axis_v:
            self._logger.warning(f"Axis {self.axis_v} != {other.axis_v}.")
            return False
        if self.axis_w != other.axis_w:
            self._logger.warning(f"Axis {self.axis_w} != {other.axis_w}.")
            return False
        x_offset = (self.corner[0] - other.corner[0]) / self.block_size[0]
        if x_offset != int(x_offset):
            self._logger.warning(f"Incompatibility in x dimension: {x_offset} != {int(x_offset)}.")
            return False
        y_offset = (self.corner[1] - other.corner[1]) / self.block_size[1]
        if y_offset != int(y_offset):
            self._logger.warning(f"Incompatibility in y dimension: {y_offset} != {int(y_offset)}.")
            return False
        z_offset = (self.corner[2] - other.corner[2]) / self.block_size[2]
        if z_offset != int(z_offset):
            self._logger.warning(f"Incompatibility in z dimension: {z_offset} != {int(z_offset)}.")
            return False
        return True
