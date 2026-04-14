"""blockmodel.py

High-level API for working with regular 3D block models backed by Parquet.

Main entry point
----------------

``ParquetBlockModel`` is a convenience wrapper around a **canonical**
``.pbm`` Parquet file. A ``.pbm`` file is just a Parquet table with:

* arbitrary attribute columns (grades, density, rock type, ...),
* optional centroid columns ``x``, ``y``, ``z`` (for backwards
  compatibility and interoperability), and
* embedded geometry metadata under the ``"parq-blockmodel"`` key.

The geometry metadata encodes a regular logical grid in terms of
``(i, j, k)`` indices via :class:`parq_blockmodel.geometry.RegularGeometry`.
That geometry is the **single source of truth** for the grid:

* ``shape`` (number of blocks along each logical axis),
* ``block_size`` and ``corner`` in world coordinates,
* ``axis_u``, ``axis_v``, ``axis_w`` as an orthonormal basis describing
  the orientation of the logical ``i, j, k`` axes in world space.

Centroid coordinates ``(x, y, z)`` are therefore a **derived view** of
``(i, j, k) + geometry``. They may still be persisted as columns in the
Parquet file today, but the long‑term design treats them as secondary to
the ijk‑first representation.
"""
import logging
import math
import shutil
import typing
import warnings
from pathlib import Path
from typing import Union, Optional, Iterator
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel.types import Shape3D, Point, BlockSize
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel, create_toy_blockmodel
from parq_blockmodel.utils.geometry_utils import angles_to_axes
from parq_blockmodel.utils.spatial_encoding import (
    MAX_XY_VALUE,
    MAX_Z_VALUE,
    decode_world_coordinates,
    encode_world_coordinates,
    get_world_id_encoding_params,
)
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from parq_tools import ParquetProfileReport, filter_parquet_file, LazyParquetDF
from parq_tools.utils import atomic_output_file

from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.reblocking.reblocking import downsample_blockmodel, upsample_blockmodel

if typing.TYPE_CHECKING:
    import pyvista as pv  # type: ignore[import]
    import plotly.graph_objects as go
    import parq_blockmodel.mesh  # type: ignore[import]


class ParquetBlockModel:
    """A class to represent a **regular** Parquet block model.

    **Canonical on-disk representation:**

    We treat ``.pbm`` files as ParquetBlockModel containers:

    - The *extension* must be ``.pbm``.
    - The *content* is a Parquet table with embedded geometry metadata
      under the ``"parq-blockmodel"`` key, describing a regular
      :class:`parq_blockmodel.geometry.RegularGeometry` grid.

    The geometry is ijk-first: logical indices ``(i, j, k)`` enumerate
    the dense grid, and centroid coordinates ``(x, y, z)`` are derived
    from those indices plus geometry (corner, block size, axis
    orientation). In non-rotated models the grid axes align with the
    global X, Y, Z directions; in rotated models they do not.

    Parquet storage order follows NumPy C-order with respect to ijk:
    ``i`` varies fastest, then ``j``, then ``k``. This aligns with
    ``numpy.ravel(order="C")`` and how :class:`RegularGeometry`
    computes row indices. For xyz-defined, non-rotated models this
    corresponds to an increasing lexicographic ordering in ``(x, y, z)``,
    but callers should treat ijk + geometry as the canonical view.

    Attributes:
        blockmodel_path (Path): Path to the canonical ``.pbm`` file backing this block model.
        name (str): The name of the block model, derived from the file name.
        geometry (RegularGeometry): Geometry of the block model, loaded from Parquet metadata or
            inferred from centroid columns.
    """

    # Positional columns describe geometry/placement and are excluded from
    # ``self.attributes``. Keep linear index helpers (index_c/index_f) as
    # attributes for debug/visual validation workflows.
    POSITION_COLUMNS = {"block_id", "world_id", "i", "j", "k", "x", "y", "z"}

    def __init__(self, blockmodel_path: Path, name: Optional[str] = None, geometry: Optional[RegularGeometry] = None):
        if blockmodel_path.suffix != ".pbm":
            raise ValueError("The provided file must have a '.pbm' extension.")
        self.blockmodel_path = blockmodel_path
        self.name = name or blockmodel_path.stem
        self.geometry = geometry or RegularGeometry.from_parquet(self.blockmodel_path)
        self.pf: ParquetFile = ParquetFile(blockmodel_path)
        # Lazy view over the backing Parquet file for large models.
        self.data: LazyParquetDF = LazyParquetDF(self.blockmodel_path)
        self.columns: list[str] = pq.read_schema(self.blockmodel_path).names
        self._centroid_index: Optional[pd.MultiIndex] = None
        self.attributes: list[str] = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._extract_column_dtypes()
        self._logger = logging.getLogger(__name__)

        if self.is_sparse:
            if not self.validate_sparse():
                raise ValueError("The sparse ParquetBlockModel is invalid. "
                                 "Sparse centroids must be a subset of the dense grid.")

    def __repr__(self):
        return f"ParquetBlockModel(name={self.name}, path={self.blockmodel_path})"

    def _extract_column_dtypes(self):
        self.column_dtypes: dict[str, np.dtype] = {}
        self._column_categorical_ordered: dict[str, bool] = {}
        schema = pq.read_schema(self.blockmodel_path)
        for col in self.columns:
            if col in ["x", "y", "z"]:
                continue
            field_type = schema.field(col).type
            if pa.types.is_dictionary(field_type):
                self.column_dtypes[col] = pd.CategoricalDtype(ordered=field_type.ordered)
                self._column_categorical_ordered[col] = field_type.ordered
            else:
                self.column_dtypes[col] = field_type.to_pandas_dtype()

    @property
    def column_categorical_ordered(self) -> dict[str, bool]:
        return self._column_categorical_ordered.copy()

    @property
    def centroid_index(self) -> pd.MultiIndex:
        """Get the centroid index of the block model as ``(x, y, z)``.

        This index is built from the centroid columns ``x``, ``y``, ``z``
        stored in the underlying ``.pbm`` Parquet file. It is primarily
        a **backwards-compatibility view** for xyz-first workflows which
        treat world-space centroids as the primary index.

        Canonical ordering is defined by the C-order ijk layout in
        :class:`RegularGeometry`, not by lexicographic sorting of
        ``(x, y, z)``. We therefore require this index to be **unique**
        but do not enforce global monotonicity in xyz.

        Returns:
            pd.MultiIndex: The MultiIndex representing centroid coordinates ``(x, y, z)``.
        """

        if self._centroid_index is None:
            centroid_cols = ["x", "y", "z"]
            if all(col in self.columns for col in centroid_cols):
                centroids: pd.DataFrame = pq.read_table(self.blockmodel_path, columns=centroid_cols).to_pandas()
                if centroids.index.names == centroid_cols:
                    index = centroids.index
                else:
                    if centroids.empty:
                        raise ValueError("Parquet file is empty or does not contain valid centroid data.")
                    index = centroids.set_index(["x", "y", "z"]).index
            else:
                block_ids = np.concatenate([ids for ids in self._iter_block_ids()])
                x, y, z = self.geometry.xyz_from_row_index(block_ids)
                index = pd.MultiIndex.from_arrays([x, y, z], names=centroid_cols)

            # Centroid coordinates must be unique, but ordering is defined
            # by the geometry's ijk grid rather than lexicographic xyz.
            if not index.is_unique:
                raise ValueError("The index of the Parquet file is not unique. "
                                 "Ensure that the centroid coordinates (x, y, z) are unique.")

            self._centroid_index = index
        return self._centroid_index

    @property
    def is_sparse(self) -> bool:
        dense_count = int(np.prod(self.geometry.local.shape))
        return int(self.pf.metadata.num_rows) < dense_count

    @property
    def sparsity(self) -> float:
        dense_count = int(np.prod(self.geometry.local.shape))
        return 1.0 - (int(self.pf.metadata.num_rows) / dense_count)

    @property
    def index_c(self) -> np.ndarray:
        """Zero-based C-order indices for the **dense ijk grid**.

        The returned array has length ``ni * nj * nk`` and corresponds to
        linearised logical indices ``(i, j, k)`` using NumPy C‑order:

        ``r = i + ni * (j + nj * k)``.

        This is mainly a low‑level helper for downstream APIs that need a
        stable mapping between a flat row index and ijk; most callers
        should use :meth:`read` with ``index="ijk"`` instead.
        """
        shape = self.geometry.local.shape
        return np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='C')

    @property
    def index_f(self) -> np.ndarray:
        """Zero-based Fortran-order indices for the **dense ijk grid**.

        Uses the same ijk logical grid as :pyattr:`index_c`, but flattened
        with ``order="F"``. This is useful when interacting with tools
        (such as some VTK/PyVista paths) that expect F‑order layouts.
        """
        shape = self.geometry.local.shape
        return np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='F')

    def validate_sparse(self) -> bool:
        dense_count = int(np.prod(self.geometry.local.shape))
        seen: set[int] = set()
        total = 0
        for block_ids in self._iter_block_ids():
            total += len(block_ids)
            if np.any(block_ids < 0) or np.any(block_ids >= dense_count):
                return False
            seen.update(block_ids.tolist())
        return len(seen) == total

    def _iter_batches(self, columns: list[str], batch_size: int = 1_000_000) -> Iterator[pa.RecordBatch]:
        pf = pq.ParquetFile(self.blockmodel_path)
        for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
            yield batch

    def _iter_block_ids(self, batch_size: int = 1_000_000) -> Iterator[np.ndarray]:
        cols = set(self.columns)
        if "block_id" in cols:
            for batch in self._iter_batches(["block_id"], batch_size=batch_size):
                block_ids = np.asarray(batch.column(0), dtype=np.uint32)
                yield block_ids
            return

        if "world_id" in cols:
            if not self.geometry.world_id_encoding:
                raise ValueError("world_id column present but metadata has no world_id_encoding payload.")
            offset, scale = get_world_id_encoding_params(self.geometry.world_id_encoding)
            for batch in self._iter_batches(["world_id"], batch_size=batch_size):
                world_ids = np.asarray(batch.column(0), dtype=np.int64)
                x, y, z = decode_world_coordinates(world_ids, offset=offset, scale=scale)
                yield self.geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
            return

        if {"x", "y", "z"}.issubset(cols):
            for batch in self._iter_batches(["x", "y", "z"], batch_size=batch_size):
                x = np.asarray(batch.column(0), dtype=float)
                y = np.asarray(batch.column(1), dtype=float)
                z = np.asarray(batch.column(2), dtype=float)
                yield self.geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
            return

        if {"i", "j", "k"}.issubset(cols):
            for batch in self._iter_batches(["i", "j", "k"], batch_size=batch_size):
                i = np.asarray(batch.column(0), dtype=np.int64)
                j = np.asarray(batch.column(1), dtype=np.int64)
                k = np.asarray(batch.column(2), dtype=np.int64)
                yield self.geometry.row_index_from_ijk(i, j, k).astype(np.uint32)
            return

        dense_count = int(np.prod(self.geometry.local.shape))
        if int(self.pf.metadata.num_rows) != dense_count:
            raise ValueError("Sparse model requires block_id, world_id, ijk, or xyz positional columns.")

        offset = 0
        for rg in range(self.pf.metadata.num_row_groups):
            n_rows = self.pf.metadata.row_group(rg).num_rows
            ids = np.arange(offset, offset + n_rows, dtype=np.uint32)
            offset += n_rows
            yield ids

    @classmethod
    def from_parquet(cls, parquet_path: Path,
                     columns: Optional[list[str]] = None,
                     overwrite: bool = False,
                     axis_azimuth: float = 0.0,
                     axis_dip: float = 0.0,
                     axis_plunge: float = 0.0,
                     chunk_size: int = 1_000_000,
                     ) -> "ParquetBlockModel":
        """Create a :class:`ParquetBlockModel` from a source Parquet file.

        This helper promotes a *source* ``.parquet`` file (typically
        xyz-centric) into a canonical ``.pbm`` container with embedded
        :class:`RegularGeometry` metadata.

        The input Parquet is expected to contain centroid columns
        ``x``, ``y``, ``z`` describing block centroids in world
        coordinates. :class:`RegularGeometry` is reconstructed from those
        centroids and/or any existing ``"parq-blockmodel"`` metadata via
        :meth:`RegularGeometry.from_parquet`. Geometry becomes the
        authoritative description of the dense ijk grid; xyz centroids
        are treated as a derived view.

        Args:
            parquet_path (Path): Path to the *source* Parquet file. If the suffix is
                ``.pbm`` and ``overwrite`` is False, a :class:`ValueError`
                is raised to avoid mutating an existing canonical container.
            columns (list[str], optional): Optional subset of columns to copy from the source file into
                the resulting ``.pbm`` file. If ``None``, all columns are copied.
            overwrite (bool, default False): If True, allows overwriting an existing ``.pbm`` file at the
                target location.
            axis_azimuth (float, default 0.0): Optional rotation angle used by
                :meth:`RegularGeometry.from_parquet` to define the
                orientation of the logical ijk axes in world coordinates
                when inferring geometry. In the common case geometry is read
                directly from existing metadata and this parameter is 0.
            axis_dip (float, default 0.0): Optional rotation angle. See ``axis_azimuth``.
            axis_plunge (float, default 0.0): Optional rotation angle. See ``axis_azimuth``.
            chunk_size (int, default 1_000_000): Batch size for reading Parquet data.

        Returns:
            ParquetBlockModel: A block model backed by a newly written ``.pbm`` file
                located alongside ``parquet_path``.
        """
        if parquet_path.suffix == ".pbm":
            if not overwrite:
                raise ValueError(
                    f"File {parquet_path} appears to be a compliant ParquetBlockModel file. "
                    f"Use the constructor directly, or pass overwrite=True to allow mutation."
                )

        geometry = RegularGeometry.from_parquet(filepath=parquet_path,
                                                axis_azimuth=axis_azimuth,
                                                axis_dip=axis_dip,
                                                axis_plunge=axis_plunge,
                                                chunk_size=chunk_size)

        cls._validate_geometry(parquet_path, geometry=geometry, chunk_size=chunk_size)

        # Canonical ParquetBlockModel files use a single ``.pbm`` suffix. We
        # write the PBM file alongside the source Parquet file by replacing
        # only the final extension.
        new_filepath: Path = parquet_path.resolve().with_suffix(".pbm")
        cls._write_canonical_pbm(input_path=parquet_path,
                                 output_path=new_filepath,
                                 geometry=geometry,
                                 columns=columns,
                                 chunk_size=chunk_size)
        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame,
                       filename: Path,
                       geometry: Optional[RegularGeometry] = None,
                       name: Optional[str] = None,
                       overwrite: bool = False,
                       axis_azimuth: float = 0.0,
                       axis_dip: float = 0.0,
                       axis_plunge: float = 0.0,
                       chunk_size: int = 1_000_000,
                       ) -> "ParquetBlockModel":
        """Create a :class:`ParquetBlockModel` from a pandas DataFrame.

        This constructor is oriented towards xyz-indexed
        DataFrames, where the index is a centroid MultiIndex with levels
        ``("x", "y", "z")`` in world coordinates.

        The DataFrame is written to a sibling ``.pbm`` file (replacing
        the ``.parquet`` suffix of ``filename``) with embedded
        :class:`RegularGeometry` metadata. Geometry is either supplied
        explicitly or inferred from the xyz centroids via
        :meth:`RegularGeometry.from_multi_index`.

        Args:
            dataframe (pandas.DataFrame): Block model data indexed by centroid coordinates with
                ``dataframe.index.names == ["x", "y", "z"]``.
            filename (Path): Path to the *source* ``.parquet`` file which will be used as
                the basis for the sibling ``.pbm`` container.
            geometry (RegularGeometry, optional): Geometry of the logical ijk grid. If omitted, it is inferred
                from the centroid MultiIndex, using the optional rotation
                angles to define the ijk axis orientation.
            name (str, optional): Optional name for the resulting :class:`ParquetBlockModel`.
                Defaults to ``filename.stem``.
            overwrite (bool, default False): If True, allows overwriting an existing ``.pbm`` file at the
                derived path.
            axis_azimuth (float, default 0.0): Optional rotation angle used when inferring geometry from
                the xyz centroids. Defines the orientation of the
                logical ijk axes relative to the world coordinate frame.
            axis_dip (float, default 0.0): Optional rotation angle. See ``axis_azimuth``.
            axis_plunge (float, default 0.0): Optional rotation angle. See ``axis_azimuth``.
            chunk_size (int, default 1_000_000): Batch size for writing Parquet data.

        Returns:
            ParquetBlockModel: A block model backed by the newly written ``.pbm`` file.

        Note:
            New ijk-first workflows are encouraged to construct a
            :class:`RegularGeometry` explicitly and use
            :meth:`ParquetBlockModel.from_geometry` or
            :meth:`ParquetBlockModel.from_parquet`. This helper remains for
            backwards compatibility with xyz-indexed DataFrames.
        """
        # Ensure MultiIndex
        if dataframe.index.names != ["x", "y", "z"]:
            raise ValueError("DataFrame index must be a MultiIndex with names ['x', 'y', 'z'].")

        # Warn if not sorted
        if not dataframe.index.is_monotonic_increasing:
            warnings.warn("DataFrame index is not sorted in ascending order.")

        # Ensure correct filename extension for the *source* Parquet; the
        # canonical ParquetBlockModel lives in a sibling ``.pbm`` file.
        if not filename.suffix == ".parquet":
            raise ValueError(f"Filename {filename} must have a '.parquet' extension.")
        pbm_path = filename.with_suffix(".pbm")
        if pbm_path.exists() and not overwrite:
            raise FileExistsError(f"File {pbm_path} already exists. Use overwrite=True to allow mutation.")

        # Infer geometry if needed
        geometry = geometry or RegularGeometry.from_multi_index(
            dataframe.index, axis_azimuth=axis_azimuth, axis_dip=axis_dip, axis_plunge=axis_plunge)
        if not isinstance(geometry, RegularGeometry):
            raise TypeError("geometry must be a RegularGeometry instance.")

        dataframe = dataframe.copy()

        if "block_id" not in dataframe.columns:
            if dataframe.index.names == ["x", "y", "z"]:
                x = dataframe.index.get_level_values("x").to_numpy()
                y = dataframe.index.get_level_values("y").to_numpy()
                z = dataframe.index.get_level_values("z").to_numpy()
                dataframe["block_id"] = geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
            elif {"x", "y", "z"}.issubset(dataframe.columns):
                dataframe["block_id"] = geometry.row_index_from_xyz(
                    dataframe["x"].to_numpy(), dataframe["y"].to_numpy(), dataframe["z"].to_numpy()
                ).astype(np.uint32)
            elif {"i", "j", "k"}.issubset(dataframe.columns):
                dataframe["block_id"] = geometry.row_index_from_ijk(
                    dataframe["i"].to_numpy(), dataframe["j"].to_numpy(), dataframe["k"].to_numpy()
                ).astype(np.uint32)
            elif len(dataframe) == int(np.prod(geometry.local.shape)):
                dataframe["block_id"] = np.arange(len(dataframe), dtype=np.uint32)
            else:
                raise ValueError("Unable to derive block_id: provide xyz or ijk coordinates.")

        if "world_id" not in dataframe.columns:
            if dataframe.index.names == ["x", "y", "z"]:
                x = dataframe.index.get_level_values("x").to_numpy(dtype=float)
                y = dataframe.index.get_level_values("y").to_numpy(dtype=float)
                z = dataframe.index.get_level_values("z").to_numpy(dtype=float)
            elif {"x", "y", "z"}.issubset(dataframe.columns):
                x = dataframe["x"].to_numpy(dtype=float)
                y = dataframe["y"].to_numpy(dtype=float)
                z = dataframe["z"].to_numpy(dtype=float)
            else:
                x, y, z = geometry.xyz_from_row_index(dataframe["block_id"].to_numpy(dtype=np.uint32))

            if geometry.world_id_encoding is None:
                geometry.world_id_encoding = cls._build_world_id_encoding_from_xyz(x, y, z)
            offset, scale = get_world_id_encoding_params(geometry.world_id_encoding)
            dataframe["world_id"] = encode_world_coordinates(x, y, z, offset=offset, scale=scale).astype(np.int64)

        # Attach geometry metadata to attrs for completeness
        dataframe.attrs["parq-blockmodel"] = geometry.to_metadata_dict()

        # Save DataFrame to Parquet and embed geometry metadata
        table = pa.Table.from_pandas(dataframe)
        meta = table.schema.metadata or {}
        meta = dict(meta)
        meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, pbm_path)

        # Validate geometry
        cls._validate_geometry(pbm_path, geometry, chunk_size=chunk_size)

        return cls(blockmodel_path=pbm_path, name=name, geometry=geometry)

    @classmethod
    def create_demo_block_model(cls, filename: Path, **demo_kwargs) -> "ParquetBlockModel":
        """Convenience helper used in tests to write a demo Parquet file and
        wrap it as a ParquetBlockModel.

        Parameters
        ----------
        filename : Path
            Target ``.parquet`` file to hold the raw demo data. The canonical
            blockmodel file will be written alongside it with a ``.pbm``
            suffix via :meth:`from_parquet`.
        **demo_kwargs
            Additional keyword arguments forwarded to
            :func:`parq_blockmodel.utils.demo_block_model.create_demo_blockmodel`,
            for example ``shape`` or ``block_size``. This allows tests and
            examples to control the logical grid used for the demo model.
        """
        from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel as _create_demo_df

        # Write the demo DataFrame to the requested Parquet file.
        df = _create_demo_blockmodel(parquet_filepath=filename, **demo_kwargs)
        if isinstance(df, Path):
            # The helper already wrote the Parquet file and returned the path.
            parquet_path = df
        else:
            # Defensive: if the helper returned a DataFrame, persist it now.
            parquet_path = filename
            df.to_parquet(parquet_path)

        # Promote the raw parquet into a canonical .pbm ParquetBlockModel.
        return cls.from_parquet(parquet_path)

    @classmethod
    def create_toy_blockmodel(cls, filename: Path,
                              shape: Shape3D = (3, 3, 3),
                              block_size: BlockSize = (1, 1, 1),
                              corner: Point = (-0.5, -0.5, -0.5),
                              axis_azimuth: float = 0.0,
                              axis_dip: float = 0.0,
                              axis_plunge: float = 0.0,
                              deposit_bearing: float = 20.0,
                              deposit_dip: float = 30.0,
                              deposit_plunge: float = 10.0,
                              grade_name: str = 'grade',
                              grade_min: float = 50.0,
                              grade_max: float = 65.0,
                              deposit_center=(10.0, 7.5, 5.0),
                              deposit_radii=(8.0, 5.0, 3.0),
                              noise_std: float = 0.2,
                              ) -> "ParquetBlockModel":
        """Create a synthetic toy block model.

        The toy model is generated on a regular ijk grid with the
        specified ``shape`` and ``block_size``, anchored at ``corner`` in
        world coordinates. A simple ellipsoidal grade distribution is
        created in xyz space and stored under ``grade_name`` alongside
        any supporting attributes.

        Parameters
        ----------
        filename : Path
            Path to the *source* ``.parquet`` file to hold the raw toy
            data. A canonical ``.pbm`` file will be written alongside it
            with the same stem.
        shape : tuple of int, default ``(3, 3, 3)``
            Number of blocks along the logical ijk axes.
        block_size : tuple of float, default ``(1, 1, 1)``
            Block dimensions along each logical axis.
        corner : tuple of float, default ``(-0.5, -0.5, -0.5)``
            World‑space coordinates of the block with indices
            ``(i=0, j=0, k=0)`` lower corner.
        axis_azimuth, axis_dip, axis_plunge : float, default 0.0
            Rotation angles defining orientation of the logical ijk axes
            relative to the world xyz frame. These are converted to an
            orthonormal basis ``(axis_u, axis_v, axis_w)`` used by
            :class:`RegularGeometry`.
        deposit_bearing, deposit_dip, deposit_plunge : float
            Orientation of the synthetic deposit in degrees.
        grade_name : str, default ``"grade"``
            Name of the column storing the simulated grade values.
        grade_min, grade_max : float
            Minimum and maximum grade values for the toy distribution.
        deposit_center : tuple of float
            Center of the ellipsoidal deposit in world xyz coordinates.
        deposit_radii : tuple of float
            Semi‑axes of the deposit ellipsoid in world units.
        noise_std : float, default 0.2
            Standard deviation of Gaussian noise added to grades.

        Returns
        -------
        ParquetBlockModel
            A :class:`ParquetBlockModel` backed by a canonical ``.pbm``
            file with the generated toy data and geometry metadata.

        Notes
        -----
        For rotated geometries (non‑zero rotation angles), xyz
        centroids will appear rotated in world coordinates relative to
        the ijk axes of the grid. Geometry validation is skipped in this
        case to avoid imposing axis‑aligned assumptions on the rotated
        layout.
        """

        create_toy_blockmodel(shape=shape, block_size=block_size, corner=corner,
                              axis_azimuth=axis_azimuth, axis_dip=axis_dip, axis_plunge=axis_plunge,
                              deposit_bearing=deposit_bearing, deposit_dip=deposit_dip, deposit_plunge=deposit_plunge,
                              grade_name=grade_name, grade_min=grade_min, grade_max=grade_max,
                              deposit_center=deposit_center, deposit_radii=deposit_radii,
                              noise_std=noise_std, parquet_filepath=filename,
                              )
        # get the orientation of the axes
        axis_u, axis_v, axis_w = angles_to_axes(
            axis_azimuth=axis_azimuth, axis_dip=axis_dip,
            axis_plunge=axis_plunge)
        # create geometry that aligns with the demo block model
        geometry = RegularGeometry(
            local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
            world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
        )

        if not geometry.is_rotated:
            cls._validate_geometry(filename)

        new_filepath = filename.resolve().with_suffix(".pbm")
        cls._write_canonical_pbm(input_path=filename,
                                 output_path=new_filepath,
                                 geometry=geometry,
                                 columns=None,
                                 chunk_size=1_000_000)

        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_geometry(cls, geometry: RegularGeometry,
                      path: Path,
                      name: Optional[str] = None
                      ) -> "ParquetBlockModel":
        """Create a :class:`ParquetBlockModel` from a geometry only.

        This constructor materialises a **dense** ijk grid defined by a
        :class:`RegularGeometry` into a Parquet file with xyz centroid
        columns but **no additional attributes**. It is useful for
        creating an empty block model skeleton that can be joined with
        external attribute data.

        Parameters
        ----------
        geometry : RegularGeometry
            Geometry of the logical ijk grid.
        path : Path
            Path to the target ``.pbm`` file. The file will contain a
            row for every ijk block with derived xyz centroid columns and
            embedded geometry metadata.
        name : str, optional
            Optional name for the resulting :class:`ParquetBlockModel`.

        Returns
        -------
        ParquetBlockModel
            A block model with geometry defined but no attribute
            columns.
        """
        centroids_df = geometry.to_dataframe()
        centroids_df["block_id"] = np.arange(int(np.prod(geometry.local.shape)), dtype=np.uint32)
        if geometry.world_id_encoding is None:
            geometry.world_id_encoding = cls._build_world_id_encoding_from_xyz(
                centroids_df["x"].to_numpy(dtype=float),
                centroids_df["y"].to_numpy(dtype=float),
                centroids_df["z"].to_numpy(dtype=float),
            )
        offset, scale = get_world_id_encoding_params(geometry.world_id_encoding)
        centroids_df["world_id"] = encode_world_coordinates(
            centroids_df["x"].to_numpy(dtype=float),
            centroids_df["y"].to_numpy(dtype=float),
            centroids_df["z"].to_numpy(dtype=float),
            offset=offset,
            scale=scale,
        ).astype(np.int64)
        centroids_df.attrs["parq-blockmodel"] = geometry.to_metadata_dict()

        table = pa.Table.from_pandas(centroids_df)
        meta = table.schema.metadata or {}
        meta = dict(meta)
        meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, path)

        return cls(blockmodel_path=path, name=name, geometry=geometry)

    def create_report(self, columns: Optional[list[str]] = None,
                      column_batch_size: int = 10,
                      show_progress: bool = True,
                      open_in_browser: bool = False
                      ) -> Path:
        """
        Create a ydata-profiling report for the block model.
        The report will be of the same name as the block model, with a '.html' extension.

        Args:
            columns: List of column names to include in the profile. If None, all columns are used.
            column_batch_size: The number of columns to process in each batch. If None, processes all columns at once.
            show_progress: bool: If True, displays a progress bar during profiling.
            open_in_browser: bool: If True, opens the report in a web browser after generation.

        Returns
            Path: The path to the generated profile report.

        """
        report: ParquetProfileReport = ParquetProfileReport(self.blockmodel_path, columns=columns,
                                                            batch_size=column_batch_size,
                                                            show_progress=show_progress).profile()
        if open_in_browser:
            report.show(notebook=False)
        if not columns:
            self.report_path = self.blockmodel_path.with_suffix('.html')
        return self.report_path

    def downsample(self, new_block_size, aggregation_config) -> "ParquetBlockModel":
        """
        Downsample the block model to a coarser grid with specified aggregation methods for each attribute.
        This function supports downsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            aggregation_config: dict mapping attribute names to aggregation methods.

        Example:
            aggregation_config = {
                'grade': {'method': 'weighted_mean', 'weight': 'dry_mass'},
                'density': {'method': 'weighted_mean', 'weight': 'volume'},
                'dry_mass': {'method': 'sum'},
                'volume': {'method': 'sum'},
                'rock_type': {'method': 'mode'}
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the downsampled grid.
        """
        return downsample_blockmodel(self, new_block_size, aggregation_config)

    def upsample(self, new_block_size, interpolation_config) -> "ParquetBlockModel":
        """
        Upsample the block model to a finer grid with specified interpolation methods for each attribute.
        This function supports upsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            interpolation_config: dict mapping attribute names to interpolation methods.

        Example:
            interpolation_config = {
                'grade': {'method': 'linear'},
                'density': {'method': 'nearest'},
                'dry_mass': {'method': 'linear'},
                'volume': {'method': 'linear'},
                'rock_type': {'method': 'nearest'}
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the upsampled grid.
        """
        return upsample_blockmodel(self, new_block_size, interpolation_config)

    def create_heatmap_from_threshold(self, attribute: str, threshold: float, axis: str = "z",
                                      return_array: bool = False) -> Union['pv.ImageData', np.ndarray]:
        """
        Create a 2D heatmap from a 3D block model at a specified attribute threshold.

        Args:
            attribute (str): The name of the attribute to threshold.
            threshold (float): The threshold value for the attribute.
            axis (str): The axis to view from ('x', 'y', or 'z'). Defaults to 'z'.
            return_array (bool): If True, returns the heatmap as a NumPy array. Defaults to False.

        Returns:
            Union[pv.ImageData, np.ndarray]: A 2D heatmap as a PyVista ImageData object or a NumPy array.
            """
        import pyvista as pv
        import numpy as np

        if attribute not in self.attributes:
            raise ValueError(f"Attribute '{attribute}' not found in the block model.")
        if axis not in {"x", "y", "z"}:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

        # Load the block model as PyVista ImageData
        mesh: pv.ImageData = self.to_pyvista(grid_type="image", attributes=[attribute])

        # Apply the threshold
        mesh.cell_data['count'] = mesh.cell_data[attribute] > threshold
        mesh.cell_data['count'] = mesh.cell_data['count'].astype(np.int8)

        # Reshape and sum along the specified axis, using Fortran order to match (x, y, z)
        reshaped_data = mesh.cell_data['count'].reshape(
            (mesh.dimensions[0] - 1, mesh.dimensions[1] - 1, mesh.dimensions[2] - 1), order='F')
        summed_data: Optional[np.ndarray] = None
        if axis == "z":
            summed_data = np.sum(reshaped_data, axis=2)
            new_dimensions = (mesh.dimensions[0], mesh.dimensions[1], 1)
        elif axis == "x":
            summed_data = np.sum(reshaped_data, axis=0)
            new_dimensions = (1, mesh.dimensions[1], mesh.dimensions[2])
        elif axis == "y":
            summed_data = np.sum(reshaped_data, axis=1)
            new_dimensions = (mesh.dimensions[0], 1, mesh.dimensions[2])
        if return_array:
            return summed_data.T  # Flip for correct orientation
        # Create a new ImageData object with correct dimensions
        new_mesh = pv.ImageData(dimensions=(summed_data.shape[0] + 1, summed_data.shape[1] + 1, 1),
                                spacing=self.geometry.local.block_size,
                                origin=self.geometry.local.corner)
        new_mesh.cell_data[attribute] = summed_data.ravel(order='F')
        return new_mesh

    def plot_heatmap(self, attribute: str, threshold: float, axis: str = "z",
                     title: Optional[str] = None) -> 'go.Figure':
        """
        Create a 2D heatmap plotly figure from a 3D block model at a specified attribute threshold.

        Args:
            attribute (str): The name of the attribute to threshold.
            threshold (float): The threshold value for the attribute.
            axis (str): The axis to view from ('x', 'y', or 'z'). Defaults to 'z'.
            title (Optional[str]): The title of the heatmap. If None, a default title is used.

        Returns:
            go.Figure: A Plotly figure containing the heatmap.
        """
        import plotly.express as px

        summed_data = self.create_heatmap_from_threshold(attribute, threshold, axis, return_array=True)

        data, labels, extents = self._get_heatmap_data(axis, summed_data)
        x_extent, y_extent = extents
        nx, ny = summed_data.shape[1], summed_data.shape[0]
        x_edges = np.linspace(x_extent[0], x_extent[1], nx + 1)
        y_edges = np.linspace(y_extent[0], y_extent[1], ny + 1)
        x_ticks = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_ticks = 0.5 * (y_edges[:-1] + y_edges[1:])
        fig = px.imshow(summed_data, origin="lower", x=x_ticks, y=y_ticks, color_continuous_scale='Viridis')

        fig.update_layout(title=f"Heatmap of {attribute}, thresholded at {threshold}",
                          xaxis_title=labels[0],
                          yaxis_title=labels[1], )
        fig.update_yaxes(range=[0, 1])
        if title is not None:
            fig.update_layout(title=title)
        return fig

    def _get_heatmap_data(self, axis, summed_data
                          ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], tuple[tuple[float, float], tuple[float, float]]]:
        ex = self.geometry.extents  # (xmin, xmax, ymin, ymax, zmin, zmax)

        if axis == "z":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[1])
            z = summed_data
            labels = "Easting", "Northing"
            x_extent = (ex[0], ex[1])
            y_extent = (ex[2], ex[3])
        elif axis == "x":
            x = np.arange(summed_data.shape[1])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Northing", "RL"
            x_extent = (ex[2], ex[3])
            y_extent = (ex[4], ex[5])
        elif axis == "y":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Easting", "RL"
            x_extent = (ex[0], ex[1])
            y_extent = (ex[4], ex[5])
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
        return (x, y, z), labels, (x_extent, y_extent)

    def plot(self, scalar: str,
             grid_type: typing.Literal["image", "structured", "unstructured"] = "image",
             threshold: bool = True, show_edges: bool = True,
             show_axes: bool = True, enable_picking: bool = False,
             picked_attributes: Optional[list[str]] = None) -> 'pv.Plotter':
        """Plot the block model using PyVista.

        Args:
            scalar: The name of the scalar attribute to visualize.
            grid_type: The type of grid to use for plotting. Options are "image", "structured", or "unstructured".
            threshold: The thresholding option for the mesh. If True, applies a threshold to the scalar values.
            show_edges: Show edges of the mesh.
            show_axes: Show the axes in the plot.
            enable_picking: If True, enables picking mode to interactively select cells in the plot.
            picked_attributes: A list of attributes that will be returned in picking mode. If None, all attributes are returned.

        Returns:

        """

        import pyvista as pv
        if scalar not in self.attributes:
            raise ValueError(f"Column '{scalar}' not found in the ParquetBlockModel."
                             f"Available columns are: {self.attributes}")

        # Create a PyVista plotter
        plotter = pv.Plotter()

        attributes = [scalar]
        if enable_picking:
            if picked_attributes is None:
                attributes = self.attributes
            else:
                attributes = picked_attributes
            if scalar not in attributes:
                attributes.append(scalar)
        mesh = self.to_pyvista(grid_type=grid_type, attributes=attributes)

        # Add a thresholded mesh to the plotter
        if threshold:
            plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, scalars=scalar, show_edges=show_edges)

        plotter.title = self.name
        if show_axes:
            plotter.show_axes()

        text_name = "cell_info_text"
        plotter.add_text("", position="upper_left", font_size=12, name=text_name)
        cell_centers = mesh.cell_centers().points  # shape: (n_cells, 3)

        if enable_picking:
            def cell_callback(picked_cell):
                if text_name in plotter.actors:
                    plotter.remove_actor(text_name)
                if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
                    if "vtkOriginalCellIds" in picked_cell.cell_data:
                        cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                        centroid = cell_centers[cell_id]  # numpy array of (x, y, z)
                        centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                        values = {attr: mesh.cell_data[attr][cell_id] for attr in attributes}
                        msg = f"Cell ID: {cell_id}, {centroid_str}, " + ", ".join(
                            f"{k}: {v}" for k, v in values.items())
                    else:
                        value = picked_cell.cell_data[scalar][0]
                        msg = f"Picked cell value: {scalar}: {value}"
                    plotter.add_text(msg, position="upper_left", font_size=12, name=text_name)
                else:
                    plotter.add_text("No valid cell picked.", position="upper_left", font_size=12, name=text_name)

            plotter.enable_cell_picking(callback=cell_callback, show_message=False, through=False)
            plotter.title = f"{self.name} - Press R and select a cell for attribute data"

        return plotter

    def read(self, columns: Optional[list[str]] = None,
             index: typing.Literal["xyz", "ijk", None] = "xyz",
             dense: bool = False) -> pd.DataFrame:
        """Read the Parquet file and return a DataFrame.

        Notes
        -----
        Current behaviour (for backwards compatibility):

        * ``index="xyz"`` (the default) returns a DataFrame indexed by
          centroid coordinates ``(x, y, z)``. This matches the original
          design where xyz are treated as canonical.
        * ``index="ijk"`` returns a DataFrame whose index is derived from
          :class:`RegularGeometry` via ``to_ijk_multi_index``.

        Option B design (future change, **not** yet enforced):

        * The canonical in-memory representation will become an
          ``(i, j, k)`` MultiIndex with attributes-only columns.
        * xyz (centroid) coordinates will be treated as a **derived view**
          computed from ``(i, j, k)`` + geometry, exposed via a helper such
          as ``as_xyz()`` or via pandas accessors that read geometry from
          ``df.attrs["parq-blockmodel"]``.
        * Plotting helpers (PyVista, Plotly) will consume ijk + geometry and
          compute xyz internally when needed, so they do not rely on xyz
          being persisted as data columns.

        Until that refactor is complete, this method preserves the existing
        xyz-first semantics so that external callers continue to work.

        Args
        ----
        columns:
            List of column names to read. If None, all columns are read.
        index:
            The index type to use for the DataFrame. Options are
            ``"xyz"`` for centroid coordinates, ``"ijk"`` for block
            indices, or ``None`` for no index.
        dense:
            If True, reads/reindexes to the full dense grid. If False,
            returns the sparse layout in the underlying file.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the block model data.

        Notes
        -----
        ``index="xyz"`` is preserved as the default for backwards
        compatibility with earlier versions of :mod:`parq_blockmodel`
        that treated centroid coordinates as canonical. New code is
        encouraged to pass ``index="ijk"`` explicitly and work in terms
        of logical grid indices plus geometry.
        """
        if columns is None:
            columns = self.columns
        df = pq.read_table(self.blockmodel_path, columns=columns).to_pandas()

        block_ids: Optional[np.ndarray] = None
        if index in {"ijk", "xyz"}:
            if "block_id" in df.columns:
                block_ids = df["block_id"].to_numpy(dtype=np.uint32)
            elif "world_id" in df.columns:
                if not self.geometry.world_id_encoding:
                    raise ValueError("world_id column present but metadata has no world_id_encoding payload.")
                offset, scale = get_world_id_encoding_params(self.geometry.world_id_encoding)
                x, y, z = decode_world_coordinates(df["world_id"].to_numpy(dtype=np.int64), offset=offset, scale=scale)
                block_ids = self.geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
            elif {"i", "j", "k"}.issubset(df.columns):
                block_ids = self.geometry.row_index_from_ijk(
                    df["i"].to_numpy(), df["j"].to_numpy(), df["k"].to_numpy()
                ).astype(np.uint32)
            elif {"x", "y", "z"}.issubset(df.columns):
                block_ids = self.geometry.row_index_from_xyz(
                    df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
                ).astype(np.uint32)
            elif int(self.pf.metadata.num_rows) == int(np.prod(self.geometry.local.shape)):
                block_ids = np.arange(len(df), dtype=np.uint32)
            else:
                ids_df = pq.read_table(self.blockmodel_path, columns=["block_id"]).to_pandas()
                block_ids = ids_df["block_id"].to_numpy(dtype=np.uint32)

        if index == "xyz":
            if {"x", "y", "z"}.issubset(df.columns):
                df.index = pd.MultiIndex.from_arrays(
                    [df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], names=["x", "y", "z"]
                )
            else:
                x, y, z = self.geometry.xyz_from_row_index(block_ids)
                df.index = pd.MultiIndex.from_arrays([x, y, z], names=["x", "y", "z"])
        elif index == "ijk":
            i, j, k = self.geometry.ijk_from_row_index(block_ids)
            df.index = pd.MultiIndex.from_arrays([i, j, k], names=["i", "j", "k"])
        elif index is None:
            pass
        else:
            raise ValueError("index must be 'xyz', 'ijk', or None")
        if dense:
            if index == "xyz":
                dense_index = self.geometry.to_multi_index_xyz()
                df = df.reindex(dense_index)
            elif index == "ijk":
                dense_index = self.geometry.to_multi_index_ijk()
                df = df.reindex(dense_index)
            else:
                raise ValueError("dense=True requires index='xyz' or index='ijk'.")

        return df

    def triangulate(
        self,
        attributes: Optional[list[str]] = None,
        surface_only: bool = True,
        sparse: Optional[bool] = None,
    ) -> 'parq_blockmodel.mesh.TriangleMesh':
        """Generate a triangulated mesh from the block model.

        Creates a triangle mesh representation of the block model geometry,
        optionally including block attributes (grades, rock types, etc.) as
        vertex or face attributes.

        Parameters
        ----------
        attributes : list[str], optional
            List of attribute columns to include in the mesh. If None,
            only geometry is included (no attributes). Attributes must
            be in :attr:`self.attributes`.
        surface_only : bool, default True
            If True, include only exterior surface faces. If False, include
            all interior faces as well. Useful for sparse models.
        sparse : bool, optional
            If True (or None and sparse model detected), include only blocks
            that exist in the data. If False, generate mesh for full dense grid.

        Returns
        -------
        parq_blockmodel.mesh.TriangleMesh
            Triangle mesh with vertices, faces, and optional attributes.

        Notes
        -----
        The mesh uses right-handed coordinates in world space, with rotation
        applied via :attr:`geometry` axis vectors. Attributes are preserved
        as per-vertex or per-face properties. For sparse models, only surface
        faces are typically needed (surface_only=True).

        Examples
        --------
        >>> pbm = ParquetBlockModel(Path("model.pbm"))
        >>> mesh = pbm.triangulate(attributes=["grade", "density"], surface_only=True)
        >>> print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
        """
        from parq_blockmodel.mesh.triangulation import BlockMeshGenerator

        generator = BlockMeshGenerator(self.geometry)

        # Read block data if attributes are requested
        if attributes:
            invalid_attrs = [a for a in attributes if a not in self.attributes]
            if invalid_attrs:
                raise ValueError(
                    f"Attributes not found: {invalid_attrs}. Available: {self.attributes}"
                )
            # Read only the requested attributes; read() with index="ijk"
            # derives the (i, j, k) MultiIndex separately, so we must NOT
            # include i/j/k in columns — otherwise reset_index() collides.
            block_data = self.read(columns=attributes, index="ijk")
            block_data = block_data.reset_index()
        else:
            block_data = None

        # Generate mesh
        mesh = generator.triangulate(
            block_data=block_data,
            surface_only=surface_only,
            sparse=sparse if sparse is not None else self.is_sparse,
        )

        return mesh

    def to_ply(
        self,
        output_path: Union[str, Path],
        attributes: Optional[list[str]] = None,
        surface_only: bool = True,
        sparse: Optional[bool] = None,
        binary: bool = False,
    ) -> Path:
        """Export the block model as a PLY (Polygon File Format) mesh.

        PLY is the canonical format for mesh storage, supporting lossless
        round-tripping of all geometry and attribute data. The output file
        includes:
        - Vertex coordinates in world space
        - Face connectivity (triangles)
        - Per-vertex and per-face attributes (grades, rock type, etc.)
        - Block logical indices (i, j, k) for traceability
        - Geometry metadata (corner, block_size, shape, axes, CRS)

        Parameters
        ----------
        output_path : str or Path
            Target PLY file path.
        attributes : list[str], optional
            Block attributes to include. If None, only geometry is exported.
        surface_only : bool, default True
            If True, export only exterior surface faces.
        sparse : bool, optional
            If True, export only existing blocks. If None, infer from model sparsity.
        binary : bool, default False
            If True, write binary PLY (smaller file, less readable).
            If False, write ASCII PLY (larger, human-readable).

        Returns
        -------
        Path
            The output file path (pathlib.Path).

        Notes
        -----
        Units and coordinate system are inherited from :attr:`geometry`.
        For rotated geometries, world-space coordinates reflect the rotation.

        Examples
        --------
        >>> pbm = ParquetBlockModel(Path("model.pbm"))
        >>> pbm.to_ply("model.ply", attributes=["grade", "density"])
        """
        from parq_blockmodel.mesh.ply import write_ply

        mesh = self.triangulate(
            attributes=attributes,
            surface_only=surface_only,
            sparse=sparse,
        )
        write_ply(mesh, output_path, binary=binary)
        return Path(output_path)

    def to_glb(
        self,
        output_path: Union[str, Path],
        attributes: Optional[list[str]] = None,
        texture_attribute: Optional[str] = None,
        colormap: str = "viridis",
        surface_only: bool = True,
        sparse: Optional[bool] = None,
    ) -> Path:
        """Export the block model as a GLB (glTF 2.0 binary) mesh.

        GLB is a derived format for external visualization in 3D viewers.
        Optionally applies vertex colors based on a scalar attribute
        (e.g., grade, density). Geometry metadata is embedded in the
        glTF extras field for reference.

        Parameters
        ----------
        output_path : str or Path
            Target GLB file path.
        attributes : list[str], optional
            Block attributes to include (stored in glTF extensions).
        texture_attribute : str, optional
            Attribute name to use for vertex coloring (e.g., "grade").
            Must be numeric. If None, uses default gray material.
        colormap : str, default "viridis"
            Matplotlib colormap name for mapping texture_attribute to colors.
            Examples: "viridis", "plasma", "coolwarm", "Greys".
        surface_only : bool, default True
            If True, export only exterior surface faces.
        sparse : bool, optional
            If True, export only existing blocks. If None, infer from model sparsity.

        Returns
        -------
        Path
            The output file path (pathlib.Path).

        Notes
        -----
        GLB format is optimized for visualization and may lose some precision
        compared to PLY. For scientific workflows requiring lossless data
        preservation, prefer :meth:`to_ply`.

        The texture_attribute is normalized to [0, 1] and mapped to the
        specified colormap. NaN values are rendered as transparent.

        Examples
        --------
        >>> pbm = ParquetBlockModel(Path("model.pbm"))
        >>> pbm.to_glb("model.glb", texture_attribute="grade", colormap="viridis")
        """
        from parq_blockmodel.mesh.glb import write_glb

        # Ensure texture_attribute is always triangulated into the mesh even
        # when the caller left the generic `attributes` list as None.
        attrs_for_mesh = list(attributes) if attributes else []
        if texture_attribute and texture_attribute not in attrs_for_mesh:
            attrs_for_mesh.append(texture_attribute)

        mesh = self.triangulate(
            attributes=attrs_for_mesh if attrs_for_mesh else None,
            surface_only=surface_only,
            sparse=sparse,
        )

        if texture_attribute and texture_attribute not in mesh.vertex_attributes:
            raise ValueError(
                f"Texture attribute '{texture_attribute}' not in mesh. "
                f"Available: {list(mesh.vertex_attributes.keys())}"
            )

        write_glb(
            mesh,
            output_path,
            texture_attribute=texture_attribute,
            colormap=colormap,
            include_metadata=True,
        )
        return Path(output_path)

    def to_pyvista(self, grid_type: typing.Literal["image", "structured", "unstructured"] = "structured",
                   attributes: Optional[list[str]] = None
                   ) -> Union['pv.ImageData', 'pv.StructuredGrid', 'pv.UnstructuredGrid']:

        if attributes is None:
            attributes = self.attributes

        if grid_type == "image":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
            # For image grids we always enable categorical encoding so that
            # object/category columns (e.g. depth_category) are mapped to
            # integer codes suitable for PyVista colour mapping. Numeric
            # columns are unaffected by this flag.
            df = self.read(columns=attributes, dense=False)
            grid = df_to_pv_image_data(df=df,
                                       geometry=self.geometry,
                                       categorical_encode=True)
        elif grid_type == "structured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_structured_grid
            grid = df_to_pv_structured_grid(df=self.read(columns=attributes, dense=False),
                                            validate_block_size=True)
        elif grid_type == "unstructured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_unstructured_grid
            grid = df_to_pv_unstructured_grid(df=self.read(columns=attributes, dense=False),
                                              block_size=self.geometry.local.block_size,
                                              validate_block_size=True)
        else:
            raise ValueError(f"Invalid grid type: {grid_type}. "
                             "Choose from 'image', 'structured', or 'unstructured'.")

        return grid

    @staticmethod
    def _validate_geometry(filepath: Path,
                           geometry: Optional[RegularGeometry] = None,
                           chunk_size: int = 1_000_000,
                           tol: float = 1e-6) -> None:
        """Validate centroid columns against a :class:`RegularGeometry`.

        This helper enforces that a Parquet file used as a
        :class:`ParquetBlockModel` has well‑formed centroid columns
        ``x``, ``y``, ``z`` and that these centroids lie on the dense
        grid implied by a :class:`RegularGeometry` instance.

        Parameters
        ----------
        filepath : Path
            Path to the Parquet file to validate (typically a ``.pbm``
            container or its source ``.parquet`` file).
        geometry : RegularGeometry, optional
            Geometry of the logical ijk grid. If omitted, it is inferred
            from ``filepath`` via :meth:`RegularGeometry.from_parquet`.

        Raises
        ------
        ValueError
            If any centroid column is missing or contains null values,
            if their lengths differ, or if the resulting sparse centroids
            are not a subset of the dense grid implied by ``geometry``.

        Notes
        -----
        For centroid-defined layouts we require explicit centroid columns
        ``x, y, z`` on disk. As the library moves toward an ijk‑first
        core, geometry will increasingly be treated as the sole source
        of truth and xyz may become purely a derived view. At that
        point this validation may be relaxed or supplemented with
        geometry‑only checks.
        """
        columns = pq.read_schema(filepath).names

        has_block_id = "block_id" in columns
        has_world_id = "world_id" in columns
        has_ijk = all(col in columns for col in ["i", "j", "k"])
        has_xyz = all(col in columns for col in ["x", "y", "z"])

        if not any([has_block_id, has_world_id, has_ijk, has_xyz]):
            raise ValueError("Dataset must contain at least one positional representation: block_id, world_id, ijk, or xyz.")

        if geometry is None:
            geometry = RegularGeometry.from_parquet(filepath)

        dense_count = int(np.prod(geometry.local.shape))
        seen: set[int] = set()

        pf = pq.ParquetFile(filepath)
        if has_block_id:
            columns_to_read = ["block_id"]
        elif has_world_id:
            columns_to_read = ["world_id"]
        elif has_xyz:
            columns_to_read = ["x", "y", "z"]
        else:
            columns_to_read = ["i", "j", "k"]

        for batch in pf.iter_batches(columns=columns_to_read, batch_size=chunk_size):
            if has_block_id:
                block_ids = np.asarray(batch.column(0), dtype=np.uint32)
            elif has_world_id:
                if not geometry.world_id_encoding:
                    raise ValueError("world_id column present but metadata has no world_id_encoding payload.")
                offset, scale = get_world_id_encoding_params(geometry.world_id_encoding)
                world_ids = np.asarray(batch.column(0), dtype=np.int64)
                x, y, z = decode_world_coordinates(world_ids, offset=offset, scale=scale)
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

        logging.info(f"Geometry validation completed successfully for {filepath}.")

    @classmethod
    def validate_xyz_parquet(cls,
                             parquet_path: Path,
                             axis_azimuth: float = 0.0,
                             axis_dip: float = 0.0,
                             axis_plunge: float = 0.0,
                             chunk_size: int = 1_000_000,
                             tol: float = 1e-6) -> RegularGeometry:
        """Validate xyz-defined Parquet input and return inferred geometry."""
        geometry = RegularGeometry.from_parquet(filepath=parquet_path,
                                                axis_azimuth=axis_azimuth,
                                                axis_dip=axis_dip,
                                                axis_plunge=axis_plunge,
                                                chunk_size=chunk_size)
        cls._validate_geometry(filepath=parquet_path, geometry=geometry, chunk_size=chunk_size, tol=tol)
        return geometry

    @classmethod
    def _build_world_id_encoding_from_xyz(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        scale: float = 10.0,
    ) -> dict[str, object]:
        """Build default world_id encoding metadata from xyz ranges."""
        ox = np.floor(float(np.min(x)) * scale) / scale
        oy = np.floor(float(np.min(y)) * scale) / scale
        oz = np.floor(float(np.min(z)) * scale) / scale

        x_span = float(np.max(x) - ox)
        y_span = float(np.max(y) - oy)
        z_span = float(np.max(z) - oz)
        if x_span > MAX_XY_VALUE or y_span > MAX_XY_VALUE or z_span > MAX_Z_VALUE:
            raise ValueError(
                "Coordinates exceed world_id span limits (24/24/16 bits at scale=10). "
                "Provide a custom encoding strategy."
            )

        return {
            "enabled": True,
            "column": "world_id",
            "frame": "world_xyz",
            "axis_order": ["x", "y", "z"],
            "quantization": {
                "scale": scale,
                "rounding": "nearest",
                "precision_decimals": 1,
            },
            "offset": {"x": ox, "y": oy, "z": oz, "units": "world"},
            "bit_layout": {
                "x_bits": 24,
                "y_bits": 24,
                "z_bits": 16,
                "x_shift": 40,
                "y_shift": 16,
                "z_shift": 0,
                "signed": False,
            },
            "range_after_offset": {
                "x_min": 0.0,
                "x_max": MAX_XY_VALUE,
                "y_min": 0.0,
                "y_max": MAX_XY_VALUE,
                "z_min": 0.0,
                "z_max": MAX_Z_VALUE,
            },
            "overflow_policy": "error",
            "null_policy": "no_nulls",
        }

    @classmethod
    def _write_canonical_pbm(cls,
                             input_path: Path,
                             output_path: Path,
                             geometry: RegularGeometry,
                             columns: Optional[list[str]],
                             chunk_size: int = 1_000_000,
                             tol: float = 1e-6) -> None:
        """Stream a source parquet file into canonical PBM with block_id and metadata."""
        pf = pq.ParquetFile(input_path)
        src_cols = pf.schema.names

        if columns is None:
            output_cols = list(src_cols)
        else:
            missing = [c for c in columns if c not in src_cols]
            if missing:
                raise ValueError(f"Requested columns not present in source parquet: {missing}")
            output_cols = list(columns)

        has_block_id = "block_id" in src_cols
        has_world_id = "world_id" in src_cols
        has_xyz = all(c in src_cols for c in ["x", "y", "z"])
        has_ijk = all(c in src_cols for c in ["i", "j", "k"])

        if not any([has_block_id, has_world_id, has_ijk, has_xyz]):
            raise ValueError("Cannot derive block_id from source parquet: requires block_id, world_id, ijk, or xyz columns.")

        if geometry.world_id_encoding is None:
            xmin, xmax, ymin, ymax, zmin, zmax = geometry.extents
            geometry.world_id_encoding = cls._build_world_id_encoding_from_xyz(
                np.array([xmin, xmax], dtype=float),
                np.array([ymin, ymax], dtype=float),
                np.array([zmin, zmax], dtype=float),
            )
        offset, scale = get_world_id_encoding_params(geometry.world_id_encoding)

        read_cols = list(dict.fromkeys(output_cols + [c for c in ["block_id", "world_id", "i", "j", "k", "x", "y", "z"] if c in src_cols]))
        if "block_id" not in output_cols:
            output_cols = ["block_id"] + output_cols
        if "world_id" not in output_cols:
            output_cols = ["block_id", "world_id"] + [c for c in output_cols if c != "block_id"]

        with atomic_output_file(output_path) as tmp_path:
            writer = None
            try:
                for batch in pf.iter_batches(columns=read_cols, batch_size=chunk_size):
                    df_batch = pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)

                    if "block_id" in df_batch.columns:
                        block_ids = df_batch["block_id"].to_numpy(dtype=np.uint32)
                    elif "world_id" in df_batch.columns:
                        world_ids = df_batch["world_id"].to_numpy(dtype=np.int64)
                        xw, yw, zw = decode_world_coordinates(world_ids, offset=offset, scale=scale)
                        block_ids = geometry.row_index_from_xyz(xw, yw, zw, tol=tol).astype(np.uint32)
                    elif {"x", "y", "z"}.issubset(df_batch.columns):
                        block_ids = geometry.row_index_from_xyz(
                            df_batch["x"].to_numpy(), df_batch["y"].to_numpy(), df_batch["z"].to_numpy(), tol=tol
                        ).astype(np.uint32)
                    else:
                        block_ids = geometry.row_index_from_ijk(
                            df_batch["i"].to_numpy(), df_batch["j"].to_numpy(), df_batch["k"].to_numpy()
                        ).astype(np.uint32)

                    if np.any(block_ids < 0) or np.any(block_ids >= int(np.prod(geometry.local.shape))):
                        raise ValueError("Source data contains positions outside geometry bounds.")

                    df_batch["block_id"] = block_ids
                    if "world_id" not in df_batch.columns:
                        if {"x", "y", "z"}.issubset(df_batch.columns):
                            x = df_batch["x"].to_numpy(dtype=float)
                            y = df_batch["y"].to_numpy(dtype=float)
                            z = df_batch["z"].to_numpy(dtype=float)
                        else:
                            x, y, z = geometry.xyz_from_row_index(block_ids)
                        df_batch["world_id"] = encode_world_coordinates(
                            x, y, z, offset=offset, scale=scale
                        ).astype(np.int64)
                    write_df = df_batch[output_cols]

                    table = pa.Table.from_pandas(write_df, preserve_index=False)
                    meta = table.schema.metadata or {}
                    meta = dict(meta)
                    meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
                    table = table.replace_schema_metadata(meta)

                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()

    @staticmethod
    def _validate_and_load_data(df, expected_num_blocks):
        """Legacy helper to ensure data is compatible with model geometry.

        The preferred layout includes explicit centroid columns ``x``,
        ``y``, ``z``. If these are missing but the row count matches the
        expected dense grid size, the function assumes the existing row
        order already aligns with the ijk C‑order implied by the
        geometry and emits a warning. Otherwise, a :class:`ValueError`
        is raised.

        This behaviour is intended **solely for backwards
        compatibility**. New code should persist centroid coordinates or
        rely on ijk indices plus geometry instead of implicit ordering.
        """
        required_cols = {'x', 'y', 'z'}
        if not required_cols.issubset(df.columns):
            if len(df) == expected_num_blocks:
                warnings.warn("Data loaded without x, y, z columns. "
                              "Order is assumed to match the block model geometry.")
            else:
                raise ValueError("Data missing x, y, z and row count does not match block model.")
        return df

    def to_dense_parquet(self, filepath: Path,
                         chunk_size: int = 100_000, show_progress: bool = False) -> None:
        """
        Export the block model as a **dense** xyz-indexed Parquet file.

        The underlying ``.pbm`` may be sparse with respect to the dense
        ijk grid encoded by :attr:`geometry`. This helper iterates over
        the on‑disk data in chunks, reindexes each chunk to the full
        dense centroid MultiIndex ``geometry.to_multi_index_xyz()`` and
        writes the result to ``filepath``.

        Parameters
        ----------
        filepath : Path
            Target path for the exported Parquet file.
        chunk_size : int, default 100_000
            Number of rows to process per chunk when reading from the
            backing ``.pbm`` file.
        show_progress : bool, default False
            If True, display a progress bar while exporting.

        Notes
        -----
        This export is primarily intended for interoperability with
        tools that expect a fully populated xyz grid. The canonical
        representation of the block model remains the ``.pbm`` file with
        embedded :class:`RegularGeometry` metadata.
        """
        columns = self.columns
        dense_index = self.geometry.to_multi_index_xyz()
        parquet_file = pq.ParquetFile(self.blockmodel_path)
        total_rows = parquet_file.metadata.num_rows
        total_batches = max(math.ceil(total_rows / chunk_size), 1)

        progress = tqdm(total=total_batches, desc="Exporting", disable=not show_progress) if show_progress else None

        with atomic_output_file(filepath) as tmp_path:
            writer = None
            try:
                for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
                    df = pa.Table.from_batches([batch]).to_pandas()
                    df = df.reindex(dense_index)
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
                    if progress:
                        progress.update(1)
            finally:
                if writer is not None:
                    writer.close()
                if progress:
                    progress.close()

# ---------------------------------------------------------------------------
# Backwards-compatible private helper for tests
# ---------------------------------------------------------------------------

def _create_demo_blockmodel(*args, **kwargs):
    """Thin wrapper around :func:`parq_blockmodel.utils.demo_block_model.create_demo_blockmodel`.

    Some tests and older scripts import this symbol from ``parq_blockmodel.blockmodel``
    directly. Re-expose it here for backward compatibility while delegating all
    behaviour to the utility implementation.
    """

    return create_demo_blockmodel(*args, **kwargs)