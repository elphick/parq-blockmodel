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
    DEFAULT_AXIS_BITS,
    ENCODED_X_BITS,
    ENCODED_Y_BITS,
    ENCODED_Z_BITS,
    decode_world_coordinates,
    encode_world_coordinates,
    get_world_id_encoding_params,
)
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from parq_tools import ParquetProfileReport, LazyParquetDF
from parq_tools.utils import atomic_output_file

from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.reporting.utils import resolve_report_columns_per_batch
from parq_blockmodel.reblocking.reblocking import downsample_blockmodel, upsample_blockmodel
from parq_blockmodel.schema.service import SchemaService
from parq_blockmodel.schema import utils as schema_utils
from parq_blockmodel.io.ingest_writer import IngestWriter
from parq_blockmodel.io import ingest_utils

if typing.TYPE_CHECKING:
    import pyvista as pv  # type: ignore[import]
    import plotly.graph_objects as go
    import parq_blockmodel.mesh  # type: ignore[import]
    from parq_blockmodel.reporting import BlockModelReport  # type: ignore[import]
    from pandera import DataFrameSchema  # type: ignore[import]
    from parq_blockmodel.polygon_field import PolygonField
    from parq_blockmodel.solid import MeshSolid

logger = logging.getLogger(__name__)

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
    # ``self.attributes``.
    POSITION_COLUMNS = {"block_id", "world_id", "i", "j", "k", "x", "y", "z"}
    INTRINSIC_VOLUME_COLUMN = "volume"
    SPECIAL_COLUMN_ORDER = ["block_id", "world_id", "i", "j", "k", "x", "y", "z"]
    PBM_METADATA_KEY = b"parq-blockmodel"
    SPECIAL_COLUMN_DTYPES = {
        "block_id": np.int32,
        "world_id": np.int64,
        "i": np.int32,
        "j": np.int32,
        "k": np.int32,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    }

    def __init__(
        self,
        blockmodel_path: Path,
        name: Optional[str] = None,
        geometry: Optional[RegularGeometry] = None,
        schema: Optional[Union[Path, "DataFrameSchema"]] = None,
        engine_initializer: Optional[typing.Callable] = None,
    ):
        """
        Initialize a ParquetBlockModel.

        Args:
            blockmodel_path: Path to the .pbm file.
            name: Optional name; defaults to file stem.
            geometry: Optional RegularGeometry; loaded from metadata if not provided.
            schema: Optional pandera DataFrameSchema or path to YAML schema.
            engine_initializer: Optional callable that configures the df-eval Engine
                for custom lookups/functions. Signature: Callable[[Engine], Engine].
                Called before applying calculated column operations.
        """
        if blockmodel_path.suffix != ".pbm":
            raise ValueError("The provided file must have a '.pbm' extension.")
        self.blockmodel_path = blockmodel_path
        self.name = name or blockmodel_path.stem
        self.geometry = geometry or RegularGeometry.from_parquet(self.blockmodel_path)
        if schema is not None:
            self.schema = schema_utils.load_schema(schema)
        else:
            self.schema = schema_utils.load_embedded_schema(self.blockmodel_path)
        self.compression = (
            schema_utils.load_embedded_compression(self.blockmodel_path)
            or schema_utils.extract_schema_compression_policy(self.schema)
        )
        self._engine_initializer = engine_initializer
        self.pf: ParquetFile = ParquetFile(blockmodel_path)
        # Lazy view over the backing Parquet file for large models.
        self.data: LazyParquetDF = LazyParquetDF(self.blockmodel_path)
        self.columns: list[str] = pq.read_schema(self.blockmodel_path).names
        self._centroid_index: Optional[pd.MultiIndex] = None
        self.attributes: list[str] = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._extract_column_dtypes()
        self._logger = logging.getLogger(__name__)
        self._assert_canonical_block_id_invariant()
        self._schema_service: Optional[SchemaService] = None

        if self.is_sparse:
            if not self.validate_sparse():
                raise ValueError("The sparse ParquetBlockModel is invalid. "
                                 "Sparse centroids must be a subset of the dense grid.")

    def configure_engine(self, initializer: typing.Callable) -> None:
        """Register custom lookups and functions with the df-eval Engine.

        This method allows post-construction configuration of the df-eval Engine
        for custom lookups and functions used in calculated columns. Can be called
        before reading calculated columns.

        Args:
            initializer: Callable that accepts an Engine, registers lookups/functions,
                and returns the configured Engine. Signature: Callable[[Engine], Engine].

        Examples:
            >>> def setup_engine(engine):
            ...     engine.register_lookup("rock_codes", lookup_df)
            ...     engine.register_function("convert_units", converter_fn)
            ...     return engine
            >>> pbm.configure_engine(setup_engine)
            >>> result = pbm.read(columns=["rock_name", "converted_value"])
        """
        self._engine_initializer = initializer
        self._schema_service = None  # Invalidate cached service

    @property
    def schema_service(self) -> SchemaService:
        """Lazy-initialized schema service (recreated if schema or engine changes).
        
        Returns
        -------
        SchemaService
            Stateful schema service bound to current geometry and schema.
        """
        if self._schema_service is None:
            self._schema_service = SchemaService(
                geometry=self.geometry,
                schema=self.schema,
                engine_initializer=self._engine_initializer,
            )
        return self._schema_service

    def _assert_canonical_block_id_invariant(self) -> None:
        """Enforce mandatory canonical identity invariant for `.pbm` files."""
        if "block_id" not in self.columns:
            raise ValueError("Canonical .pbm requires a 'block_id' column.")

        dense_count = int(np.prod(self.geometry.local.shape))
        seen: set[int] = set()
        total = 0
        for batch in self._iter_batches(["block_id"]):
            block_ids = np.asarray(batch.column(0), dtype=np.int64)
            if np.any(block_ids < 0) or np.any(block_ids >= dense_count):
                raise ValueError("Canonical .pbm has block_id values outside geometry bounds.")
            if np.unique(block_ids).size != block_ids.size:
                raise ValueError("Canonical .pbm requires unique block_id values.")
            seen_before = len(seen)
            seen.update(block_ids.tolist())
            if len(seen) - seen_before != block_ids.size:
                raise ValueError("Canonical .pbm requires unique block_id values.")
            total += int(block_ids.size)

        if total != int(self.pf.metadata.num_rows):
            raise ValueError("Canonical .pbm block_id validation could not cover all rows.")

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
    def index_columns(self) -> list[str]:
        return ["i", "j", "k"]

    @property
    def spatial_columns(self) -> list[str]:
        return ["x", "y", "z"]

    @property
    def id_column(self) -> str:
        return "block_id"

    @property
    def world_id_column(self) -> str:
        return "world_id"

    @property
    def position_columns(self) -> list[str]:
        """Persisted positional/identity columns present on disk."""
        return [c for c in self.SPECIAL_COLUMN_ORDER if c in self.columns]

    @property
    def persisted_columns(self) -> list[str]:
        """Column names physically persisted in the backing parquet."""
        return list(self.columns)

    @property
    def persisted_attributes(self) -> list[str]:
        """Persisted block-property columns (non-positional)."""
        return [c for c in self.columns if c not in self.POSITION_COLUMNS]

    @property
    def calculated_columns(self) -> list[str]:
        """df-eval columns available for materialization (schema + intrinsic)."""
        operations = self._available_df_eval_operations()
        return list(operations)

    @property
    def calculated_attributes(self) -> list[str]:
        """Calculated block-property columns classified as attributes."""
        flags = self._calculated_attribute_flags()
        return [
            c for c in self.calculated_columns
            if c not in self.POSITION_COLUMNS and flags.get(c, True)
        ]

    @property
    def available_columns(self) -> list[str]:
        """Persisted + calculated columns that can be read from this model."""
        columns = list(self.columns)
        columns.extend([c for c in self.calculated_columns if c not in columns])
        return self._ordered_columns(columns)

    @property
    def available_attributes(self) -> list[str]:
        """Persisted + calculated attribute columns."""
        attrs = list(self.attributes)
        attrs.extend([c for c in self.calculated_attributes if c not in attrs])
        return attrs

    @classmethod
    def _ordered_columns(cls, columns: list[str]) -> list[str]:
        ordered = [c for c in cls.SPECIAL_COLUMN_ORDER if c in columns]
        ordered.extend([c for c in columns if c not in cls.SPECIAL_COLUMN_ORDER])
        return ordered

    @classmethod
    def _coerce_special_column_dtypes(
        cls,
        dataframe: pd.DataFrame,
        columns: Optional[typing.Iterable[str]] = None,
    ) -> pd.DataFrame:
        selected = set(columns) if columns is not None else set(cls.SPECIAL_COLUMN_DTYPES)
        for column, dtype in cls.SPECIAL_COLUMN_DTYPES.items():
            if column not in selected:
                continue
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(dtype, copy=False)
        return dataframe

    @classmethod
    def _load_schema(cls, schema: Union[Path, "DataFrameSchema"]) -> "DataFrameSchema":
        """Load and return a pandera DataFrameSchema.
        
        Delegates to schema_utils.load_schema for backwards compatibility.
        """
        return schema_utils.load_schema(schema)

    @classmethod
    def _dump_schema_yaml(cls, schema: "DataFrameSchema") -> str:
        """Dump schema to YAML text.
        
        Delegates to schema_utils.dump_schema_yaml for backwards compatibility.
        """
        return schema_utils.dump_schema_yaml(schema)

    @classmethod
    def _load_embedded_schema(cls, parquet_path: Path) -> Optional["DataFrameSchema"]:
        """Load embedded schema from Parquet metadata.
        
        Delegates to schema_utils.load_embedded_schema for backwards compatibility.
        """
        return schema_utils.load_embedded_schema(parquet_path)

    @classmethod
    def _build_schema_metadata(
        cls,
        *,
        geometry: RegularGeometry,
        schema: Optional["DataFrameSchema"],
        base_metadata: Optional[dict[bytes, bytes]] = None,
        compression: Optional[dict[str, typing.Any]] = None,
    ) -> dict[bytes, bytes]:
        """Build schema metadata dict.
        
        Delegates to schema_utils.build_schema_metadata for backwards compatibility.
        """
        return schema_utils.build_schema_metadata(
            geometry,
            schema,
            base_metadata,
            compression=compression,
        )

    @classmethod
    def _df_eval_operations_from_schema(cls, schema: "DataFrameSchema") -> dict[str, dict[str, typing.Any]]:
        """Extract df-eval operations from schema.
        
        Delegates to schema_utils for backwards compatibility.
        """
        return schema_utils.df_eval_operations_from_schema(schema)

    def _intrinsic_df_eval_operations(self) -> dict[str, dict[str, typing.Any]]:
        """Get intrinsic df-eval operations.
        
        Delegates to schema_service.
        """
        return self.schema_service.intrinsic_df_eval_operations()

    def _available_df_eval_operations(self) -> dict[str, dict[str, typing.Any]]:
        """Get all available df-eval operations (schema + intrinsic).
        
        Delegates to schema_service.
        """
        return self.schema_service.available_df_eval_operations()

    def _schema_column_attribute_flags(self) -> dict[str, bool]:
       """Get attribute classification flags from schema.
        
       Delegates to schema_service.
       """
       return self.schema_service.schema_column_attribute_flags()

    def _get_schema_column_names(self) -> list[str]:
       """Extract schema column names in definition order.
        
       Delegates to schema_service.
       """
       return self.schema_service.get_schema_column_names()

    @staticmethod
    def _extract_non_empty_schema_field(
       target: typing.Any,
       field: str,
       *,
       include_metadata: bool = True,
    ) -> Optional[str]:
       """Extract a non-empty field from a schema object.
        
       Delegates to schema_service helper.
       """
       return SchemaService._extract_non_empty_schema_field(target, field, include_metadata=include_metadata)

    def _report_schema_metadata(
       self,
       selected_columns: list[str],
    ) -> tuple[Optional[dict[str, str]], Optional[dict[str, str]]]:
       """Extract metadata descriptions for reporting.
        
       Delegates to schema_service.
       """
       return self.schema_service.report_schema_metadata(selected_columns)

    def _calculated_attribute_flags(self) -> dict[str, bool]:
       """Classify calculated columns as attributes or positional.
        
       Delegates to schema_service.
       """
       return self.schema_service.calculated_attribute_flags(self.calculated_columns)

    def _apply_intrinsic_operations(
        self,
        dataframe: pd.DataFrame,
        operations: dict[str, dict[str, typing.Any]],
    ) -> pd.DataFrame:
        """Apply intrinsic operations to a DataFrame.
        
        Delegates to schema_service.
        """
        return self.schema_service.apply_intrinsic_operations(dataframe, operations)

    @classmethod
    def _apply_df_eval_operations(
        cls,
        dataframe: pd.DataFrame,
        schema: "DataFrameSchema",
        operations: Optional[dict[str, dict[str, typing.Any]]] = None,
        engine_initializer: Optional[typing.Callable] = None,
    ) -> pd.DataFrame:
        """Apply df-eval operations (aliases, decimals, expressions).
        
        This is a compatibility wrapper. For internal use with geometry context,
        use schema_service.apply_df_eval_operations() instead.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to transform
        schema : DataFrameSchema
            Pandera schema with optional df-eval metadata
        operations : dict, optional
            Pre-extracted operations (if None, extracted from schema)
        engine_initializer : Callable, optional
            Optional engine customization function
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        try:
            from df_eval.pandera import apply_aliases, apply_decimals, df_eval_operations_from_pandera
            from df_eval import Engine
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Calculated column support requires 'df-eval'. Install it with: "
                "pip install 'parq-blockmodel[schema]'"
            ) from exc

        # Step 1: Apply alias transforms (rename columns)
        df = apply_aliases(dataframe, schema)

        # Step 2: Apply decimals transforms
        df = apply_decimals(df, schema)

        # Step 3: Apply df-eval operations (expr/lookup/function)
        ops = operations if operations is not None else df_eval_operations_from_pandera(schema)
        if not ops:
            return df

        # Create engine, potentially customized via engine_initializer
        engine = Engine()
        if engine_initializer is not None:
            engine = engine_initializer(engine)

        # Apply operations using the engine
        return engine.apply_operations(df, ops)

    @classmethod
    def _select_df_eval_operations(
        cls,
        operations: dict[str, dict[str, typing.Any]],
        targets: typing.Iterable[str],
    ) -> dict[str, dict[str, typing.Any]]:
        """Recursively select df-eval operations and dependencies.
        
        Delegates to schema_utils for backwards compatibility.
        """
        return schema_utils.select_df_eval_operations(operations, targets)

    @classmethod
    def _extract_required_columns_from_schema(cls, schema: "DataFrameSchema") -> set[str]:
        """Extract required columns from schema.
        
        Delegates to schema_utils for backwards compatibility.
        """
        return schema_utils.extract_required_columns_from_schema(schema)

    @classmethod
    def _extract_column_aliases_from_schema(cls, schema: "DataFrameSchema") -> dict[str, str]:
        """Extract column aliases from schema.
        
        Delegates to schema_utils for backwards compatibility.
        """
        return schema_utils.extract_column_aliases_from_schema(schema)

    @classmethod
    def _validate_chunk(cls, dataframe: pd.DataFrame, schema: "DataFrameSchema") -> pd.DataFrame:
        """Validate and coerce a DataFrame chunk.
        
        Delegates to schema_utils for backwards compatibility.
        """
        return schema_utils.validate_chunk(dataframe, schema)

    def validate(self, schema: Optional[Union[Path, "DataFrameSchema"]] = None, sample_chunks: int = 0) -> bool:
        """Validate the block model data against a pandera schema.

        Reads the backing ``.pbm`` file in chunks and validates each chunk
        against the supplied (or stored) schema.

        Args:
            schema: A pandera ``DataFrameSchema`` to validate against.  When
                ``None`` (default) ``self.schema`` is used.
            sample_chunks: Number of chunks to validate.  ``0`` (default)
                validates every chunk; a positive integer limits validation to
                that many leading chunks, which is useful for quick spot-checks
                on large models.

        Returns:
            bool: ``True`` when all validated chunks pass.

        Raises:
            ValueError: If no schema is available (neither passed nor stored).
            pandera.errors.SchemaErrors: If any chunk fails validation (when
                pandera raises in lazy mode).
        """
        if sample_chunks < 0:
            raise ValueError("sample_chunks must be >= 0")

        if schema is None:
            active_schema = self.schema
        else:
            active_schema = self._load_schema(schema)
        if active_schema is None:
            raise ValueError(
                "No schema provided. Either pass a schema to validate() or set self.schema "
                "via the constructor / entry-point methods."
            )

        chunks_seen = 0
        for batch in self._iter_batches(self.columns):
            df_batch = pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)
            self._validate_chunk(df_batch, active_schema)
            chunks_seen += 1
            if sample_chunks and chunks_seen >= sample_chunks:
                break

        return True

    def _persist_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        compression_policy: Optional[dict[str, typing.Any]] = None,
    ) -> None:
        dataframe = dataframe[self._ordered_columns(list(dataframe.columns))]
        dataframe = self._coerce_special_column_dtypes(dataframe)
        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        active_policy = compression_policy or schema_utils.resolve_active_compression_policy("fast")
        meta = self._build_schema_metadata(
            geometry=self.geometry,
            schema=self.schema,
            base_metadata=dict(table.schema.metadata or {}),
            compression=active_policy,
        )
        table = table.replace_schema_metadata(meta)
        pq.write_table(
            table,
            self.blockmodel_path,
            **schema_utils.build_parquet_compression_kwargs(table.column_names, active_policy),
        )

        self.pf = ParquetFile(self.blockmodel_path)
        self.data = LazyParquetDF(self.blockmodel_path)
        self.columns = pq.read_schema(self.blockmodel_path).names
        self.attributes = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._centroid_index = None
        self._extract_column_dtypes()
        self.compression = active_policy

    def recompute_spatial(self) -> None:
        """Recompute and persist canonical spatial columns (x, y, z, i, j, k)."""
        df = pq.read_table(self.blockmodel_path).to_pandas()
        if "block_id" in df.columns:
            block_ids = df["block_id"].to_numpy(dtype=np.uint32)
        elif {"i", "j", "k"}.issubset(df.columns):
            block_ids = self.geometry.row_index_from_ijk(
                df["i"].to_numpy(dtype=np.int64),
                df["j"].to_numpy(dtype=np.int64),
                df["k"].to_numpy(dtype=np.int64),
            ).astype(np.uint32)
        elif {"x", "y", "z"}.issubset(df.columns):
            block_ids = self.geometry.row_index_from_xyz(
                df["x"].to_numpy(dtype=float),
                df["y"].to_numpy(dtype=float),
                df["z"].to_numpy(dtype=float),
            ).astype(np.uint32)
        else:
            raise ValueError("Cannot recompute spatial columns without block_id, ijk, or xyz.")

        x, y, z = self.geometry.xyz_from_row_index(block_ids)
        i, j, k = self.geometry.ijk_from_row_index(block_ids)
        df["x"] = x
        df["y"] = y
        df["z"] = z
        df["i"] = i.astype(np.int32)
        df["j"] = j.astype(np.int32)
        df["k"] = k.astype(np.int32)
        self._persist_dataframe(df, compression_policy=self.compression)

    def ensure_spatial(self) -> None:
        """Ensure canonical spatial columns exist; recompute when missing."""
        if not {"x", "y", "z", "i", "j", "k"}.issubset(self.columns):
            self.recompute_spatial()

    def recompute_world_id(self) -> None:
        """Recompute and persist world_id from current spatial columns and encoding metadata."""
        df = pq.read_table(self.blockmodel_path).to_pandas()
        if not {"x", "y", "z"}.issubset(df.columns):
            self.recompute_spatial()
            df = pq.read_table(self.blockmodel_path).to_pandas()

        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        z = df["z"].to_numpy(dtype=float)
        if self.geometry.world_id_encoding is None:
            self.geometry.world_id_encoding = self._build_world_id_encoding_from_xyz(x, y, z)

        offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)
        df["world_id"] = encode_world_coordinates(
            x, y, z, offset=offset, scale=scale, bits_per_axis=bits_per_axis
        ).astype(np.int64)
        self._persist_dataframe(df, compression_policy=self.compression)

    def ensure_world_id(self) -> None:
        """Ensure world_id exists; recompute when missing."""
        if "world_id" not in self.columns:
            self.recompute_world_id()

    def validate_special_columns(self, sample_size: int = 1000) -> bool:
        """Validate consistency of special columns against geometry and encoding metadata.

        Returns ``False`` when any available positional representation disagrees with another
        (for example block_id vs ijk/xyz, or world_id vs xyz + world_id_encoding).
        ``sample_size`` controls validation sampling for large datasets (set ``0`` to validate all rows).
        """
        cols = [c for c in self.SPECIAL_COLUMN_ORDER if c in self.columns]
        if not cols:
            return True

        df = pq.read_table(self.blockmodel_path, columns=cols).to_pandas()
        if sample_size > 0 and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=0)

        if "block_id" in df.columns and {"i", "j", "k"}.issubset(df.columns):
            expected = self.geometry.row_index_from_ijk(
                df["i"].to_numpy(dtype=np.int64),
                df["j"].to_numpy(dtype=np.int64),
                df["k"].to_numpy(dtype=np.int64),
            ).astype(np.uint32)
            if not np.array_equal(df["block_id"].to_numpy(dtype=np.uint32), expected):
                return False

        if "block_id" in df.columns and {"x", "y", "z"}.issubset(df.columns):
            expected = self.geometry.row_index_from_xyz(
                df["x"].to_numpy(dtype=float),
                df["y"].to_numpy(dtype=float),
                df["z"].to_numpy(dtype=float),
            ).astype(np.uint32)
            if not np.array_equal(df["block_id"].to_numpy(dtype=np.uint32), expected):
                return False

        if "world_id" in df.columns and {"x", "y", "z"}.issubset(df.columns):
            if self.geometry.world_id_encoding is None:
                return False
            offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)
            expected = encode_world_coordinates(
                df["x"].to_numpy(dtype=float),
                df["y"].to_numpy(dtype=float),
                df["z"].to_numpy(dtype=float),
                offset=offset,
                scale=scale,
                bits_per_axis=bits_per_axis,
            ).astype(np.int64)
            if not np.array_equal(df["world_id"].to_numpy(dtype=np.int64), expected):
                return False

        return True

    @property
    def corner(self) -> Point:
        """Backward-compatible access to the local grid corner."""
        return self.geometry.corner

    @property
    def origin(self) -> Point:
        """Backward-compatible access to the world-frame origin."""
        return self.geometry.origin

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
            offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)
            for batch in self._iter_batches(["world_id"], batch_size=batch_size):
                world_ids = np.asarray(batch.column(0), dtype=np.int64)
                x, y, z = decode_world_coordinates(world_ids, offset=offset, scale=scale, bits_per_axis=bits_per_axis)
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
    def from_parquet(
        cls,
        parquet_path: Path,
        columns: Optional[list[str]] = None,
        overwrite: bool = False,
        axis_azimuth: float = 0.0,
        axis_dip: float = 0.0,
        axis_plunge: float = 0.0,
        chunk_size: int = 1_000_000,
        schema: Optional[Union[Path, "DataFrameSchema"]] = None,
        engine_initializer: Optional[typing.Callable] = None,
        compression: typing.Any = "fast",
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
            schema (DataFrameSchema or Path, optional): Optional pandera schema used to
                validate and coerce each chunk during the canonical write.  Accepts a
                :class:`~pandera.DataFrameSchema` object or a :class:`~pathlib.Path` to
                a YAML schema file.  When provided the schema is also stored on the
                resulting instance (``self.schema``).

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

        ingest_utils.validate_geometry(parquet_path, geometry=geometry, chunk_size=chunk_size)

        loaded_schema = cls._load_schema(schema) if schema is not None else None

        # Canonical ParquetBlockModel files use a single ``.pbm`` suffix. We
        # write the PBM file alongside the source Parquet file by replacing
        # only the final extension.
        new_filepath: Path = parquet_path.resolve().with_suffix(".pbm")
        
        writer = IngestWriter(
            input_path=parquet_path,
            output_path=new_filepath,
            geometry=geometry,
            schema=loaded_schema,
            engine_initializer=engine_initializer,
            compression=compression,
        )
        writer.write(columns=columns, chunk_size=chunk_size)
        
        return cls(
            blockmodel_path=new_filepath,
            geometry=geometry,
            schema=loaded_schema,
            engine_initializer=engine_initializer,
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        filename: Path,
        geometry: Optional[RegularGeometry] = None,
        name: Optional[str] = None,
        overwrite: bool = False,
        axis_azimuth: float = 0.0,
        axis_dip: float = 0.0,
        axis_plunge: float = 0.0,
        chunk_size: int = 1_000_000,
        schema: Optional[Union[Path, "DataFrameSchema"]] = None,
        engine_initializer: Optional[typing.Callable] = None,
        compression: typing.Any = "fast",
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
            schema (DataFrameSchema or Path, optional): Optional pandera schema used to
                validate and coerce the DataFrame before writing.  Accepts a
                :class:`~pandera.DataFrameSchema` object or a :class:`~pathlib.Path` to
                a YAML schema file.  When provided the schema is also stored on the
                resulting instance (``self.schema``).

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

        loaded_schema = cls._load_schema(schema) if schema is not None else None

        # Convert MultiIndex to regular columns (preserves names)
        dataframe_work = dataframe.reset_index()

        writer = IngestWriter(
            input_path=None,  # No input file; we're writing from DataFrame
            output_path=pbm_path,
            geometry=geometry,
            schema=loaded_schema,
            engine_initializer=engine_initializer,
            compression=compression,
        )
        writer.write_dataframe(dataframe_work)

        # Validate geometry
        ingest_utils.validate_geometry(pbm_path, geometry, chunk_size=chunk_size)

        return cls(
            blockmodel_path=pbm_path,
            name=name,
            geometry=geometry,
            schema=loaded_schema,
            engine_initializer=engine_initializer,
        )

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
                              noise_std: float = 0.0,
                              noise_rel: float | None = None,
                              noise_seed: int | None = None,
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
        noise_std : float, default 0.0
            Absolute standard deviation of Gaussian noise added to grades.
        noise_rel : float, optional
            Relative standard deviation expressed as a fraction of
            ``(grade_max - grade_min)``. Mutually exclusive with
            ``noise_std``.
        noise_seed : int, optional
            Seed used for reproducible Gaussian noise.

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
                              noise_std=noise_std, noise_rel=noise_rel,
                              noise_seed=noise_seed, parquet_filepath=filename,
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
            from parq_blockmodel.io.ingest_utils import validate_geometry
            validate_geometry(filepath=filename, geometry=geometry)

        new_filepath = filename.resolve().with_suffix(".pbm")
        from parq_blockmodel.io.ingest_writer import IngestWriter
        writer = IngestWriter(
            input_path=filename,
            output_path=new_filepath,
            geometry=geometry,
        )
        writer.write(columns=None, chunk_size=1_000_000)

        return cls(blockmodel_path=new_filepath, geometry=geometry)

    @classmethod
    def from_geometry(
        cls,
        geometry: RegularGeometry,
        path: Path,
        name: Optional[str] = None,
        schema: Optional[Union[Path, "DataFrameSchema"]] = None,
        engine_initializer: Optional[typing.Callable] = None,
        compression: typing.Any = "fast",
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
        schema : DataFrameSchema or Path, optional
            Optional pandera schema stored on the resulting instance
            (``self.schema``).  Accepts a
            :class:`~pandera.DataFrameSchema` object or a
            :class:`~pathlib.Path` to a YAML schema file.

        Returns
        -------
        ParquetBlockModel
            A block model with geometry defined but no attribute
            columns.
        """
        centroids_df = geometry.to_dataframe()
        centroids_df["block_id"] = np.arange(int(np.prod(geometry.local.shape)), dtype=np.uint32)
        i, j, k = geometry.ijk_from_row_index(centroids_df["block_id"].to_numpy(dtype=np.uint32))
        centroids_df["i"] = i.astype(np.int32)
        centroids_df["j"] = j.astype(np.int32)
        centroids_df["k"] = k.astype(np.int32)

        loaded_schema = cls._load_schema(schema) if schema is not None else None

        writer = IngestWriter(
            input_path=None,  # No input file; we're writing from DataFrame
            output_path=path,
            geometry=geometry,
            schema=None,  # Don't apply schema validation for geometry-only writes
            engine_initializer=engine_initializer,
            compression=compression,
        )
        writer.write_dataframe(centroids_df)

        return cls(
            blockmodel_path=path,
            name=name,
            geometry=geometry,
            schema=loaded_schema,
            engine_initializer=engine_initializer,
        )

    def rename(self, new_pbm_filepath: Path, rename_to_new_pbm_stem: bool = True) -> "ParquetBlockModel":
        """Rename the underlying .pbm file and refresh path-bound caches.

        Parameters
        ----------
        new_pbm_filepath : Path
            Target .pbm file path.
        rename_to_new_pbm_stem : bool, default True
            If True, update self.name to the new file stem.

        Returns
        -------
        ParquetBlockModel
            Self (mutated).

        Notes
        -----
        This method only relocates the file on disk and refreshes internal
        caches. Any computed columns held in memory (not persisted to the
        original .pbm) will not be saved to the new location. Persist them first:

        >>> pbm.write(df_with_new_columns).rename(Path("new_location.pbm"))
        """
        new_pbm_filepath = Path(new_pbm_filepath)

        if new_pbm_filepath.suffix != ".pbm":
            raise ValueError("new_pbm_filepath must have a '.pbm' extension.")
        if not self.blockmodel_path.exists():
            raise FileNotFoundError(f"Current .pbm file does not exist: {self.blockmodel_path}")
        if new_pbm_filepath.exists():
            raise FileExistsError(f"Target file already exists: {new_pbm_filepath}")
        if not new_pbm_filepath.parent.exists():
            raise FileNotFoundError(f"Target directory does not exist: {new_pbm_filepath.parent}")

        if self.blockmodel_path.resolve() == new_pbm_filepath.resolve():
            if rename_to_new_pbm_stem:
                self.name = new_pbm_filepath.stem
            return self

        # Release file-backed handles before rename (important on Windows).
        self.pf = None
        self.data = None

        self.blockmodel_path.rename(new_pbm_filepath)

        # Refresh path-dependent state.
        self.blockmodel_path = new_pbm_filepath
        self.pf = ParquetFile(self.blockmodel_path)
        self.data = LazyParquetDF(self.blockmodel_path)
        self.columns = pq.read_schema(self.blockmodel_path).names
        self.attributes = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._centroid_index = None
        self._extract_column_dtypes()

        if rename_to_new_pbm_stem:
            self.name = self.blockmodel_path.stem

        return self

    def create_report(self, columns: Optional[list[str]] = None,
                      columns_per_batch: Optional[int] = 10,
                      show_progress: bool = True,
                      open_in_browser: bool = False,
                      report_path: Optional[Union[str, Path]] = None,
                      write_html: bool = True,
                      memory_budget: Optional[Union[int, float, str]] = None,
                      ) -> "BlockModelReport":
        """
        Create a ydata-profiling report for the block model.
        By default the generated HTML is saved beside the block model with a
        ``.html`` suffix, and the returned report object can be re-saved later.

        Args:
            columns: List of column names to include in the profile. If None, all columns are used.
            columns_per_batch: Number of columns to process in each profiling batch.
                Use None to profile all requested columns in one pass.
            show_progress: bool: If True, displays a progress bar during profiling.
            open_in_browser: bool: If True, opens the report in a web browser after generation.
            report_path: Optional output path for the HTML report.
            write_html: If True, save the report HTML before returning it.
            memory_budget: Optional heuristic per-batch memory budget in bytes or
                as a string such as ``"512MB"`` or ``"2GiB"``. When provided,
                the method derives ``columns_per_batch`` automatically from Parquet
                column-chunk metadata. If the full selected column set fits within
                the budget estimate, profiling runs without column chunking.

        Returns
            BlockModelReport: The generated report wrapper. Use
            :meth:`BlockModelReport.save` to write it again.

        """
        selected_columns = list(columns) if columns is not None else list(self.columns)
        column_descriptions, dataset_metadata = self._report_schema_metadata(selected_columns)
        
        logging.basicConfig(level=logging.DEBUG)
        
        effective_columns_per_batch, memory_budget_bytes = resolve_report_columns_per_batch(
            parquet_file=self.pf,
            columns=selected_columns,
            columns_per_batch=columns_per_batch,
            memory_budget=memory_budget,
        )
        resolved_output_path = Path(report_path) if report_path is not None else self.blockmodel_path.with_suffix('.html')

        from parq_blockmodel.reporting import BlockModelReport

        raw_report: ParquetProfileReport = ParquetProfileReport(
            self.blockmodel_path,
            columns=selected_columns,
            batch_size=effective_columns_per_batch,
            show_progress=show_progress,
            title=f"{self.name} Profile Report",
            dataset_metadata=dataset_metadata,
            column_descriptions=column_descriptions,
        ).profile()
        report = BlockModelReport(
            report=raw_report,
            output_path=resolved_output_path,
            columns=selected_columns,
            columns_per_batch=effective_columns_per_batch,
            memory_budget_bytes=memory_budget_bytes,
        )

        if write_html:
            report.save()
        if open_in_browser:
            report.show(notebook=False)
        self.report_path = report.output_path
        return report

    def downsample(self, new_block_size, aggregation_config) -> "ParquetBlockModel":
        """
        Downsample the block model to a coarser grid with specified aggregation methods for each attribute.
        This function supports downsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            aggregation_config: dict mapping attribute names to aggregation methods.
                Use ``basis`` for ``weighted_mean`` configurations.

        Example:
            aggregation_config = {
                'grade': {'method': 'weighted_mean', 'basis': 'dry_mass'},
                'density': {'method': 'weighted_mean', 'basis': 'volume'},
                'dry_mass': {'method': 'sum'},
                'volume': {'method': 'sum'},
                'rock_type': {'method': 'mode'}
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the downsampled grid.
        """
        return downsample_blockmodel(self, new_block_size, aggregation_config)

    def upsample(self, new_block_size, upsample_config=None, interpolation_config=None) -> "ParquetBlockModel":
        """
        Upsample the block model to a finer grid with specified methods for each attribute.
        This function supports upsampling of both categorical and numeric attributes.
        Args:
            new_block_size: tuple of floats (dx, dy, dz) for the new block size.
            upsample_config: dict mapping attribute names to upsampling methods
                (``linear``, ``nearest``, ``mode``, ``parent``).
            interpolation_config: deprecated alias of ``upsample_config``.

        Example:
            upsample_config = {
                'grade': 'linear',
                'density': 'linear',
                'dry_mass': 'linear',
                'volume': 'linear',
                'rock_type': 'mode'
            }
        Returns:
            ParquetBlockModel: A new ParquetBlockModel instance with the upsampled grid.
        """
        return upsample_blockmodel(
            self,
            new_block_size,
            upsample_config=upsample_config,
            interpolation_config=interpolation_config,
        )

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

        if attribute not in self.available_columns:
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
        ex = self.geometry.extents  # Extents object with xmin, xmax, ymin, ymax, zmin, zmax properties

        if axis == "z":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[1])
            z = summed_data
            labels = "Easting", "Northing"
            x_extent = (ex.xmin, ex.xmax)
            y_extent = (ex.ymin, ex.ymax)
        elif axis == "x":
            x = np.arange(summed_data.shape[1])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Northing", "RL"
            x_extent = (ex.ymin, ex.ymax)
            y_extent = (ex.zmin, ex.zmax)
        elif axis == "y":
            x = np.arange(summed_data.shape[0])
            y = np.arange(summed_data.shape[2])
            z = summed_data
            labels = "Easting", "RL"
            x_extent = (ex.xmin, ex.xmax)
            y_extent = (ex.zmin, ex.zmax)
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
        return (x, y, z), labels, (x_extent, y_extent)

    def plot(self, scalar: str,
             grid_type: typing.Literal["image", "structured", "unstructured"] = "image",
             frame: typing.Literal["world", "local"] = "world",
             threshold: bool = True, show_edges: bool = True,
             show_axes: bool = True, enable_picking: bool = False,
             picked_attributes: Optional[list[str]] = None,
             engine: Optional[typing.Any] = None) -> typing.Any:
        """Plot the block model using PyVista.

        Args:
            scalar: The name of the scalar attribute to visualize.
            grid_type: The type of grid to use for plotting. Options are "image", "structured", or "unstructured".
            frame: Coordinate frame for image grids. ``"world"`` applies geometry axis vectors;
                ``"local"`` uses axis-aligned local ijk orientation. Only supported for ``grid_type="image"``.
            threshold: The thresholding option for the mesh. If True, applies a threshold to the scalar values.
            show_edges: Show edges of the mesh.
            show_axes: Show the axes in the plot.
            enable_picking: If True, enables picking mode to interactively select cells in the plot.
            picked_attributes: A list of attributes that will be returned in picking mode. If None, all attributes are returned.

        Returns:

        """

        from parq_blockmodel.visualization.blockmodel_plot import PyVistaBlockModelPlotEngine

        if engine is None:
            engine = PyVistaBlockModelPlotEngine()
        return engine.plot(
            self,
            scalar=scalar,
            grid_type=grid_type,
            frame=frame,
            threshold=threshold,
            show_edges=show_edges,
            show_axes=show_axes,
            enable_picking=enable_picking,
            picked_attributes=picked_attributes,
        )

    def read(
        self,
        columns: Optional[list[str]] = None,
        index: typing.Literal["xyz", "ijk", None] = "xyz",
        dense: bool = False,
        include_calculated: bool = False,
    ) -> pd.DataFrame:
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
        include_calculated:
            If True, include available calculated columns (schema-defined
            and intrinsic) in the default no-argument read. Explicit column
            requests still behave as before.

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
        requested_columns = list(columns) if columns is not None else list(self.columns)
        operations = self._available_df_eval_operations()
        if columns is None and include_calculated:
            requested_columns = list(self.columns)
            requested_columns.extend(
                [col for col in operations if col not in self.columns]
            )

        # align column order with the schema
        if columns is None and self.schema is not None:
            schema_column_names = self._get_schema_column_names()
            schema_cols_in_data = [col for col in schema_column_names if col in requested_columns]
            extra_cols = [col for col in requested_columns if col not in schema_column_names]
            requested_columns = schema_cols_in_data + extra_cols

        missing_requested = [col for col in requested_columns if col not in self.columns]
        read_columns = requested_columns
        required_operations: Optional[dict[str, dict[str, typing.Any]]] = None
        if missing_requested:
            undefined = [col for col in missing_requested if col not in operations]
            if undefined:
                raise ValueError(
                    "Requested columns are neither on-disk nor defined by available df-eval operations: "
                    f"{undefined}"
                )
            required_operations = self._select_df_eval_operations(operations, missing_requested)
            read_columns = list(self.columns)

        df = pq.read_table(self.blockmodel_path, columns=read_columns).to_pandas()
        if missing_requested:
            if required_operations is None:  # defensive
                raise ValueError("No df-eval operations selected for requested calculated columns.")
            if self.schema is not None:
                df = self._apply_df_eval_operations(
                    df,
                    self.schema,
                    operations=required_operations,
                    engine_initializer=self._engine_initializer,
                )
            else:
                df = self._apply_intrinsic_operations(df, required_operations)
            missing_after_eval = [col for col in requested_columns if col not in df.columns]
            if missing_after_eval:
                raise ValueError(
                    "Calculated column evaluation did not produce requested columns: "
                    f"{missing_after_eval}"
                )
            df = df[requested_columns]

        block_ids: Optional[np.ndarray] = None
        if index in {"ijk", "xyz"}:
            if "block_id" in df.columns:
                block_ids = df["block_id"].to_numpy(dtype=np.uint32)
            elif "world_id" in df.columns:
                if not self.geometry.world_id_encoding:
                    raise ValueError("world_id column present but metadata has no world_id_encoding payload.")
                offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)
                x, y, z = decode_world_coordinates(
                    df["world_id"].to_numpy(dtype=np.int64),
                    offset=offset,
                    scale=scale,
                    bits_per_axis=bits_per_axis,
                )
                block_ids = self.geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
            elif {"i", "j", "k"}.issubset(df.columns):
                block_ids = self.geometry.row_index_from_ijk(
                    df["i"].to_numpy(), df["j"].to_numpy(), df["k"].to_numpy()
                ).astype(np.uint32)
            elif {"x", "y", "z"}.issubset(df.columns):
                block_ids = self.geometry.row_index_from_xyz(
                    df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
                ).astype(np.uint32)
            else:
                # Requested columns may omit positional fields (block_id/world_id/ijk/xyz).
                # Recover row ids from the backing dataset to preserve correct mapping even
                # when on-disk row order is not canonical ijk C-order.
                block_id_batches = [ids for ids in self._iter_block_ids()]
                if not block_id_batches:
                    derived_block_ids = np.empty(0, dtype=np.uint32)
                else:
                    derived_block_ids = np.concatenate(block_id_batches).astype(np.uint32, copy=False)
                block_ids = derived_block_ids
                if len(derived_block_ids) != len(df):
                    raise ValueError(
                        "Unable to align requested columns with positional rows: "
                        f"loaded {len(df)} data rows but derived {len(derived_block_ids)} block ids."
                    )

        if index in {"ijk", "xyz"} and block_ids is None:
            raise RuntimeError("Failed to derive block ids for requested indexed read.")

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

    @staticmethod
    def _resolve_inside_label(
        evaluator: typing.Any,
        inside_value: typing.Any,
    ) -> typing.Any:
        if inside_value is not None:
            return inside_value
        name = getattr(evaluator, "name", None)
        return name if name is not None else True

    def _flag_evaluator(
        self,
        evaluator: typing.Any,
        *,
        column: str,
        inside_value: typing.Any = None,
        outside_value: typing.Any = pd.NA,
        as_categorical: Optional[bool] = None,
    ) -> np.ndarray:
        mask = np.asarray(evaluator.evaluate(self.geometry), dtype=bool)
        expected_len = int(np.prod(self.geometry.local.shape))
        if mask.shape != (expected_len,):
            raise ValueError(
                f"Evaluator mask length mismatch: expected ({expected_len},), got {mask.shape}."
            )

        inside = self._resolve_inside_label(evaluator, inside_value)
        if as_categorical is None:
            as_categorical = isinstance(inside, str) or outside_value is pd.NA

        rows = self.read(columns=["block_id"], index=None)
        block_ids = rows["block_id"].to_numpy(dtype=np.int64)
        values = np.full(len(rows), outside_value, dtype=object)
        values[mask[block_ids]] = inside

        payload = pd.DataFrame({"block_id": block_ids})
        if as_categorical:
            categories = [inside]
            if outside_value is not pd.NA:
                categories.append(outside_value)
            payload[column] = pd.Categorical(values, categories=categories, ordered=False)
        else:
            payload[column] = pd.Series(values)

        self.write(payload, merge=True)
        return mask

    def flag_polygon(
        self,
        polygon: "PolygonField",
        *,
        column: str,
        inside_value: typing.Any = None,
        outside_value: typing.Any = pd.NA,
        as_categorical: Optional[bool] = None,
    ) -> np.ndarray:
        """Evaluate a PolygonField and persist a new flag/domain column.

        Returns the dense boolean inside/outside mask aligned to geometry C-order.
        """
        return self._flag_evaluator(
            polygon,
            column=column,
            inside_value=inside_value,
            outside_value=outside_value,
            as_categorical=as_categorical,
        )

    def flag_solid(
        self,
        solid: "MeshSolid",
        *,
        column: str,
        inside_value: typing.Any = None,
        outside_value: typing.Any = pd.NA,
        as_categorical: Optional[bool] = None,
    ) -> np.ndarray:
        """Evaluate a MeshSolid and persist a new flag/domain column.

        Returns the dense boolean inside/outside mask aligned to geometry C-order.
        """
        return self._flag_evaluator(
            solid,
            column=column,
            inside_value=inside_value,
            outside_value=outside_value,
            as_categorical=as_categorical,
        )

    def write(
            self,
            dataframe: pd.DataFrame,
            *,
            merge: bool = False,
            index_columns: Optional[list[str]] = None,
            batch_size: int = 1_000_000,
            allow_overwrite: bool = True,
            show_progress: bool = False,
            compression: typing.Any = "fast",
            **pq_write_kwargs: object,
    ) -> "ParquetBlockModel":
        """Persist data to the backing ``.pbm`` file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Data to persist.
        merge : bool, default False
            If False, rewrite the file using only ``dataframe`` columns.
            If True, append columns from ``dataframe`` to existing parquet rows
            using key alignment (streamed via parq-tools).
        index_columns : list[str], optional
            Key columns used for merge alignment. Defaults to ``["block_id"]`` when
            available, then ``["i", "j", "k"]``, then ``["x", "y", "z"]``.
        batch_size : int, default 1_000_000
            Streaming batch size used when ``merge=True``.
        allow_overwrite : bool, default True
            Forwarded to merge helper.
        show_progress : bool, default False
            Forwarded to merge helper.
        compression : str or int, default "fast"
            Active write compression policy. "fast"/0 maps to snappy; integers
            greater than zero map to zstd at that level.
        **pq_write_kwargs : object
            Extra args forwarded to parquet writer (merge mode).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        # Windows cannot atomically replace files while parquet readers are alive.
        self.pf = None  # type: ignore[assignment]
        self.data = None  # type: ignore[assignment]
        active_compression = schema_utils.resolve_active_compression_policy(compression)

        if merge:
            if index_columns is None:
                if "block_id" in self.columns:
                    index_columns = ["block_id"]
                elif all(c in self.columns for c in ["i", "j", "k"]):
                    index_columns = ["i", "j", "k"]
                elif all(c in self.columns for c in ["x", "y", "z"]):
                    index_columns = ["x", "y", "z"]
                else:
                    raise ValueError(
                        "Unable to infer index_columns for merge. "
                        "Provide index_columns explicitly."
                    )

            from parq_tools import concat_parquet_file_with_dataframe

            concat_parquet_file_with_dataframe(
                parquet_path=self.blockmodel_path,
                df=dataframe,
                output_path=None,
                index_columns=index_columns,
                batch_size=batch_size,
                allow_overwrite=allow_overwrite,
                show_progress=show_progress,
                **{
                    **schema_utils.build_parquet_compression_kwargs(dataframe.columns, active_compression),
                    **pq_write_kwargs,
                },
            )
        else:
            if len(dataframe) != int(ParquetFile(self.blockmodel_path).metadata.num_rows):
                raise ValueError(
                    f"DataFrame row count ({len(dataframe)}) does not match "
                    f"block model rows ({int(ParquetFile(self.blockmodel_path).metadata.num_rows)})."
                )
            missing_columns = [col for col in self.columns if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(
                    "DataFrame is missing required on-disk columns for strict write: "
                    f"{missing_columns}. Use merge=True to append only new columns."
                )
            self._persist_dataframe(dataframe, compression_policy=active_compression)
            return self

        schema = pq.read_schema(self.blockmodel_path)
        current_meta = dict(schema.metadata or {})
        desired_meta = self._build_schema_metadata(
            geometry=self.geometry,
            schema=self.schema,
            base_metadata=current_meta,
            compression=active_compression,
        )
        if current_meta != desired_meta:
            table = pq.read_table(self.blockmodel_path)
            meta = dict(table.schema.metadata or {})
            meta = self._build_schema_metadata(
                geometry=self.geometry,
                schema=self.schema,
                base_metadata=meta,
                compression=active_compression,
            )
            table = table.replace_schema_metadata(meta)
            pq.write_table(
                table,
                self.blockmodel_path,
                **schema_utils.build_parquet_compression_kwargs(table.column_names, active_compression),
            )

        self.pf = ParquetFile(self.blockmodel_path)
        self.data = LazyParquetDF(self.blockmodel_path)
        self.columns = pq.read_schema(self.blockmodel_path).names
        self.attributes = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._centroid_index = None
        self._extract_column_dtypes()
        self.compression = active_compression

        return self

    def compress(
        self,
        level: int = 7,
        policy: Optional[dict[str, typing.Any]] = None,
    ) -> "ParquetBlockModel":
        """Rewrite the backing file with archive compression."""
        if level < 0:
            raise ValueError("level must be >= 0")

        archive_policy = schema_utils.resolve_archive_compression_policy(
            schema=self.schema,
            level=level,
            policy=policy,
        )

        self.pf = None  # type: ignore[assignment]
        self.data = None  # type: ignore[assignment]

        with atomic_output_file(self.blockmodel_path) as tmp_path:
            parquet_file = pq.ParquetFile(self.blockmodel_path)
            schema = parquet_file.schema_arrow
            metadata = schema_utils.build_schema_metadata(
                geometry=self.geometry,
                schema=self.schema,
                base_metadata=dict(schema.metadata or {}),
                compression=archive_policy,
            )
            writer = pq.ParquetWriter(
                tmp_path,
                schema.with_metadata(metadata),
                **schema_utils.build_parquet_compression_kwargs(schema.names, archive_policy),
            )
            try:
                for batch in parquet_file.iter_batches(batch_size=1_000_000):
                    writer.write_batch(batch)
            finally:
                writer.close()
                try:
                    parquet_file.close()
                except Exception:
                    pass

        self.pf = ParquetFile(self.blockmodel_path)
        self.data = LazyParquetDF(self.blockmodel_path)
        self.columns = pq.read_schema(self.blockmodel_path).names
        self.attributes = [col for col in self.columns if col not in self.POSITION_COLUMNS]
        self._centroid_index = None
        self._extract_column_dtypes()
        self.compression = archive_policy
        return self

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
            invalid_attrs = [a for a in attributes if a not in self.available_columns]
            if invalid_attrs:
                raise ValueError(
                    f"Attributes not found: {invalid_attrs}. Available: {self.available_columns}"
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
                   attributes: Optional[list[str]] = None,
                   frame: typing.Literal["world", "local"] = "world"
                   ) -> Union['pv.ImageData', 'pv.StructuredGrid', 'pv.UnstructuredGrid']:

        if attributes is None:
            attributes = self.available_attributes

        if grid_type == "image":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
            # Read in canonical ijk order (rotation-invariant) so values map
            # to cells correctly for both local and world frames.
            df = self.read(columns=attributes, index="ijk", dense=True)
            grid = df_to_pv_image_data(
                df=df,
                geometry=self.geometry,
                categorical_encode=True,
                frame=frame,
            )
        elif grid_type == "structured":
            if frame != "world":
                raise ValueError("frame='local' is only supported for grid_type='image'.")
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_structured_grid
            grid = df_to_pv_structured_grid(
                df=self.read(columns=attributes, dense=False),
                validate_block_size=True,
            )
        elif grid_type == "unstructured":
            if frame != "world":
                raise ValueError("frame='local' is only supported for grid_type='image'.")
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_unstructured_grid
            grid = df_to_pv_unstructured_grid(
                df=self.read(columns=attributes, dense=False),
                block_size=self.geometry.local.block_size,
                validate_block_size=True,
            )
        else:
            raise ValueError(f"Invalid grid type: {grid_type}. "
                             "Choose from 'image', 'structured', or 'unstructured'.")

        return grid

    @classmethod
    def validate_xyz_parquet(cls,
                             parquet_path: Path,
                             axis_azimuth: float = 0.0,
                             axis_dip: float = 0.0,
                             axis_plunge: float = 0.0,
                             chunk_size: int = 1_000_000,
                             tol: float = 1e-6) -> RegularGeometry:
        """Validate xyz-defined Parquet input and return inferred geometry.
        
        This is a backwards-compatible wrapper that delegates to ingest_utils.validate_xyz_parquet().
        """
        from parq_blockmodel.io.ingest_utils import validate_xyz_parquet
        return validate_xyz_parquet(
            parquet_path=parquet_path,
            axis_azimuth=axis_azimuth,
            axis_dip=axis_dip,
            axis_plunge=axis_plunge,
            chunk_size=chunk_size,
            tol=tol
        )

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
