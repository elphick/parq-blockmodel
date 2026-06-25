"""Stateful ingestion writer for streaming Parquet data into canonical .pbm files.

This module provides the IngestWriter class, which handles streaming data from
a source Parquet file (or DataFrame) into a canonical .pbm file with automatic
block_id derivation, schema validation, and df-eval operations support.
"""
import logging
import typing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel.geometry import RegularGeometry
from parq_blockmodel.io.ingest_utils import assert_block_id_xyz_consistent, build_world_id_encoding_from_xyz
from parq_blockmodel.utils.spatial_encoding import (
    decode_world_coordinates,
    encode_world_coordinates,
    get_world_id_encoding_params,
)
from parq_tools.utils import atomic_output_file

if typing.TYPE_CHECKING:
    from pandera import DataFrameSchema

logger = logging.getLogger(__name__)


class IngestWriter:
    """Stateful writer for streaming Parquet data into canonical .pbm files.

    This class manages the ingestion of data from a source Parquet file into
    a canonical .pbm (Parquet block model) file with automatic block_id
    derivation, spatial column coercion, and optional schema validation and
    df-eval operations.

    The writer handles:
    - Streaming batches from source Parquet files
    - Deriving block_id from positional columns (block_id, world_id, ijk, or xyz)
    - Validating consistency between block_id and xyz if both present
    - Ensuring all spatial columns are present in output
    - Applying schema validation and df-eval operations if schema provided
    - Atomic file writing via parq_tools

    Attributes
    ----------
    input_path : Path
        Path to the source Parquet file.
    output_path : Path
        Path to the output .pbm file.
    geometry : RegularGeometry
        Geometry of the block model (defines block size, shape, and orientation).
    schema : Optional[DataFrameSchema]
        Optional pandera schema for validation and df-eval operations.
    engine_initializer : Optional[Callable]
        Optional callable to configure the df-eval Engine.
        Signature: Callable[[Engine], Engine].
    """

    def __init__(
        self,
        input_path: typing.Union[str, Path],
        output_path: typing.Union[str, Path],
        geometry: RegularGeometry,
        schema: Optional["DataFrameSchema"] = None,
        engine_initializer: Optional[typing.Callable] = None,
    ) -> None:
        """Initialize the IngestWriter with input/output paths and geometry.

        Parameters
        ----------
        input_path : str or Path
            Path to the source Parquet file. Use None for DataFrame-based writes.
        output_path : str or Path
            Path to the output .pbm file.
        geometry : RegularGeometry
            Geometry of the block model.
        schema : Optional[DataFrameSchema], default None
            Optional pandera schema for validation and df-eval operations.
        engine_initializer : Optional[Callable], default None
            Optional callable to configure the df-eval Engine.

        Raises
        ------
        ValueError
            If input_path is provided but does not exist or is not a valid Parquet file.
        """
        self.input_path = Path(input_path) if input_path is not None else None
        self.output_path = Path(output_path)
        self.geometry = geometry
        self.schema = schema
        self.engine_initializer = engine_initializer

        if self.input_path is not None and not self.input_path.exists():
            raise ValueError(f"Input file does not exist: {self.input_path}")

    def write(
        self,
        columns: Optional[list[str]] = None,
        chunk_size: int = 1_000_000,
        tol: float = 1e-6,
    ) -> None:
        """Stream data from source Parquet file into canonical .pbm file.

        Orchestrates the streaming write process:
        1. Determines source columns and output columns
        2. Builds world_id_encoding if needed (using geometry extents)
        3. Iterates through batches of the source Parquet file
        4. For each batch:
           - Derives block_id from positional columns
           - Validates consistency between block_id and xyz if both present
           - Ensures all spatial columns are present
           - Applies schema validation and df-eval operations if schema provided
           - Writes to output Parquet

        Parameters
        ----------
        columns : Optional[list[str]], default None
            Optional subset of columns to persist. If None, all columns in the
            source file plus any missing special columns are persisted.
        chunk_size : int, default 1_000_000
            Batch size for reading Parquet data.
        tol : float, default 1e-6
            Tolerance for xyz to ijk conversion when deriving block_id.

        Raises
        ------
        ValueError
            If source file lacks required positional columns, if block_id
            values are inconsistent with xyz, if block_id values are outside
            geometry bounds, if canonical constraint (unique block_ids) is
            violated, or if requested columns are not present in source file.
        """
        # Lazy import to avoid circular dependencies
        from parq_blockmodel.blockmodel import ParquetBlockModel

        pf = pq.ParquetFile(self.input_path)
        src_cols = pf.schema.names

        # Determine output columns
        if columns is None:
            output_cols = list(src_cols)
            for special_col in ParquetBlockModel.SPECIAL_COLUMN_ORDER:
                if special_col not in output_cols:
                    output_cols.append(special_col)
        else:
            missing = [c for c in columns if c not in src_cols]
            if missing:
                raise ValueError(f"Requested columns not present in source parquet: {missing}")
            output_cols = list(columns)

        # Check available positional representations
        has_block_id = "block_id" in src_cols
        has_world_id = "world_id" in src_cols
        has_xyz = all(c in src_cols for c in ["x", "y", "z"])
        has_ijk = all(c in src_cols for c in ["i", "j", "k"])

        if not any([has_block_id, has_world_id, has_ijk, has_xyz]):
            raise ValueError(
                "Cannot derive block_id from source parquet: requires block_id, world_id, ijk, or xyz columns."
            )

        # Build world_id_encoding if needed
        if self.geometry.world_id_encoding is None:
            xmin = self.geometry.extents.xmin
            xmax = self.geometry.extents.xmax
            ymin = self.geometry.extents.ymin
            ymax = self.geometry.extents.ymax
            zmin = self.geometry.extents.zmin
            zmax = self.geometry.extents.zmax
            self.geometry.world_id_encoding = build_world_id_encoding_from_xyz(
                np.array([xmin, xmax], dtype=float),
                np.array([ymin, ymax], dtype=float),
                np.array([zmin, zmax], dtype=float),
            )
        offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)

        # Order columns and prepare read columns
        output_cols = ParquetBlockModel._ordered_columns(output_cols)
        read_cols = list(dict.fromkeys(output_cols + [c for c in ParquetBlockModel.SPECIAL_COLUMN_ORDER if c in src_cols]))
        if "block_id" not in output_cols:
            output_cols = ["block_id"] + output_cols
        if "world_id" not in output_cols:
            output_cols = ["block_id", "world_id"] + [c for c in output_cols if c != "block_id"]
        output_cols = ParquetBlockModel._ordered_columns(output_cols)

        # Handle schema-based column selection
        selected_persist_operations: dict[str, dict[str, typing.Any]] = {}
        if self.schema is not None:
            required_cols = ParquetBlockModel._extract_required_columns_from_schema(self.schema)
            schema_ops = dict(ParquetBlockModel._df_eval_operations_from_schema(self.schema))
            schema_columns = getattr(self.schema, "columns", {})
            schema_column_names = list(schema_columns.keys()) if isinstance(schema_columns, dict) else []
            persist_targets = [name for name in required_cols if name in schema_ops]
            if persist_targets:
                all_operations = dict(schema_ops)
                all_operations.setdefault(
                    ParquetBlockModel.INTRINSIC_VOLUME_COLUMN,
                    {"kind": "expr", "expr": repr(self.geometry.block_volume)},
                )
                selected_persist_operations = ParquetBlockModel._select_df_eval_operations(
                    all_operations,
                    persist_targets,
                )

            if required_cols:
                aliases = ParquetBlockModel._extract_column_aliases_from_schema(self.schema)
                inverse_aliases = {v: k for k, v in aliases.items()}

                output_cols = [
                    c
                    for c in output_cols
                    if (c in ParquetBlockModel.POSITION_COLUMNS
                        or c in required_cols
                        or (c not in schema_ops and c not in inverse_aliases))
                ]
                for col_name in required_cols:
                    if col_name not in output_cols and (col_name in schema_ops or col_name in aliases):
                        output_cols.append(col_name)

                special_cols = [c for c in ParquetBlockModel.SPECIAL_COLUMN_ORDER if c in output_cols]
                schema_cols = [
                    c for c in schema_column_names
                    if c in output_cols and c not in ParquetBlockModel.SPECIAL_COLUMN_ORDER
                ]
                remaining_cols = [
                    c for c in output_cols
                    if c not in special_cols and c not in schema_cols
                ]
                output_cols = special_cols + schema_cols + remaining_cols

            read_cols = list(dict.fromkeys(src_cols + [c for c in ParquetBlockModel.SPECIAL_COLUMN_ORDER if c in src_cols]))

        # Stream data through batches
        with atomic_output_file(self.output_path) as tmp_path:
            writer = None
            seen_block_ids: set[int] = set()
            try:
                for batch in pf.iter_batches(columns=read_cols, batch_size=chunk_size):
                    df_batch = pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)

                    # Validate consistency if both block_id and xyz present
                    if "block_id" in df_batch.columns and {"x", "y", "z"}.issubset(df_batch.columns):
                        assert_block_id_xyz_consistent(
                            block_ids=df_batch["block_id"].to_numpy(dtype=np.uint32),
                            x=df_batch["x"].to_numpy(dtype=float),
                            y=df_batch["y"].to_numpy(dtype=float),
                            z=df_batch["z"].to_numpy(dtype=float),
                            geometry=self.geometry,
                            tol=tol,
                            context=f"canonical write for {self.input_path}",
                        )

                    # Derive block_id from available positional columns
                    if "block_id" in df_batch.columns:
                        block_ids = df_batch["block_id"].to_numpy(dtype=np.uint32)
                    elif "world_id" in df_batch.columns:
                        world_ids = df_batch["world_id"].to_numpy(dtype=np.int64)
                        xw, yw, zw = decode_world_coordinates(
                            world_ids, offset=offset, scale=scale, bits_per_axis=bits_per_axis
                        )
                        block_ids = self.geometry.row_index_from_xyz(xw, yw, zw, tol=tol).astype(np.uint32)
                    elif {"x", "y", "z"}.issubset(df_batch.columns):
                        block_ids = self.geometry.row_index_from_xyz(
                            df_batch["x"].to_numpy(),
                            df_batch["y"].to_numpy(),
                            df_batch["z"].to_numpy(),
                            tol=tol,
                        ).astype(np.uint32)
                    else:
                        block_ids = self.geometry.row_index_from_ijk(
                            df_batch["i"].to_numpy(),
                            df_batch["j"].to_numpy(),
                            df_batch["k"].to_numpy(),
                        ).astype(np.uint32)

                    # Validate block_ids are within bounds
                    if np.any(block_ids < 0) or np.any(block_ids >= int(np.prod(self.geometry.local.shape))):
                        raise ValueError("Source data contains positions outside geometry bounds.")

                    # Validate uniqueness within batch
                    if np.unique(block_ids).size != block_ids.size:
                        raise ValueError("Canonical .pbm requires unique block_id values.")

                    # Validate global uniqueness across all batches
                    seen_before = len(seen_block_ids)
                    seen_block_ids.update(block_ids.tolist())
                    if len(seen_block_ids) - seen_before != block_ids.size:
                        raise ValueError("Canonical .pbm requires unique block_id values.")

                    # Ensure all spatial columns are present
                    df_batch["block_id"] = block_ids
                    if not {"x", "y", "z"}.issubset(df_batch.columns):
                        x, y, z = self.geometry.xyz_from_row_index(block_ids)
                        df_batch["x"] = x
                        df_batch["y"] = y
                        df_batch["z"] = z
                    if not {"i", "j", "k"}.issubset(df_batch.columns):
                        i, j, k = self.geometry.ijk_from_row_index(block_ids)
                        df_batch["i"] = i.astype(np.int32)
                        df_batch["j"] = j.astype(np.int32)
                        df_batch["k"] = k.astype(np.int32)

                    # Coerce spatial column dtypes
                    df_batch = ParquetBlockModel._coerce_special_column_dtypes(
                        df_batch,
                        columns=["block_id", "i", "j", "k", "x", "y", "z"],
                    )

                    # Ensure world_id is present
                    if "world_id" not in df_batch.columns:
                        x = df_batch["x"].to_numpy(dtype=float)
                        y = df_batch["y"].to_numpy(dtype=float)
                        z = df_batch["z"].to_numpy(dtype=float)
                        df_batch["world_id"] = encode_world_coordinates(
                            x, y, z, offset=offset, scale=scale, bits_per_axis=bits_per_axis
                        ).astype(np.int64)

                    df_batch = ParquetBlockModel._coerce_special_column_dtypes(df_batch)

                    # Apply schema validation and df-eval operations if schema provided
                    if self.schema is not None:
                        df_batch = ParquetBlockModel._apply_df_eval_operations(
                            df_batch,
                            self.schema,
                            operations=selected_persist_operations,
                            engine_initializer=self.engine_initializer,
                        )
                        df_batch = ParquetBlockModel._validate_chunk(df_batch, self.schema)

                    # Select output columns, guarding against any cols absent from df_batch
                    ordered = ParquetBlockModel._ordered_columns(output_cols)
                    ordered = [c for c in ordered if c in df_batch.columns]
                    write_df = df_batch[ordered]

                    table = pa.Table.from_pandas(write_df, preserve_index=False)
                    meta = ParquetBlockModel._build_schema_metadata(
                        geometry=self.geometry,
                        schema=self.schema,
                        base_metadata=dict(table.schema.metadata or {}),
                    )
                    table = table.replace_schema_metadata(meta)

                    # Write batch (first batch initializes writer)
                    if writer is None:
                        # todo: extract compression to a config file.
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd",  compression_level=3)
                    else:
                        if table.schema != writer.schema:
                            table = table.cast(writer.schema)
                    writer.write_table(table)

            finally:
                if writer is not None:
                    writer.close()
                try:
                    pf.close()
                except Exception:
                    pass

        logger.info(
            f"Successfully wrote canonical .pbm file to {self.output_path} "
            f"from source {self.input_path}"
        )

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[list[str]] = None,
        tol: float = 1e-6,
    ) -> None:
        """Write a DataFrame directly into canonical .pbm file.

        Unlike write() which streams from a Parquet file, this method
        processes a pandas DataFrame that's already in memory. The entire
        DataFrame is processed in one pass (not chunked).

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with positional columns (xyz, ijk, block_id, or world_id).
        columns : Optional[list[str]], default None
            Optional subset of columns to persist. If None, all columns are persisted.
        tol : float, default 1e-6
            Tolerance for xyz to ijk conversion when deriving block_id.

        Raises
        ------
        ValueError
            If DataFrame lacks required positional columns or other validation fails.
        """
        from parq_blockmodel.blockmodel import ParquetBlockModel

        df_work = dataframe.copy()
        src_cols = list(df_work.columns)

        # Determine output columns
        if columns is None:
            output_cols = list(src_cols)
            for special_col in ParquetBlockModel.SPECIAL_COLUMN_ORDER:
                if special_col not in output_cols:
                    output_cols.append(special_col)
        else:
            missing = [c for c in columns if c not in src_cols]
            if missing:
                raise ValueError(f"Requested columns not present in source dataframe: {missing}")
            output_cols = list(columns)

        # Check available positional representations
        has_block_id = "block_id" in src_cols
        has_world_id = "world_id" in src_cols
        has_xyz = all(c in src_cols for c in ["x", "y", "z"])
        has_ijk = all(c in src_cols for c in ["i", "j", "k"])

        if not any([has_block_id, has_world_id, has_ijk, has_xyz]):
            raise ValueError(
                "Cannot derive block_id from source dataframe: requires block_id, world_id, ijk, or xyz columns."
            )

        # Build world_id_encoding if needed
        if self.geometry.world_id_encoding is None:
            xmin = self.geometry.extents.xmin
            xmax = self.geometry.extents.xmax
            ymin = self.geometry.extents.ymin
            ymax = self.geometry.extents.ymax
            zmin = self.geometry.extents.zmin
            zmax = self.geometry.extents.zmax
            self.geometry.world_id_encoding = build_world_id_encoding_from_xyz(
                np.array([xmin, xmax], dtype=float),
                np.array([ymin, ymax], dtype=float),
                np.array([zmin, zmax], dtype=float),
            )
        offset, scale, bits_per_axis = get_world_id_encoding_params(self.geometry.world_id_encoding)

        # Derive block_id from available positional columns
        if "block_id" in df_work.columns:
            block_ids = df_work["block_id"].to_numpy(dtype=np.uint32)
        elif "world_id" in df_work.columns:
            world_ids = df_work["world_id"].to_numpy(dtype=np.int64)
            xw, yw, zw = decode_world_coordinates(
                world_ids, offset=offset, scale=scale, bits_per_axis=bits_per_axis
            )
            block_ids = self.geometry.row_index_from_xyz(xw, yw, zw, tol=tol).astype(np.uint32)
        elif {"x", "y", "z"}.issubset(df_work.columns):
            block_ids = self.geometry.row_index_from_xyz(
                df_work["x"].to_numpy(),
                df_work["y"].to_numpy(),
                df_work["z"].to_numpy(),
                tol=tol,
            ).astype(np.uint32)
        else:
            block_ids = self.geometry.row_index_from_ijk(
                df_work["i"].to_numpy(),
                df_work["j"].to_numpy(),
                df_work["k"].to_numpy(),
            ).astype(np.uint32)

        # Validate block_ids
        dense_count = int(np.prod(self.geometry.local.shape))
        if np.any(block_ids < 0) or np.any(block_ids >= dense_count):
            raise ValueError("Source data contains positions outside geometry bounds.")
        if np.unique(block_ids).size != block_ids.size:
            raise ValueError("Canonical .pbm requires unique block_id values.")

        # Populate all positional columns
        df_work["block_id"] = block_ids
        if not {"x", "y", "z"}.issubset(df_work.columns):
            x, y, z = self.geometry.xyz_from_row_index(block_ids)
            df_work["x"] = x
            df_work["y"] = y
            df_work["z"] = z
        if not {"i", "j", "k"}.issubset(df_work.columns):
            i, j, k = self.geometry.ijk_from_row_index(block_ids)
            df_work["i"] = i.astype(np.int32)
            df_work["j"] = j.astype(np.int32)
            df_work["k"] = k.astype(np.int32)

        df_work = ParquetBlockModel._coerce_special_column_dtypes(
            df_work,
            columns=["block_id", "i", "j", "k", "x", "y", "z"],
        )

        # Ensure world_id is present
        if "world_id" not in df_work.columns:
            x = df_work["x"].to_numpy(dtype=float)
            y = df_work["y"].to_numpy(dtype=float)
            z = df_work["z"].to_numpy(dtype=float)
            df_work["world_id"] = encode_world_coordinates(
                x, y, z, offset=offset, scale=scale, bits_per_axis=bits_per_axis
            ).astype(np.int64)

        df_work = ParquetBlockModel._coerce_special_column_dtypes(df_work)

        # Apply schema validation and df-eval operations if schema provided
        if self.schema is not None:
            required_cols = ParquetBlockModel._extract_required_columns_from_schema(self.schema)
            schema_ops = dict(ParquetBlockModel._df_eval_operations_from_schema(self.schema))
            persist_targets = [name for name in required_cols if name in schema_ops]

            if persist_targets:
                all_operations = dict(schema_ops)
                if ParquetBlockModel.INTRINSIC_VOLUME_COLUMN not in df_work.columns:
                    all_operations.setdefault(
                        ParquetBlockModel.INTRINSIC_VOLUME_COLUMN,
                        {"kind": "expr", "expr": repr(self.geometry.block_volume)},
                    )
                selected_operations = ParquetBlockModel._select_df_eval_operations(
                    all_operations,
                    persist_targets,
                )

                if selected_operations:
                    df_work = ParquetBlockModel._apply_df_eval_operations(
                        df_work,
                        self.schema,
                        operations=selected_operations,
                        engine_initializer=self.engine_initializer,
                    )

            # Validate/coerce after required calculated columns are materialized
            df_work = ParquetBlockModel._validate_chunk(df_work, self.schema)
            if required_cols:
                # required=False is used for non-persisted calculated columns.
                # Keep regular source columns even if schema marks them optional.
                df_work = df_work[
                    [
                        c
                        for c in df_work.columns
                        if c in ParquetBlockModel.POSITION_COLUMNS
                        or c in required_cols
                        or c not in schema_ops
                    ]
                ]
            
            # Update output_cols to include any calculated columns that were materialized
            if columns is None:
                # If no specific columns were requested, include all columns from df_work
                output_cols = list(df_work.columns)

        # Select output columns, guarding against any cols absent from df_work
        ordered = ParquetBlockModel._ordered_columns(output_cols)
        ordered = [c for c in ordered if c in df_work.columns]
        write_df = df_work[ordered]

        # Write to Parquet with metadata
        table = pa.Table.from_pandas(write_df, preserve_index=False)
        meta = ParquetBlockModel._build_schema_metadata(
            geometry=self.geometry,
            schema=self.schema,
            base_metadata=dict(table.schema.metadata or {}),
        )
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, self.output_path)

        logger.info(f"Successfully wrote canonical .pbm file to {self.output_path} from DataFrame.")
