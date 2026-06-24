"""Stateful schema service for ParquetBlockModel operations.

The SchemaService class provides geometry and schema-aware operations that
require maintaining context across multiple method calls.
"""

import typing
from typing import Optional

import pandas as pd

from parq_blockmodel.schema.utils import (
    df_eval_operations_from_schema,
    select_df_eval_operations,
    validate_chunk as validate_chunk_util,
)

if typing.TYPE_CHECKING:
    from parq_blockmodel.geometry import RegularGeometry  # type: ignore[import]
    from pandera import DataFrameSchema  # type: ignore[import]


class SchemaService:
    """Manages schema, geometry, and df-eval operations for a block model.

    This service provides stateful operations that depend on both a schema
    and geometry context. It handles:
    - Intrinsic operations (e.g., volume calculation)
    - Available df-eval operations (schema + intrinsic)
    - Calculated column classification
    - Application of df-eval operations and validation

    Parameters
    ----------
    geometry : RegularGeometry
        The block model geometry.
    schema : DataFrameSchema or None
        Optional pandera schema.
    engine_initializer : Callable, optional
        Optional callable to customize the df-eval Engine.
        Signature: Callable[[Engine], Engine].
    intrinsic_volume_column : str, default "volume"
        Name of the intrinsic volume column.
    """

    INTRINSIC_VOLUME_COLUMN = "volume"
    POSITION_COLUMNS = {"block_id", "world_id", "i", "j", "k", "x", "y", "z"}

    def __init__(
        self,
        geometry: "RegularGeometry",
        schema: Optional["DataFrameSchema"] = None,
        engine_initializer: Optional[typing.Callable] = None,
        intrinsic_volume_column: str = "volume",
    ):
        """Initialize the schema service.

        Parameters
        ----------
        geometry : RegularGeometry
            The block model geometry.
        schema : DataFrameSchema, optional
            Pandera schema for validation and calculated columns.
        engine_initializer : Callable, optional
            Optional engine configuration function.
        intrinsic_volume_column : str, default "volume"
            Name for the intrinsic volume column.
        """
        self.geometry = geometry
        self.schema = schema
        self.engine_initializer = engine_initializer
        self.intrinsic_volume_column = intrinsic_volume_column

    def intrinsic_df_eval_operations(self) -> dict[str, dict[str, typing.Any]]:
        """Get intrinsic df-eval operations (e.g., volume calculation).

        Returns
        -------
        dict[str, dict[str, typing.Any]]
            Operations not from schema but built into the model (e.g., volume).
        """
        if self.intrinsic_volume_column in ["x", "y", "z"]:
            # Don't compute volume if using reserved column names
            return {}

        block_volume = self.geometry.block_volume
        return {
            self.intrinsic_volume_column: {
                "kind": "expr",
                "expr": repr(block_volume),
            }
        }

    def available_df_eval_operations(self) -> dict[str, dict[str, typing.Any]]:
        """Get all available df-eval operations (schema + intrinsic).

        Returns
        -------
        dict[str, dict[str, typing.Any]]
            Combined operations from schema and intrinsic sources.
        """
        schema_operations: dict[str, dict[str, typing.Any]] = {}
        if self.schema is not None:
            schema_operations = df_eval_operations_from_schema(self.schema)
            # Include alias placeholders
            aliases = self._extract_column_aliases_from_schema()
            for target_col, source_col in aliases.items():
                if target_col not in schema_operations:
                    schema_operations[target_col] = {
                        "kind": "alias",
                        "alias": source_col,
                    }

        operations = dict(schema_operations)
        for name, operation in self.intrinsic_df_eval_operations().items():
            operations.setdefault(name, operation)
        return operations

    def schema_column_attribute_flags(self) -> dict[str, bool]:
        """Get attribute classification flags from schema metadata.

        Returns
        -------
        dict[str, bool]
            Mapping of column names to is_attribute boolean.
        """
        if self.schema is None:
            return {}
        flags: dict[str, bool] = {}
        schema_columns = getattr(self.schema, "columns", {})
        if not isinstance(schema_columns, dict):
            return flags
        for name, column in schema_columns.items():
            metadata = getattr(column, "metadata", None) or {}
            if not isinstance(metadata, dict):
                continue
            pbm_metadata = metadata.get("parq-blockmodel")
            if not isinstance(pbm_metadata, dict):
                continue
            is_attribute = pbm_metadata.get("is_attribute")
            if isinstance(is_attribute, bool):
                flags[name] = is_attribute
        return flags

    def get_schema_column_names(self) -> list[str]:
        """Extract schema column names in definition order.

        Returns
        -------
        list[str]
            Column names from schema, or empty list if no schema.
        """
        if self.schema is None:
            return []
        schema_columns = getattr(self.schema, "columns", {})
        if not isinstance(schema_columns, dict):
            return []
        return list(schema_columns.keys())

    def calculated_attribute_flags(self, calculated_columns: list[str]) -> dict[str, bool]:
        """Classify calculated columns as attributes or positional.

        Parameters
        ----------
        calculated_columns : list[str]
            List of available calculated column names.

        Returns
        -------
        dict[str, bool]
            Mapping of column names to is_attribute boolean.
        """
        flags = {name: True for name in calculated_columns}
        for name, is_attribute in self.schema_column_attribute_flags().items():
            if name in flags:
                flags[name] = is_attribute
        intrinsic = self.intrinsic_df_eval_operations()
        for name in intrinsic:
            flags[name] = False
        return flags

    def report_schema_metadata(
        self,
        selected_columns: list[str],
    ) -> tuple[Optional[dict[str, str]], Optional[dict[str, str]]]:
        """Extract metadata descriptions for reporting.

        Parameters
        ----------
        selected_columns : list[str]
            Column names to extract descriptions for.

        Returns
        -------
        tuple[dict[str, str], dict[str, str]]
            (column_descriptions, dataset_metadata) tuples or (None, None).
        """
        if self.schema is None:
            return None, None

        schema_columns = getattr(self.schema, "columns", {})
        column_descriptions: dict[str, str] = {}
        if isinstance(schema_columns, dict):
            for column_name in selected_columns:
                schema_column = schema_columns.get(column_name)
                if schema_column is None:
                    continue
                title = self._extract_non_empty_schema_field(
                    schema_column,
                    "title",
                    include_metadata=False,
                )
                description = self._extract_non_empty_schema_field(
                    schema_column,
                    "description",
                    include_metadata=False,
                )
                pieces = [piece for piece in (title, description) if piece]
                if pieces:
                    column_descriptions[column_name] = " - ".join(pieces)

        dataset_payload: dict[str, str] = {}
        for key in ("name", "title", "description"):
            value = self._extract_non_empty_schema_field(self.schema, key, include_metadata=False)
            if value is not None:
                dataset_payload[key] = value
        dataset_metadata = {"description": str(dataset_payload)} if dataset_payload else None

        return (column_descriptions or None), dataset_metadata

    def apply_intrinsic_operations(
        self,
        dataframe: pd.DataFrame,
        operations: dict[str, dict[str, typing.Any]],
    ) -> pd.DataFrame:
        """Apply intrinsic operations to a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame.
        operations : dict[str, dict[str, typing.Any]]
            Operations to apply.

        Returns
        -------
        pd.DataFrame
            DataFrame with operations applied.
        """
        import numpy as np

        for column_name in operations:
            if column_name == self.intrinsic_volume_column:
                dataframe[column_name] = np.float64(self.geometry.block_volume)
            else:
                raise ValueError(
                    f"Column '{column_name}' requires schema-backed calculated expressions."
                )
        return dataframe

    def apply_df_eval_operations(
        self,
        dataframe: pd.DataFrame,
        operations: Optional[dict[str, dict[str, typing.Any]]] = None,
    ) -> pd.DataFrame:
        """Apply df-eval operations including alias, decimals, and expressions.

        Pipeline order:
        1. Apply alias transforms (rename columns)
        2. Apply decimals transforms (round output)
        3. Apply df-eval operations (expr/lookup/function)

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to transform.
        operations : dict[str, dict[str, typing.Any]], optional
            Pre-extracted operations. If None, extracted from schema.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with all operations applied.

        Raises
        ------
        ImportError
            If df-eval or pandera is not installed.
        """
        try:
            from df_eval.pandera import apply_aliases, apply_decimals
            from df_eval import Engine
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Calculated column support requires 'df-eval'. Install it with: "
                "pip install 'parq-blockmodel[schema]'"
            ) from exc

        if self.schema is None:
            return dataframe

        # Step 1: Apply alias transforms (rename columns)
        df = apply_aliases(dataframe, self.schema)

        # Step 2: Apply decimals transforms
        df = apply_decimals(df, self.schema)

        # Step 3: Apply df-eval operations (expr/lookup/function)
        ops = operations if operations is not None else df_eval_operations_from_schema(self.schema)
        if not ops:
            return df

        # Create engine, potentially customized via engine_initializer
        engine = Engine()
        if self.engine_initializer is not None:
            engine = self.engine_initializer(engine)

        # Apply operations using the engine
        return engine.apply_operations(df, ops)

    def validate_chunk(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate and coerce a DataFrame chunk against the schema.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Chunk to validate.

        Returns
        -------
        pd.DataFrame
            Validated (and possibly coerced) DataFrame.

        Raises
        ------
        ValueError
            If no schema is available.
        pandera.errors.SchemaErrors
            If validation fails.
        """
        if self.schema is None:
            raise ValueError("Schema service has no schema available for validation.")
        return validate_chunk_util(dataframe, self.schema)

    def _extract_column_aliases_from_schema(self) -> dict[str, str]:
        """Extract alias mappings from schema df-eval metadata.

        Returns
        -------
        dict[str, str]
            Mapping of {target_column: source_column} for aliased columns.
        """
        if self.schema is None:
            return {}

        schema_columns = getattr(self.schema, "columns", {})
        if not isinstance(schema_columns, dict):
            return {}

        aliases: dict[str, str] = {}
        for col_name, col_spec in schema_columns.items():
            metadata = getattr(col_spec, "metadata", None) or {}
            if not isinstance(metadata, dict):
                continue
            df_eval_meta = metadata.get("df-eval")
            if not isinstance(df_eval_meta, dict):
                continue
            alias = df_eval_meta.get("alias")
            if isinstance(alias, str) and alias:
                aliases[col_name] = alias

        return aliases

    @staticmethod
    def _extract_non_empty_schema_field(
        target: typing.Any,
        field: str,
        *,
        include_metadata: bool = True,
    ) -> Optional[str]:
        """Extract a non-empty field from a schema object.

        Parameters
        ----------
        target : Any
            Object to extract from (e.g., schema or column).
        field : str
            Field name.
        include_metadata : bool, default True
            Whether to also check the metadata dict.

        Returns
        -------
        str or None
            The field value if non-empty, else None.
        """
        value = getattr(target, field, None)
        if value is None and include_metadata:
            metadata = getattr(target, "metadata", None)
            if isinstance(metadata, dict):
                value = metadata.get(field)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
