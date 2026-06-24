"""Pure utility functions for schema handling and validation.

Functions in this module are stateless and can be used independently
of any ParquetBlockModel or SchemaService instance.
"""

import json
import typing
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow.parquet as pq

if typing.TYPE_CHECKING:
    from pandera import DataFrameSchema  # type: ignore[import]


SCHEMA_METADATA_KEY = b"parq-blockmodel-schema-yaml"
GEOMETRY_METADATA_KEY = b"parq-blockmodel"


def load_schema(schema: Union[Path, "DataFrameSchema"]) -> "DataFrameSchema":
    """Load and return a pandera DataFrameSchema.

    Accepts either a ready-made DataFrameSchema object or a Path pointing to a
    YAML schema file (loaded via df_eval.utils.pandera_io_compat.from_yaml to
    preserve custom metadata).

    Parameters
    ----------
    schema : DataFrameSchema or Path
        A pandera DataFrameSchema instance, or a Path to a YAML schema file.

    Returns
    -------
    DataFrameSchema
        The loaded schema.

    Raises
    ------
    ImportError
        If pandera or df-eval is not installed.
    TypeError
        If schema is not a recognised type.
    """
    try:
        from pandera import DataFrameSchema as _DataFrameSchema
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Schema support requires 'pandera'. Install it with: "
            "pip install 'parq-blockmodel[schema]'"
        ) from exc

    if isinstance(schema, _DataFrameSchema):
        return schema

    if isinstance(schema, (str, Path)):
        try:
            from df_eval.pandera import load_pandera_schema_yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading a schema from a YAML file requires 'df-eval'. Install it with: "
                "pip install 'parq-blockmodel[schema]'"
            ) from exc
        return load_pandera_schema_yaml(schema)

    raise TypeError(
        f"schema must be a pandera DataFrameSchema or a path to a YAML file, got {type(schema)!r}"
    )


def dump_schema_yaml(schema: "DataFrameSchema") -> str:
    """Serialize a pandera schema to YAML text.

    Parameters
    ----------
    schema : DataFrameSchema
        The schema to serialize.

    Returns
    -------
    str
        YAML representation of the schema.

    Raises
    ------
    ImportError
        If df-eval is not installed.
    ValueError
        If serialization fails.
    """
    try:
        from df_eval.pandera import dump_pandera_schema_yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Schema serialization requires 'df-eval'. Install it with: "
            "pip install 'parq-blockmodel[schema]'"
        ) from exc
    yaml_text = dump_pandera_schema_yaml(schema)
    if not isinstance(yaml_text, str):
        raise ValueError("Expected df-eval to return schema YAML text.")
    return yaml_text


def load_embedded_schema(parquet_path: Path) -> Optional["DataFrameSchema"]:
    """Extract and load a schema embedded in Parquet metadata.

    Parameters
    ----------
    parquet_path : Path
        Path to the Parquet file.

    Returns
    -------
    DataFrameSchema or None
        The embedded schema, or None if not present.
    """
    metadata = pq.read_metadata(parquet_path).metadata or {}
    payload = metadata.get(SCHEMA_METADATA_KEY)
    if payload is None:
        return None
    yaml_text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    return load_schema(yaml_text)


def extract_required_columns_from_schema(schema: "DataFrameSchema") -> set[str]:
    """Extract the set of column names that are required (should be persisted).

    Columns with required=False are non-persisted calculated columns and
    should be excluded from output during write operations.

    Parameters
    ----------
    schema : DataFrameSchema
        The pandera schema.

    Returns
    -------
    set[str]
        Column names with required=True, or all schema columns if required
        is not explicitly set (defaults to True).
    """
    if schema is None:
        return set()
    schema_columns = getattr(schema, "columns", {})
    if not isinstance(schema_columns, dict):
        return set()
    required = set()
    for col_name, col_spec in schema_columns.items():
        col_required = getattr(col_spec, "required", True)
        if col_required is not False:
            required.add(col_name)
    return required


def extract_column_aliases_from_schema(schema: "DataFrameSchema") -> dict[str, str]:
    """Extract alias mappings from schema df-eval metadata.

    Returns a dict mapping target_column -> source_column for all columns
    that have an alias specified in their df-eval metadata.

    Parameters
    ----------
    schema : DataFrameSchema
        The pandera schema.

    Returns
    -------
    dict[str, str]
        Mapping of {target_column: source_column} for aliased columns.
        Example: {"deposit_code": "deposit"} if deposit_code has alias "deposit".
    """
    if schema is None:
        return {}

    schema_columns = getattr(schema, "columns", {})
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


def validate_chunk(dataframe: pd.DataFrame, schema: "DataFrameSchema") -> pd.DataFrame:
    """Validate and coerce a single DataFrame chunk against a schema.

    The schema is applied with lazy=True so all column errors are collected
    before raising, and inplace=False so the original chunk is not mutated.
    Coercion (type casting) is performed when the schema has coerce=True set
    on its columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas DataFrame chunk to validate.
    schema : DataFrameSchema
        The pandera schema to validate against.

    Returns
    -------
    pd.DataFrame
        The validated (and possibly coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaErrors
        If the chunk fails validation (when pandera raises in lazy mode).
    """
    return schema.validate(dataframe, lazy=True)


def build_schema_metadata(
    geometry: "RegularGeometry",  # noqa: F821
    schema: Optional["DataFrameSchema"],
    base_metadata: Optional[dict[bytes, bytes]] = None,
) -> dict[bytes, bytes]:
    """Build combined metadata dict for Parquet schema (geometry + schema YAML).

    Parameters
    ----------
    geometry : RegularGeometry
        The geometry object to embed.
    schema : DataFrameSchema or None
        The pandera schema to embed, or None to remove embedded schema.
    base_metadata : dict[bytes, bytes], optional
        Existing metadata to preserve.

    Returns
    -------
    dict[bytes, bytes]
        Combined metadata with geometry and schema YAML keys.
    """
    metadata = dict(base_metadata or {})
    metadata[GEOMETRY_METADATA_KEY] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
    if schema is None:
        metadata.pop(SCHEMA_METADATA_KEY, None)
    else:
        metadata[SCHEMA_METADATA_KEY] = dump_schema_yaml(schema).encode("utf-8")
    return metadata


def df_eval_operations_from_schema(schema: "DataFrameSchema") -> dict[str, dict[str, typing.Any]]:
    """Extract df-eval operations from a pandera schema.

    Parameters
    ----------
    schema : DataFrameSchema
        The pandera schema.

    Returns
    -------
    dict[str, dict[str, typing.Any]]
        Dictionary of operation specs keyed by column name.

    Raises
    ------
    ImportError
        If df-eval is not installed.
    """
    try:
        from df_eval.pandera import df_eval_operations_from_pandera
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Calculated column support requires 'df-eval'. Install it with: "
            "pip install 'parq-blockmodel[schema]'"
        ) from exc
    return df_eval_operations_from_pandera(schema)


def select_df_eval_operations(
    operations: dict[str, dict[str, typing.Any]],
    targets: typing.Iterable[str],
) -> dict[str, dict[str, typing.Any]]:
    """Recursively select df-eval operations and their dependencies.

    Walks the operation dependency tree to include all required operations
    for the specified targets.

    Parameters
    ----------
    operations : dict[str, dict[str, typing.Any]]
        Available operations keyed by column name.
    targets : Iterable[str]
        Target column names to include.

    Returns
    -------
    dict[str, dict[str, typing.Any]]
        Subset of operations including targets and all dependencies.
    """
    selected: dict[str, dict[str, typing.Any]] = {}
    in_progress: set[str] = set()

    def include_operation(column: str) -> None:
        if column in selected or column in in_progress:
            return
        operation = operations.get(column)
        if operation is None:
            return

        in_progress.add(column)
        kind = operation.get("kind")

        if kind == "expr":
            expr_text = operation.get("expr")
            if isinstance(expr_text, str):
                try:
                    float(expr_text)
                except ValueError:
                    try:
                        from df_eval.expr import Expression
                    except ImportError as exc:  # pragma: no cover
                        raise ImportError(
                            "Calculated column support requires 'df-eval'. Install it with: "
                            "pip install 'parq-blockmodel[schema]'"
                        ) from exc
                    for dependency in Expression(expr_text).dependencies:
                        include_operation(dependency)
        elif kind == "lookup":
            lookup_spec = operation.get("lookup") or {}
            key_column = lookup_spec.get("key")
            if isinstance(key_column, str):
                include_operation(key_column)
        elif kind == "function":
            function_spec = operation.get("function") or {}
            inputs = function_spec.get("inputs")
            if isinstance(inputs, list):
                for dependency in inputs:
                    if isinstance(dependency, str):
                        include_operation(dependency)

        in_progress.remove(column)
        selected[column] = operation

    for target in targets:
        include_operation(target)

    return {name: operations[name] for name in operations if name in selected}
