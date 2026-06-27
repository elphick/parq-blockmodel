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


PBM_METADATA_KEY = b"parq-blockmodel"


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
    payload = metadata.get(PBM_METADATA_KEY)
    if payload is None:
        return None
    payload_text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    payload_dict = json.loads(payload_text)
    if not isinstance(payload_dict, dict):
        raise ValueError("Embedded PBM metadata must decode to a dictionary.")
    schema_yaml = payload_dict.get("schema_yaml")
    if schema_yaml is None:
        return None
    return load_schema(schema_yaml)


def load_embedded_compression(parquet_path: Path) -> Optional[dict[str, typing.Any]]:
    """Extract compression policy metadata embedded in a Parquet file."""
    metadata = pq.read_metadata(parquet_path).metadata or {}
    payload = metadata.get(PBM_METADATA_KEY)
    if payload is None:
        return None
    json_text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
    compression_payload = json.loads(json_text)
    if not isinstance(compression_payload, dict):
        raise ValueError("Embedded PBM metadata must decode to a dictionary.")
    compression = compression_payload.get("compression")
    if compression is None:
        return None
    if not isinstance(compression, dict):
        raise ValueError("Embedded compression metadata must decode to a dictionary.")
    return compression


def _normalize_compression_spec(
    value: typing.Any,
    *,
    default_codec: str,
    default_level: Optional[int],
) -> dict[str, typing.Any]:
    if isinstance(value, dict):
        codec = str(value.get("codec", default_codec)).strip().lower()
        level = value.get("level", default_level)
        if level is not None:
            level = int(level)
        return {"codec": codec, "level": level}

    if value is None:
        return {"codec": default_codec, "level": default_level}

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "fast":
            return {"codec": "snappy", "level": None}
        if normalized == "snappy":
            return {"codec": "snappy", "level": None}
        if normalized == "zstd":
            return {"codec": "zstd", "level": default_level}
        raise ValueError(f"Unsupported compression value: {value!r}")

    if isinstance(value, int):
        if value < 0:
            raise ValueError("compression level must be >= 0")
        if value == 0:
            return {"codec": "snappy", "level": None}
        return {"codec": "zstd", "level": int(value)}

    raise TypeError(f"Unsupported compression specification type: {type(value)!r}")


def resolve_active_compression_policy(compression: typing.Any = "fast") -> dict[str, typing.Any]:
    """Resolve a simple user-facing compression choice for active writes."""
    return {
        "mode": "active",
        "default": _normalize_compression_spec(compression, default_codec="snappy", default_level=None),
        "columns": {},
    }


def extract_schema_compression_policy(schema: "DataFrameSchema") -> Optional[dict[str, typing.Any]]:
    """Read compression policy from schema metadata if present."""
    if schema is None:
        return None
    schema_metadata = getattr(schema, "metadata", None)
    if not isinstance(schema_metadata, dict):
        return None
    pbm_metadata = schema_metadata.get("parq-blockmodel")
    if not isinstance(pbm_metadata, dict):
        return None
    compression = pbm_metadata.get("compression")
    if not isinstance(compression, dict):
        return None
    return compression


def resolve_archive_compression_policy(
    *,
    schema: Optional["DataFrameSchema"] = None,
    level: int = 7,
    policy: Optional[dict[str, typing.Any]] = None,
) -> dict[str, typing.Any]:
    """Resolve archive compression using schema metadata or explicit policy."""
    candidate = policy
    if candidate is None:
        candidate = extract_schema_compression_policy(schema)

    if isinstance(candidate, dict) and any(key in candidate for key in ("default", "archive", "columns")):
        default_spec = candidate.get("archive", candidate.get("default", {"codec": "zstd", "level": level}))
        default = _normalize_compression_spec(default_spec, default_codec="zstd", default_level=level)
        columns = candidate.get("columns") or {}
        if not isinstance(columns, dict):
            raise TypeError("compression columns overrides must be a dictionary")
        normalized_columns = {
            str(column_name): _normalize_compression_spec(spec, default_codec=default["codec"], default_level=default["level"])
            for column_name, spec in columns.items()
        }
        return {"mode": "archive", "default": default, "columns": normalized_columns}

    if isinstance(candidate, dict):
        default = _normalize_compression_spec(candidate, default_codec="zstd", default_level=level)
        return {"mode": "archive", "default": default, "columns": {}}

    return {
        "mode": "archive",
        "default": _normalize_compression_spec(level, default_codec="zstd", default_level=level),
        "columns": {},
    }


def build_parquet_compression_kwargs(
    columns: typing.Iterable[str],
    policy: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Build Parquet writer kwargs from a resolved compression policy."""
    ordered_columns = list(columns)
    default_spec = policy.get("default", {"codec": "snappy", "level": None})
    column_specs = policy.get("columns", {}) or {}

    specs = [
        column_specs.get(column_name, default_spec)
        for column_name in ordered_columns
    ]
    if not specs:
        kwargs: dict[str, typing.Any] = {"compression": default_spec["codec"]}
        if default_spec.get("level") is not None:
            kwargs["compression_level"] = default_spec["level"]
        return kwargs

    first_spec = specs[0]
    if all(spec == first_spec for spec in specs):
        kwargs: dict[str, typing.Any] = {"compression": first_spec["codec"]}
        if first_spec.get("level") is not None:
            kwargs["compression_level"] = first_spec["level"]
        return kwargs

    compression_map = {column_name: spec["codec"] for column_name, spec in zip(ordered_columns, specs)}
    level_map = {
        column_name: spec["level"]
        for column_name, spec in zip(ordered_columns, specs)
        if spec.get("level") is not None
    }
    kwargs = {"compression": compression_map}
    if level_map:
        kwargs["compression_level"] = level_map
    return kwargs


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
    compression: Optional[dict[str, typing.Any]] = None,
) -> dict[bytes, bytes]:
    """Build combined metadata dict for Parquet schema (geometry + schema YAML + compression).

    Parameters
    ----------
    geometry : RegularGeometry
        The geometry object to embed.
    schema : DataFrameSchema or None
        The pandera schema to embed, or None to remove embedded schema.
    base_metadata : dict[bytes, bytes], optional
        Existing metadata to preserve.
    compression : dict[str, Any], optional
        Compression policy metadata to persist alongside geometry and schema.

    Returns
    -------
    dict[bytes, bytes]
        Combined metadata with geometry and schema YAML keys.
    """
    metadata = dict(base_metadata or {})
    payload: dict[str, typing.Any] = {}
    existing = metadata.get(PBM_METADATA_KEY)
    if existing is not None:
        existing_text = existing.decode("utf-8") if isinstance(existing, (bytes, bytearray)) else str(existing)
        existing_payload = json.loads(existing_text)
        if not isinstance(existing_payload, dict):
            raise ValueError("Embedded PBM metadata must decode to a dictionary.")
        payload.update(existing_payload)

    payload["geometry"] = geometry.to_metadata_dict()
    if schema is None:
        payload.pop("schema_yaml", None)
    else:
        payload["schema_yaml"] = dump_schema_yaml(schema)
    if compression is None:
        payload.pop("compression", None)
    else:
        payload["compression"] = compression
    metadata[PBM_METADATA_KEY] = json.dumps(payload).encode("utf-8")
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
