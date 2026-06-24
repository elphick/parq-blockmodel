"""Schema service module for ParquetBlockModel.

Provides utilities and services for managing pandera schemas, df-eval operations,
and calculated column evaluation within the block model pipeline.

Public API
----------
SchemaService : Class for stateful schema/geometry-aware operations
"""

from parq_blockmodel.schema.service import SchemaService
from parq_blockmodel.schema.utils import (
    load_schema,
    dump_schema_yaml,
    load_embedded_schema,
    extract_required_columns_from_schema,
    extract_column_aliases_from_schema,
    validate_chunk,
    build_schema_metadata,
    df_eval_operations_from_schema,
    select_df_eval_operations,
)

__all__ = [
    "SchemaService",
    "load_schema",
    "dump_schema_yaml",
    "load_embedded_schema",
    "extract_required_columns_from_schema",
    "extract_column_aliases_from_schema",
    "validate_chunk",
    "build_schema_metadata",
    "df_eval_operations_from_schema",
    "select_df_eval_operations",
]
