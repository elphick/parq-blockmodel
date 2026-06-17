"""Utility helpers for profiling report generation."""

from __future__ import annotations

import logging
import math
import re
from numbers import Integral, Real
from typing import Optional, Union

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


_MEMORY_FACTORS = {
    "b": 1,
    "k": 1_000,
    "kb": 1_000,
    "ki": 1_024,
    "kib": 1_024,
    "m": 1_000_000,
    "mb": 1_000_000,
    "mi": 1_048_576,
    "mib": 1_048_576,
    "g": 1_000_000_000,
    "gb": 1_000_000_000,
    "gi": 1_073_741_824,
    "gib": 1_073_741_824,
    "t": 1_000_000_000_000,
    "tb": 1_000_000_000_000,
    "ti": 1_099_511_627_776,
    "tib": 1_099_511_627_776,
}


def parse_memory_budget(memory_budget: Union[int, float, str]) -> int:
    """Convert a user-provided memory budget into bytes."""
    if isinstance(memory_budget, bool):
        raise TypeError("memory_budget must be an int, float, or string.")

    if isinstance(memory_budget, Integral):
        budget = int(memory_budget)
    elif isinstance(memory_budget, Real):
        if not math.isfinite(memory_budget):
            raise ValueError("memory_budget must be a finite positive number of bytes.")
        budget = int(memory_budget)
    elif isinstance(memory_budget, str):
        cleaned = memory_budget.strip().replace("_", "")
        match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)", cleaned)
        if match is None:
            raise ValueError(
                "memory_budget must be an integer byte count or a string such as '512MB' or '2GiB'."
            )
        value = float(match.group(1))
        unit = match.group(2).lower() or "b"
        budget = int(value * _MEMORY_FACTORS[unit])
    else:
        raise TypeError("memory_budget must be an int, float, or string.")

    if budget <= 0:
        raise ValueError("memory_budget must be greater than zero.")
    return budget


def estimate_report_column_uncompressed_sizes(parquet_file: pq.ParquetFile, columns: list[str]) -> dict[str, int]:
    """Estimate per-column size from Parquet row-group metadata."""
    metadata = parquet_file.metadata
    column_names = parquet_file.schema_arrow.names
    name_to_index = {name: idx for idx, name in enumerate(column_names)}

    missing_columns = [column for column in columns if column not in name_to_index]
    if missing_columns:
        raise ValueError(f"Unknown report columns: {missing_columns}")

    size_by_column: dict[str, int] = {}
    for column in columns:
        column_index = name_to_index[column]
        estimated_bytes = 0
        for row_group_index in range(metadata.num_row_groups):
            column_meta = metadata.row_group(row_group_index).column(column_index)
            estimated_bytes += max(
                int(column_meta.total_uncompressed_size or 0),
                int(getattr(column_meta, "total_compressed_size", 0) or 0),
                0,
            )
        size_by_column[column] = estimated_bytes or max(int(metadata.num_rows), 1)
    return size_by_column


def max_report_batch_size_within_budget(
    columns: list[str],
    estimated_sizes: dict[str, int],
    memory_budget_bytes: int,
) -> int:
    """Find the largest contiguous batch size that stays within budget."""
    best_batch_size = 1
    for candidate in range(1, len(columns) + 1):
        running_total = sum(estimated_sizes[column] for column in columns[:candidate])
        if running_total > memory_budget_bytes:
            break

        window_max = running_total
        for right_index in range(candidate, len(columns)):
            left_index = right_index - candidate
            running_total += estimated_sizes[columns[right_index]]
            running_total -= estimated_sizes[columns[left_index]]
            if running_total > window_max:
                window_max = running_total

        if window_max > memory_budget_bytes:
            break
        best_batch_size = candidate

    return best_batch_size


def resolve_report_columns_per_batch(
    parquet_file: pq.ParquetFile,
    columns: list[str],
    columns_per_batch: Optional[int],
    memory_budget: Optional[Union[int, float, str]],
) -> tuple[Optional[int], Optional[int]]:
    """Resolve the effective batch size and any parsed memory budget."""
    if columns_per_batch is not None and columns_per_batch <= 0:
        raise ValueError("columns_per_batch must be a positive integer or None.")

    if memory_budget is None:
        return columns_per_batch, None

    memory_budget_bytes = parse_memory_budget(memory_budget)
    estimated_sizes = estimate_report_column_uncompressed_sizes(parquet_file, columns)
    total_estimated_bytes = sum(estimated_sizes.values())
    
    logger.debug(f"Memory budget resolution: total_estimated_bytes={total_estimated_bytes} bytes, memory_budget_bytes={memory_budget_bytes} bytes")
    logger.debug(f"Per-column sizes: {estimated_sizes}")
    logger.debug(f"Fits in budget? {total_estimated_bytes <= memory_budget_bytes}")
    
    if total_estimated_bytes <= memory_budget_bytes:
        logger.debug(f"All columns fit within budget -> returning batch_size=None (native profiler)")
        return None, memory_budget_bytes

    batch_size = max_report_batch_size_within_budget(columns, estimated_sizes, memory_budget_bytes)
    logger.debug(f"Budget insufficient -> calculated batch_size={batch_size}")
    return batch_size, memory_budget_bytes

