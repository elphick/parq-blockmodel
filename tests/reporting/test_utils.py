import pandas as pd
import pyarrow.parquet as pq
import pytest

from parq_blockmodel.reporting.utils import (
    max_report_batch_size_within_budget,
    parse_memory_budget,
    resolve_report_columns_per_batch,
)


def test_parse_memory_budget_accepts_numeric_and_string_units():
    assert parse_memory_budget(512) == 512
    assert parse_memory_budget(1.5) == 1
    assert parse_memory_budget("256MB") == 256_000_000
    assert parse_memory_budget("2GiB") == 2_147_483_648


@pytest.mark.parametrize("value", [0, -1, "0B", "nonsense", float("inf")])
def test_parse_memory_budget_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        parse_memory_budget(value)  # type: ignore[arg-type]


def test_resolve_report_columns_per_batch_from_memory_budget(tmp_path, monkeypatch):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).to_parquet(path, index=False)
    parquet_file = pq.ParquetFile(path)

    monkeypatch.setattr(
        "parq_blockmodel.reporting.utils.estimate_report_column_uncompressed_sizes",
        lambda parquet_file, columns: {column: size for column, size in zip(columns, [100, 150, 100])},
    )

    columns_per_batch, memory_budget_bytes = resolve_report_columns_per_batch(
        parquet_file=parquet_file,
        columns=["a", "b", "c"],
        columns_per_batch=10,
        memory_budget="260B",
    )

    assert columns_per_batch == 2
    assert memory_budget_bytes == 260


def test_max_report_batch_size_within_budget_prefers_larger_contiguous_windows():
    assert max_report_batch_size_within_budget(
        ["a", "b", "c"],
        {"a": 100, "b": 150, "c": 100},
        260,
    ) == 2

