from pathlib import Path

import pandas as pd
import pytest

from parq_blockmodel.blockmodel import ParquetBlockModel


def _make_demo_pbm(tmp_path: Path) -> ParquetBlockModel:
    src_parquet = tmp_path / "demo.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(src_parquet)
    assert pbm.blockmodel_path.exists()
    return pbm


def test_write_merge_true_preserves_unmentioned_columns_and_adds_new_one(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    before = pbm.read(index=None)
    assert "block_id" in before.columns

    # Merge mode appends NEW columns only; existing columns are left untouched.
    subset = before[["block_id"]].copy()
    subset["new_metric"] = 1.234

    pbm.write(subset, merge=True)

    after = pbm.read(index=None)

    # Existing untouched columns should still exist (no destructive narrow write)
    for col in before.columns:
        assert col in after.columns

    # New column should be appended
    assert "new_metric" in after.columns
    assert (after["new_metric"] == 1.234).all()


def test_write_merge_false_requires_full_column_set(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    df = pbm.read(index=None)
    # Intentionally narrow to provoke strict-mode failure
    narrowed = df[["block_id"]].copy()

    with pytest.raises(ValueError, match="missing|Missing|required on-disk columns"):
        pbm.write(narrowed, merge=False)


def test_write_merge_true_duplicate_column_collision_raises(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    df = pbm.read(index=None)

    # Create a duplicate column name intentionally (pandas allows duplicates)
    # Keep block_id for alignment.
    candidate_existing = None
    for col in df.columns:
        if col != "block_id":
            candidate_existing = col
            break
    if candidate_existing is None:
        pytest.skip("Unable to construct duplicate-column test; no non-key column found.")

    dup_df = df[["block_id", candidate_existing]].copy()
    dup_df = pd.concat([dup_df, dup_df[[candidate_existing]]], axis=1)
    assert dup_df.columns.duplicated().any()

    with pytest.raises((ValueError, KeyError), match="already present|already exists|duplicate|collision"):
        pbm.write(dup_df, merge=True)


def test_write_merge_true_uses_block_id_default_alignment(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    df = pbm.read(columns=["block_id"], index=None).copy()
    df["aligned_flag"] = 7

    # No index_columns passed -> should default to ["block_id"]
    pbm.write(df, merge=True)

    out = pbm.read(columns=["block_id", "aligned_flag"], index=None)
    assert "aligned_flag" in out.columns
    assert (out["aligned_flag"] == 7).all()