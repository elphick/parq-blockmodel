from pathlib import Path

import pytest

from parq_blockmodel.blockmodel import ParquetBlockModel


def _make_demo_pbm(tmp_path: Path) -> ParquetBlockModel:
    # Creates demo.parquet and canonical demo.pbm via helper already in codebase.
    src_parquet = tmp_path / "demo.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(src_parquet)
    assert pbm.blockmodel_path.exists()
    assert pbm.blockmodel_path.suffix == ".pbm"
    return pbm


def test_rename_updates_file_and_refreshes_object_state(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    old_path = pbm.blockmodel_path
    old_name = pbm.name
    new_path = tmp_path / "renamed_model.pbm"

    result = pbm.rename(new_pbm_filepath=new_path, rename_to_new_pbm_stem=True)

    # Method contract: mutate in place and return self
    assert result is pbm

    # File system state
    assert not old_path.exists()
    assert new_path.exists()

    # Object state refreshed
    assert pbm.blockmodel_path == new_path
    assert pbm.name == "renamed_model"
    assert isinstance(pbm.columns, list)
    assert "block_id" in pbm.columns
    assert pbm._centroid_index is None

    # Ensure path-bound readers were refreshed and still functional
    df = pbm.read(columns=["block_id"], index=None)
    assert not df.empty
    assert "block_id" in df.columns
    assert pbm.name != old_name


def test_rename_can_keep_existing_name(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)
    pbm.name = "custom_name"

    new_path = tmp_path / "another_name.pbm"
    pbm.rename(new_pbm_filepath=new_path, rename_to_new_pbm_stem=False)

    assert pbm.blockmodel_path == new_path
    assert pbm.name == "custom_name"
    assert new_path.exists()


def test_rename_rejects_non_pbm_extension(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    with pytest.raises(ValueError, match=r"\.pbm"):
        pbm.rename(new_pbm_filepath=tmp_path / "bad_name.parquet")


def test_rename_rejects_existing_target(tmp_path: Path) -> None:
    pbm = _make_demo_pbm(tmp_path)

    existing_target = tmp_path / "already_exists.pbm"
    existing_target.write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        pbm.rename(new_pbm_filepath=existing_target)