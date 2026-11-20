"""PR3: ParquetBlockModel wrapper integration contract tests.

Targets uncovered branches in parq_blockmodel/blockmodel.py:
- Constructor guard (non-.pbm suffix)
- centroid_index property (xyz-column and derived-from-block_id paths)
- sparsity property
- from_dataframe (entire method)
- from_geometry (entire method)
- from_parquet overwrite=True
- validate_xyz_parquet
- triangulate with invalid/valid attributes
- to_glb missing-texture-attribute validation and auto-include behaviour
- read() with index=None and invalid index value
- _write_canonical_pbm column validation paths
- _validate_and_load_data legacy helper
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.utils import create_demo_blockmodel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pbm(tmp_path, shape=(3, 3, 3), name="model"):
    return ParquetBlockModel.create_demo_block_model(
        tmp_path / f"{name}.parquet", shape=shape
    )


def _xyz_source(shape=(2, 2, 2)):
    return create_demo_blockmodel(shape=shape).reset_index(drop=True).drop(columns="block_id")


# ===========================================================================
# Constructor guard
# ===========================================================================


def test_constructor_rejects_non_pbm_extension(tmp_path):
    fake = tmp_path / "model.parquet"
    fake.touch()
    with pytest.raises(ValueError, match=r"\.pbm"):
        ParquetBlockModel(fake)


def test_blockmodel_lazy_dataframe_does_not_lock_file(tmp_path):
    pbm = _make_pbm(tmp_path)
    data = pbm.data
    assert len(data["depth"]) == 27
    assert data.head().shape[0] == 5
    moved = pbm.blockmodel_path.rename(tmp_path / "moved.pbm")
    moved.rename(pbm.blockmodel_path)

    pbm.close()
    assert pbm.data is None

    pbm._refresh_path_bound_state()
    assert pbm.data is not None


@pytest.mark.parametrize("sparse", [False, True])
def test_constructor_reads_only_one_footer(tmp_path, monkeypatch, sparse):
    source = _make_pbm(tmp_path)
    if sparse:
        table = pq.read_table(source.blockmodel_path)
        pq.write_table(table.slice(0, 20), source.blockmodel_path)
    read_metadata = pq.read_metadata
    metadata_reads = []

    def counted_metadata(*args, **kwargs):
        metadata_reads.append(args[0])
        return read_metadata(*args, **kwargs)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Construction should only read the footer")

    monkeypatch.setattr(pq, "read_metadata", counted_metadata)
    monkeypatch.setattr(pq, "read_schema", unexpected_read)
    monkeypatch.setattr(pq, "ParquetFile", unexpected_read)
    monkeypatch.setattr(pq, "read_table", unexpected_read)
    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    monkeypatch.setattr(ParquetBlockModel, "_iter_batches", unexpected_read)
    model = ParquetBlockModel(source.blockmodel_path)
    assert metadata_reads == [source.blockmodel_path]
    assert model.columns == source.columns
    assert model.column_dtypes == source.column_dtypes
    assert model.compression == source.compression
    assert model._num_rows == (20 if sparse else 27)
    assert model.is_sparse == sparse
    assert model.sparsity == pytest.approx(7 / 27 if sparse else 0)


# ===========================================================================
# centroid_index property
# ===========================================================================


def test_centroid_index_from_xyz_columns(tmp_path):
    """centroid_index should build from x/y/z columns when they exist in the file."""
    pbm = _make_pbm(tmp_path)
    idx = pbm.centroid_index
    assert idx.names == ["x", "y", "z"]
    assert idx.is_unique
    assert len(idx) == int(np.prod(pbm.geometry.local.shape))


def test_centroid_index_derived_when_xyz_absent(tmp_path):
    """centroid_index should be derived from geometry when x/y/z columns are absent."""
    source_parquet = tmp_path / "source.parquet"
    _xyz_source().to_parquet(source_parquet, index=False)
    # Build a .pbm that retains only block_id + depth (no xyz)
    pbm = ParquetBlockModel.from_parquet(source_parquet, columns=["depth"])
    assert "x" not in pbm.columns
    idx = pbm.centroid_index
    assert idx.names == ["x", "y", "z"]
    assert len(idx) == 8


def test_centroid_index_cached_on_second_access(tmp_path):
    """centroid_index should be populated from the cache on second access."""
    pbm = _make_pbm(tmp_path)
    idx1 = pbm.centroid_index
    idx2 = pbm.centroid_index
    assert idx1 is idx2


# ===========================================================================
# sparsity property
# ===========================================================================


def test_sparsity_is_zero_for_dense_model(tmp_path):
    pbm = _make_pbm(tmp_path)
    assert pbm.sparsity == 0.0


def test_sparsity_nonzero_for_sparse_model(tmp_path):
    parquet_path = tmp_path / "sparse.parquet"
    blocks = _xyz_source(shape=(3, 3, 3))
    blocks.iloc[:-5].to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert 0.0 < pbm.sparsity < 1.0


def _xyz_df(shape=(2, 2, 2)):
    """Return a demo DataFrame with xyz MultiIndex as required by from_dataframe."""
    df = create_demo_blockmodel(shape=shape)
    return df.reset_index().set_index(["x", "y", "z"])


# ===========================================================================
# from_dataframe
# ===========================================================================


def test_from_dataframe_basic_roundtrip(tmp_path):
    df = _xyz_df()
    filename = tmp_path / "from_df.parquet"
    pbm = ParquetBlockModel.from_dataframe(df, filename)
    assert isinstance(pbm, ParquetBlockModel)
    assert pbm.geometry.local.shape == (2, 2, 2)
    assert len(pbm.attributes) > 0


def test_from_dataframe_rejects_wrong_index(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})  # no xyz index
    with pytest.raises(ValueError, match="MultiIndex"):
        ParquetBlockModel.from_dataframe(df, tmp_path / "bad.parquet")


def test_from_dataframe_rejects_non_parquet_filename(tmp_path):
    df = _xyz_df()
    with pytest.raises(ValueError, match=r"'.parquet'"):
        ParquetBlockModel.from_dataframe(df, tmp_path / "bad.pbm")


def test_from_dataframe_raises_if_pbm_exists_and_no_overwrite(tmp_path):
    df = _xyz_df()
    filename = tmp_path / "exists.parquet"
    ParquetBlockModel.from_dataframe(df, filename)
    with pytest.raises(FileExistsError):
        ParquetBlockModel.from_dataframe(df, filename, overwrite=False)


def test_from_dataframe_warns_on_unsorted_index(tmp_path):
    df = _xyz_df()
    df_shuffled = df.sample(frac=1, random_state=42)
    filename = tmp_path / "shuffled.parquet"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ParquetBlockModel.from_dataframe(df_shuffled, filename, overwrite=True)
    assert any("sorted" in str(w.message).lower() for w in caught)


def test_from_dataframe_with_explicit_geometry(tmp_path):
    df = _xyz_df()
    # Infer geometry via from_parquet on a temp file, then pass it explicitly
    tmp_parquet = tmp_path / "tmp_for_geom.parquet"
    df.reset_index().to_parquet(tmp_parquet, index=False)
    geometry = RegularGeometry.from_parquet(tmp_parquet)
    filename = tmp_path / "with_geom.parquet"
    pbm = ParquetBlockModel.from_dataframe(df, filename, geometry=geometry, name="explicit")
    assert pbm.name == "explicit"
    assert pbm.geometry.local.shape == (2, 2, 2)


# ===========================================================================
# from_csv
# ===========================================================================


def test_from_csv_basic_roundtrip(tmp_path):
    df = _xyz_df().reset_index()
    csv_path = tmp_path / "from_csv.csv"
    df.to_csv(csv_path, index=False)

    pbm = ParquetBlockModel.from_csv(csv_path)
    assert isinstance(pbm, ParquetBlockModel)
    assert pbm.geometry.local.shape == (2, 2, 2)
    assert pbm.blockmodel_path == csv_path.with_suffix(".pbm")


def test_from_csv_rejects_missing_xyz_columns(tmp_path):
    csv_path = tmp_path / "missing_xyz.csv"
    pd.DataFrame({"x": [1.0], "y": [2.0], "grade": [3.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        ParquetBlockModel.from_csv(csv_path)


# ===========================================================================
# from_geometry
# ===========================================================================


def test_from_geometry_creates_dense_grid(tmp_path):
    geometry = RegularGeometry(
        local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=(3, 3, 3)),
        world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1)),
    )
    pbm_path = tmp_path / "from_geom.pbm"
    pbm = ParquetBlockModel.from_geometry(geometry, pbm_path, name="skeleton")

    assert pbm.name == "skeleton"
    assert pbm.geometry.local.shape == (3, 3, 3)
    assert not pbm.is_sparse
    assert "x" in pbm.columns
    assert "block_id" in pbm.columns


# ===========================================================================
# from_parquet overwrite
# ===========================================================================


def test_from_parquet_overwrite_existing_pbm(tmp_path):
    source = tmp_path / "source.parquet"
    _xyz_source().to_parquet(source, index=False)
    pbm1 = ParquetBlockModel.from_parquet(source)
    pbm_path = pbm1.blockmodel_path

    with pytest.raises(ValueError, match="overwrite"):
        ParquetBlockModel.from_parquet(pbm_path, overwrite=False)

    pbm2 = ParquetBlockModel.from_parquet(pbm_path, overwrite=True)
    pd.testing.assert_frame_equal(pbm1.read(index=None), pbm2.read(index=None))
    assert pbm_path.rename(tmp_path / "moved.pbm").exists()


@pytest.mark.parametrize("indexed", [False, True])
def test_from_parquet_rejects_ambiguous_roundtrip_with_working_fix(tmp_path, monkeypatch, indexed):
    pbm = _make_pbm(tmp_path)
    df = pbm.read(index=None)
    if indexed:
        df = df.set_index(["i", "j", "k"])
    source = tmp_path / "mutated.parquet"
    df.to_parquet(source)

    with pytest.raises(ValueError, match=r"found block_id, world_id, i, j, k alongside x, y, z") as exc:
        ParquetBlockModel.from_parquet(source, columns=["depth"])

    message = str(exc.value)
    assert 'df = df.drop(columns=["block_id", "world_id", "i", "j", "k"])' in message
    assert not source.with_suffix(".pbm").exists()
    monkeypatch.chdir(tmp_path)
    recipe = message.split("\n\n")[-1]
    exec("\n".join(line.strip() for line in recipe.splitlines()), {"pd": pd})
    clean = ParquetBlockModel.from_parquet(tmp_path / "mutated_clean.parquet")
    assert clean._num_rows == pbm._num_rows


def test_from_parquet_fix_without_xyz_preserves_geometry_metadata(tmp_path, monkeypatch):
    pbm = _make_pbm(tmp_path)
    source = tmp_path / "positions.parquet"
    table = pq.read_table(pbm.blockmodel_path)
    pq.write_table(table.select(["block_id", "world_id", "depth"]), source)

    with pytest.raises(ValueError, match=r"found world_id alongside block_id") as exc:
        ParquetBlockModel.from_parquet(source)

    monkeypatch.chdir(tmp_path)
    recipe = str(exc.value).split("\n\n")[-1]
    exec("\n".join(line.strip() for line in recipe.splitlines()), {"pd": pd})
    clean = ParquetBlockModel.from_parquet(tmp_path / "positions_clean.parquet")
    assert clean.geometry.local == pbm.geometry.local
    assert clean._num_rows == pbm._num_rows


@pytest.mark.parametrize("positions", [["block_id"], ["world_id"], ["i", "j", "k"], ["x", "y", "z"]])
def test_from_parquet_accepts_each_single_positional_representation(tmp_path, positions):
    original = _make_pbm(tmp_path)
    source = tmp_path / "source.parquet"
    table = pq.read_table(original.blockmodel_path)
    pq.write_table(table.select([*positions, *original.attributes]), source)

    pbm = ParquetBlockModel.from_parquet(source, chunk_size=4)
    assert pbm._num_rows == 27
    renamed = pbm.blockmodel_path.rename(tmp_path / "renamed.pbm")
    renamed.rename(pbm.blockmodel_path)
    assert list(map(len, pbm._iter_batches(["block_id"], batch_size=10))) == [10, 10, 7]
    pd.testing.assert_frame_equal(original.read(index=None), pbm.read(index=None))
    assert pbm.blockmodel_path.rename(tmp_path / "renamed.pbm").exists()


def test_write_and_compress_refresh_row_count(tmp_path):
    pbm = _make_pbm(tmp_path)
    df = pbm.read(index=None)
    pbm.write(df)
    assert pbm._num_rows == len(df)

    # Replace the backing file with a sparse subset before rewriting it.
    table = pq.read_table(pbm.blockmodel_path).slice(0, 20)
    pq.write_table(table, pbm.blockmodel_path)
    pbm.compress()
    assert pbm._num_rows == 20
    assert pbm.sparsity == pytest.approx(7 / 27)


# ===========================================================================
# validate_xyz_parquet
# ===========================================================================


def test_validate_xyz_parquet_returns_geometry(tmp_path):
    source = tmp_path / "xyz.parquet"
    create_demo_blockmodel(shape=(2, 3, 4), block_size=(2.0, 1.0, 0.5)).reset_index()[
        ["x", "y", "z", "depth"]
    ].to_parquet(source, index=False)

    geometry = ParquetBlockModel.validate_xyz_parquet(source)
    assert geometry.local.shape == (2, 3, 4)
    assert geometry.local.block_size == (2.0, 1.0, 0.5)


# ===========================================================================
# triangulate invalid attribute
# ===========================================================================


def test_triangulate_raises_on_invalid_attribute(tmp_path):
    pbm = _make_pbm(tmp_path, shape=(2, 2, 2))
    with pytest.raises(ValueError, match="not found"):
        pbm.triangulate(attributes=["nonexistent_attr"])


def test_triangulate_with_valid_attribute_produces_mesh(tmp_path):
    pbm = _make_pbm(tmp_path, shape=(2, 2, 2))
    mesh = pbm.triangulate(attributes=["depth"], surface_only=False, sparse=False)
    assert "depth" in mesh.vertex_attributes
    assert mesh.n_vertices > 0


# ===========================================================================
# to_glb validation
# ===========================================================================


def test_to_glb_raises_when_texture_attribute_not_in_pbm_attributes(tmp_path):
    """to_glb with texture_attribute that does not exist in the model should raise at triangulation."""
    pbm = _make_pbm(tmp_path, shape=(2, 2, 2))
    glb_path = tmp_path / "bad_texture.glb"
    # "nonexistent_attr" is not in pbm.attributes at all → triangulate raises first
    with pytest.raises(ValueError, match="not found"):
        pbm.to_glb(glb_path, attributes=["nonexistent_attr"], texture_attribute="nonexistent_attr")


def test_to_glb_auto_includes_texture_attribute(tmp_path):
    """texture_attribute supplied without attributes list should be auto-added."""
    pbm = _make_pbm(tmp_path, shape=(2, 2, 2))
    glb_path = tmp_path / "auto_texture.glb"
    result = pbm.to_glb(glb_path, attributes=None, texture_attribute="depth")
    assert result == glb_path
    assert glb_path.exists()


# ===========================================================================
# read() with index=None and invalid index
# ===========================================================================


def test_read_with_index_none_returns_flat_dataframe(tmp_path):
    pbm = _make_pbm(tmp_path)
    df = pbm.read(columns=["depth"], index=None)
    assert df.index.names == [None]  # default RangeIndex
    assert "depth" in df.columns


def test_read_with_invalid_index_raises(tmp_path):
    pbm = _make_pbm(tmp_path)
    with pytest.raises(ValueError, match="index must be"):
        pbm.read(index="bad_value")  # type: ignore[arg-type]


def test_read_dense_ijk_reindexes_to_full_grid(tmp_path):
    parquet_path = tmp_path / "sparse_read.parquet"
    blocks = _xyz_source(shape=(3, 3, 3))
    blocks.iloc[:-3].to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert pbm.is_sparse

    df = pbm.read(columns=["depth"], index="ijk", dense=True)
    assert len(df) == 27  # full dense grid


# ===========================================================================
# _write_canonical_pbm column validation
# ===========================================================================


def test_write_canonical_pbm_raises_on_missing_requested_column(tmp_path):
    """_write_canonical_pbm with a requested column absent from source should raise."""
    source = tmp_path / "source_col.parquet"
    _xyz_source().to_parquet(source, index=False)

    with pytest.raises(ValueError, match="not present"):
        ParquetBlockModel.from_parquet(source, columns=["nonexistent_col"])


# ===========================================================================
# _validate_and_load_data legacy helper
# ===========================================================================


def test_validate_and_load_data_warns_without_xyz_if_count_matches():
    df = pd.DataFrame({"a": range(8)})  # no x/y/z, 8 rows = 2×2×2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ParquetBlockModel._validate_and_load_data(df, expected_num_blocks=8)
    assert result is df
    assert any("without x, y, z" in str(w.message).lower() for w in caught)


def test_validate_and_load_data_raises_without_xyz_if_count_mismatch():
    df = pd.DataFrame({"a": range(5)})
    with pytest.raises(ValueError, match="missing x, y, z"):
        ParquetBlockModel._validate_and_load_data(df, expected_num_blocks=8)


# ===========================================================================
# repr
# ===========================================================================


def test_repr_contains_name_and_path(tmp_path):
    pbm = _make_pbm(tmp_path, name="my_model")
    r = repr(pbm)
    assert "my_model" in r
    assert ".pbm" in r

