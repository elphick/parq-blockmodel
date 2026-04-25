"""PR3: ParquetBlockModel wrapper integration contract tests.

Targets uncovered branches in parq_blockmodel/blockmodel.py:
- Constructor guard (non-.pbm suffix)
- centroid_index property (xyz-column and derived-from-block_id paths)
- sparsity / index_c / index_f properties
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


# ===========================================================================
# Constructor guard
# ===========================================================================


def test_constructor_rejects_non_pbm_extension(tmp_path):
    fake = tmp_path / "model.parquet"
    fake.touch()
    with pytest.raises(ValueError, match=r"\.pbm"):
        ParquetBlockModel(fake)


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
    create_demo_blockmodel(shape=(2, 2, 2)).reset_index().to_parquet(source_parquet, index=False)
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
# sparsity / index_c / index_f properties
# ===========================================================================


def test_sparsity_is_zero_for_dense_model(tmp_path):
    pbm = _make_pbm(tmp_path)
    assert pbm.sparsity == 0.0


def test_sparsity_nonzero_for_sparse_model(tmp_path):
    parquet_path = tmp_path / "sparse.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 3)).reset_index()
    blocks.iloc[:-5].to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)
    assert 0.0 < pbm.sparsity < 1.0


def test_index_c_and_index_f_shape(tmp_path):
    pbm = _make_pbm(tmp_path, shape=(2, 3, 4))
    total = 2 * 3 * 4
    assert pbm.index_c.shape == (total,)
    assert pbm.index_f.shape == (total,)
    # C-order and F-order differ (non-cube grid)
    assert not np.array_equal(pbm.index_c, pbm.index_f)


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
    """from_parquet on a .pbm path with overwrite=False must raise ValueError."""
    source = tmp_path / "source.parquet"
    create_demo_blockmodel(shape=(2, 2, 2)).reset_index().to_parquet(source, index=False)
    pbm1 = ParquetBlockModel.from_parquet(source)
    pbm_path = pbm1.blockmodel_path  # the .pbm file

    # Calling from_parquet on the .pbm with overwrite=False should raise
    with pytest.raises(ValueError, match="overwrite"):
        ParquetBlockModel.from_parquet(pbm_path, overwrite=False)


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
    blocks = create_demo_blockmodel(shape=(3, 3, 3)).reset_index()
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
    create_demo_blockmodel(shape=(2, 2, 2)).reset_index().to_parquet(source, index=False)

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







