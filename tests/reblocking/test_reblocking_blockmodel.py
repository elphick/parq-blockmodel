import numpy as np
import pytest

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_demo_blockmodel


@pytest.mark.integration
def test_downsample_blockmodel_rejects_non_nested_config(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "downsample_invalid_config.parquet",
        shape=(4, 4, 4),
    )

    with pytest.raises(ValueError, match="dict of dicts"):
        pbm.downsample((2.0, 2.0, 2.0), {"depth": "mean"})


@pytest.mark.integration
def test_upsample_blockmodel_rejects_non_string_methods(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_invalid_config.parquet",
        shape=(2, 2, 2),
    )

    with pytest.raises(ValueError, match="dict of interpolation methods"):
        pbm.upsample((0.5, 0.5, 0.5), upsample_config={"depth": {"method": "linear"}})


@pytest.mark.integration
def test_upsample_blockmodel_requires_config_for_every_attribute(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_missing_attr_config.parquet",
        shape=(2, 2, 2),
    )

    with pytest.raises(ValueError, match="must specify a method for every attribute"):
        pbm.upsample((0.5, 0.5, 0.5), upsample_config={"depth": "linear"})


@pytest.mark.integration
def test_downsample_blockmodel_builds_expected_geometry_and_block_ids(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "downsample_success.parquet",
        shape=(4, 4, 4),
    )

    downsampled = pbm.downsample(
        (2.0, 2.0, 2.0),
        {
            "depth": {"method": "mean"},
            "depth_category": {"method": "mode"},
        },
    )

    assert downsampled.geometry.local.shape == (2, 2, 2)
    assert downsampled.geometry.local.block_size == (2.0, 2.0, 2.0)

    df = downsampled.read(columns=["block_id", "depth"], index="ijk", dense=True)
    assert len(df) == 8
    np.testing.assert_array_equal(df["block_id"].to_numpy(), np.arange(8, dtype=np.uint32))


@pytest.mark.integration
def test_upsample_blockmodel_builds_expected_geometry_and_block_ids(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_success.parquet",
        shape=(2, 2, 2),
    )

    upsampled = pbm.upsample(
        (0.5, 0.5, 0.5),
        upsample_config={
            "depth": "linear",
            "depth_category": "nearest",
        },
    )

    assert upsampled.geometry.local.shape == (4, 4, 4)
    assert upsampled.geometry.local.block_size == (0.5, 0.5, 0.5)

    df = upsampled.read(columns=["block_id", "depth"], index="ijk", dense=True)
    assert len(df) == 64
    assert df["block_id"].is_unique
    np.testing.assert_array_equal(df["block_id"].to_numpy(), np.arange(64, dtype=np.uint32))


@pytest.mark.integration
def test_upsample_blockmodel_allows_mode_for_class_attributes(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_mode_for_class.parquet",
        shape=(2, 2, 2),
    )

    upsampled = pbm.upsample(
        (0.5, 0.5, 0.5),
        upsample_config={
            "depth": "linear",
            "depth_category": "mode",
        },
    )

    out = upsampled.read(columns=["depth_category"], index="ijk", dense=True)
    assert out["depth_category"].notna().all()
    assert set(out["depth_category"].astype(str).unique()) <= {"shallow", "deep"}


@pytest.mark.integration
def test_upsample_blockmodel_preserves_float32_attribute_dtype(tmp_path):
    df = create_demo_blockmodel(shape=(2, 2, 2), index_type="world_centroids")
    df["depth"] = df["depth"].astype(np.float32)
    pbm = ParquetBlockModel.from_dataframe(df[["depth"]], tmp_path / "upsample_float32.parquet")
    pbm.geometry.world_id_encoding = None

    upsampled = pbm.upsample((0.5, 0.5, 0.5), upsample_config={"depth": "linear"})
    out = upsampled.read(columns=["depth"], index="ijk", dense=True)
    assert out["depth"].dtype == np.float32


@pytest.mark.integration
def test_upsample_blockmodel_parent_inherits_parent_blocks(tmp_path):
    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_parent_method.parquet",
        shape=(2, 2, 2),
    )

    upsampled = pbm.upsample(
        (0.5, 0.5, 0.5),
        upsample_config={
            "depth": "parent",
            "depth_category": "parent",
        },
    )

    coarse = pbm.read(columns=["depth"], index="ijk", dense=True)["depth"].to_numpy().reshape(2, 2, 2, order="C")
    fine = upsampled.read(columns=["depth"], index="ijk", dense=True)["depth"].to_numpy().reshape(4, 4, 4, order="C")
    expected = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(fine, expected)


@pytest.mark.integration
def test_downsample_blockmodel_preserves_float32_attribute_dtype(tmp_path):
    df = create_demo_blockmodel(shape=(4, 4, 4), index_type="world_centroids")
    df["depth"] = df["depth"].astype(np.float32)
    pbm = ParquetBlockModel.from_dataframe(df[["depth"]], tmp_path / "downsample_float32.parquet")
    pbm.geometry.world_id_encoding = None

    downsampled = pbm.downsample((2.0, 2.0, 2.0), {"depth": {"method": "mean"}})
    out = downsampled.read(columns=["depth"], index="ijk", dense=True)
    assert out["depth"].dtype == np.float32


@pytest.mark.integration
def test_downsample_blockmodel_accepts_calculated_basis_and_target(tmp_path):
    pytest.importorskip("df_eval", reason="df-eval not installed")
    pandera = pytest.importorskip("pandera", reason="pandera not installed")
    DataFrameSchema = pandera.DataFrameSchema
    Column = pandera.Column

    df = create_demo_blockmodel(shape=(4, 4, 4), index_type="world_centroids")
    count = len(df)
    df["grade"] = np.linspace(0.2, 1.2, count)
    df["density"] = np.linspace(2.0, 3.0, count)

    schema = DataFrameSchema(
        columns={
            "grade": Column(float, coerce=True, nullable=True),
            "density": Column(float, coerce=True, nullable=True),
            "tonnes": Column(
                float,
                coerce=True,
                nullable=True,
                required=False,
                metadata={"df-eval": {"expr": "density * volume"}},
            ),
            "metal": Column(
                float,
                coerce=True,
                nullable=True,
                required=False,
                metadata={"df-eval": {"expr": "tonnes * grade"}},
            ),
        },
        strict=False,
    )

    pbm = ParquetBlockModel.from_dataframe(
        df[["grade", "density"]],
        filename=tmp_path / "downsample_calculated_inputs.parquet",
        schema=schema,
    )

    downsampled = pbm.downsample(
        (2.0, 2.0, 2.0),
        {
            "grade": {"method": "mean"},
            "metal": {"method": "weighted_mean", "basis": "tonnes"},
        },
    )
    out = downsampled.read(columns=["grade", "metal"], index="ijk", dense=True)
    assert "tonnes" not in out.columns

    source = pbm.read(columns=["grade", "tonnes", "metal"], index="ijk", dense=True)
    shape = (4, 4, 4)
    coarse_shape = (2, 2, 2)
    reshape_shape = (coarse_shape[0], 2, coarse_shape[1], 2, coarse_shape[2], 2)
    transpose_axes = (0, 2, 4, 1, 3, 5)

    grade = source["grade"].to_numpy().reshape(shape, order="C").reshape(reshape_shape).transpose(transpose_axes)
    tonnes = source["tonnes"].to_numpy().reshape(shape, order="C").reshape(reshape_shape).transpose(transpose_axes)
    metal = source["metal"].to_numpy().reshape(shape, order="C").reshape(reshape_shape).transpose(transpose_axes)

    expected_grade = np.nanmean(grade, axis=(3, 4, 5))
    weighted_sum = np.nansum(metal * tonnes, axis=(3, 4, 5))
    total_tonnes = np.nansum(tonnes, axis=(3, 4, 5))
    expected_metal = np.divide(
        weighted_sum,
        total_tonnes,
        out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
        where=total_tonnes != 0,
    )

    np.testing.assert_allclose(out["grade"].to_numpy().reshape(coarse_shape, order="C"), expected_grade)
    np.testing.assert_allclose(out["metal"].to_numpy().reshape(coarse_shape, order="C"), expected_metal)


@pytest.mark.integration
def test_downsample_blockmodel_calculated_basis_requires_schema(tmp_path):
    df = create_demo_blockmodel(shape=(4, 4, 4), index_type="world_centroids")
    df["grade"] = np.linspace(0.2, 1.2, len(df))
    df["density"] = np.linspace(2.0, 3.0, len(df))

    pbm = ParquetBlockModel.from_dataframe(
        df[["grade", "density"]],
        filename=tmp_path / "downsample_missing_schema_inputs.parquet",
    )

    with pytest.raises(ValueError, match="defined by available df-eval operations|calculated"):
        pbm.downsample(
            (2.0, 2.0, 2.0),
            {
                "grade": {"method": "weighted_mean", "basis": "tonnes"},
                "density": {"method": "weighted_mean", "basis": "volume"},
            },
        )


@pytest.mark.integration
def test_downsample_blockmodel_intrinsic_volume_basis_without_schema(tmp_path):
    df = create_demo_blockmodel(shape=(4, 4, 4), index_type="world_centroids")
    df["density"] = np.linspace(2.0, 3.0, len(df))

    pbm = ParquetBlockModel.from_dataframe(
        df[["density"]],
        filename=tmp_path / "downsample_intrinsic_volume_basis.parquet",
    )

    downsampled = pbm.downsample(
        (2.0, 2.0, 2.0),
        {
            "density": {"method": "weighted_mean", "basis": "volume"},
        },
    )
    out = downsampled.read(columns=["density"], index="ijk", dense=True)

    source = pbm.read(columns=["density"], index="ijk", dense=True)["density"].to_numpy().reshape(4, 4, 4, order="C")
    expected = source.reshape(2, 2, 2, 2, 2, 2).transpose(0, 2, 4, 1, 3, 5).mean(axis=(3, 4, 5))
    np.testing.assert_allclose(out["density"].to_numpy().reshape(2, 2, 2, order="C"), expected)
