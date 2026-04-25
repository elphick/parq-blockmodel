import numpy as np
import pytest

from parq_blockmodel import ParquetBlockModel


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
        pbm.upsample((0.5, 0.5, 0.5), {"depth": {"method": "linear"}})


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
        {
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

