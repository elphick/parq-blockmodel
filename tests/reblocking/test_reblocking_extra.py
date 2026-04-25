"""Additional reblocking tests to improve coverage of reblocking.py"""
import pytest
from parq_blockmodel.reblocking.reblocking import _validate_params


def test_validate_params_bad_block_size_not_tuple():
    with pytest.raises(ValueError, match="tuple of three floats"):
        _validate_params({}, [1.0, 2.0, 3.0])


def test_validate_params_bad_block_size_wrong_length():
    with pytest.raises(ValueError, match="tuple of three floats"):
        _validate_params({}, (1.0, 2.0))


def test_validate_params_non_positive_dimension():
    with pytest.raises(ValueError, match="positive"):
        _validate_params({}, (1.0, 0.0, 1.0))


def test_validate_params_config_not_dict():
    with pytest.raises(ValueError, match="dictionary"):
        _validate_params("not_a_dict", (1.0, 1.0, 1.0))


def test_validate_params_valid():
    # Should not raise
    _validate_params({"attr": {"method": "mean"}}, (5.0, 5.0, 5.0))


def test_downsample_blockmodel_config_must_be_dict_of_dicts(tmp_path):
    from parq_blockmodel import ParquetBlockModel
    from parq_blockmodel.reblocking.reblocking import downsample_blockmodel

    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "ds_config.parquet", shape=(4, 4, 4)
    )
    # flat string values instead of dicts
    with pytest.raises(ValueError, match="dict of dicts"):
        downsample_blockmodel(pbm, (20.0, 20.0, 20.0), {"au": "mean"})


def test_upsample_blockmodel_config_not_str_values(tmp_path):
    from parq_blockmodel import ParquetBlockModel
    from parq_blockmodel.reblocking.reblocking import upsample_blockmodel

    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "us_config.parquet", shape=(4, 4, 4)
    )
    # block_size (1,1,1) → new (0.5,0.5,0.5) → factor 0.5 → upsampling branch
    # dict values instead of strings → should raise "interpolation methods"
    with pytest.raises(ValueError, match="interpolation methods"):
        upsample_blockmodel(pbm, (0.5, 0.5, 0.5), {"au": {"method": "linear"}})


def test_downsample_blockmodel_implies_upsample_raises(tmp_path):
    from parq_blockmodel import ParquetBlockModel
    from parq_blockmodel.reblocking.reblocking import downsample_blockmodel

    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "ds_up.parquet", shape=(4, 4, 4)
    )
    # block_size (1,1,1) → new (0.5,0.5,0.5) → factor 0.5 → implies upsampling
    # via downsample_blockmodel should raise
    with pytest.raises(ValueError, match="upsampling"):
        downsample_blockmodel(pbm, (0.5, 0.5, 0.5), {})


def test_upsample_blockmodel_implies_downsample_raises(tmp_path):
    from parq_blockmodel import ParquetBlockModel
    from parq_blockmodel.reblocking.reblocking import upsample_blockmodel

    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "us_down.parquet", shape=(4, 4, 4)
    )
    # larger block size → downsampling via upsample_blockmodel should raise
    with pytest.raises(ValueError, match="downsampling"):
        upsample_blockmodel(pbm, (20.0, 20.0, 20.0), {})


